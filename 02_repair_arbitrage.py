# 02 Static-arbitrage repair
#
# Repairs the European-equivalent quotes date by date with the linear-programming method of
# Cohen, Reisinger and Wang (2020) (`arbitragerepair`). Calls are repaired directly; puts are converted to
# call-equivalent prices with put-call parity, repaired, and converted back.

# %%
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from arbitragerepair import constraints, repair
from cvxopt import solvers

solvers.options["glpk"] = {"msg_lev": "GLP_MSG_OFF"}

OPTIONS_DIR = Path("options_dir")
TICKERS = ["AMZN", "GOOGL", "JNJ", "JPM", "MSFT", "PG", "TSLA", "V"]

MIN_VOLUME = 1
MIN_DTE_BUS, MAX_DTE_BUS = 5, 252
MAX_REL_SPREAD = 2.0
FLOAT_EPS = np.finfo(float).eps

# %% [markdown]
# ## Quote preparation and filters

# %%
def prepare_option_base(opt):
    opt = opt.copy()
    opt["ticker"] = opt["ticker"].astype(str).str.upper()
    opt["cp_flag"] = opt["cp_flag"].astype(str).str.upper()
    opt["date"] = pd.to_datetime(opt["date"], errors="coerce").dt.normalize()
    opt["exdate"] = pd.to_datetime(opt["exdate"], errors="coerce").dt.normalize()

    opt["bid"] = pd.to_numeric(opt["bid_european"], errors="coerce")
    opt["ask"] = pd.to_numeric(opt["ask_european"], errors="coerce")
    opt["mid_price"] = pd.to_numeric(opt["mid_european"], errors="coerce")
    opt["C_mid"] = np.where(opt["cp_flag"].eq("C"), opt["mid_price"], np.nan)
    opt["P_mid"] = np.where(opt["cp_flag"].eq("P"), opt["mid_price"], np.nan)

    for col in ["K", "F", "dte_bus", "tau_cal_365", "r"]:
        opt[col] = pd.to_numeric(opt[col], errors="coerce")
    opt["dte"] = opt["dte_bus"]
    opt["T"] = opt["tau_cal_365"]
    if opt["r"].isna().any():
        raise ValueError("Missing risk-free rates in the option base table.")
    opt["DF"] = np.exp(-opt["r"] * opt["T"])
    opt["volume"] = pd.to_numeric(opt["volume"], errors="coerce").fillna(0.0)
    return opt


def prep_side(opt, ticker, side, price_col):
    """Quote filters for one option side, then one quote per (date, expiry, strike)."""
    x = opt[opt["ticker"].eq(ticker) & opt["cp_flag"].eq(side)].copy()
    x["vol"] = x["volume"]
    x["spread"] = (x["ask"] - x["bid"]).clip(lower=0.0)
    x[price_col] = pd.to_numeric(x[price_col], errors="coerce")

    x = x[
        x["date"].notna() & x["exdate"].notna()
        & np.isfinite(x["T"]) & np.isfinite(x["K"]) & np.isfinite(x["F"]) & np.isfinite(x["DF"])
        & np.isfinite(x["bid"]) & np.isfinite(x["ask"]) & np.isfinite(x[price_col])
        & (x["T"] > 0) & (x["K"] > 0) & (x["F"] > 0)
        & (x["bid"] >= 0) & (x["ask"] >= x["bid"]) & (x["ask"] > 0) & (x[price_col] > 0)
    ].copy()

    x = x[x["vol"] >= MIN_VOLUME]
    x = x[(x["dte"] >= MIN_DTE_BUS) & (x["dte"] <= MAX_DTE_BUS)]

    # no-arbitrage bounds
    if side == "C":
        lower = x["DF"] * np.maximum(x["F"] - x["K"], 0.0)
        upper = x["DF"] * x["F"]
    else:
        lower = x["DF"] * np.maximum(x["K"] - x["F"], 0.0)
        upper = x["DF"] * x["K"]
    x = x[(x[price_col] >= lower) & (x[price_col] <= upper)]

    rel_spread = x["spread"] / x[price_col].replace(0, np.nan)
    x = x[np.isfinite(rel_spread) & (rel_spread <= MAX_REL_SPREAD)].copy()

    # duplicates: keep the tighter spread, then the higher volume
    x["K_r"] = x["K"].round(4)
    x = (
        x.sort_values(["date", "exdate", "K_r", "spread", "vol"], ascending=[True, True, True, True, False])
        .drop_duplicates(["date", "exdate", "K_r"], keep="first")
        .reset_index(drop=True)
    )
    return x

# %% [markdown]
# ## LP repair
#
# Prices are undiscounted (divided by the discount factor) before `Normalise`, as required by
# `arbitragerepair`.

# %%
def as_epsilon(eps, n):
    if eps is None or (isinstance(eps, list) and len(eps) == 0):
        return np.zeros(n)
    eps = np.asarray(eps, dtype=float).reshape(-1)
    if eps.size == 0:
        return np.zeros(n)
    if eps.size != n:
        raise ValueError(f"eps size {eps.size} != {n}")
    return eps


def lp_repair(T, K, F, C, bid, ask):
    normaliser = constraints.Normalise(min_price=FLOAT_EPS)
    normaliser.fit(T, K, C, F)
    T1, K1, C1 = normaliser.transform(T, K, C)
    _, _, bid1 = normaliser.transform(T, K, bid)
    _, _, ask1 = normaliser.transform(T, K, ask)

    # bid-ask tolerances in the normalised scale
    floor = np.spacing(np.maximum(1.0, np.abs(C1)))
    d_ask = np.where(np.isfinite(ask1 - C1) & (ask1 - C1 > 0), ask1 - C1, floor)
    d_bid = np.where(np.isfinite(C1 - bid1) & (C1 - bid1 > 0), C1 - bid1, floor)

    mat_A, vec_b, _, _ = constraints.detect(T1, K1, C1, verbose=False)
    if np.shape(mat_A)[0] == 0:
        return None, "no_violation"

    eps = None
    try:
        e = as_epsilon(repair.l1ba(mat_A, vec_b, C1, spread=[d_ask, d_bid]), len(C1))
        if np.any(np.abs(e) > 0):
            eps = e
    except Exception as err:
        print("  l1ba failed:", err)

    if eps is None:
        try:
            eps = as_epsilon(repair.l1(mat_A, vec_b, C1), len(C1))
        except Exception as err:
            print("  l1 failed:", err)
            return None, "solver_failed"

    _, C0 = normaliser.inverse_transform(K1, C1 + eps)
    repaired = C.copy()
    repaired[normaliser._order_mask] = C0
    return repaired, "repaired"


def _repair_date(T, K, F, DF, C, bid, ask):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return lp_repair(T, K, F, C / DF, bid / DF, ask / DF)


def repair_calls(opt, ticker):
    calls = prep_side(opt, ticker, "C", "C_mid")
    calls["C_rep"] = np.nan
    calls["repair_status"] = ""

    for _, g in calls.groupby("date", sort=False):
        g = g.sort_values(["T", "K"])
        T, K, F, DF = (g[c].to_numpy(dtype=float) for c in ["T", "K", "F", "DF"])
        C, bid, ask = (g[c].to_numpy(dtype=float) for c in ["C_mid", "bid", "ask"])

        ok = (np.isfinite(T) & np.isfinite(K) & np.isfinite(F) & np.isfinite(DF)
              & np.isfinite(C) & np.isfinite(bid) & np.isfinite(ask)
              & (T > 0) & (K > 0) & (F > 0) & (DF > 0) & (ask >= bid))
        if not ok.all():
            calls.loc[g.index, "repair_status"] = "invalid_input"
            continue
        if len(g) < 3:
            calls.loc[g.index, "C_rep"] = C
            calls.loc[g.index, "repair_status"] = "too_few_keep_mid"
            continue

        try:
            rep, status = _repair_date(T, K, F, DF, C, bid, ask)
            # keep the midpoint when no repair is needed or the solver fails
            C_rep = C if rep is None else np.minimum(np.maximum(DF * rep, 0.0), DF * F)
        except Exception as err:
            C_rep, status = C, f"exception_keep_mid: {err}"

        calls.loc[g.index, "C_rep"] = C_rep
        calls.loc[g.index, "repair_status"] = status

    calls = calls[calls["C_rep"].notna()].copy()
    calls["price_clean"] = calls["C_rep"]
    return calls


def repair_puts(opt, ticker):
    # Puts are repaired as call-equivalents C_eq = P + DF (F - K)
    puts = prep_side(opt, ticker, "P", "P_mid")
    puts["P_rep"] = np.nan
    puts["repair_status"] = ""

    for d, g in puts.groupby("date", sort=False):
        g = g.sort_values(["T", "K"])
        T, K, F, DF = (g[c].to_numpy(dtype=float) for c in ["T", "K", "F", "DF"])
        P, bid_p, ask_p = (g[c].to_numpy(dtype=float) for c in ["P_mid", "bid", "ask"])

        shift = DF * (F - K)
        C, bid, ask = P + shift, bid_p + shift, ask_p + shift

        ok = (np.isfinite(T) & np.isfinite(K) & np.isfinite(F) & np.isfinite(DF)
              & np.isfinite(P) & np.isfinite(bid_p) & np.isfinite(ask_p)
              & np.isfinite(C) & np.isfinite(bid) & np.isfinite(ask)
              & (T > 0) & (K > 0) & (F > 0) & (DF > 0)
              & (ask_p >= bid_p) & (ask >= bid) & (C > 0) & (ask > 0))
        if not ok.all():
            puts.loc[g.index, "repair_status"] = "invalid_input"
            continue
        if len(g) < 3:
            puts.loc[g.index, "P_rep"] = P
            puts.loc[g.index, "repair_status"] = "too_few_keep_mid"
            continue

        try:
            rep, status = _repair_date(T, K, F, DF, C, bid, ask)
            if rep is None:
                P_rep = P
            else:
                # bounds on C_eq that keep the converted put non-negative
                C_rep = np.minimum(np.maximum(DF * rep, np.maximum(0.0, shift)), DF * F)
                P_rep = np.minimum(np.maximum(C_rep - shift, 0.0), DF * K)
        except Exception as err:
            P_rep, status = P, f"exception_keep_mid: {err}"

        puts.loc[g.index, "P_rep"] = P_rep
        puts.loc[g.index, "repair_status"] = status

    puts = puts[puts["P_rep"].notna()].copy()
    puts["price_clean"] = puts["P_rep"]
    return puts

# %%
def repair_ticker(ticker):
    opt = prepare_option_base(pd.read_parquet(OPTIONS_DIR / f"{ticker}_option_base.parquet"))

    calls = repair_calls(opt, ticker)
    puts = repair_puts(opt, ticker)

    out = pd.concat([calls, puts], ignore_index=True, sort=False)
    out = out[np.isfinite(out["price_clean"]) & (out["price_clean"] >= 0)]
    out = out.sort_values(["date", "exdate", "cp_flag", "K"]).reset_index(drop=True)

    out.to_parquet(OPTIONS_DIR / f"{ticker}_options_repaired.parquet", index=False)
    out.to_csv(OPTIONS_DIR / f"{ticker}_calls_and_puts_repaired.csv", index=False)
    print(f"{ticker}: {len(calls):,} calls, {len(puts):,} puts")

# %%
for ticker in TICKERS:
    repair_ticker(ticker)
