# 04 Downside-risk premium
#
# For every stock, date and expiry:
# 1. Q_down: risk-neutral probability of S_T <= K_thr, with K_thr = F exp(-2 sigma_ATM sqrt(tau)).
# 2. P_down: the same probability under the physical-distribution proxy, obtained by exponential tilting of
#    the risk-neutral masses to match the historical mean and variance of log(S_T/F).
# 3. DRP = log((Q_down + eps) / (P_down + eps))

# %%
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import root

OPTIONS_DIR = Path("options_dir")
STOCK_DIR = Path("stocks_dir")
HIST_MOMENT_FILE = OPTIONS_DIR / "historical_moments_daily_by_ticker.csv"
TICKERS = ["AMZN", "GOOGL", "JNJ", "JPM", "MSFT", "PG", "TSLA", "V"]

TARGET_DTE = 20.0            # centre of the maturity gate (business days)
Z_SIGMA = 2.0                # threshold in ATM standard deviations
EPS = np.finfo(float).tiny   # numerical floor in the log ratio
LOOKBACK_DAYS = 252
MIN_PERIODS = 126

# %% [markdown]
# ## Historical moments

# %%
def historical_moments(path):
    df = pd.read_csv(path)
    df["ticker"] = df["ticker"].astype(str).str.upper()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["price"] = pd.to_numeric(df["close_yahoo_split_adjusted"], errors="coerce")
    df["spot_contract"] = pd.to_numeric(df["spot"], errors="coerce")
    df = df.dropna(subset=["ticker", "date", "price", "spot_contract"])
    df = df[(df["price"] > 0) & (df["spot_contract"] > 0)].sort_values(["ticker", "date"]).reset_index(drop=True)

    parts = []
    for _, g in df.groupby("ticker"):
        g = g.sort_values("date").copy()
        past = np.log(g["price"] / g["price"].shift(1)).shift(1)
        g["mu_daily"] = past.rolling(LOOKBACK_DAYS, min_periods=MIN_PERIODS).mean()
        g["var_daily"] = past.rolling(LOOKBACK_DAYS, min_periods=MIN_PERIODS).var(ddof=1)
        parts.append(g[["ticker", "date", "mu_daily", "var_daily", "spot_contract"]])

    out = pd.concat(parts, ignore_index=True).dropna(subset=["mu_daily", "var_daily"])
    return out[out["var_daily"] > 0]


hist_moments = pd.concat(
    [historical_moments(f) for f in sorted(STOCK_DIR.glob("*_stock.csv"))], ignore_index=True
)
hist_moments.to_csv(HIST_MOMENT_FILE, index=False)
print("Historical moments:", len(hist_moments), "rows")

# %% [markdown]
# ## Risk-neutral masses and downside threshold

# %%
def prepare_slice(g, z_sigma=Z_SIGMA):
    # Discrete risk-neutral masses on the strike grid and the dynamic threshold K_thr
    g = g.dropna(subset=["ticker", "date", "exdate", "K", "F"])
    g = g[(g["K"] > 0) & (g["F"] > 0)]
    if len(g) < 2:
        return None
    g = g.sort_values("K").drop_duplicates(subset=["K"], keep="last")

    K = g["K"].to_numpy(dtype=float)
    F = float(np.nanmedian(g["F"].to_numpy(dtype=float)))
    if not np.isfinite(F) or F <= 0:
        return None
    p = g["risk_neutral_pdf_projected"].to_numpy(dtype=float) * g["integration_weight"].to_numpy(dtype=float)
    ok = np.isfinite(K) & np.isfinite(p) & (K > 0) & (p >= 0)
    K, p = K[ok], p[ok]
    if len(K) < 2 or not np.isfinite(p.sum()) or p.sum() <= 0:
        return None

    tau = float(np.nanmedian(g["tau"]))
    atm = pd.to_numeric(g["atm_iv"], errors="coerce")
    atm = atm[np.isfinite(atm) & (atm > 0)]
    atm_iv = float(np.nanmedian(atm)) if len(atm) else np.nan
    K_thr, alpha_dyn = dynamic_threshold(F, tau, atm_iv, z_sigma)

    meta = {
        "ticker": g["ticker"].iloc[0],
        "date": pd.to_datetime(g["date"].iloc[0]),
        "exdate": pd.to_datetime(g["exdate"].iloc[0]),
        "F": F,
        "DF": float(np.nanmedian(g["DF"])),
        "dte": float(np.nanmedian(g["dte"])),
        "tau": tau,
        "atm_iv": atm_iv,
        "K_thr": K_thr,
        "alpha_dyn": alpha_dyn,
        "n_support_points": int(len(K)),
    }
    return K, p / p.sum(), meta


def dynamic_threshold(F, tau, atm_iv, z_sigma=Z_SIGMA):
    if not (np.isfinite(F) and F > 0 and np.isfinite(tau) and tau > 0
            and np.isfinite(atm_iv) and atm_iv > 0 and np.isfinite(z_sigma) and z_sigma > 0):
        return np.nan, np.nan
    alpha_dyn = float(np.exp(-z_sigma * atm_iv * np.sqrt(tau)))
    return float(alpha_dyn * F), alpha_dyn


def attach_historical_moments(meta, hist):
    # Target moments of log(S_T/F): mu = d mu_daily - log(F/S_t), var = d var_daily
    parts = []
    for ticker, m in meta.groupby("ticker", dropna=False):
        m = m.sort_values("date").copy()
        hh = hist[hist["ticker"].astype(str).str.upper() == str(ticker).upper()].sort_values("date")
        if hh.empty:
            m["mu_hist"], m["var_hist"] = np.nan, np.nan
            parts.append(m)
            continue
        m = pd.merge_asof(m, hh[["date", "mu_daily", "var_daily", "spot_contract"]], on="date", direction="backward")
        ok = (np.isfinite(m["mu_daily"]) & np.isfinite(m["var_daily"]) & (m["var_daily"] > 0)
              & np.isfinite(m["spot_contract"]) & (m["spot_contract"] > 0)
              & np.isfinite(m["F"]) & (m["F"] > 0) & np.isfinite(m["dte"]) & (m["dte"] > 0))
        m["mu_hist"] = np.where(ok, m["mu_daily"] * m["dte"] - np.log(m["F"] / m["spot_contract"]), np.nan)
        m["var_hist"] = np.where(ok, m["var_daily"] * m["dte"], np.nan)
        parts.append(m.drop(columns=["mu_daily", "var_daily", "spot_contract"]))
    return pd.concat(parts, ignore_index=True)

# %% [markdown]
# ## Physical-distribution proxy
#
# p_P proportional to p_Q exp(theta x + eta x^2) with x = log(K/F); (theta, eta) solve the two moment conditions.

# %%
def moments(p, x):
    mu = float(np.sum(p * x))
    return mu, float(np.sum(p * (x - mu) ** 2))


def tilted_probs(p_q, x, theta, eta):
    logw = np.log(np.maximum(p_q, np.finfo(float).tiny)) + theta * x + eta * x**2
    w = np.exp(logw - np.nanmax(logw))
    s = np.sum(w)
    if not np.isfinite(s) or s <= 0:
        return np.full_like(p_q, np.nan, dtype=float)
    return w / s


def fit_exponential_tilt(p_q, x, target_mean, target_var):
    ok = np.isfinite(p_q) & np.isfinite(x) & (p_q >= 0)
    p_q, x = p_q[ok], x[ok]
    if len(p_q) < 2 or np.sum(p_q) <= 0:
        return None
    p_q = p_q / np.sum(p_q)
    if not np.isfinite(target_mean) or not np.isfinite(target_var) or target_var <= 0:
        return None
    if target_mean < float(np.min(x)) or target_mean > float(np.max(x)):
        return None

    def equations(params):
        pp = tilted_probs(p_q, x, *params)
        if np.any(~np.isfinite(pp)):
            return np.array([1e10, 1e10])
        mu, var = moments(pp, x)
        return np.array([mu - target_mean, var - target_var])

    best = None
    for start in ([0.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]):
        try:
            sol = root(equations, np.array(start))
            theta, eta = sol.x
            pp = tilted_probs(p_q, x, theta, eta)
            if np.any(~np.isfinite(pp)):
                continue
            mu, var = moments(pp, x)
            resid = float(np.linalg.norm(np.array([mu - target_mean, var - target_var], dtype=float)))
            if best is None or resid < best["resid"]:
                best = {"success": bool(sol.success and resid < 1e-6), "theta": float(theta),
                        "eta": float(eta), "p_p": pp, "resid": resid}
        except Exception:
            continue
    return best if best is not None and best["success"] else None

# %% [markdown]
# ## Downside probabilities with the Pareto tail

# %%
def empirical_prob(K, p, K_thr):
    if not np.isfinite(K_thr) or K_thr <= 0:
        return np.nan
    ok = np.isfinite(K) & np.isfinite(p) & (p >= 0) & (K > 0)
    K, p = K[ok], p[ok]
    if len(K) == 0 or np.sum(p) <= 0:
        return np.nan
    p = p / np.sum(p)
    return float(np.sum(p[K <= K_thr]))


def pareto_ok(row):
    return row.get("pareto_status") in {"ok", "ok_alpha_clipped"}


def q_down(K, p_q, F, K_thr, row, use_pareto=True):
    empirical = empirical_prob(K, p_q, K_thr)
    if not use_pareto or not pareto_ok(row):
        return empirical
    if not (np.isfinite(F) and np.isfinite(K_thr) and F > 0 and K_thr > 0):
        return empirical
    alpha, u, mass = row.get("pareto_alpha", np.nan), row.get("pareto_u", np.nan), row.get("pareto_tail_mass", np.nan)
    if not (np.isfinite(alpha) and np.isfinite(u) and np.isfinite(mass) and alpha > 0 and u > 0 and mass >= 0):
        return empirical
    ell = F / K_thr
    return float(mass * (u / ell) ** alpha) if ell >= u else empirical


def p_down(K, p_p, F, K_thr, row, use_pareto=True):
    # Same threshold u as the risk-neutral tail; mass and Hill index re-estimated under p_P
    empirical = empirical_prob(K, p_p, K_thr)
    if not use_pareto or not pareto_ok(row):
        return empirical
    u = row.get("pareto_u", np.nan)
    if not (np.isfinite(u) and u > 0 and np.isfinite(F) and F > 0 and np.isfinite(K_thr) and K_thr > 0):
        return empirical

    ok = np.isfinite(K) & np.isfinite(p_p) & (K > 0) & (p_p >= 0)
    K, p_p = K[ok], p_p[ok]
    if len(K) == 0 or np.sum(p_p) <= 0:
        return empirical
    p_p = p_p / np.sum(p_p)

    L, ell = F / K, F / K_thr
    if ell < u:
        return empirical
    tail = np.isfinite(L) & (L >= u)
    if tail.sum() < 2:
        return empirical
    pi_u = float(np.sum(p_p[tail]))
    if not np.isfinite(pi_u) or pi_u <= 0:
        return empirical
    denom = float(np.sum(p_p[tail] / pi_u * np.log(L[tail] / u)))
    if not np.isfinite(denom) or denom <= 0:
        return empirical
    return float(np.clip(pi_u * (u / ell) ** (1.0 / denom), 0.0, 1.0))


def premium_one_slice(row, K, p_q, use_pareto=True):
    F, K_thr = float(row["F"]), row["K_thr"]
    if not np.isfinite(K_thr) or K_thr <= 0:
        return None
    if not np.isfinite(row["mu_hist"]) or not np.isfinite(row["var_hist"]) or row["var_hist"] <= 0:
        return None

    tilt = fit_exponential_tilt(p_q, np.log(K / F), float(row["mu_hist"]), float(row["var_hist"]))
    if tilt is None:
        return None
    Q = q_down(K, p_q, F, K_thr, row, use_pareto)
    P = p_down(K, tilt["p_p"], F, K_thr, row, use_pareto)
    if not np.isfinite(Q) or not np.isfinite(P):
        return None

    keep = ["ticker", "date", "exdate", "dte", "tau", "F", "DF", "atm_iv", "K_thr", "alpha_dyn",
            "n_support_points", "pareto_status", "pareto_alpha", "pareto_u", "pareto_tail_mass"]
    out = {c: row.get(c, np.nan) for c in keep}
    out.update({
        "Q_crash": Q,
        "P_crash": P,
        "crash_premium": float(np.log((Q + EPS) / (P + EPS))),
        "theta": tilt["theta"],
        "eta": tilt["eta"],
        "mu_hist": row["mu_hist"],
        "var_hist": row["var_hist"],
    })
    return out

# %% [markdown]
# ## 20-day gated premium
#
# Probabilities are averaged across expiries with Gaussian weights centred on 20 business days; the
# bandwidth is the median spacing between that date's expiries.

# %%
def gate_bandwidth(dte):
    d = np.sort(dte[np.isfinite(dte)])
    unique = np.unique(d)
    if len(d) <= 1 or len(unique) <= 1:
        return np.inf
    spacing = np.diff(unique)
    spacing = spacing[np.isfinite(spacing) & (spacing > 0)]
    return float(np.nanmedian(spacing)) if len(spacing) else np.inf


def gaussian_gate_weights(dte, target):
    h = gate_bandwidth(dte)
    w = np.ones_like(dte, dtype=float) if not np.isfinite(h) or h <= 0 else np.exp(-0.5 * ((dte - target) / h) ** 2)
    w = np.where(np.isfinite(w), w, 0.0)
    if np.sum(w) <= 0:
        w = np.ones_like(dte, dtype=float)
    return w / np.sum(w), h


def weighted_average(values, weights):
    ok = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    return float(np.sum(values[ok] * weights[ok]) / np.sum(weights[ok])) if ok.any() else np.nan


def add_maturity_gates(term, target_dte=TARGET_DTE):
    parts = []
    for _, g in term.groupby(["ticker", "date"], dropna=False):
        g = g.sort_values("dte").copy()
        w, h = gaussian_gate_weights(g["dte"].to_numpy(dtype=float), target_dte)
        q, p = g["Q_crash"].to_numpy(dtype=float), g["P_crash"].to_numpy(dtype=float)
        ok_q, ok_p = np.isfinite(q) & np.isfinite(w), np.isfinite(p) & np.isfinite(w)
        gate_q = float(np.sum(w[ok_q] * q[ok_q]) / np.sum(w[ok_q])) if ok_q.any() and np.sum(w[ok_q]) > 0 else np.nan
        gate_p = float(np.sum(w[ok_p] * p[ok_p]) / np.sum(w[ok_p])) if ok_p.any() and np.sum(w[ok_p]) > 0 else np.nan
        g["Gate_Q"], g["Gate_P"] = gate_q, gate_p
        g["Gate_premium"] = (float(np.log((gate_q + EPS) / (gate_p + EPS)))
                             if np.isfinite(gate_q) and np.isfinite(gate_p) else np.nan)
        g["gate_target_dte"], g["gate_bandwidth_dte"] = target_dte, h
        parts.append(g)
    return pd.concat(parts, ignore_index=True)

# %%
def normal_down_prob(alpha_dyn, F, spot, dte, mu_daily, var_daily):
    vals = [alpha_dyn, F, spot, dte, mu_daily, var_daily]
    if not all(np.isfinite(v) for v in vals) or min(alpha_dyn, F, spot, dte, var_daily) <= 0:
        return np.nan
    z = (math.log(alpha_dyn) - (mu_daily * dte - math.log(F / spot))) / math.sqrt(var_daily * dte)
    return min(max(0.5 * (1.0 + math.erf(z / math.sqrt(2.0))), 0.0), 1.0)


def truncated_down_prob(g, K_thr):
    K = g["K"].to_numpy(dtype=float)
    f = g["risk_neutral_pdf_truncated"].to_numpy(dtype=float)
    w = g["integration_weight"].to_numpy(dtype=float)
    ok = np.isfinite(K) & np.isfinite(f) & np.isfinite(w) & np.isfinite(K_thr) & (K > 0) & (f >= 0) & (w > 0)
    if not ok.any():
        return np.nan
    K, f, w = K[ok], f[ok], w[ok]
    mass = float(np.sum(f * w))
    if not np.isfinite(mass) or mass <= 0:
        return np.nan
    return min(max(float(np.sum(f[K <= K_thr] * w[K <= K_thr]) / mass), 0.0), 1.0)


def fallback_gate_weights(dte, target):
    out = np.full_like(dte, np.nan, dtype=float)
    ok = np.isfinite(dte) & (dte > 0)
    if not ok.any():
        return out
    unique = np.sort(np.unique(dte[ok]))
    spacing = np.diff(unique)[np.diff(unique) > 0] if len(unique) >= 2 else np.array([])
    h = max(float(np.median(spacing)) if len(spacing) else 20.0, 5.0)
    out[ok] = np.exp(-0.5 * ((dte[ok] - target) / h) ** 2)
    total = np.nansum(out)
    return out / total if np.isfinite(total) and total > 0 else out


def fallback_daily(g_date, h, target_dte, z_sigma=Z_SIGMA):
    rows = []
    for exdate, s in g_date.groupby("exdate", sort=True):
        date = s["date"].iloc[0]
        dte = float(np.nanmedian(s["dte"]))
        F = float(np.nanmedian(s["F"]))
        atm_iv = float(np.nanmedian(s["atm_iv"]))
        days = (pd.Timestamp(exdate) - pd.Timestamp(date)).days
        K_thr, alpha_dyn = dynamic_threshold(F, days / 365.0 if days > 0 else np.nan, atm_iv, z_sigma)
        if not np.isfinite(K_thr) or K_thr <= 0:
            K_thr, alpha_dyn = float(np.nanmedian(s["K_thr"])), float(np.nanmedian(s["alpha_dyn"]))
        Q = truncated_down_prob(s, K_thr)
        P = normal_down_prob(alpha_dyn, F, float(h["spot_contract"]), dte, float(h["mu_daily"]), float(h["var_daily"]))
        if np.isfinite(Q) and np.isfinite(P):
            rows.append({"dte": dte, "Q": Q, "P": P})

    if not rows:
        return {"estimation_status": "unavailable_no_valid_rnd_expiry", "n_expiries_used": 0,
                "Q_crash": np.nan, "P_crash": np.nan, "crash_premium": np.nan, "mean_dte_used": np.nan}
    e = pd.DataFrame(rows)
    w = fallback_gate_weights(e["dte"].to_numpy(dtype=float), target_dte)
    gq, gp = weighted_average(e["Q"].to_numpy(), w), weighted_average(e["P"].to_numpy(), w)
    return {
        "estimation_status": "fallback_historical_normal",
        "n_expiries_used": len(e),
        "Q_crash": gq, "P_crash": gp,
        "crash_premium": float(np.log((gq + EPS) / (gp + EPS))) if np.isfinite(gq) and np.isfinite(gp) else np.nan,
        "mean_dte_used": float(np.nanmean(e["dte"])),
    }


def build_daily_output(rnd, hist, term, target_dte=TARGET_DTE, z_sigma=Z_SIGMA):
    hist = hist[["ticker", "date", "mu_daily", "var_daily", "spot_contract"]].drop_duplicates(["ticker", "date"])
    daily = rnd[["ticker", "date"]].drop_duplicates().merge(hist, on=["ticker", "date"], how="inner")
    daily = daily.sort_values(["ticker", "date"]).reset_index(drop=True)
    daily = daily[["ticker", "date", "mu_daily", "var_daily"]]

    # dates with a tilted term structure use its 20-day gate
    strict = pd.DataFrame()
    if len(term):
        s = term[term["Gate_premium"].notna()].sort_values(["ticker", "date", "dte"])
        strict = s.groupby(["ticker", "date"], as_index=False).agg(
            Q_crash=("Gate_Q", "first"), P_crash=("Gate_P", "first"), crash_premium=("Gate_premium", "first"),
            gate_target_dte=("gate_target_dte", "first"), gate_bandwidth_dte=("gate_bandwidth_dte", "first"),
            n_expiries_used=("dte", "count"), n_support_points=("n_support_points", "sum"),
            mean_dte_used=("dte", "mean"),
        )
        strict["estimation_status"] = "strict_entropy_tilt"
        strict["fallback_used"] = False

    if len(strict):
        daily = daily.merge(strict, on=["ticker", "date"], how="left")
    else:
        for c in ["Q_crash", "P_crash", "crash_premium", "gate_target_dte", "gate_bandwidth_dte",
                  "n_expiries_used", "mean_dte_used", "estimation_status", "fallback_used"]:
            daily[c] = np.nan

    hist_lookup = hist.set_index(["ticker", "date"])
    need = set(zip(daily.loc[daily["crash_premium"].isna(), "ticker"], daily.loc[daily["crash_premium"].isna(), "date"]))
    fb = [
        {"ticker": t, "date": d, **fallback_daily(g, hist_lookup.loc[(t, d)], target_dte, z_sigma),
         "gate_target_dte": target_dte, "gate_bandwidth_dte": np.nan, "fallback_used": True}
        for (t, d), g in rnd.groupby(["ticker", "date"], sort=True) if (t, d) in need
    ]
    if fb:
        daily = daily.merge(pd.DataFrame(fb), on=["ticker", "date"], how="left", suffixes=("", "_fb"))
        use_fb = daily["crash_premium"].isna()
        for col in ["Q_crash", "P_crash", "crash_premium", "gate_target_dte", "gate_bandwidth_dte",
                    "n_expiries_used", "mean_dte_used", "estimation_status", "fallback_used"]:
            daily[col] = daily[col].where(~use_fb, daily[col + "_fb"])
        daily = daily.drop(columns=[c for c in daily.columns if c.endswith("_fb")])

    daily["forced_output"] = True
    daily["output_available"] = daily["Q_crash"].notna() & daily["P_crash"].notna() & daily["crash_premium"].notna()
    valid = daily["Q_crash"].notna() & daily["P_crash"].notna() & (daily["Q_crash"] >= 0) & (daily["P_crash"] >= 0)
    daily.loc[valid, "crash_premium"] = np.log((daily.loc[valid, "Q_crash"] + EPS) / (daily.loc[valid, "P_crash"] + EPS))
    daily.loc[daily["estimation_status"].isna(), "estimation_status"] = "unavailable_no_valid_rnd_expiry"
    daily.loc[daily["fallback_used"].isna(), "fallback_used"] = True
    return daily.sort_values(["ticker", "date"]).reset_index(drop=True)

# %% [markdown]
# ## Run per stock

# %%
def run_downside_premium(ticker, out_root, use_pareto=True, target_dte=TARGET_DTE, z_sigma=Z_SIGMA):
    rnd = pd.read_parquet(OPTIONS_DIR / f"{ticker}_breeden_litzenberger_rnd_density.parquet")
    pareto = pd.read_parquet(OPTIONS_DIR / f"{ticker}_pareto_tail_adjustment_summary.parquet")
    hist = pd.read_csv(HIST_MOMENT_FILE)
    for df in (rnd, pareto, hist):
        df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    rnd["exdate"] = pd.to_datetime(rnd["exdate"], errors="coerce")
    pareto["exdate"] = pd.to_datetime(pareto["exdate"], errors="coerce")

    # stock-dates with historical moments only
    rnd = rnd.merge(hist[["ticker", "date"]].dropna().drop_duplicates(), on=["ticker", "date"], how="inner")

    slices, metas = {}, []
    for key, g in rnd.groupby(["ticker", "date", "exdate"], dropna=False):
        prepared = prepare_slice(g, z_sigma)
        if prepared is not None:
            slices[key] = prepared[:2]
            metas.append(prepared[2])

    pareto_cols = ["ticker", "date", "exdate", "pareto_status", "pareto_alpha", "pareto_u", "pareto_tail_mass"]
    meta = pd.DataFrame(metas).merge(pareto[pareto_cols], on=["ticker", "date", "exdate"], how="left")
    meta = attach_historical_moments(meta, hist)

    rows = []
    for _, row in meta.iterrows():
        K, p_q = slices[(row["ticker"], row["date"], row["exdate"])]
        res = premium_one_slice(row, K, p_q, use_pareto)
        if res is not None:
            rows.append(res)
    term = pd.DataFrame(rows)
    if len(term):
        term = add_maturity_gates(term, target_dte)
        term = term.sort_values(["ticker", "date", "dte", "exdate"]).reset_index(drop=True)

    daily = build_daily_output(rnd, hist, term, target_dte, z_sigma)

    out_dir = Path(out_root) / ticker
    out_dir.mkdir(parents=True, exist_ok=True)
    term.to_parquet(out_dir / "strict_crash_premium_term_structure.parquet", index=False)
    daily.to_parquet(out_dir / "forced_daily_crash_premium.parquet", index=False)
    print(f"{ticker}: downside risk premium saved")

# %%
for ticker in TICKERS:
    run_downside_premium(ticker, OPTIONS_DIR / "crash_premium_outputs", use_pareto=True)

# %%
# no Pareto tail
for ticker in TICKERS:
    run_downside_premium(ticker, OPTIONS_DIR / "crash_premium_outputs_no_pareto", use_pareto=False)
