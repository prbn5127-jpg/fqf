# 03 Implied-volatility smoothing, risk-neutral density and Pareto tail
#
# 1. Build the out-of-the-money call-equivalent surface from the repaired quotes and invert it to Black
#    implied volatilities and total variance.
# 2. Smooth total variance with a 2D local-linear kernel regression in (log-moneyness, maturity) for each
#    stock and date, convert back to call prices and keep only slices that pass the shape checks.
# 3. Breeden-Litzenberger density on each usable slice, projected onto non-negative densities.
# 4. Pareto tail fitted to the downside-loss variable L = F/K.

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

OPTIONS_DIR = Path("options_dir")
TICKERS = ["AMZN", "GOOGL", "JNJ", "JPM", "MSFT", "PG", "TSLA", "V"]

# At least three second differences are needed, i.e. five distinct strikes.
MIN_INTERIOR_DENSITY_POINTS = 3
MIN_DISTINCT_STRIKES_FOR_RND = MIN_INTERIOR_DENSITY_POINTS + 2

# Local-linear bandwidth rate
BANDWIDTH_RATE_EXPONENT = -1.0 / (2 + 4.0)
NORMAL_IQR = norm.ppf(0.75) - norm.ppf(0.25)
MACHINE_EPS = np.finfo(float).eps
SMOOTHED = "smoothed_2d_local_linear_kernel"

Z_SIGMA = 2.0
Q_TAIL = 0.10

# %% [markdown]
# ## Call-equivalent surface and implied volatility

# %%
def prepare_repaired_option_data(df):
    x = df.copy()
    x["date"] = pd.to_datetime(x["date"], errors="coerce")
    x["exdate"] = pd.to_datetime(x["exdate"], errors="coerce")
    x["cp_flag"] = x["cp_flag"].astype(str).str.upper().str.strip()
    for col in ["K", "F", "DF", "dte", "tau_cal_365", "C_rep", "P_rep"]:
        x[col] = pd.to_numeric(x[col], errors="coerce")
    x["tau"] = x["tau_cal_365"]

    valid = (
        x["date"].notna() & x["exdate"].notna() & x["cp_flag"].isin(["C", "P"])
        & np.isfinite(x["K"]) & np.isfinite(x["F"]) & np.isfinite(x["DF"])
        & np.isfinite(x["dte"]) & np.isfinite(x["tau"])
        & (x["K"] > 0) & (x["F"] > 0) & (x["DF"] > 0) & (x["dte"] > 0) & (x["tau"] > 0)
    )
    return x.loc[valid].copy()


def black_call(F, K, tau, DF, sigma):
    F, K, DF = (np.asarray(v, dtype=float) for v in (F, K, DF))
    tau = np.maximum(np.asarray(tau, dtype=float), MACHINE_EPS)
    sigma = np.maximum(np.asarray(sigma, dtype=float), MACHINE_EPS)
    vol = sigma * np.sqrt(tau)
    d1 = (np.log(F / K) + 0.5 * sigma**2 * tau) / vol
    return DF * (F * norm.cdf(d1) - K * norm.cdf(d1 - vol))


def black_vega(F, K, tau, DF, sigma):
    F, K, DF = (np.asarray(v, dtype=float) for v in (F, K, DF))
    tau = np.maximum(np.asarray(tau, dtype=float), MACHINE_EPS)
    sigma = np.maximum(np.asarray(sigma, dtype=float), MACHINE_EPS)
    vol = sigma * np.sqrt(tau)
    d1 = (np.log(F / K) + 0.5 * sigma**2 * tau) / vol
    return DF * F * norm.pdf(d1) * np.sqrt(tau)


def implied_vol(price, F, K, tau, DF):
    if not all(np.isfinite(v) for v in (price, F, K, tau, DF)):
        return np.nan
    if price <= 0 or F <= 0 or K <= 0 or tau <= 0 or DF <= 0:
        return np.nan
    if price <= max(0.0, DF * (F - K)) or price >= DF * F:
        return np.nan

    def objective(sigma):
        return black_call(F, K, tau, DF, sigma) - price

    lo, hi = np.sqrt(MACHINE_EPS), 1.0
    f_lo, f_hi = objective(lo), objective(hi)
    while np.isfinite(f_hi) and f_hi < 0 and hi < 1.0 / np.sqrt(MACHINE_EPS):
        hi *= 2.0
        f_hi = objective(hi)
    if not (np.isfinite(f_lo) and np.isfinite(f_hi)) or f_lo * f_hi > 0:
        return np.nan
    try:
        return brentq(objective, lo, hi, maxiter=200)
    except Exception:
        return np.nan


def build_call_equivalent_surface(repaired):
    """OTM puts below the forward (via put-call parity) and OTM calls at or above it."""
    x = prepare_repaired_option_data(repaired)
    x["log_moneyness"] = np.log(x["K"] / x["F"])

    puts = x[x["cp_flag"].eq("P") & (x["K"] < x["F"]) & np.isfinite(x["P_rep"])].copy()
    puts["call_equiv_price"] = puts["P_rep"] + puts["DF"] * (puts["F"] - puts["K"])
    calls = x[x["cp_flag"].eq("C") & (x["K"] >= x["F"]) & np.isfinite(x["C_rep"])].copy()
    calls["call_equiv_price"] = calls["C_rep"]

    surf = pd.concat([puts, calls], ignore_index=True)
    lower = np.maximum(0.0, surf["DF"] * (surf["F"] - surf["K"]))
    upper = surf["DF"] * surf["F"]
    valid = (
        np.isfinite(surf["log_moneyness"]) & np.isfinite(surf["call_equiv_price"])
        & (surf["call_equiv_price"] > 0)
        & (surf["call_equiv_price"] >= lower) & (surf["call_equiv_price"] <= upper)
    )
    surf = surf.loc[valid].copy()

    surf["iv_call_equiv"] = [
        implied_vol(p, F, K, tau, DF)
        for p, F, K, tau, DF in zip(surf["call_equiv_price"], surf["F"], surf["K"], surf["tau"], surf["DF"])
    ]
    surf = surf[np.isfinite(surf["iv_call_equiv"]) & (surf["iv_call_equiv"] > 0)].copy()
    surf["total_variance"] = surf["iv_call_equiv"] ** 2 * surf["tau"]
    return surf[np.isfinite(surf["total_variance"]) & (surf["total_variance"] > 0)].copy()


def add_quote_weights(surf):
    """Inverse noise variance of total variance implied by the bid-ask spread (delta method).

    Mid-price noise is uniform on the spread, Var = s^2/12, and dw/dC = 2 sigma tau / vega.
    Weights are scaled by their median.
    """
    x = surf.copy()
    spread = (pd.to_numeric(x["ask"], errors="coerce") - pd.to_numeric(x["bid"], errors="coerce")).to_numpy(dtype=float)
    positive = spread[np.isfinite(spread) & (spread > 0)]
    if len(positive) == 0:
        x["obs_weight"] = 1.0
        return x

    spread = np.where(np.isfinite(spread) & (spread > 0), spread, np.nanmin(positive))
    sigma = x["iv_call_equiv"].to_numpy(dtype=float)
    tau = x["tau"].to_numpy(dtype=float)
    vega = black_vega(x["F"].to_numpy(dtype=float), x["K"].to_numpy(dtype=float), tau,
                      x["DF"].to_numpy(dtype=float), sigma)
    dw_dC = 2.0 * sigma * tau / np.maximum(vega, MACHINE_EPS)
    raw = 1.0 / np.maximum(dw_dC**2 * spread**2 / 12.0, MACHINE_EPS)

    pos = raw[np.isfinite(raw) & (raw > 0)]
    med = np.nanmedian(pos) if len(pos) else np.nan
    weight = raw / med if np.isfinite(med) and med > 0 else np.ones(len(x))
    x["obs_weight"] = np.where(np.isfinite(weight) & (weight > 0), weight, 1.0)
    return x

# %% [markdown]
# ## 2D local-linear smoothing of total variance

# %%
def robust_scale(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.inf
    q25, q75 = np.nanpercentile(x, [25, 75])
    if np.isfinite(q75 - q25) and q75 - q25 > 0:
        return (q75 - q25) / NORMAL_IQR
    sd = np.nanstd(x, ddof=1)
    return sd if np.isfinite(sd) and sd > 0 else np.inf


def bandwidths(k, tau, weights):
    w = weights[np.isfinite(weights) & (weights > 0)]
    n_eff = w.sum() ** 2 / np.sum(w**2) if len(w) else 0.0
    if n_eff <= 1:
        return np.nan, np.nan
    rate = n_eff**BANDWIDTH_RATE_EXPONENT
    return robust_scale(k) * rate, robust_scale(tau) * rate


def local_linear_2d(k_train, tau_train, w_train, weights, k_eval, tau_eval, h_k, h_tau):
    # Weighted local-linear fit w = a + b_k (k - k0) + b_tau (tau - tau0)
    out = np.empty(len(k_eval))
    use_k = np.isfinite(h_k) and h_k > 0
    use_tau = np.isfinite(h_tau) and h_tau > 0

    for j in range(len(k_eval)):
        k0, tau0 = k_eval[j], tau_eval[j]
        dk = (k_train - k0) / h_k if use_k else np.zeros_like(k_train)
        dt = (tau_train - tau0) / h_tau if use_tau else np.zeros_like(tau_train)
        local_w = weights * np.exp(-0.5 * (dk**2 + dt**2))

        valid = (np.isfinite(k_train) & np.isfinite(tau_train) & np.isfinite(w_train)
                 & np.isfinite(local_w) & (w_train > 0) & (local_w > MACHINE_EPS))
        if valid.sum() < 3:
            out[j] = w_train[np.nanargmin(dk**2 + dt**2)]
            continue

        X = np.column_stack([np.ones(valid.sum()), k_train[valid] - k0, tau_train[valid] - tau0])
        sw = np.sqrt(local_w[valid])
        try:
            beta, *_ = np.linalg.lstsq(X * sw[:, None], w_train[valid] * sw, rcond=None)
            if not np.isfinite(beta[0]):
                raise ValueError("non-finite fit")
            out[j] = beta[0]
        except Exception:
            denom = local_w[valid].sum()
            if denom > MACHINE_EPS:
                out[j] = np.sum(local_w[valid] * w_train[valid]) / denom
            else:
                out[j] = w_train[np.nanargmin(dk**2 + dt**2)]
    return out


def smooth_one_date(g):
    # Smooth the surface of one stock on one date jointly across strikes and maturities
    g = g.copy()
    # one quote per (tau, K): highest weight first, volume as tie-breaker
    g["volume"] = pd.to_numeric(g["volume"], errors="coerce")
    g = g.sort_values(["tau", "K", "obs_weight", "volume"], ascending=[True, True, False, False])
    g = g.drop_duplicates(subset=["tau", "K"], keep="first").sort_values(["tau", "K"]).copy()

    def unsmoothed(status):
        g["total_variance_smooth"] = g["total_variance"]
        g["iv_smooth"] = g["iv_call_equiv"]
        g["call_price_smooth"] = g["call_equiv_price"]
        g["smoothing_status"] = status
        return g

    if len(g) < MIN_DISTINCT_STRIKES_FOR_RND:
        return unsmoothed("too_few_observations")

    k = g["log_moneyness"].to_numpy(dtype=float)
    tau = g["tau"].to_numpy(dtype=float)
    w = g["total_variance"].to_numpy(dtype=float)
    weights = g["obs_weight"].to_numpy(dtype=float)
    valid = (np.isfinite(k) & np.isfinite(tau) & np.isfinite(w) & np.isfinite(weights)
             & (tau > 0) & (w > 0) & (weights > 0))
    if valid.sum() < MIN_DISTINCT_STRIKES_FOR_RND:
        return unsmoothed("too_few_valid_observations")

    h_k, h_tau = bandwidths(k[valid], tau[valid], weights[valid])
    if not (((np.isfinite(h_k) and h_k > 0) or np.isinf(h_k))
            and ((np.isfinite(h_tau) and h_tau > 0) or np.isinf(h_tau))):
        return unsmoothed("invalid_bandwidth")

    try:
        w_smooth = local_linear_2d(k[valid], tau[valid], w[valid], weights[valid], k, tau, h_k, h_tau)
        g["total_variance_smooth"] = np.maximum(w_smooth, MACHINE_EPS)
        g["iv_smooth"] = np.sqrt(g["total_variance_smooth"] / g["tau"])
        g["call_price_smooth"] = black_call(
            g["F"].to_numpy(dtype=float), g["K"].to_numpy(dtype=float), g["tau"].to_numpy(dtype=float),
            g["DF"].to_numpy(dtype=float), g["iv_smooth"].to_numpy(dtype=float),
        )
        g["smoothing_status"] = SMOOTHED
    except Exception as err:
        return unsmoothed(f"smoothing_failed_{type(err).__name__}")
    return g


def check_slice(g):
    # Static bounds, monotonicity and convexity of one smoothed expiry slice
    g = g.sort_values("K")
    K = g["K"].to_numpy(dtype=float)
    C = g["call_price_smooth"].to_numpy(dtype=float)
    DF, F = g["DF"].to_numpy(dtype=float), g["F"].to_numpy(dtype=float)
    finite = np.isfinite(K).all() and np.isfinite(C).all()

    bounds_ok = finite and np.all(C >= np.maximum(0.0, DF * (F - K))) and np.all(C <= DF * F)
    monotone_ok = finite and len(g) >= 2 and int(np.sum(np.diff(C) > 0)) == 0
    convex_ok = False
    if finite and len(g) >= 3 and np.all(np.diff(K) > 0):
        convex_ok = int(np.sum(np.diff(np.diff(C) / np.diff(K)) < 0)) == 0

    logm = g["log_moneyness"].to_numpy(dtype=float)
    has_left = np.isfinite(logm).any() and np.nanmin(logm) < 0
    has_right = np.isfinite(logm).any() and np.nanmax(logm) > 0
    status = g["smoothing_status"].iloc[0] if len(g) else "unknown"

    return {
        "n_distinct_strikes": g["K"].nunique(),
        "bounds_ok": bounds_ok,
        "monotone_ok": monotone_ok,
        "convex_ok": convex_ok,
        "has_left_side": has_left,
        "has_right_side": has_right,
        "slice_usable_for_rnd": (
            g["K"].nunique() >= MIN_DISTINCT_STRIKES_FOR_RND and finite and bounds_ok and monotone_ok
            and convex_ok and has_left and has_right and status == SMOOTHED
        ),
    }


def run_smoothing(ticker):
    repaired = pd.read_parquet(OPTIONS_DIR / f"{ticker}_options_repaired.parquet")
    surf = add_quote_weights(build_call_equivalent_surface(repaired))

    smoothed = pd.concat([smooth_one_date(g) for _, g in surf.groupby(["ticker", "date"])], ignore_index=True)
    quality = pd.DataFrame([
        {"ticker": t, "date": d, "exdate": e, **check_slice(g)}
        for (t, d, e), g in smoothed.groupby(["ticker", "date", "exdate"])
    ])
    smoothed = smoothed.merge(quality, on=["ticker", "date", "exdate"], how="left")
    rnd_input = smoothed[smoothed["slice_usable_for_rnd"].fillna(False)].copy()

    smoothed.to_parquet(OPTIONS_DIR / f"{ticker}_call_equiv_surface_smoothed.parquet", index=False)
    rnd_input.to_parquet(OPTIONS_DIR / f"{ticker}_rnd_input_smoothed.parquet", index=False)
    print(f"{ticker}: smoothing done")
    return smoothed

# %%
smoothed_df = pd.concat([run_smoothing(t) for t in TICKERS], ignore_index=True)

# %% [markdown]
# Roughness is the median absolute second derivative of total variance in log-moneyness. Before smoothing,
# call prices are rebuilt from the raw total variance with the same Black formula.

# %%
def roughness(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3:
        return np.nan
    order = np.argsort(x)
    x, y = x[order], y[order]
    keep = np.r_[True, np.diff(x) > 0]
    x, y = x[keep], y[keep]
    if len(x) < 3:
        return np.nan
    dx_l, dx_r = x[1:-1] - x[:-2], x[2:] - x[1:-1]
    second = 2.0 * ((y[2:] - y[1:-1]) / dx_r - (y[1:-1] - y[:-2]) / dx_l) / (dx_l + dx_r)
    second = second[(dx_l > 0) & (dx_r > 0)]
    return float(np.nanmedian(np.abs(second))) if len(second) else np.nan


def shape_checks(K, C):
    ok = np.isfinite(K) & np.isfinite(C)
    K, C = K[ok], C[ok]
    if len(K) >= 3:
        order = np.argsort(K)
        K, C = K[order], C[order]
        keep = np.r_[True, np.diff(K) > 0]
        K, C = K[keep], C[keep]
    if len(K) < 3:
        return False, False
    dC = np.diff(C)
    return int(np.sum(dC > 0)) == 0, int(np.sum(np.diff(dC / np.diff(K)) < 0)) == 0


def before_after(g):
    g = g.sort_values("K").drop_duplicates("K")
    K, k = g["K"].to_numpy(dtype=float), g["log_moneyness"].to_numpy(dtype=float)
    tau = g["tau"].to_numpy(dtype=float)
    tv_before = g["total_variance"].to_numpy(dtype=float)
    tv_after = g["total_variance_smooth"].to_numpy(dtype=float)

    iv_before = np.full(len(g), np.nan)
    ok = np.isfinite(tv_before) & np.isfinite(tau) & (tv_before > 0) & (tau > 0)
    iv_before[ok] = np.sqrt(tv_before[ok] / tau[ok])
    call_before = black_call(g["F"].to_numpy(dtype=float), K, tau, g["DF"].to_numpy(dtype=float), iv_before)

    mono_b, conv_b = shape_checks(K, call_before)
    mono_a, conv_a = shape_checks(K, g["call_price_smooth"].to_numpy(dtype=float))
    iv_change = np.abs(g["iv_smooth"].to_numpy(dtype=float) - g["iv_call_equiv"].to_numpy(dtype=float))
    return {
        "roughness_before": roughness(k, tv_before),
        "roughness_after": roughness(k, tv_after),
        "monotone_before": mono_b, "monotone_after": mono_a,
        "convex_before": conv_b, "convex_after": conv_a,
        "joint_before": mono_b and conv_b, "joint_after": mono_a and conv_a,
        "median_abs_iv_change": float(np.nanmedian(iv_change)) if np.isfinite(iv_change).any() else np.nan,
    }


evidence = pd.DataFrame([
    before_after(g) for _, g in smoothed_df[smoothed_df["smoothing_status"] == SMOOTHED].groupby(["ticker", "date", "exdate"])
])

table1 = pd.DataFrame({
    "Before": [
        evidence["roughness_before"].median(),
        100 * evidence["monotone_before"].mean(),
        100 * evidence["convex_before"].mean(),
        100 * evidence["joint_before"].mean(),
    ],
    "After": [
        evidence["roughness_after"].median(),
        100 * evidence["monotone_after"].mean(),
        100 * evidence["convex_after"].mean(),
        100 * evidence["joint_after"].mean(),
    ],
}, index=["Median roughness", "Monotonicity pass rate (%)", "Convexity pass rate (%)", "Joint pass rate (%)"])
print(table1.round(6))
print("Median absolute IV adjustment:", evidence["median_abs_iv_change"].median())

# %% [markdown]
# ## Breeden-Litzenberger density

# %%
def integration_weights(x):
    w = np.empty_like(x, dtype=float)
    w[0] = 0.5 * (x[1] - x[0])
    w[-1] = 0.5 * (x[-1] - x[-2])
    if len(x) > 2:
        w[1:-1] = 0.5 * (x[2:] - x[:-2])
    return w


def project_nonnegative(f_raw, weights, target_mass, tol=1e-12, max_iter=200):
    """min sum w (f - f_raw)^2 s.t. f >= 0, sum w f = target_mass; solution f = max(f_raw - lam, 0)."""
    positive = np.maximum(f_raw, 0.0)
    if not np.isfinite(target_mass) or target_mass <= tol:
        if np.sum(weights * positive) > tol:
            return positive
        return np.ones_like(f_raw) / float(np.sum(weights))
    if np.all(f_raw >= 0) and abs(np.sum(weights * f_raw) - target_mass) <= tol:
        return f_raw.copy()

    lower = float(np.min(f_raw) - target_mass / np.min(weights) - 1.0)
    upper = float(np.max(f_raw))
    lam = 0.0
    for _ in range(max_iter):
        lam = 0.5 * (lower + upper)
        mass = float(np.sum(weights * np.maximum(f_raw - lam, 0.0)))
        if abs(mass - target_mass) <= tol * max(1.0, target_mass):
            break
        if mass > target_mass:
            lower = lam
        else:
            upper = lam

    f = np.maximum(f_raw - lam, 0.0)
    mass = float(np.sum(weights * f))
    if mass > tol:
        f *= target_mass / mass
    return f


def rnd_one_slice(g):
    g = g.sort_values("K")   # strikes are unique within a slice after smoothing
    if g["K"].nunique() < MIN_DISTINCT_STRIKES_FOR_RND:
        return pd.DataFrame()

    K = g["K"].to_numpy(dtype=float)
    C = g["call_price_smooth"].to_numpy(dtype=float)
    F, DF = float(np.nanmedian(g["F"])), float(np.nanmedian(g["DF"]))
    tau, dte = float(np.nanmedian(g["tau"])), float(np.nanmedian(g["dte"]))

    # second derivative on the uneven strike grid 
    h_l, h_r = K[1:-1] - K[:-2], K[2:] - K[1:-1]
    ok = (h_l > 0) & (h_r > 0)
    ckk = np.full(len(K) - 2, np.nan)
    ckk[ok] = 2.0 * ((C[2:][ok] - C[1:-1][ok]) / h_r[ok] - (C[1:-1][ok] - C[:-2][ok]) / h_l[ok]) / (h_l[ok] + h_r[ok])
    K_mid = K[1:-1]
    finite = np.isfinite(K_mid) & np.isfinite(ckk)
    K_mid, ckk = K_mid[finite], ckk[finite]
    if len(K_mid) < MIN_INTERIOR_DENSITY_POINTS or not np.isfinite(DF) or DF <= 0:
        return pd.DataFrame()

    pdf_raw = ckk / DF
    weights = integration_weights(K_mid)
    pdf_proj = project_nonnegative(pdf_raw, weights, float(np.sum(weights * pdf_raw)))
    mass = float(np.sum(weights * pdf_proj))
    if np.isfinite(mass) and mass > 0:
        pdf_trunc = pdf_proj / mass
        status = "pass" if np.all(pdf_raw >= 0) else "pass_after_projection"
    else:
        pdf_trunc = np.full_like(pdf_proj, np.nan)
        status = "fail_zero_mass"

    iv_mid = g["iv_smooth"].to_numpy(dtype=float)[1:-1][finite]
    atm = pd.DataFrame({"K": K_mid, "iv": iv_mid})
    atm = atm[np.isfinite(atm["K"]) & np.isfinite(atm["iv"]) & (atm["K"] > 0) & (atm["iv"] > 0)]
    if len(atm) and np.isfinite(F) and F > 0:
        atm_iv = float(atm.assign(d=np.abs(np.log(atm["K"] / F))).sort_values("d").head(2)["iv"].median())
    else:
        atm_iv = np.nan

    if np.isfinite(F) and F > 0 and np.isfinite(tau) and tau > 0 and np.isfinite(atm_iv) and atm_iv > 0:
        alpha_dyn = float(np.exp(-Z_SIGMA * atm_iv * np.sqrt(tau)))
        K_thr = float(F * alpha_dyn)
    else:
        alpha_dyn, K_thr = np.nan, np.nan

    return pd.DataFrame({
        "ticker": str(g["ticker"].iloc[0]),
        "date": pd.to_datetime(g["date"].iloc[0]),
        "exdate": pd.to_datetime(g["exdate"].iloc[0]),
        "tau": tau, "dte": dte, "F": F, "DF": DF,
        "K": K_mid,
        "log_moneyness": np.log(K_mid / F),
        "call_price_smooth": g["call_price_smooth"].to_numpy(dtype=float)[1:-1][finite],
        "iv_smooth": iv_mid,
        "atm_iv": atm_iv, "K_thr": K_thr, "alpha_dyn": alpha_dyn,
        "risk_neutral_pdf_raw": pdf_raw,
        "risk_neutral_pdf_projected": pdf_proj,
        "risk_neutral_pdf_truncated": pdf_trunc,
        "integration_weight": weights,
        "rnd_status": status,
    })


def run_rnd(ticker):
    x = pd.read_parquet(OPTIONS_DIR / f"{ticker}_rnd_input_smoothed.parquet")
    x["date"] = pd.to_datetime(x["date"], errors="coerce")
    x["exdate"] = pd.to_datetime(x["exdate"], errors="coerce")
    x = x[
        x["ticker"].notna() & x["date"].notna() & x["exdate"].notna()
        & np.isfinite(x["K"]) & np.isfinite(x["F"]) & np.isfinite(x["DF"]) & np.isfinite(x["tau"])
        & np.isfinite(x["call_price_smooth"])
        & (x["K"] > 0) & (x["F"] > 0) & (x["DF"] > 0) & (x["tau"] > 0)
    ]
    parts = [rnd_one_slice(g) for _, g in x.groupby(["ticker", "date", "exdate"], sort=True)]
    rnd = pd.concat([p for p in parts if len(p)], ignore_index=True)
    rnd.to_parquet(OPTIONS_DIR / f"{ticker}_breeden_litzenberger_rnd_density.parquet", index=False)
    print(f"{ticker}: {rnd.groupby(['date', 'exdate']).ngroups:,} density slices")
    return rnd


rnd_df = pd.concat([run_rnd(t) for t in TICKERS], ignore_index=True)

# %% [markdown]
# ## Pareto downside tail
#
# u is the (1 - q) quantile of L = F/K under the discrete risk-neutral masses; the tail index is the
# probability-weighted Hill estimator, alpha = 1 / E[log(L/u) | L >= u].

# %%
def weighted_quantile(values, weights, q):
    ok = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values, weights = values[ok], weights[ok]
    if len(values) == 0:
        return np.nan
    order = np.argsort(values)
    cum = np.cumsum(weights[order])
    return np.interp(q, cum / cum[-1], values[order])


def fit_pareto_tail(g, q_tail=Q_TAIL, min_tail_points=3, min_tail_mass=1e-8, alpha_min=0.25, alpha_max=50.0):
    g = g.sort_values("K")
    fit = {"pareto_status": "", "pareto_alpha": np.nan, "pareto_alpha_raw": np.nan, "pareto_u": np.nan,
           "pareto_tail_mass": np.nan, "pareto_tail_points": 0, "pareto_q_tail": q_tail,
           "pareto_K_threshold": np.nan, "pareto_logm_threshold": np.nan}

    F_values = g["F"].dropna().to_numpy(dtype=float)
    if len(F_values) == 0:
        return {**fit, "pareto_status": "missing_forward"}
    F = float(np.nanmedian(F_values))

    K = g["K"].to_numpy(dtype=float)
    f = g["risk_neutral_pdf_truncated"].to_numpy(dtype=float)
    dK = g["integration_weight"].to_numpy(dtype=float)
    ok = (np.isfinite(K) & np.isfinite(f) & np.isfinite(dK) & np.isfinite(F)
          & (K > 0) & (f >= 0) & (dK > 0) & (F > 0))
    K, f, dK = K[ok], f[ok], dK[ok]
    if len(K) < min_tail_points + 2:
        return {**fit, "pareto_status": "insufficient_grid"}

    p = f * dK
    if not np.isfinite(p.sum()) or p.sum() <= 0:
        return {**fit, "pareto_status": "invalid_probability_mass"}
    p = p / p.sum()

    L = F / K
    u = weighted_quantile(L, p, 1.0 - q_tail)
    if not np.isfinite(u) or u <= 0:
        return {**fit, "pareto_status": "invalid_tail_threshold", "pareto_u": u}

    tail = L >= u
    fit.update({
        "pareto_u": float(u),
        "pareto_tail_mass": float(np.sum(p[tail])),
        "pareto_tail_points": int(tail.sum()),
        "pareto_K_threshold": F / u,
        "pareto_logm_threshold": np.log((F / u) / F),
    })
    if tail.sum() < min_tail_points:
        return {**fit, "pareto_status": "insufficient_tail_points"}
    if fit["pareto_tail_mass"] <= min_tail_mass:
        return {**fit, "pareto_status": "insufficient_tail_mass"}

    mean_log_excess = float(np.sum(p[tail] / fit["pareto_tail_mass"] * np.log(L[tail] / u)))
    if not np.isfinite(mean_log_excess) or mean_log_excess <= 0:
        return {**fit, "pareto_status": "degenerate_tail_excess"}

    alpha_raw = 1.0 / mean_log_excess
    alpha = float(np.clip(alpha_raw, alpha_min, alpha_max))
    fit.update({
        "pareto_status": "ok" if np.isclose(alpha, alpha_raw) else "ok_alpha_clipped",
        "pareto_alpha": alpha,
        "pareto_alpha_raw": float(alpha_raw),
    })
    return fit


rows = []
for (ticker, date, exdate), g in rnd_df.groupby(["ticker", "date", "exdate"]):
    rows.append({"ticker": ticker, "date": date, "exdate": exdate, "dte": float(g["dte"].median()),
                 **fit_pareto_tail(g)})
pareto_tail_df = pd.DataFrame(rows)

for ticker, g in pareto_tail_df.groupby("ticker"):
    g.sort_values(["date", "exdate"]).reset_index(drop=True).to_parquet(
        OPTIONS_DIR / f"{ticker}_pareto_tail_adjustment_summary.parquet", index=False
    )

# %% [markdown]
# ## RND and Pareto tail for GOOGL
#
# The slice with the widest left support among GOOGL expiries of 30-90 business days with a valid tail fit
# (1 October 2021, expiry 21 January 2022).

# %%
cand = pareto_tail_df[
    pareto_tail_df["pareto_status"].isin(["ok", "ok_alpha_clipped"])
    & pareto_tail_df["ticker"].eq("GOOGL")
    & pareto_tail_df["dte"].between(30, 90)
]
cand_rnd = rnd_df[rnd_df["ticker"].eq("GOOGL")].merge(cand[["date", "exdate"]], on=["date", "exdate"])
width = cand_rnd.groupby(["date", "exdate"])["log_moneyness"].agg(["min", "max"])
width["width"] = width["max"] - width["min"]
date0, exdate0 = width.sort_values(["min", "width"], ascending=[True, False]).index[0]

tail_row = cand[(cand["date"] == date0) & (cand["exdate"] == exdate0)].iloc[0]
g = rnd_df[rnd_df["ticker"].eq("GOOGL") & (rnd_df["date"] == date0) & (rnd_df["exdate"] == exdate0)].sort_values("K")
F = float(np.nanmedian(g["F"]))
title = f"Date: {date0.date()}, Expiry: {exdate0.date()}"

# (a) density in log-moneyness, f_m(m) = K f_K(K)
logm = np.log(g["K"] / F).to_numpy()
pdf_logm = (g["risk_neutral_pdf_truncated"] * g["K"]).to_numpy()
trapezoid = getattr(np, "trapezoid", None)
pdf_logm = pdf_logm / trapezoid(pdf_logm, logm)
thr = float(tail_row["pareto_logm_threshold"])

plt.figure(figsize=(9, 5))
plt.plot(logm, pdf_logm, linewidth=2, label="Projected truncated RND")
plt.fill_between(logm[logm <= thr], pdf_logm[logm <= thr], alpha=0.25, label="Pareto tail region")
plt.axvline(thr, linestyle="--", linewidth=1.8, label="Pareto threshold")
plt.xlabel(r"Log-moneyness, $\log(K/F)$")
plt.ylabel("Truncated RND density w.r.t. log-moneyness")
plt.title(f"GOOGL RND with Pareto Tail Region\n{title}, alpha={tail_row['pareto_alpha']:.2f}, "
          f"tail mass={tail_row['pareto_tail_mass']:.2%}")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("figure4a_GOOGL.png", dpi=300, bbox_inches="tight")
plt.show()

# (b) empirical survival of L = F/K and the fitted tail
p = (g["risk_neutral_pdf_truncated"] * g["integration_weight"]).to_numpy()
p = p / p.sum()
L = F / g["K"].to_numpy()
order = np.argsort(L)
L, p = L[order], p[order]
survival = np.array([p[L >= ell].sum() for ell in L])
u, alpha, mass = tail_row["pareto_u"], tail_row["pareto_alpha"], tail_row["pareto_tail_mass"]
grid = np.linspace(u, L.max(), 200)

plt.figure(figsize=(9, 5))
plt.plot(L, survival, marker="o", markersize=3, linewidth=1.5, label="Empirical RND survival")
plt.plot(grid, mass * (u / grid) ** alpha, linewidth=2, label="Fitted Pareto tail")
plt.axvline(u, linestyle="--", linewidth=1.8, label=r"Tail threshold $u$")
plt.yscale("log")
plt.xlabel(r"Downside-loss variable, $L=F/K$")
plt.ylabel(r"Survival probability, $Q(L \geq \ell)$")
plt.title(f"GOOGL Pareto Downside Tail Fit\n{title}, alpha={alpha:.2f}")
plt.legend()
plt.grid(True, alpha=0.3, which="both")
plt.tight_layout()
plt.savefig("figure4b_GOOGL.png", dpi=300, bbox_inches="tight")
plt.show()
