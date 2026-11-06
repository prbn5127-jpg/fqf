# Functions from file 03 used for ablation A9

from pathlib import Path
import gc
import time
import warnings
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq
from scipy.interpolate import UnivariateSpline

MACHINE_EPS = np.finfo(float).eps
MIN_INTERIOR_DENSITY_POINTS = 3
MIN_DISTINCT_STRIKES_FOR_RND = MIN_INTERIOR_DENSITY_POINTS + 2
UNIFORM_VARIANCE_DENOMINATOR = 12.0
SMOOTHING_SUCCESS_STATUS = "smoothed_1d_spline_per_expiry"
SPLINE_IV_DIR = Path("options_dir_iv_spline")
OPTIONS_DIR = Path("options_dir_iv_spline")
OUTPUT_DIR = OPTIONS_DIR
SAVE_RND_DENSITY_CSV = False


def _first_existing_col(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _copy_alias_if_missing(df, canonical, aliases):
    if canonical in df.columns:
        return df

    found = _first_existing_col(df, aliases)

    if found is not None:
        df[canonical] = df[found]

    return df


def prepare_repaired_option_data(df):
    """
    Standardize column names while retaining two maturity measures:

        dte = business-day maturity,
        tau = tau_cal_365 = ACT/365 pricing maturity.

    """
    x = df.copy()

    if "price_basis" not in x.columns:
        raise ValueError(
            "Missing price_basis in repaired data. "
            "BL extraction requires European-equivalent option prices."
        )

    valid_basis = (
        x["price_basis"]
        .astype(str)
        .eq("european_equivalent_vendor_iv")
    )

    if not valid_basis.all():
        raise ValueError(
            "BL input contains non-European prices. "
            f"Invalid rows: {(~valid_basis).sum()}"
        )

    alias_map = {
        "ticker": ["ticker", "symbol", "underlying"],
        "date": ["date", "trade_date", "valuation_date"],
        "exdate": ["exdate", "expiry_date", "expiration_date", "maturity_date"],
        "cp_flag": ["cp_flag", "option_type", "optiontype", "right", "put_call"],
        "K": ["K", "strike", "strike_price", "strike price"],
        "F": ["F", "forward", "forward_price"],
        "DF": ["DF", "df", "discount_factor", "discount"],
        "C_rep": ["C_rep", "call_repaired", "call_price_repaired", "C_repaired"],
        "P_rep": ["P_rep", "put_repaired", "put_price_repaired", "P_repaired"],
        "dte": ["dte", "DTE", "business_days_to_expiry", "bdays_to_expiry"],
        "tau_cal_365": ["tau_cal_365", "T"],
    }

    for canonical, aliases in alias_map.items():
        x = _copy_alias_if_missing(x, canonical, aliases)

    required_initial = [
        "ticker", "date", "exdate", "cp_flag",
        "K", "F", "DF", "dte", "tau_cal_365"
    ]
    missing_initial = [col for col in required_initial if col not in x.columns]

    if missing_initial:
        raise ValueError(f"Missing required columns: {missing_initial}")

    x["date"] = pd.to_datetime(x["date"], errors="coerce")
    x["exdate"] = pd.to_datetime(x["exdate"], errors="coerce")

    x["cp_flag"] = (
        x["cp_flag"]
        .astype(str)
        .str.upper()
        .str.strip()
        .replace({
            "CALL": "C",
            "PUT": "P",
            "CE": "C",
            "PE": "P",
        })
    )

    for col in ["K", "F", "DF", "dte", "tau_cal_365", "C_rep", "P_rep"]:
        if col in x.columns:
            x[col] = _safe_numeric(x[col])

    if ("C_rep" not in x.columns) or ("P_rep" not in x.columns):
        repaired_col = _first_existing_col(
            x,
            ["repaired_price", "price_repaired", "mid_repaired", "price_clean"]
        )

        if repaired_col is not None:
            repaired_price = _safe_numeric(x[repaired_col])

            if "C_rep" not in x.columns:
                x["C_rep"] = np.where(x["cp_flag"] == "C", repaired_price, np.nan)

            if "P_rep" not in x.columns:
                x["P_rep"] = np.where(x["cp_flag"] == "P", repaired_price, np.nan)

    required_final = [
        "ticker", "date", "exdate", "cp_flag",
        "K", "F", "DF", "dte", "C_rep", "P_rep"
    ]
    missing_final = [col for col in required_final if col not in x.columns]

    if missing_final:
        raise ValueError(f"Missing required columns after standardization: {missing_final}")

    x["tau"] = x["tau_cal_365"]

    valid = (
        x["date"].notna() &
        x["exdate"].notna() &
        x["cp_flag"].isin(["C", "P"]) &
        np.isfinite(x["K"]) &
        np.isfinite(x["F"]) &
        np.isfinite(x["DF"]) &
        np.isfinite(x["dte"]) &
        np.isfinite(x["tau"]) &
        (x["K"] > 0) &
        (x["F"] > 0) &
        (x["DF"] > 0) &
        (x["dte"] > 0) &
        (x["tau"] > 0)
    )

    return x.loc[valid].copy()


def black_call_price_forward(F, K, tau, DF, sigma):
    """
    Forward Black-76 call price:

        C = DF [F N(d1) - K N(d2)].
    """

    F = np.asarray(F, dtype=float)
    K = np.asarray(K, dtype=float)
    tau = np.asarray(tau, dtype=float)
    DF = np.asarray(DF, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    tau = np.maximum(tau, MACHINE_EPS)
    sigma = np.maximum(sigma, MACHINE_EPS)

    vol_sqrt = sigma * np.sqrt(tau)

    d1 = (np.log(F / K) + 0.5 * sigma**2 * tau) / vol_sqrt
    d2 = d1 - vol_sqrt

    return DF * (F * norm.cdf(d1) - K * norm.cdf(d2))


def black_vega_forward(F, K, tau, DF, sigma):
    """
    Forward Black-76 vega:

        Vega = DF * F * phi(d1) * sqrt(tau).
    """

    F = np.asarray(F, dtype=float)
    K = np.asarray(K, dtype=float)
    tau = np.asarray(tau, dtype=float)
    DF = np.asarray(DF, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    tau = np.maximum(tau, MACHINE_EPS)
    sigma = np.maximum(sigma, MACHINE_EPS)

    vol_sqrt = sigma * np.sqrt(tau)
    d1 = (np.log(F / K) + 0.5 * sigma**2 * tau) / vol_sqrt

    return DF * F * norm.pdf(d1) * np.sqrt(tau)


def implied_vol_call_forward(price, F, K, tau, DF):

    vals = [price, F, K, tau, DF]

    if any(not np.isfinite(v) for v in vals):
        return np.nan

    if price <= 0 or F <= 0 or K <= 0 or tau <= 0 or DF <= 0:
        return np.nan

    lower = max(0.0, DF * (F - K))
    upper = DF * F

    if price <= lower or price >= upper:
        return np.nan

    def objective(sigma):
        return black_call_price_forward(F, K, tau, DF, sigma) - price

    lo = np.sqrt(MACHINE_EPS)
    hi = 1.0

    f_lo = objective(lo)
    f_hi = objective(hi)

    max_numerical_hi = 1.0 / np.sqrt(MACHINE_EPS)

    while np.isfinite(f_hi) and f_hi < 0 and hi < max_numerical_hi:
        hi *= 2.0
        f_hi = objective(hi)

    if not np.isfinite(f_lo) or not np.isfinite(f_hi):
        return np.nan

    if f_lo * f_hi > 0:
        return np.nan

    try:
        return brentq(objective, lo, hi, maxiter=200)
    except Exception:
        return np.nan


def build_otm_call_equivalent_surface(repaired_df):
    """
    Construct unified call-equivalent surface.

    For K < F:
        use repaired OTM puts and convert through put-call parity:
        C_eq = P_rep + DF(F-K)

    For K >= F:
        use repaired OTM calls directly:
        C_eq = C_rep
    """

    x = prepare_repaired_option_data(repaired_df)

    x["log_moneyness"] = np.log(x["K"] / x["F"])

    left_puts = x[
        (x["cp_flag"] == "P") &
        (x["K"] < x["F"]) &
        np.isfinite(x["P_rep"])
    ].copy()

    left_puts["call_equiv_price"] = (
        left_puts["P_rep"] +
        left_puts["DF"] * (left_puts["F"] - left_puts["K"])
    )
    left_puts["surface_source"] = "OTM_put_to_synthetic_call"

    right_calls = x[
        (x["cp_flag"] == "C") &
        (x["K"] >= x["F"]) &
        np.isfinite(x["C_rep"])
    ].copy()

    right_calls["call_equiv_price"] = right_calls["C_rep"]
    right_calls["surface_source"] = "OTM_call_direct"

    surf = pd.concat([left_puts, right_calls], ignore_index=True)

    surf["call_lower_bound"] = np.maximum(
        0.0,
        surf["DF"] * (surf["F"] - surf["K"])
    )
    surf["call_upper_bound"] = surf["DF"] * surf["F"]

    valid = (
        np.isfinite(surf["log_moneyness"]) &
        np.isfinite(surf["call_equiv_price"]) &
        (surf["call_equiv_price"] > 0) &
        (surf["call_equiv_price"] >= surf["call_lower_bound"]) &
        (surf["call_equiv_price"] <= surf["call_upper_bound"])
    )

    return surf.loc[valid].copy()


def add_call_equiv_iv_and_total_variance(surf):
    """
    Recompute implied volatility from repaired call-equivalent price, then compute:

        w = sigma^2 * tau.
    """

    x = surf.copy()

    x["iv_call_equiv"] = [
        implied_vol_call_forward(price, F, K, tau, DF)
        for price, F, K, tau, DF in zip(
            x["call_equiv_price"],
            x["F"],
            x["K"],
            x["tau"],
            x["DF"],
        )
    ]

    x = x[
        np.isfinite(x["iv_call_equiv"]) &
        (x["iv_call_equiv"] > 0)
    ].copy()

    x["total_variance"] = x["iv_call_equiv"] ** 2 * x["tau"]

    x = x[
        np.isfinite(x["total_variance"]) &
        (x["total_variance"] > 0)
    ].copy()

    return x


def add_quote_reliability_weights(surf):
    """
    Construct inverse-noise-variance weights in total-variance space.

    Midpoint price noise:
        Var(epsilon_C) ≈ spread^2 / 12.

    Delta method:
        dw/dC = 2 sigma tau / Vega.

    Total-variance noise:
        Var(epsilon_w) ≈ (dw/dC)^2 Var(epsilon_C).

    Weight:
        omega_i ∝ 1 / Var(epsilon_w).
    """

    x = surf.copy()

    bid_col = _first_existing_col(
        x,
        ["bid", "best_bid", "nbbobid", "option_bid"]
    )

    ask_col = _first_existing_col(
        x,
        ["ask", "best_offer", "best_ask", "nbboask", "option_ask"]
    )

    if bid_col is not None and ask_col is not None:
        bid = _safe_numeric(x[bid_col]).to_numpy(dtype=float)
        ask = _safe_numeric(x[ask_col]).to_numpy(dtype=float)
        spread = ask - bid
    else:
        spread = np.full(len(x), np.nan)

    positive_spreads = spread[np.isfinite(spread) & (spread > 0)]

    if len(positive_spreads) == 0:
        x["obs_weight"] = 1.0
        x["price_noise_var"] = np.nan
        x["total_variance_noise_var"] = np.nan
        x["weight_reason"] = "equal_weight_no_valid_bid_ask"
        return x

    min_positive_spread = np.nanmin(positive_spreads)

    spread = np.where(
        np.isfinite(spread) & (spread > 0),
        spread,
        min_positive_spread,
    )

    price_noise_var = spread**2 / UNIFORM_VARIANCE_DENOMINATOR

    vega = black_vega_forward(
        F=x["F"].to_numpy(dtype=float),
        K=x["K"].to_numpy(dtype=float),
        tau=x["tau"].to_numpy(dtype=float),
        DF=x["DF"].to_numpy(dtype=float),
        sigma=x["iv_call_equiv"].to_numpy(dtype=float),
    )

    sigma = x["iv_call_equiv"].to_numpy(dtype=float)
    tau = x["tau"].to_numpy(dtype=float)

    dw_dC = 2.0 * sigma * tau / np.maximum(vega, MACHINE_EPS)

    total_variance_noise_var = (dw_dC**2) * price_noise_var

    raw_weight = 1.0 / np.maximum(total_variance_noise_var, MACHINE_EPS)

    positive_weights = raw_weight[np.isfinite(raw_weight) & (raw_weight > 0)]

    if len(positive_weights) == 0:
        weight = np.ones(len(x))
    else:
        median_weight = np.nanmedian(positive_weights)

        if np.isfinite(median_weight) and median_weight > 0:
            weight = raw_weight / median_weight
        else:
            weight = np.ones(len(x))

    weight = np.where(np.isfinite(weight) & (weight > 0), weight, 1.0)

    x["obs_weight"] = weight
    x["price_noise_var"] = price_noise_var
    x["total_variance_noise_var"] = total_variance_noise_var
    x["weight_reason"] = "inverse_delta_method_total_variance_noise"

    return x


def check_one_smoothed_slice(g):
    """
    Diagnostics for one fixed ticker-date-expiry slice.

    Breeden-Litzenberger is applied maturity by maturity, so after
    ticker-date-level smoothing, validation is done by date-expiry.
    """

    g = g.sort_values("K").copy()

    K = g["K"].to_numpy(dtype=float)
    C = g["call_price_smooth"].to_numpy(dtype=float)

    n_points = len(g)
    n_distinct_strikes = g["K"].nunique()

    finite_ok = np.isfinite(K).all() and np.isfinite(C).all()

    lower = np.maximum(
        0.0,
        g["DF"].to_numpy(dtype=float) *
        (g["F"].to_numpy(dtype=float) - K)
    )

    upper = g["DF"].to_numpy(dtype=float) * g["F"].to_numpy(dtype=float)

    price_scale = np.nanmedian(np.abs(C)) if len(C) > 0 else 1.0

    if not np.isfinite(price_scale) or price_scale <= 0:
        price_scale = 1.0

    bounds_ok = (
        finite_ok and
        np.all(C >= lower) and
        np.all(C <= upper)
    )

    if n_points >= 2 and finite_ok:
        dC = np.diff(C)
        monotone_violations = int(np.sum(dC > 0))
        monotone_ok = monotone_violations == 0
    else:
        monotone_violations = np.nan
        monotone_ok = False

    if n_points >= 3 and finite_ok:
        dK = np.diff(K)

        if np.all(dK > 0):
            slopes = np.diff(C) / dK
            convex_violations = int(np.sum(np.diff(slopes) < 0))
            convex_ok = convex_violations == 0
        else:
            convex_violations = np.nan
            convex_ok = False
    else:
        convex_violations = np.nan
        convex_ok = False

    log_mny = g["log_moneyness"].to_numpy(dtype=float)

    if np.isfinite(log_mny).any():
        min_log_moneyness = np.nanmin(log_mny)
        max_log_moneyness = np.nanmax(log_mny)
    else:
        min_log_moneyness = np.nan
        max_log_moneyness = np.nan

    has_left_side = np.isfinite(min_log_moneyness) and (min_log_moneyness < 0)
    has_right_side = np.isfinite(max_log_moneyness) and (max_log_moneyness > 0)

    smoothing_status = (
        g["smoothing_status"].iloc[0]
        if "smoothing_status" in g.columns and len(g) > 0
        else "unknown"
    )

    slice_usable_for_rnd = (
        (n_distinct_strikes >= MIN_DISTINCT_STRIKES_FOR_RND) and
        finite_ok and
        bounds_ok and
        monotone_ok and
        convex_ok and
        has_left_side and
        has_right_side and
        (smoothing_status == SMOOTHING_SUCCESS_STATUS)
    )

    return pd.Series({
        "n_points": n_points,
        "n_distinct_strikes": n_distinct_strikes,
        "finite_ok": finite_ok,
        "bounds_ok": bounds_ok,
        "monotone_ok": monotone_ok,
        "convex_ok": convex_ok,
        "monotone_violations": monotone_violations,
        "convex_violations": convex_violations,
        "min_log_moneyness": min_log_moneyness,
        "max_log_moneyness": max_log_moneyness,
        "has_left_side": has_left_side,
        "has_right_side": has_right_side,
        "smoothing_status": smoothing_status,
        "slice_usable_for_rnd": slice_usable_for_rnd,
    })


def add_smoothed_slice_quality(smoothed):
    """
    Attach date-expiry-level RND usability diagnostics.
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)

        slice_quality = (
            smoothed.groupby(["ticker", "date", "exdate"])
            .apply(check_one_smoothed_slice)
            .reset_index()
        )

    merge_cols = [
        "ticker", "date", "exdate",
        "n_points",
        "n_distinct_strikes",
        "finite_ok",
        "bounds_ok",
        "monotone_ok",
        "convex_ok",
        "has_left_side",
        "has_right_side",
        "slice_usable_for_rnd",
    ]

    smoothed = smoothed.merge(
        slice_quality[merge_cols],
        on=["ticker", "date", "exdate"],
        how="left",
    )

    return smoothed, slice_quality


def save_smoothing_outputs(smoothed_df, slice_quality, rnd_input, output_dir, ticker):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    smoothed_parquet = output_dir / f"{ticker}_call_equiv_surface_2d_local_linear_smoothed.parquet"
    smoothed_csv = output_dir / f"{ticker}_call_equiv_surface_2d_local_linear_smoothed.csv"

    quality_csv = output_dir / f"{ticker}_2d_local_linear_smoothing_slice_quality.csv"

    rnd_parquet = output_dir / f"{ticker}_rnd_input_2d_local_linear_smoothed.parquet"
    rnd_csv = output_dir / f"{ticker}_rnd_input_2d_local_linear_smoothed.csv"

    smoothed_df.to_parquet(smoothed_parquet, index=False)
    smoothed_df.to_csv(smoothed_csv, index=False)

    slice_quality.to_csv(quality_csv, index=False)

    rnd_input.to_parquet(rnd_parquet, index=False)
    rnd_input.to_csv(rnd_csv, index=False)

    print("Saved:")
    print("  ", smoothed_parquet)
    print("  ", smoothed_csv)
    print("  ", quality_csv)
    print("  ", rnd_parquet)
    print("  ", rnd_csv)


def smoothing_output_paths(output_dir, ticker):
    output_dir = Path(output_dir)

    return {
        "smoothed_parquet": output_dir / f"{ticker}_call_equiv_surface_2d_local_linear_smoothed.parquet",
        "smoothed_csv": output_dir / f"{ticker}_call_equiv_surface_2d_local_linear_smoothed.csv",
        "quality_csv": output_dir / f"{ticker}_2d_local_linear_smoothing_slice_quality.csv",
        "rnd_parquet": output_dir / f"{ticker}_rnd_input_2d_local_linear_smoothed.parquet",
        "rnd_csv": output_dir / f"{ticker}_rnd_input_2d_local_linear_smoothed.csv",
    }


def load_existing_smoothing_outputs(output_dir, ticker):
    paths = smoothing_output_paths(output_dir, ticker)

    required = ["smoothed_parquet", "quality_csv", "rnd_parquet"]
    missing = [name for name in required if not paths[name].exists()]

    if missing:
        raise FileNotFoundError(
            f"Cannot load existing smoothing outputs for {ticker}; missing {missing}."
        )

    smoothed = pd.read_parquet(paths["smoothed_parquet"])
    quality = pd.read_csv(paths["quality_csv"])
    rnd_in = pd.read_parquet(paths["rnd_parquet"])

    return smoothed, quality, rnd_in


def spline_1d_total_variance_predict(
    k_train,
    w_train,
    obs_weights,
    k_eval,
    smoothing_multiplier=1.0,
):
    """
    Per-expiry spline baseline.

    Inputs:
        k_train     = observed log-moneyness
        w_train     = observed total implied variance
        obs_weights = quote reliability weights
        k_eval      = log-moneyness values where we evaluate the spline
    """

    k_train = np.asarray(k_train, dtype=float)
    w_train = np.asarray(w_train, dtype=float)
    obs_weights = np.asarray(obs_weights, dtype=float)
    k_eval = np.asarray(k_eval, dtype=float)

    valid = (
        np.isfinite(k_train)
        & np.isfinite(w_train)
        & np.isfinite(obs_weights)
        & (w_train > 0)
        & (obs_weights > 0)
    )

    k = k_train[valid]
    w = w_train[valid]
    wt = obs_weights[valid]

    if len(k) == 0:
        return np.full(len(k_eval), np.nan)

    order = np.argsort(k)
    k = k[order]
    w = w[order]
    wt = wt[order]

    ku, inv = np.unique(k, return_inverse=True)

    wu_num = np.zeros(len(ku), dtype=float)
    wu_den = np.zeros(len(ku), dtype=float)

    for i, j in enumerate(inv):
        wu_num[j] += wt[i] * w[i]
        wu_den[j] += wt[i]

    wu = wu_num / np.maximum(wu_den, MACHINE_EPS)
    wtu = wu_den.copy()

    if len(ku) == 1:
        return np.repeat(wu[0], len(k_eval))

    if len(ku) < 4:
        return np.interp(k_eval, ku, wu, left=wu[0], right=wu[-1])

    s = float(smoothing_multiplier) * len(ku) * np.nanvar(wu)

    if not np.isfinite(s) or s < 0:
        s = 0.0

    try:
        spline = UnivariateSpline(
            ku,
            wu,
            w=np.sqrt(np.maximum(wtu, MACHINE_EPS)),
            k=3,
            s=s,
        )

        out = spline(k_eval)
        out = np.asarray(out, dtype=float)

        out = np.maximum(out, MACHINE_EPS)

        return out

    except Exception:
        return np.interp(k_eval, ku, wu, left=wu[0], right=wu[-1])


def smooth_one_ticker_date_expiry_spline(g, smoothing_multiplier=1.0):
    """
    Smooth one ticker-date-expiry slice.

    This is deliberately weaker than the full method because it does not
    borrow information across maturities.
    """

    g = g.copy().sort_values(["log_moneyness", "K"])

    if "total_variance_for_smoothing" not in g.columns:
        g["total_variance_for_smoothing"] = g["total_variance"]

    n_obs = len(g)
    n_distinct_strikes = g["K"].nunique()
    n_maturities = g["exdate"].nunique()
    n_distinct_tau = g["tau"].nunique()

    k = g["log_moneyness"].to_numpy(dtype=float)
    w = g["total_variance_for_smoothing"].to_numpy(dtype=float)
    obs_weights = g["obs_weight"].to_numpy(dtype=float)

    valid = (
        np.isfinite(k)
        & np.isfinite(w)
        & np.isfinite(obs_weights)
        & (w > 0)
        & (obs_weights > 0)
    )

    if valid.sum() < MIN_DISTINCT_STRIKES_FOR_RND or n_distinct_strikes < MIN_DISTINCT_STRIKES_FOR_RND:
        g["total_variance_smooth"] = g["total_variance"]
        g["iv_smooth"] = g["iv_call_equiv"]
        g["call_price_smooth"] = g["call_equiv_price"]
        g["smoothing_status"] = "too_few_strikes_spline_fallback_raw"

    else:
        w_smooth = spline_1d_total_variance_predict(
            k_train=k[valid],
            w_train=w[valid],
            obs_weights=obs_weights[valid],
            k_eval=k,
            smoothing_multiplier=smoothing_multiplier,
        )

        if np.isfinite(w_smooth).sum() != len(w_smooth):
            g["total_variance_smooth"] = g["total_variance"]
            g["iv_smooth"] = g["iv_call_equiv"]
            g["call_price_smooth"] = g["call_equiv_price"]
            g["smoothing_status"] = "spline_failed_fallback_raw"

        else:
            g["total_variance_smooth"] = np.maximum(w_smooth, MACHINE_EPS)
            g["iv_smooth"] = np.sqrt(
                g["total_variance_smooth"].to_numpy(dtype=float)
                / np.maximum(g["tau"].to_numpy(dtype=float), MACHINE_EPS)
            )

            g["call_price_smooth"] = black_call_price_forward(
                F=g["F"].to_numpy(dtype=float),
                K=g["K"].to_numpy(dtype=float),
                tau=g["tau"].to_numpy(dtype=float),
                DF=g["DF"].to_numpy(dtype=float),
                sigma=g["iv_smooth"].to_numpy(dtype=float),
            )

            g["smoothing_status"] = "smoothed_1d_spline_per_expiry"

    g["h_k"] = np.nan
    g["h_tau"] = np.nan
    g["h_tau_calendar_days"] = np.nan
    g["n_eff"] = int(valid.sum())
    g["n_date_surface_obs"] = n_obs
    g["n_date_surface_maturities"] = n_maturities
    g["n_date_surface_distinct_tau"] = n_distinct_tau

    return g


def smooth_surface_1d_spline_baseline(surf, smoothing_multiplier=1.0):
    """
    Apply 1D spline smoothing separately for each ticker-date-expiry slice.
    """

    x = surf.copy()

    x["total_variance_for_smoothing"] = x["total_variance"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)

        smoothed = (
            x.groupby(["ticker", "date", "exdate"], group_keys=False)
            .apply(
                lambda g: smooth_one_ticker_date_expiry_spline(
                    g,
                    smoothing_multiplier=smoothing_multiplier,
                )
            )
            .reset_index(drop=True)
        )

    return smoothed


def run_smoothing_pipeline_1d_spline_baseline(
    repaired_df,
    smoothing_multiplier=1.0,
):
    """
    Spline-IV baseline pipeline.

    Same as the full smoothing pipeline except:
        2D local-linear total-variance smoother
    is replaced by:
        1D per-expiry spline total-variance smoother.
    """

    surf = build_otm_call_equivalent_surface(repaired_df)

    surf = add_call_equiv_iv_and_total_variance(surf)

    surf = add_quote_reliability_weights(surf)

    smoothed = smooth_surface_1d_spline_baseline(
        surf,
        smoothing_multiplier=smoothing_multiplier,
    )

    smoothed, slice_quality = add_smoothed_slice_quality(smoothed)

    rnd_input = smoothed[
        smoothed["slice_usable_for_rnd"].fillna(False)
    ].copy()

    return smoothed, slice_quality, rnd_input


def run_spline_iv_smoothing_for_ticker(
    ticker,
    repaired_options_dir=Path("options_dir"),
    output_dir=SPLINE_IV_DIR,
    smoothing_multiplier=1.0,
    skip_existing=False,
):
    """
    Run IV-spline smoothing for one ticker.

    Important:
    - Input repaired option file is read from repaired_options_dir.
    - Output is saved to output_dir.
    - Filenames intentionally remain compatible with the existing RND extraction cell.
    """

    ticker = str(ticker).strip().upper()
    repaired_options_dir = Path(repaired_options_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    repaired_path = repaired_options_dir / f"{ticker}_options_repaired.parquet"

    out_paths = smoothing_output_paths(output_dir, ticker)

    if skip_existing and all(
        out_paths[name].exists()
        for name in ["smoothed_parquet", "quality_csv", "rnd_parquet"]
    ):
        print(f"\nSkipping {ticker}: existing spline-IV smoothing outputs found.")
        smoothed, quality, rnd_in = load_existing_smoothing_outputs(output_dir, ticker)
        status = "loaded_existing"

    else:
        if not repaired_path.exists():
            raise FileNotFoundError(f"Missing repaired option file: {repaired_path}")

        print("\n" + "=" * 72)
        print(f"Running IV-spline smoothing baseline for {ticker}")
        print(f"Input repaired options: {repaired_path}")
        print(f"Output folder:          {output_dir}")
        print("=" * 72)

        df_repaired = pd.read_parquet(repaired_path)

        smoothed, quality, rnd_in = run_smoothing_pipeline_1d_spline_baseline(
            df_repaired,
            smoothing_multiplier=smoothing_multiplier,
        )

        save_smoothing_outputs(
            smoothed_df=smoothed,
            slice_quality=quality,
            rnd_input=rnd_in,
            output_dir=output_dir,
            ticker=ticker,
        )

        status = "success"

    n_usable = (
        int(quality["slice_usable_for_rnd"].sum())
        if len(quality) and "slice_usable_for_rnd" in quality.columns
        else 0
    )

    usable_rate = (
        float(quality["slice_usable_for_rnd"].mean())
        if len(quality) and "slice_usable_for_rnd" in quality.columns
        else np.nan
    )

    status_counts = (
        smoothed["smoothing_status"].value_counts(dropna=False).to_dict()
        if len(smoothed) and "smoothing_status" in smoothed.columns
        else {}
    )

    record = {
        "ticker": ticker,
        "status": status,
        "baseline": "iv_1d_spline_per_expiry",
        "smoothing_multiplier": smoothing_multiplier,
        "smoothed_rows": int(len(smoothed)),
        "rnd_input_rows": int(len(rnd_in)),
        "n_date_expiry_slices": int(len(quality)),
        "n_usable_rnd_slices": n_usable,
        "usable_rnd_rate": usable_rate,
        "smoothing_status_counts": status_counts,
        "repaired_input_path": str(repaired_path),
        "smoothed_output_path": str(out_paths["smoothed_parquet"]),
        "rnd_input_output_path": str(out_paths["rnd_parquet"]),
    }

    print("\nSummary for", ticker)
    print("Rows in smoothed surface:", record["smoothed_rows"])
    print("Usable RND slices:", record["n_usable_rnd_slices"])
    print("Usable RND rate:", record["usable_rnd_rate"])
    print("Smoothing status counts:", status_counts)

    return {
        "ticker": ticker,
        "record": record,
        "smoothed_df": smoothed,
        "slice_quality": quality,
        "rnd_input": rnd_in,
    }


def rnd_output_paths(output_dir, ticker):
    output_dir = Path(output_dir)
    return {
        "input_parquet": output_dir / f"{ticker}_rnd_input_2d_local_linear_smoothed.parquet",
        "density_parquet": output_dir / f"{ticker}_breeden_litzenberger_rnd_density.parquet",
        "density_csv": output_dir / f"{ticker}_breeden_litzenberger_rnd_density.csv",
        "quality_csv": output_dir / f"{ticker}_breeden_litzenberger_rnd_quality.csv",
    }


def _safe_numeric(x):
    return pd.to_numeric(x, errors="coerce")


def _cumulative_trapezoid(y, x):
    """
    Cumulative trapezoidal integral without requiring scipy.
    Returns an array with the same length as x and starts at zero.
    """

    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)

    out = np.zeros_like(y, dtype=float)

    if len(y) <= 1:
        return out

    increments = 0.5 * (y[1:] + y[:-1]) * np.diff(x)
    out[1:] = np.cumsum(increments)

    return out


def integration_weights_1d(x):
    """
    Trapezoidal integration weights for an irregular one-dimensional grid.

    For a density f evaluated on x, sum(weights * f) is equivalent to the
    trapezoidal-rule integral over the retained strike support.
    """

    x = np.asarray(x, dtype=float)

    if x.ndim != 1 or len(x) < 2:
        raise ValueError("x must be a one-dimensional grid with at least two points.")

    if not np.all(np.isfinite(x)):
        raise ValueError("x must contain only finite values.")

    if not np.all(np.diff(x) > 0):
        raise ValueError("x must be strictly increasing.")

    w = np.empty_like(x, dtype=float)
    w[0] = 0.5 * (x[1] - x[0])
    w[-1] = 0.5 * (x[-1] - x[-2])

    if len(x) > 2:
        w[1:-1] = 0.5 * (x[2:] - x[:-2])

    return w


def weighted_nonnegative_density_projection(
    f_raw,
    weights,
    target_mass=None,
    tol=1e-12,
    max_iter=200,
):
    """
    Weighted projection of a raw BL density onto the nonnegative density cone
    with a fixed total mass on the retained strike grid.

    It solves:
        min_f sum_i weights_i (f_i - f_raw_i)^2
        subject to f_i >= 0 and sum_i weights_i f_i = target_mass.

    The solution has the thresholding form f_i = max(f_raw_i - lambda, 0),
    where lambda is chosen by bisection so that the weighted mass constraint
    is satisfied.
    """

    f_raw = np.asarray(f_raw, dtype=float)
    weights = np.asarray(weights, dtype=float)

    if f_raw.shape != weights.shape:
        raise ValueError("f_raw and weights must have the same shape.")

    if len(f_raw) == 0:
        return f_raw.copy()

    if np.any(~np.isfinite(f_raw)):
        raise ValueError("f_raw must contain only finite values.")

    if np.any(~np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("weights must be positive and finite.")

    raw_signed_mass = float(np.sum(weights * f_raw))
    positive_part = np.maximum(f_raw, 0.0)
    positive_mass = float(np.sum(weights * positive_part))

    if target_mass is None:
        target_mass = raw_signed_mass

    if (not np.isfinite(target_mass)) or target_mass <= tol:
        if positive_mass > tol:
            return positive_part
        return np.ones_like(f_raw) / float(np.sum(weights))

    if np.all(f_raw >= 0) and abs(raw_signed_mass - target_mass) <= tol:
        return f_raw.copy()

    lower = float(np.min(f_raw) - target_mass / np.min(weights) - 1.0)
    upper = float(np.max(f_raw))

    lam = 0.0
    for _ in range(max_iter):
        lam = 0.5 * (lower + upper)
        candidate = np.maximum(f_raw - lam, 0.0)
        mass = float(np.sum(weights * candidate))

        if abs(mass - target_mass) <= tol * max(1.0, target_mass):
            break

        if mass > target_mass:
            lower = lam
        else:
            upper = lam

    f_projected = np.maximum(f_raw - lam, 0.0)
    projected_mass = float(np.sum(weights * f_projected))

    if projected_mass > tol:
        f_projected *= target_mass / projected_mass

    return f_projected


def second_derivative_nonuniform_grid(K, C):
    """
    Non-uniform finite-difference approximation to d^2C/dK^2.

    For interior strike K_i:
        C''(K_i) ≈ 2/(K_{i+1}-K_{i-1}) *
                  [(C_{i+1}-C_i)/(K_{i+1}-K_i)
                   - (C_i-C_{i-1})/(K_i-K_{i-1})]

    This is the natural local quadratic curvature estimate on an irregular
    strike grid, so it does not require arbitrary resampling before applying
    Breeden-Litzenberger.
    """

    K = np.asarray(K, dtype=float)
    C = np.asarray(C, dtype=float)

    if len(K) < 3:
        return (
            np.array([], dtype=float),
            np.array([], dtype=float),
            np.array([], dtype=float),
            np.array([], dtype=float),
        )

    h_left = K[1:-1] - K[:-2]
    h_right = K[2:] - K[1:-1]

    valid_spacing = (h_left > 0) & (h_right > 0)

    slope_left = np.full(len(K) - 2, np.nan)
    slope_right = np.full(len(K) - 2, np.nan)
    ckk = np.full(len(K) - 2, np.nan)

    slope_left[valid_spacing] = (
        C[1:-1][valid_spacing] - C[:-2][valid_spacing]
    ) / h_left[valid_spacing]

    slope_right[valid_spacing] = (
        C[2:][valid_spacing] - C[1:-1][valid_spacing]
    ) / h_right[valid_spacing]

    ckk[valid_spacing] = (
        2.0 * (slope_right[valid_spacing] - slope_left[valid_spacing])
        / (h_left[valid_spacing] + h_right[valid_spacing])
    )

    return K[1:-1], ckk, h_left, h_right


def prepare_rnd_input_for_bl(rnd_input):
    """
    Prepare the output of the smoothing stage for Breeden-Litzenberger.

    Required fields:
        ticker, date, exdate, K, F, DF, tau, call_price_smooth

    The smoothed call price is a discounted call price. Therefore:
        risk-neutral density = (1 / DF) * d^2C/dK^2.
    """

    x = rnd_input.copy()

    required = [
        "ticker", "date", "exdate",
        "K", "F", "DF", "tau", "dte",
        "call_price_smooth"
    ]
    missing = [c for c in required if c not in x.columns]

    if missing:
        raise ValueError(f"Missing required columns for BL extraction: {missing}")

    x["date"] = pd.to_datetime(x["date"], errors="coerce")
    x["exdate"] = pd.to_datetime(x["exdate"], errors="coerce")

    numeric_cols = [
        "K", "F", "DF", "tau", "dte",
        "call_price_smooth"
    ]

    if "log_moneyness" not in x.columns:
        x["log_moneyness"] = np.log(x["K"] / x["F"])

    for col in numeric_cols + ["log_moneyness"]:
        x[col] = _safe_numeric(x[col])

    x = x[
        x["ticker"].notna() &
        x["date"].notna() &
        x["exdate"].notna() &
        np.isfinite(x["K"]) &
        np.isfinite(x["F"]) &
        np.isfinite(x["DF"]) &
        np.isfinite(x["tau"]) &
        np.isfinite(x["call_price_smooth"]) &
        (x["K"] > 0) &
        (x["F"] > 0) &
        (x["DF"] > 0) &
        (x["tau"] > 0)
    ].copy()

    return x


def extract_bl_rnd_one_slice(g):
    """
    Extract one date-expiry risk-neutral density slice from smoothed call prices.

    Output density columns:
        state_price_density:
            d^2C/dK^2, using discounted call prices.

        risk_neutral_pdf_raw:
            (1/DF) d^2C/dK^2 on the observed strike support. This is the direct
            Breeden-Litzenberger estimate and may contain small negative values
            because numerical differentiation amplifies residual curvature noise.

        risk_neutral_pdf_projected:
            closest nonnegative density on the retained strike support under a
            weighted least-squares projection with a fixed within-support mass.

        risk_neutral_pdf_truncated:
            projected density renormalized to integrate to one over the observed
            interior strike range. This is useful for plotting and within-support
            probability summaries, but it should be interpreted as a truncated-
            support distribution, not the full risk-neutral density.
    """

    g = g.copy()
    g = g.sort_values("K")

    if g["K"].duplicated().any():
        first_cols = ["ticker", "date", "exdate"]
        median_cols = [
            "K",
            "F",
            "DF",
            "tau",
            "dte",
            "log_moneyness",
            "call_price_smooth",
            "iv_smooth",
            "iv_call_equiv",
            "total_variance_smooth",
        ]
        agg_spec = {col: "first" for col in first_cols if col in g.columns}
        agg_spec.update({col: "median" for col in median_cols if col in g.columns})

        g = (
            g.groupby("K", as_index=False)
            .agg(agg_spec)
            .sort_values("K")
            .reset_index(drop=True)
        )

    ticker = str(g["ticker"].iloc[0])
    date = pd.to_datetime(g["date"].iloc[0])
    exdate = pd.to_datetime(g["exdate"].iloc[0])

    n_input_points = len(g)
    n_distinct_strikes = g["K"].nunique()

    base_quality = {
        "ticker": ticker,
        "date": date,
        "exdate": exdate,
        "n_input_points": int(n_input_points),
        "n_distinct_strikes": int(n_distinct_strikes),
        "min_K": float(g["K"].min()) if n_input_points else np.nan,
        "max_K": float(g["K"].max()) if n_input_points else np.nan,
        "min_log_moneyness": float(g["log_moneyness"].min()) if n_input_points else np.nan,
        "max_log_moneyness": float(g["log_moneyness"].max()) if n_input_points else np.nan,
        "F": float(np.nanmedian(g["F"])) if n_input_points else np.nan,
        "DF": float(np.nanmedian(g["DF"])) if n_input_points else np.nan,
        "tau": float(np.nanmedian(g["tau"])) if n_input_points else np.nan,
        "dte": float(np.nanmedian(g["dte"])) if n_input_points else np.nan,
    }

    if n_distinct_strikes < MIN_DISTINCT_STRIKES_FOR_RND:
        quality = {
            **base_quality,
            "rnd_status": "fail_too_few_distinct_strikes",
            "n_density_points": 0,
            "n_negative_raw_density_points": 0,
            "n_material_negative_raw_density_points": 0,
            "raw_density_signed_mass_observed_support": np.nan,
            "raw_density_positive_mass_observed_support": np.nan,
            "projected_density_mass_observed_support": np.nan,
            "projection_l1_adjustment": np.nan,
            "projection_l2_adjustment": np.nan,
            "projection_relative_l1_adjustment": np.nan,
            "truncated_mean": np.nan,
            "truncated_variance": np.nan,
        }
        return pd.DataFrame(), pd.Series(quality)

    K = g["K"].to_numpy(dtype=float)
    C = g["call_price_smooth"].to_numpy(dtype=float)

    F = float(np.nanmedian(g["F"]))
    DF = float(np.nanmedian(g["DF"]))
    tau = float(np.nanmedian(g["tau"]))
    dte = float(np.nanmedian(g["dte"]))

    K_mid, ckk, h_left, h_right = second_derivative_nonuniform_grid(K, C)

    finite = np.isfinite(K_mid) & np.isfinite(ckk)

    K_mid = K_mid[finite]
    ckk = ckk[finite]
    h_left = h_left[finite]
    h_right = h_right[finite]

    if len(K_mid) < MIN_INTERIOR_DENSITY_POINTS or not np.isfinite(DF) or DF <= 0:
        quality = {
            **base_quality,
            "rnd_status": "fail_too_few_finite_density_points_or_invalid_discount_factor",
            "n_density_points": int(len(K_mid)),
            "n_negative_raw_density_points": 0,
            "n_material_negative_raw_density_points": 0,
            "raw_density_signed_mass_observed_support": np.nan,
            "raw_density_positive_mass_observed_support": np.nan,
            "projected_density_mass_observed_support": np.nan,
            "projection_l1_adjustment": np.nan,
            "projection_l2_adjustment": np.nan,
            "projection_relative_l1_adjustment": np.nan,
            "truncated_mean": np.nan,
            "truncated_variance": np.nan,
        }
        return pd.DataFrame(), pd.Series(quality)

    state_price_density = ckk
    risk_neutral_pdf_raw = ckk / DF

    weights = integration_weights_1d(K_mid)

    positive_scale = 1.0 / max(F, MACHINE_EPS)
    finite_raw = risk_neutral_pdf_raw[np.isfinite(risk_neutral_pdf_raw)]
    density_scale = max(
        positive_scale,
        np.nanmedian(np.abs(finite_raw)) if len(finite_raw) else positive_scale,
    )
    density_tol = 100.0 * np.sqrt(MACHINE_EPS) * density_scale

    negative_raw = risk_neutral_pdf_raw < 0
    material_negative_raw = risk_neutral_pdf_raw < -density_tol

    raw_signed_mass = float(np.sum(weights * risk_neutral_pdf_raw))
    raw_positive_part = np.maximum(risk_neutral_pdf_raw, 0.0)
    raw_positive_mass = float(np.sum(weights * raw_positive_part))
    raw_negative_mass = float(np.sum(weights * np.maximum(-risk_neutral_pdf_raw, 0.0)))

    risk_neutral_pdf_projected = weighted_nonnegative_density_projection(
        risk_neutral_pdf_raw,
        weights,
        target_mass=raw_signed_mass,
    )

    projected_mass = float(np.sum(weights * risk_neutral_pdf_projected))

    projection_l1_adjustment = float(
        np.sum(weights * np.abs(risk_neutral_pdf_projected - risk_neutral_pdf_raw))
    )
    projection_l2_adjustment = float(
        np.sqrt(np.sum(weights * (risk_neutral_pdf_projected - risk_neutral_pdf_raw) ** 2))
    )
    projection_relative_l1_adjustment = (
        projection_l1_adjustment / raw_positive_mass
        if np.isfinite(raw_positive_mass) and raw_positive_mass > 0
        else np.nan
    )

    if np.isfinite(projected_mass) and projected_mass > 0:
        risk_neutral_pdf_truncated = risk_neutral_pdf_projected / projected_mass
        cdf_truncated = _cumulative_trapezoid(risk_neutral_pdf_truncated, K_mid)

        if len(cdf_truncated) > 0 and cdf_truncated[-1] > 0:
            cdf_truncated = cdf_truncated / cdf_truncated[-1]

        truncated_mean = np.sum(weights * K_mid * risk_neutral_pdf_truncated)
        truncated_second_moment = np.sum(weights * (K_mid ** 2) * risk_neutral_pdf_truncated)
        truncated_variance = truncated_second_moment - truncated_mean ** 2
    else:
        risk_neutral_pdf_truncated = np.full_like(risk_neutral_pdf_projected, np.nan)
        cdf_truncated = np.full_like(risk_neutral_pdf_projected, np.nan)
        truncated_mean = np.nan
        truncated_variance = np.nan

    if not np.isfinite(projected_mass) or projected_mass <= 0:
        rnd_status = "fail_zero_or_invalid_projected_density_mass"
    elif material_negative_raw.any():
        rnd_status = "pass_after_nonnegative_projection_material_adjustment"
    elif negative_raw.any():
        rnd_status = "pass_after_nonnegative_projection_small_adjustment"
    else:
        rnd_status = "pass"

    call_price_smooth_mid = g["call_price_smooth"].to_numpy(dtype=float)[1:-1][finite]

    if "iv_smooth" in g.columns:
        iv_smooth_mid = pd.to_numeric(g["iv_smooth"], errors="coerce").to_numpy(dtype=float)[1:-1][finite]
    else:
        iv_smooth_mid = np.full_like(K_mid, np.nan, dtype=float)

    iv_mid = iv_smooth_mid.copy()

    tmp_atm = pd.DataFrame({
        "K": K_mid,
        "iv_smooth": iv_smooth_mid,
    })
    tmp_atm = tmp_atm[
        np.isfinite(tmp_atm["K"]) &
        np.isfinite(tmp_atm["iv_smooth"]) &
        (tmp_atm["K"] > 0) &
        (tmp_atm["iv_smooth"] > 0)
    ].copy()

    if len(tmp_atm) > 0 and np.isfinite(F) and F > 0:
        tmp_atm["abs_log_moneyness"] = np.abs(np.log(tmp_atm["K"] / F))
        atm_iv = float(
            tmp_atm.sort_values("abs_log_moneyness")
            .head(min(2, len(tmp_atm)))["iv_smooth"]
            .median()
        )
        atm_iv_source = "nearest_smoothed_iv"
    else:
        atm_iv = np.nan
        atm_iv_source = "missing_iv"

    z_sigma_crash = 2.0

    if (
        np.isfinite(F) and F > 0 and
        np.isfinite(tau) and tau > 0 and
        np.isfinite(atm_iv) and atm_iv > 0
    ):
        alpha_dyn = float(np.exp(-z_sigma_crash * atm_iv * np.sqrt(tau)))
        K_thr = float(F * alpha_dyn)
        threshold_source = "dynamic_atm_iv_from_smoothed_surface"
    else:
        alpha_dyn = np.nan
        K_thr = np.nan
        threshold_source = "missing"

    density = pd.DataFrame({
        "ticker": ticker,
        "date": date,
        "exdate": exdate,

        "tau": tau,
        "dte": dte,
        "F": F,
        "DF": DF,

        "K": K_mid,
        "log_moneyness": np.log(K_mid / F),

        "call_price_smooth": call_price_smooth_mid,
        "iv_smooth": iv_smooth_mid,
        "iv": iv_mid,
        "atm_iv": atm_iv,
        "atm_iv_source": atm_iv_source,
        "K_thr": K_thr,
        "alpha_dyn": alpha_dyn,
        "z_sigma_crash": z_sigma_crash,
        "threshold_source": threshold_source,

        "state_price_density": state_price_density,
        "risk_neutral_pdf_raw": risk_neutral_pdf_raw,
        "risk_neutral_pdf_projected": risk_neutral_pdf_projected,
        "risk_neutral_pdf_nonnegative": risk_neutral_pdf_projected,
        "risk_neutral_pdf_truncated": risk_neutral_pdf_truncated,
        "risk_neutral_cdf_truncated": cdf_truncated,
        "integration_weight": weights,
        "left_strike_spacing": h_left,
        "right_strike_spacing": h_right,
        "density_tolerance": density_tol,
        "negative_raw_density": negative_raw,
        "material_negative_raw_density": material_negative_raw,
        "rnd_status": rnd_status,
    })

    quality = {
        **base_quality,
        "rnd_status": rnd_status,
        "n_density_points": int(len(density)),
        "n_negative_raw_density_points": int(negative_raw.sum()),
        "n_material_negative_raw_density_points": int(material_negative_raw.sum()),
        "raw_density_signed_mass_observed_support": raw_signed_mass,
        "raw_density_positive_mass_observed_support": raw_positive_mass,
        "raw_density_negative_mass_observed_support": raw_negative_mass,
        "projected_density_mass_observed_support": projected_mass,
        "raw_density_mass_observed_support": projected_mass,
        "projection_l1_adjustment": projection_l1_adjustment,
        "projection_l2_adjustment": projection_l2_adjustment,
        "projection_relative_l1_adjustment": projection_relative_l1_adjustment,
        "truncated_mean": float(truncated_mean) if np.isfinite(truncated_mean) else np.nan,
        "truncated_variance": float(truncated_variance) if np.isfinite(truncated_variance) else np.nan,
        "truncated_std": float(np.sqrt(max(truncated_variance, 0.0))) if np.isfinite(truncated_variance) else np.nan,
        "forward_minus_truncated_mean": (
            float(F - truncated_mean)
            if np.isfinite(F) and np.isfinite(truncated_mean)
            else np.nan
        ),
    }

    return density, pd.Series(quality)


def extract_bl_rnd_one_ticker(rnd_input):
    """
    Extract Breeden-Litzenberger RND slices for one ticker.
    """

    x = prepare_rnd_input_for_bl(rnd_input)

    density_parts = []
    quality_parts = []

    grouped = x.groupby(["ticker", "date", "exdate"], sort=True)

    for _, g in grouped:
        density_g, quality_g = extract_bl_rnd_one_slice(g)

        if len(density_g) > 0:
            density_parts.append(density_g)

        quality_parts.append(quality_g)

    if density_parts:
        rnd_density = pd.concat(density_parts, ignore_index=True)
    else:
        rnd_density = pd.DataFrame()

    if quality_parts:
        rnd_quality = pd.DataFrame(quality_parts)
    else:
        rnd_quality = pd.DataFrame()

    return rnd_density, rnd_quality


def save_rnd_outputs_automated(
    rnd_density,
    rnd_quality,
    output_dir,
    ticker,
    save_density_csv=False,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = rnd_output_paths(output_dir, ticker)

    rnd_density.to_parquet(paths["density_parquet"], index=False)
    rnd_quality.to_csv(paths["quality_csv"], index=False)

    if save_density_csv:
        rnd_density.to_csv(paths["density_csv"], index=False)

    return paths


def run_rnd_extraction_one_ticker(
    ticker,
    options_dir=OPTIONS_DIR,
    output_dir=OUTPUT_DIR,
    save_density_csv=SAVE_RND_DENSITY_CSV,
):
    paths = rnd_output_paths(options_dir, ticker)

    if not paths["input_parquet"].exists():
        raise FileNotFoundError(
            f"Missing smoothed RND input for {ticker}: {paths['input_parquet']}"
        )

    print(f"\n{'=' * 72}")
    print(f"Extracting Breeden-Litzenberger RND for {ticker}")
    print(f"Input: {paths['input_parquet']}")
    print(f"{'=' * 72}")

    start = time.time()

    rnd_input = pd.read_parquet(paths["input_parquet"])

    rnd_density, rnd_quality = extract_bl_rnd_one_ticker(rnd_input)

    output_paths = save_rnd_outputs_automated(
        rnd_density=rnd_density,
        rnd_quality=rnd_quality,
        output_dir=output_dir,
        ticker=ticker,
        save_density_csv=save_density_csv,
    )

    elapsed = time.time() - start

    status_counts = (
        rnd_quality["rnd_status"].value_counts(dropna=False).to_dict()
        if len(rnd_quality) and "rnd_status" in rnd_quality.columns
        else {}
    )

    record = {
        "ticker": ticker,
        "status": "success",
        "input_rows": int(len(rnd_input)),
        "density_rows": int(len(rnd_density)),
        "quality_rows": int(len(rnd_quality)),
        "n_dates": int(rnd_quality["date"].nunique()) if len(rnd_quality) else 0,
        "n_slices": int(len(rnd_quality)),
        "rnd_status_counts": status_counts,
        "elapsed_seconds": elapsed,
        "input_path": str(paths["input_parquet"]),
        "density_output_path": str(output_paths["density_parquet"]),
        "quality_output_path": str(output_paths["quality_csv"]),
    }

    print(f"Finished {ticker} in {elapsed:.1f} seconds.")
    print(f"Density rows: {record['density_rows']:,}")
    print(f"Slice statuses: {status_counts}")

    del rnd_input, rnd_density, rnd_quality
    gc.collect()

    return record


def weighted_quantile(values, weights, q):
    """
    Weighted empirical quantile.
    """

    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values = values[mask]
    weights = weights[mask]

    if len(values) == 0:
        return np.nan

    order = np.argsort(values)
    values = values[order]
    weights = weights[order]

    cum_w = np.cumsum(weights)
    cum_w = cum_w / cum_w[-1]

    return np.interp(q, cum_w, values)


def fit_pareto_downside_tail_one_slice(
    g,
    k_col="K",
    pdf_col="risk_neutral_pdf_truncated",
    weight_col="integration_weight",
    forward_col="F",
    q_tail=0.10,
    min_tail_points=3,
    min_tail_mass=1e-8,
    alpha_min=0.25,
    alpha_max=50.0,
):
    """
    Fit Pareto tail to the downside-loss variable:

        L = F / K

    Larger L means a more severe downside outcome.

    The Pareto model is fitted only on the upper tail of L,
    which corresponds to the far-left tail of terminal stock prices.
    """

    g = g.copy().sort_values(k_col)

    required_cols = [k_col, pdf_col, weight_col, forward_col]
    missing = [c for c in required_cols if c not in g.columns]

    if missing:
        return {
            "pareto_status": f"missing_columns_{missing}",
            "pareto_alpha": np.nan,
            "pareto_alpha_raw": np.nan,
            "pareto_u": np.nan,
            "pareto_tail_mass": np.nan,
            "pareto_tail_points": 0,
            "pareto_q_tail": q_tail,
            "pareto_forward": np.nan,
            "pareto_K_threshold": np.nan,
            "pareto_logm_threshold": np.nan,
        }

    K = g[k_col].to_numpy(dtype=float)
    f = g[pdf_col].to_numpy(dtype=float)
    dK = g[weight_col].to_numpy(dtype=float)

    F_values = g[forward_col].dropna().to_numpy(dtype=float)

    if len(F_values) == 0:
        return {
            "pareto_status": "missing_forward",
            "pareto_alpha": np.nan,
            "pareto_alpha_raw": np.nan,
            "pareto_u": np.nan,
            "pareto_tail_mass": np.nan,
            "pareto_tail_points": 0,
            "pareto_q_tail": q_tail,
            "pareto_forward": np.nan,
            "pareto_K_threshold": np.nan,
            "pareto_logm_threshold": np.nan,
        }

    F = float(np.nanmedian(F_values))

    valid = (
        np.isfinite(K)
        & np.isfinite(f)
        & np.isfinite(dK)
        & np.isfinite(F)
        & (K > 0)
        & (f >= 0)
        & (dK > 0)
        & (F > 0)
    )

    K = K[valid]
    f = f[valid]
    dK = dK[valid]

    if len(K) < min_tail_points + 2:
        return {
            "pareto_status": "insufficient_grid",
            "pareto_alpha": np.nan,
            "pareto_alpha_raw": np.nan,
            "pareto_u": np.nan,
            "pareto_tail_mass": np.nan,
            "pareto_tail_points": 0,
            "pareto_q_tail": q_tail,
            "pareto_forward": F,
            "pareto_K_threshold": np.nan,
            "pareto_logm_threshold": np.nan,
        }

    p = f * dK
    total_mass = float(np.sum(p))

    if not np.isfinite(total_mass) or total_mass <= 0:
        return {
            "pareto_status": "invalid_probability_mass",
            "pareto_alpha": np.nan,
            "pareto_alpha_raw": np.nan,
            "pareto_u": np.nan,
            "pareto_tail_mass": np.nan,
            "pareto_tail_points": 0,
            "pareto_q_tail": q_tail,
            "pareto_forward": F,
            "pareto_K_threshold": np.nan,
            "pareto_logm_threshold": np.nan,
        }

    p = p / total_mass

    L = F / K

    u = weighted_quantile(L, p, 1.0 - q_tail)

    if not np.isfinite(u) or u <= 0:
        return {
            "pareto_status": "invalid_tail_threshold",
            "pareto_alpha": np.nan,
            "pareto_alpha_raw": np.nan,
            "pareto_u": u,
            "pareto_tail_mass": np.nan,
            "pareto_tail_points": 0,
            "pareto_q_tail": q_tail,
            "pareto_forward": F,
            "pareto_K_threshold": np.nan,
            "pareto_logm_threshold": np.nan,
        }

    tail_mask = L >= u
    tail_points = int(np.sum(tail_mask))
    tail_mass = float(np.sum(p[tail_mask]))

    K_threshold = F / u
    logm_threshold = np.log(K_threshold / F)

    if tail_points < min_tail_points:
        return {
            "pareto_status": "insufficient_tail_points",
            "pareto_alpha": np.nan,
            "pareto_alpha_raw": np.nan,
            "pareto_u": float(u),
            "pareto_tail_mass": tail_mass,
            "pareto_tail_points": tail_points,
            "pareto_q_tail": q_tail,
            "pareto_forward": F,
            "pareto_K_threshold": K_threshold,
            "pareto_logm_threshold": logm_threshold,
        }

    if tail_mass <= min_tail_mass:
        return {
            "pareto_status": "insufficient_tail_mass",
            "pareto_alpha": np.nan,
            "pareto_alpha_raw": np.nan,
            "pareto_u": float(u),
            "pareto_tail_mass": tail_mass,
            "pareto_tail_points": tail_points,
            "pareto_q_tail": q_tail,
            "pareto_forward": F,
            "pareto_K_threshold": K_threshold,
            "pareto_logm_threshold": logm_threshold,
        }

    L_tail = L[tail_mask]
    p_tail = p[tail_mask] / tail_mass

    log_excess = np.log(L_tail / u)
    mean_log_excess = float(np.sum(p_tail * log_excess))

    if not np.isfinite(mean_log_excess) or mean_log_excess <= 0:
        return {
            "pareto_status": "degenerate_tail_excess",
            "pareto_alpha": np.nan,
            "pareto_alpha_raw": np.nan,
            "pareto_u": float(u),
            "pareto_tail_mass": tail_mass,
            "pareto_tail_points": tail_points,
            "pareto_q_tail": q_tail,
            "pareto_forward": F,
            "pareto_K_threshold": K_threshold,
            "pareto_logm_threshold": logm_threshold,
        }

    alpha_raw = 1.0 / mean_log_excess
    alpha = float(np.clip(alpha_raw, alpha_min, alpha_max))

    status = "ok" if np.isclose(alpha, alpha_raw) else "ok_alpha_clipped"

    return {
        "pareto_status": status,
        "pareto_alpha": alpha,
        "pareto_alpha_raw": float(alpha_raw),
        "pareto_u": float(u),
        "pareto_tail_mass": tail_mass,
        "pareto_tail_points": tail_points,
        "pareto_q_tail": q_tail,
        "pareto_forward": F,
        "pareto_K_threshold": K_threshold,
        "pareto_logm_threshold": logm_threshold,
    }


def run_pareto_tail_fit_for_q(rnd_df, output_dir, q_tail=0.10):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tail_rows = []

    for keys, g in rnd_df.groupby(["ticker", "date", "exdate"], dropna=False):
        ticker, date, exdate = keys

        tail_fit = fit_pareto_downside_tail_one_slice(
            g,
            q_tail=q_tail,
            min_tail_points=3,
        )

        row = {
            "ticker": ticker,
            "date": date,
            "exdate": exdate,
            "dte": float(g["dte"].median()) if "dte" in g.columns else np.nan,
            **tail_fit,
        }
        tail_rows.append(row)

    out = pd.DataFrame(tail_rows)
    out["q_tail_setting"] = float(q_tail)

    out.to_csv(output_dir / "pareto_tail_adjustment_summary.csv", index=False)
    out.to_parquet(output_dir / "pareto_tail_adjustment_summary.parquet", index=False)

    for ticker, g in out.groupby("ticker"):
        ticker = str(ticker).upper()
        g.to_csv(output_dir / f"{ticker}_pareto_tail_adjustment_summary.csv", index=False)
        g.to_parquet(output_dir / f"{ticker}_pareto_tail_adjustment_summary.parquet", index=False)

    return out
