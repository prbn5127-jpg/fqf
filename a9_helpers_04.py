# Functions from file 04 used for ablation A9

from pathlib import Path
import math
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import root

RND_DIR = Path("options_dir")
RND_FILE = None
PARETO_FILE = None
HIST_MOMENT_FILE = "options_dir/historical_moments_daily_by_ticker.csv"
TICKERS = None
OUTPUT_DIR = Path("options_dir")
TARGET_DTE = 20.0
USE_DYNAMIC_CRASH_THRESHOLD = True
Z_SIGMA_CRASH = 2.0
THRESHOLD_PRIORITY = ("dynamic_atm_iv", "input_column", "configured_K_over_F")
CRASH_K_OVER_F = None
ALLOW_FIXED_THRESHOLD_FALLBACK = False
CRASH_PROB_EPS = np.finfo(float).tiny


def intersect_rnd_with_historical_moment_dates(
    rnd,
    hist,
    output_dir=None,
    save_skipped=True,
):
    """
    Keep only RND rows whose ticker-date pair exists in the historical
    moment file.

    Match key:
        ticker, date

    Important:
        We match option valuation date, not option expiry date.
        Therefore exdate is NOT used in the intersection.
    """

    if hist is None or len(hist) == 0:
        print("Historical moment file is missing or empty. No intersection applied.")
        return rnd.copy(), pd.DataFrame()

    rnd = rnd.copy()
    hist = hist.copy()

    required_rnd = ["ticker", "date"]
    required_hist = ["ticker", "date"]

    missing_rnd = [c for c in required_rnd if c not in rnd.columns]
    missing_hist = [c for c in required_hist if c not in hist.columns]

    if missing_rnd:
        raise ValueError(f"RND data missing required columns for intersection: {missing_rnd}")

    if missing_hist:
        raise ValueError(f"Historical moments missing required columns for intersection: {missing_hist}")

    rnd["ticker"] = rnd["ticker"].astype(str).str.upper().str.strip()
    hist["ticker"] = hist["ticker"].astype(str).str.upper().str.strip()

    rnd["date"] = pd.to_datetime(rnd["date"], errors="coerce").dt.normalize()
    hist["date"] = pd.to_datetime(hist["date"], errors="coerce").dt.normalize()

    valid_pairs = (
        hist[["ticker", "date"]]
        .dropna()
        .drop_duplicates()
        .assign(has_hist_moment=True)
    )

    before_rows = len(rnd)

    before_slices = (
        rnd[["ticker", "date", "exdate"]]
        .drop_duplicates()
        .shape[0]
        if {"ticker", "date", "exdate"}.issubset(rnd.columns)
        else np.nan
    )

    tagged = rnd.merge(
        valid_pairs,
        on=["ticker", "date"],
        how="left",
    )

    kept = (
        tagged[tagged["has_hist_moment"].eq(True)]
        .drop(columns=["has_hist_moment"])
        .copy()
    )

    skipped = (
        tagged[tagged["has_hist_moment"].isna()]
        .drop(columns=["has_hist_moment"])
        .copy()
    )

    after_rows = len(kept)

    after_slices = (
        kept[["ticker", "date", "exdate"]]
        .drop_duplicates()
        .shape[0]
        if {"ticker", "date", "exdate"}.issubset(kept.columns)
        else np.nan
    )

    skipped_rows = len(skipped)

    skipped_slices = (
        skipped[["ticker", "date", "exdate"]]
        .drop_duplicates()
        .shape[0]
        if {"ticker", "date", "exdate"}.issubset(skipped.columns) and len(skipped) > 0
        else 0
    )

    print("=" * 80)
    print("Ticker-date intersection with historical moment dates")
    print(f"RND rows before:      {before_rows:,}")
    print(f"RND rows after:       {after_rows:,}")
    print(f"Skipped RND rows:     {skipped_rows:,}")
    print(f"RND slices before:    {before_slices:,}")
    print(f"RND slices after:     {after_slices:,}")
    print(f"Skipped RND slices:   {skipped_slices:,}")
    print("=" * 80)

    if save_skipped and output_dir is not None and len(skipped) > 0:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        skipped_summary = (
            skipped[["ticker", "date", "exdate"]]
            .drop_duplicates()
            .sort_values(["ticker", "date", "exdate"])
        )

        skipped_path = output_dir / "skipped_rnd_slices_missing_stock_date.csv"
        skipped_summary.to_csv(skipped_path, index=False)

        print(f"Skipped slice log saved to: {skipped_path}")

    return kept, skipped


def _norm_col(c):
    return str(c).strip().lower().replace(" ", "_").replace("-", "_")


def _first_existing_col(df, candidates):
    cols = set(df.columns)
    for c in candidates:
        c_norm = _norm_col(c)
        if c_norm in cols:
            return c_norm
    return None


def standardize_columns(df):
    """
    Rename common column variants to a canonical schema.
    """
    x = df.copy()
    x.columns = [_norm_col(c) for c in x.columns]

    alias_map = {
        "ticker": [
            "ticker", "symbol", "underlying", "underlying_symbol"
        ],
        "date": [
            "date", "trade_date", "valuation_date", "obs_date"
        ],
        "exdate": [
            "exdate", "expiry", "expiry_date", "expiration",
            "expiration_date", "maturity_date"
        ],
        "K": [
            "k", "strike", "strike_price", "strike price"
        ],
        "F": [
            "f", "forward", "forward_price", "fwd", "fwd_price",
            "undiscounted_forward"
        ],
        "F_prepaid": [
            "f_prepaid", "prepaid_forward", "forward_prepaid",
            "prepaid_fwd", "prepaid_forward_price"
        ],
        "DF": [
            "df", "discount", "discount_factor"
        ],
        "dte": [
            "dte", "days_to_expiry", "business_days_to_expiry",
            "bdays_to_expiry", "days_to_maturity"
        ],
        "tau": [
            "tau", "ttm", "time_to_maturity", "maturity",
            "tau_years", "years_to_expiry"
        ],
        "density_q": [
            "risk_neutral_pdf_projected",
            "risk_neutral_pdf_truncated",
            "projected_pdf",
            "projected_density",
            "density_projected",
            "density_q",
            "q_density",
            "rnd_density",
            "f_projected",
            "f_hat",
            "density"
        ],
        "prob_q": [
            "prob_q",
            "q_prob",
            "p_q",
            "p",
            "prob",
            "probability",
            "risk_neutral_probability",
            "risk_neutral_prob",
            "projected_prob",
            "projected_probability",
            "mass"
        ],
        "integration_weight": [
            "integration_weight",
            "delta_k",
            "dk",
            "quad_weight",
            "grid_weight",
            "weight_k"
        ],
        "K_thr": [
            "k_thr",
            "dynamic_threshold",
            "crash_threshold",
            "threshold_k",
            "k_threshold",
            "crash_k",
            "k_crash"
        ],
        "atm_iv": [
            "atm_iv",
            "iv_atm",
            "atm_vol",
            "atm_implied_vol",
            "sigma_atm",
            "atm_sigma",
            "smoothed_atm_iv"
        ],
        "iv": [
            "iv",
            "implied_volatility",
            "impl_volatility",
            "sigma",
            "vol",
            "smoothed_iv",
            "iv_smooth",
            "sigma_smooth",
            "sigma_eq",
            "call_equivalent_iv"
        ],
        "mu_hist": [
            "mu_hist", "target_mean", "hist_mean",
            "historical_mean", "mean_target"
        ],
        "var_hist": [
            "var_hist", "target_var", "hist_var",
            "historical_variance", "variance_target"
        ],
        "mu_daily": [
            "mu_daily", "daily_mean", "hist_daily_mean"
        ],
        "var_daily": [
            "var_daily", "daily_var", "daily_variance",
            "hist_daily_var"
        ],
        "pareto_status": [
            "pareto_status"
        ],
        "pareto_alpha": [
            "pareto_alpha", "alpha"
        ],
        "pareto_alpha_raw": [
            "pareto_alpha_raw", "alpha_raw"
        ],
        "pareto_u": [
            "pareto_u", "u"
        ],
        "pareto_tail_mass": [
            "pareto_tail_mass", "tail_mass", "pi_u"
        ],
        "pareto_tail_points": [
            "pareto_tail_points", "tail_points"
        ],
        "pareto_q_tail": [
            "pareto_q_tail", "q_tail"
        ],
        "pareto_K_threshold": [
            "pareto_k_threshold"
        ],
        "pareto_logm_threshold": [
            "pareto_logm_threshold"
        ],
    }

    rename = {}
    for canonical, aliases in alias_map.items():
        found = _first_existing_col(x, aliases)
        if found is not None:
            rename[found] = canonical

    return x.rename(columns=rename)


def ensure_dates(df):
    x = df.copy()
    if "date" in x.columns:
        x["date"] = pd.to_datetime(x["date"], errors="coerce")
    if "exdate" in x.columns:
        x["exdate"] = pd.to_datetime(x["exdate"], errors="coerce")
    return x


def _read_table(path):
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path}")


def find_rnd_files(rnd_dir):
    rnd_dir = Path(rnd_dir)

    patterns = [
        "*rnd*density*.parquet",
        "*RND*density*.parquet",
        "*risk_neutral*.parquet",
        "*projected*rnd*.parquet",
        "*projected*density*.parquet",
        "*rnd*.parquet",
        "*rnd*density*.csv",
        "*RND*density*.csv",
        "*risk_neutral*.csv",
        "*projected*rnd*.csv",
        "*projected*density*.csv",
        "*rnd*.csv",
    ]

    files = []
    for pat in patterns:
        files.extend(rnd_dir.glob(pat))

    return sorted(set(files))


def load_rnd_data(rnd_file=None, rnd_dir=RND_DIR):
    if rnd_file is not None:
        df = _read_table(rnd_file)
        df = standardize_columns(df)
        df = ensure_dates(df)
        return df

    files = find_rnd_files(rnd_dir)
    if not files:
        raise FileNotFoundError(
            f"No RND files found in {rnd_dir}. Set RND_FILE explicitly."
        )

    tables = []

    for f in files:
        try:
            d = _read_table(f)
            d = standardize_columns(d)
            d = ensure_dates(d)
            d["source_file"] = str(f)
            tables.append(d)
        except Exception as exc:
            print(f"Skipped {f}: {type(exc).__name__}: {exc}")

    if not tables:
        raise RuntimeError("RND files were found, but none could be loaded.")

    return pd.concat(tables, ignore_index=True)


def find_pareto_file(rnd_dir=RND_DIR):
    if PARETO_FILE is not None:
        return Path(PARETO_FILE)

    candidates = []

    for base in [Path(rnd_dir), OUTPUT_DIR, Path(".")]:
        candidates.extend(base.glob("*pareto_tail_adjustment_summary*.parquet"))
        candidates.extend(base.glob("*pareto_tail_adjustment_summary*.csv"))
        candidates.extend(base.glob("*pareto*tail*.parquet"))
        candidates.extend(base.glob("*pareto*tail*.csv"))

    candidates = sorted(set(candidates))

    if not candidates:
        return None

    return candidates[0]


def load_pareto_summary(pareto_file=None, rnd_dir=RND_DIR):
    path = Path(pareto_file) if pareto_file is not None else find_pareto_file(rnd_dir)

    if path is None:
        print("No Pareto summary file found. Q_crash will use empirical projected RND only.")
        return None

    df = _read_table(path)
    df = standardize_columns(df)
    df = ensure_dates(df)

    required = [
        "ticker", "date", "exdate",
        "pareto_status", "pareto_alpha",
        "pareto_u", "pareto_tail_mass"
    ]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Pareto summary is missing required columns: {missing}")

    print(f"Loaded Pareto summary: {path}")
    return df


def load_historical_moments(hist_file):
    if hist_file is None:
        return None

    df = _read_table(hist_file)
    df = standardize_columns(df)
    df = ensure_dates(df)

    required_base = ["ticker", "date"]
    missing_base = [c for c in required_base if c not in df.columns]

    if missing_base:
        raise ValueError(f"Historical moment file is missing: {missing_base}")

    has_horizon = {"mu_hist", "var_hist"}.issubset(df.columns)
    has_daily = {"mu_daily", "var_daily"}.issubset(df.columns)

    if not has_horizon and not has_daily:
        raise ValueError(
            "Historical moment file must contain either mu_hist/var_hist "
            "or mu_daily/var_daily."
        )

    return df


def compute_trapezoid_weights(K):
    """
    Trapezoidal integration weights for an irregular strike grid.
    Used only if no integration weights are supplied by the RND notebook.
    """
    K = np.asarray(K, dtype=float)
    n = len(K)

    if n < 2:
        return np.full(n, np.nan)

    w = np.empty(n, dtype=float)
    w[0] = 0.5 * (K[1] - K[0])
    w[-1] = 0.5 * (K[-1] - K[-2])

    if n > 2:
        w[1:-1] = 0.5 * (K[2:] - K[:-2])

    return np.maximum(w, 0.0)


def business_dte(date, exdate):
    if pd.isna(date) or pd.isna(exdate):
        return np.nan

    try:
        return float(np.busday_count(pd.Timestamp(date).date(), pd.Timestamp(exdate).date()))
    except Exception:
        return np.nan


def estimate_atm_iv_from_slice(g, F):
    """
    Estimate expiry-specific ATM IV for a ticker-date-expiry slice.

    Priority:
      1. Use existing atm_iv column if available.
      2. Otherwise use per-strike implied volatility column and take the
         median of the nearest bracketing strikes around K/F = 1.
      3. If no bracket exists, use the closest available IV observations.

    This avoids imposing a fixed number of nearest strikes when the retained
    grid is sparse or asymmetric.
    """
    if not np.isfinite(F) or F <= 0:
        return np.nan, "invalid_forward"

    if "atm_iv" in g.columns:
        atm = pd.to_numeric(g["atm_iv"], errors="coerce")
        atm = atm[np.isfinite(atm) & (atm > 0)]

        if len(atm) > 0:
            return float(np.nanmedian(atm)), "input_atm_iv"

    if "iv" not in g.columns:
        return np.nan, "missing_iv"

    tmp = g[["K", "iv"]].copy()
    tmp["K"] = pd.to_numeric(tmp["K"], errors="coerce")
    tmp["iv"] = pd.to_numeric(tmp["iv"], errors="coerce")
    tmp = tmp[np.isfinite(tmp["K"]) & np.isfinite(tmp["iv"])]
    tmp = tmp[(tmp["K"] > 0) & (tmp["iv"] > 0)].copy()

    if tmp.empty:
        return np.nan, "invalid_iv"

    tmp["log_moneyness"] = np.log(tmp["K"] / F)
    tmp = tmp[np.isfinite(tmp["log_moneyness"])].copy()

    if tmp.empty:
        return np.nan, "invalid_log_moneyness"

    selected_idx = []

    below = tmp[tmp["log_moneyness"] <= 0].copy()
    above = tmp[tmp["log_moneyness"] >= 0].copy()

    if not below.empty:
        selected_idx.append(below["log_moneyness"].idxmax())

    if not above.empty:
        selected_idx.append(above["log_moneyness"].idxmin())

    selected_idx = list(dict.fromkeys(selected_idx))

    if len(selected_idx) >= 2:
        atm_iv = float(np.nanmedian(tmp.loc[selected_idx, "iv"]))
        return atm_iv, "bracketing_iv_median"

    tmp = tmp.assign(abs_logm=np.abs(tmp["log_moneyness"]))
    tmp = tmp.sort_values("abs_logm")

    n_take = min(2, len(tmp))
    atm_iv = float(np.nanmedian(tmp.head(n_take)["iv"]))

    return atm_iv, "nearest_iv_median_one_sided"


def compute_dynamic_crash_threshold(F, tau, atm_iv, z_sigma=Z_SIGMA_CRASH):
    """
    Dynamic crash threshold:
        K_thr = F * exp(-z_sigma * atm_iv * sqrt(tau))

    Also returns:
        alpha_dyn = exp(-z_sigma * atm_iv * sqrt(tau))
    """
    if not (
        np.isfinite(F) and F > 0
        and np.isfinite(tau) and tau > 0
        and np.isfinite(atm_iv) and atm_iv > 0
        and np.isfinite(z_sigma) and z_sigma > 0
    ):
        return np.nan, np.nan

    alpha_dyn = float(np.exp(-z_sigma * atm_iv * np.sqrt(tau)))
    K_thr = float(alpha_dyn * F)

    return K_thr, alpha_dyn


def choose_crash_threshold(
    g,
    meta,
    z_sigma_crash=2.0,
    threshold_priority=("dynamic_atm_iv", "input_column", "configured_K_over_F"),
):
    """
    Choose K_thr according to THRESHOLD_PRIORITY.

    Recommended priority:
      1. dynamic_atm_iv
      2. input_column
      3. configured_K_over_F
    """
    F = meta.get("F", np.nan)
    tau = meta.get("tau", np.nan)

    atm_iv, atm_iv_source = estimate_atm_iv_from_slice(g, F)
    meta["atm_iv"] = atm_iv
    meta["atm_iv_source"] = atm_iv_source

    K_dynamic, alpha_dyn = compute_dynamic_crash_threshold(
        F=F,
        tau=tau,
        atm_iv=atm_iv,
        z_sigma=z_sigma_crash,
    )

    K_input = np.nan
    if "K_thr" in g.columns:
        k_in = pd.to_numeric(g["K_thr"], errors="coerce")
        k_in = k_in[np.isfinite(k_in) & (k_in > 0)]

        if len(k_in) > 0:
            K_input = float(np.nanmedian(k_in))

    K_fixed = np.nan
    if ALLOW_FIXED_THRESHOLD_FALLBACK and CRASH_K_OVER_F is not None:
        if np.isfinite(F) and F > 0 and np.isfinite(CRASH_K_OVER_F) and CRASH_K_OVER_F > 0:
            K_fixed = float(CRASH_K_OVER_F * F)

    for source in THRESHOLD_PRIORITY:
        if source == "dynamic_atm_iv" and USE_DYNAMIC_CRASH_THRESHOLD:
            if np.isfinite(K_dynamic) and K_dynamic > 0:
                meta["K_thr"] = K_dynamic
                meta["alpha_dyn"] = alpha_dyn
                meta["z_sigma_crash"] = float(z_sigma_crash)
                meta["threshold_source"] = "dynamic_atm_iv"
                return meta

        if source == "input_column":
            if np.isfinite(K_input) and K_input > 0:
                meta["K_thr"] = K_input
                meta["alpha_dyn"] = np.nan
                meta["z_sigma_crash"] = np.nan
                meta["threshold_source"] = "input_column"
                return meta

        if source == "configured_K_over_F":
            if np.isfinite(K_fixed) and K_fixed > 0:
                meta["K_thr"] = K_fixed
                meta["alpha_dyn"] = float(CRASH_K_OVER_F)
                meta["z_sigma_crash"] = np.nan
                meta["threshold_source"] = "configured_K_over_F"
                return meta

    meta["K_thr"] = np.nan
    meta["alpha_dyn"] = np.nan
    meta["z_sigma_crash"] = np.nan
    meta["threshold_source"] = "missing"
    meta["z_sigma_crash"] = float(z_sigma_crash)
    return meta


def prepare_one_rnd_slice(
    g,
    z_sigma_crash=2.0,
    threshold_priority=("dynamic_atm_iv", "input_column", "configured_K_over_F"),
):
    """
    Convert one ticker-date-expiry slice into sorted K, p_Q, dK and metadata.
    """
    g = g.copy()

    numeric_candidates = [
        "K", "F", "F_prepaid", "DF", "dte", "tau",
        "density_q", "prob_q", "integration_weight", "K_thr",
        "atm_iv", "iv", "mu_hist", "var_hist", "mu_daily", "var_daily"
    ]

    for col in numeric_candidates:
        if col in g.columns:
            g[col] = pd.to_numeric(g[col], errors="coerce")

    if "F" not in g.columns and {"F_prepaid", "DF"}.issubset(g.columns):
        g["F"] = g["F_prepaid"] / g["DF"]

    required = ["ticker", "date", "exdate", "K", "F"]
    missing = [c for c in required if c not in g.columns]

    if missing:
        return None, f"missing_required_columns_{missing}"

    g = g.dropna(subset=["ticker", "date", "exdate", "K", "F"])
    g = g[(g["K"] > 0) & (g["F"] > 0)].copy()

    if len(g) < 2:
        return None, "insufficient_grid"

    g = g.sort_values("K").drop_duplicates(subset=["K"], keep="last").copy()

    K = g["K"].to_numpy(dtype=float)
    F = float(np.nanmedian(g["F"].to_numpy(dtype=float)))

    if not np.isfinite(F) or F <= 0:
        return None, "invalid_forward"

    if "prob_q" in g.columns and g["prob_q"].notna().sum() >= 2:
        p = g["prob_q"].to_numpy(dtype=float)

        if "integration_weight" in g.columns:
            dK = g["integration_weight"].to_numpy(dtype=float)
        else:
            dK = compute_trapezoid_weights(K)

    elif "density_q" in g.columns:
        f = g["density_q"].to_numpy(dtype=float)

        if "integration_weight" in g.columns and g["integration_weight"].notna().sum() == len(g):
            dK = g["integration_weight"].to_numpy(dtype=float)
        else:
            dK = compute_trapezoid_weights(K)

        p = f * dK

    else:
        return None, "missing_probability_or_density"

    valid = (
        np.isfinite(K)
        & np.isfinite(p)
        & np.isfinite(F)
        & (K > 0)
        & (p >= 0)
        & (F > 0)
    )

    K = K[valid]
    p = p[valid]

    if len(dK) == len(valid):
        dK = dK[valid]
    else:
        dK = compute_trapezoid_weights(K)

    if len(K) < 2:
        return None, "insufficient_valid_grid"

    total = float(np.sum(p))

    if not np.isfinite(total) or total <= 0:
        return None, "invalid_probability_mass"

    p_q = p / total

    date = pd.to_datetime(g["date"].iloc[0], errors="coerce")
    exdate = pd.to_datetime(g["exdate"].iloc[0], errors="coerce")

    dte = float(np.nanmedian(g["dte"])) if "dte" in g.columns else np.nan
    tau = float(np.nanmedian(g["tau"])) if "tau" in g.columns else np.nan

    if not np.isfinite(dte):
        dte = business_dte(date, exdate)

    if not np.isfinite(tau):
        if pd.notna(date) and pd.notna(exdate):
            calendar_days = (exdate - date).days
            if calendar_days > 0:
                tau = calendar_days / 365.0

    meta = {
        "ticker": g["ticker"].iloc[0],
        "date": date,
        "exdate": exdate,
        "F": F,
        "DF": float(np.nanmedian(g["DF"])) if "DF" in g.columns else np.nan,
        "F_prepaid": float(np.nanmedian(g["F_prepaid"])) if "F_prepaid" in g.columns else np.nan,
        "dte": dte,
        "tau": tau,
        "n_support_points": int(len(K)),
        "source_file": g["source_file"].iloc[0] if "source_file" in g.columns else "",
    }

    for col in ["mu_hist", "var_hist", "mu_daily", "var_daily", "spot_contract",]:
        if col in g.columns:
            vals = pd.to_numeric(g[col], errors="coerce")
            vals = vals[np.isfinite(vals)]
            meta[col] = float(np.nanmedian(vals)) if len(vals) > 0 else np.nan
        else:
            meta[col] = np.nan

    meta = choose_crash_threshold(
        g,
        meta,
        z_sigma_crash=z_sigma_crash,
        threshold_priority=threshold_priority,
    )

    return {"K": K, "p_q": p_q, "dK": dK, "meta": meta}, "ok"


def attach_historical_moments(slice_meta, hist_moments):
    """
    Attach historical target mean/variance to each ticker-date-expiry slice.

    Priority:
      1. horizon moments already present in RND metadata,
      2. daily moments already present in RND metadata, scaled by DTE,
      3. external horizon moments by ticker-date,
      4. external daily moments by ticker-date, scaled by DTE.

    External merge is backward-looking by date.
    """
    meta = slice_meta.copy()
    meta["date"] = pd.to_datetime(meta["date"], errors="coerce")

    for col in ["mu_hist", "var_hist", "mu_daily", "var_daily", "spot_contract",]:
        if col not in meta.columns:
            meta[col] = np.nan
        meta[col] = pd.to_numeric(meta[col], errors="coerce")

    meta["hist_moment_source"] = "missing"

    horizon_mask = (
        np.isfinite(meta["mu_hist"])
        & np.isfinite(meta["var_hist"])
        & (meta["var_hist"] > 0)
    )

    meta.loc[horizon_mask, "hist_moment_source"] = "rnd_horizon_moments"

    daily_mask = (
        ~horizon_mask
        & np.isfinite(meta["mu_daily"])
        & np.isfinite(meta["var_daily"])
        & (meta["var_daily"] > 0)
        & np.isfinite(meta["dte"])
        & (meta["dte"] > 0)
        & np.isfinite(meta["F"])
        & (meta["F"] > 0)
        & np.isfinite(meta["spot_contract"])
        & (meta["spot_contract"] > 0)
    )

    meta.loc[daily_mask, "mu_hist"] = (
        meta.loc[daily_mask, "mu_daily"]
        * meta.loc[daily_mask, "dte"]
        - np.log(
            meta.loc[daily_mask, "F"]
            / meta.loc[daily_mask, "spot_contract"]
        )
    )
    meta.loc[daily_mask, "var_hist"] = meta.loc[daily_mask, "var_daily"] * meta.loc[daily_mask, "dte"]
    meta.loc[daily_mask, "hist_moment_source"] = "rnd_daily_moments_scaled_by_dte"

    needs_external = meta["hist_moment_source"].eq("missing")

    if hist_moments is None or not needs_external.any():
        return meta

    h = hist_moments.copy()
    h["date"] = pd.to_datetime(h["date"], errors="coerce")

    for col in ["mu_hist", "var_hist", "mu_daily", "var_daily", "spot_contract",]:
        if col in h.columns:
            h[col] = pd.to_numeric(h[col], errors="coerce")

    out_parts = []

    for ticker, m in meta.groupby("ticker", dropna=False):
        m = m.sort_values("date").copy()

        hh = h[h["ticker"].astype(str).str.upper() == str(ticker).upper()].copy()

        if hh.empty:
            out_parts.append(m)
            continue

        hh = hh.sort_values("date").copy()

        h_use = pd.DataFrame({"date": hh["date"]})

        if {"mu_hist", "var_hist"}.issubset(hh.columns):
            h_use["mu_hist_ext"] = hh["mu_hist"]
            h_use["var_hist_ext"] = hh["var_hist"]
            h_use["hist_source_ext"] = "external_horizon_moments"

        elif {"mu_daily", "var_daily", "spot_contract"}.issubset(hh.columns):
            h_use["mu_daily_ext"] = hh["mu_daily"]
            h_use["var_daily_ext"] = hh["var_daily"]
            h_use["spot_contract_ext"] = hh["spot_contract"]
            h_use["hist_source_ext"] = "external_daily_moments_scaled_by_dte"

        merged = pd.merge_asof(
            m,
            h_use.sort_values("date"),
            on="date",
            direction="backward",
            allow_exact_matches=True,
        )

        missing_mask = merged["hist_moment_source"].eq("missing")

        if {"mu_hist_ext", "var_hist_ext"}.issubset(merged.columns):
            valid_ext = (
                missing_mask
                & np.isfinite(merged["mu_hist_ext"])
                & np.isfinite(merged["var_hist_ext"])
                & (merged["var_hist_ext"] > 0)
            )

            merged.loc[valid_ext, "mu_hist"] = (
                merged.loc[valid_ext, "mu_hist_ext"]
            )

            merged.loc[valid_ext, "var_hist"] = (
                merged.loc[valid_ext, "var_hist_ext"]
            )

            merged.loc[
                valid_ext,
                "hist_moment_source"
            ] = "external_horizon_moments"

        elif {"mu_daily_ext", "var_daily_ext"}.issubset(merged.columns):
            valid_ext = (
                missing_mask
                & np.isfinite(merged["mu_daily_ext"])
                & np.isfinite(merged["var_daily_ext"])
                & (merged["var_daily_ext"] > 0)
                & np.isfinite(merged["spot_contract_ext"])
                & (merged["spot_contract_ext"] > 0)
                & np.isfinite(merged["F"])
                & (merged["F"] > 0)
                & np.isfinite(merged["dte"])
                & (merged["dte"] > 0)
            )

            merged.loc[valid_ext, "mu_hist"] = (
                merged.loc[valid_ext, "mu_daily_ext"]
                * merged.loc[valid_ext, "dte"]
                - np.log(
                    merged.loc[valid_ext, "F"]
                    / merged.loc[valid_ext, "spot_contract_ext"]
                )
            )
            merged.loc[valid_ext, "var_hist"] = merged.loc[valid_ext, "var_daily_ext"] * merged.loc[valid_ext, "dte"]
            merged.loc[valid_ext, "hist_moment_source"] = "external_daily_moments_scaled_by_dte"

        drop_cols = [
            c for c in merged.columns
            if c.endswith("_ext") or c in {"hist_source_ext"}
        ]
        merged = merged.drop(columns=drop_cols, errors="ignore")

        out_parts.append(merged)

    return pd.concat(out_parts, ignore_index=True)


def moments_from_probs(p, x):
    p = np.asarray(p, dtype=float)
    x = np.asarray(x, dtype=float)

    mu = float(np.sum(p * x))
    var = float(np.sum(p * (x - mu) ** 2))

    return mu, var


def tilted_probs(p_q, x, theta, eta):
    """
    Stable computation of:
        p_P(x_i) proportional to p_Q(x_i) exp(theta x_i + eta x_i^2).
    """
    p_q = np.asarray(p_q, dtype=float)
    x = np.asarray(x, dtype=float)

    logw = np.log(np.maximum(p_q, np.finfo(float).tiny)) + theta * x + eta * x**2

    m = np.nanmax(logw)
    w = np.exp(logw - m)
    s = np.sum(w)

    if not np.isfinite(s) or s <= 0:
        return np.full_like(p_q, np.nan, dtype=float)

    return w / s


def fit_exponential_tilt(p_q, x, target_mean, target_var):
    """
    Solve for theta and eta so the tilted distribution matches the target
    mean and variance.
    """
    p_q = np.asarray(p_q, dtype=float)
    x = np.asarray(x, dtype=float)

    valid = np.isfinite(p_q) & np.isfinite(x) & (p_q >= 0)
    p_q = p_q[valid]
    x = x[valid]

    if len(p_q) < 2 or np.sum(p_q) <= 0:
        return {
            "success": False,
            "theta": np.nan,
            "eta": np.nan,
            "p_p": np.full_like(p_q, np.nan),
            "matched_mean": np.nan,
            "matched_var": np.nan,
            "tilt_residual_mean": np.nan,
            "tilt_residual_var": np.nan,
            "tilt_residual_norm": np.nan,
            "message": "invalid_support",
        }

    p_q = p_q / np.sum(p_q)

    if not np.isfinite(target_mean) or not np.isfinite(target_var) or target_var <= 0:
        return {
            "success": False,
            "theta": np.nan,
            "eta": np.nan,
            "p_p": np.full_like(p_q, np.nan),
            "matched_mean": np.nan,
            "matched_var": np.nan,
            "tilt_residual_mean": np.nan,
            "tilt_residual_var": np.nan,
            "tilt_residual_norm": np.nan,
            "message": "invalid_target_moments",
        }

    x_min = float(np.min(x))
    x_max = float(np.max(x))

    if target_mean < x_min or target_mean > x_max:
        return {
            "success": False,
            "theta": np.nan,
            "eta": np.nan,
            "p_p": np.full_like(p_q, np.nan),
            "matched_mean": np.nan,
            "matched_var": np.nan,
            "tilt_residual_mean": np.nan,
            "tilt_residual_var": np.nan,
            "tilt_residual_norm": np.nan,
            "message": "target_mean_outside_support",
        }

    def equations(params):
        theta, eta = params
        pp = tilted_probs(p_q, x, theta, eta)

        if np.any(~np.isfinite(pp)):
            return np.array([1e10, 1e10])

        mu, var = moments_from_probs(pp, x)

        return np.array([mu - target_mean, var - target_var])

    starts = [
        np.array([0.0, 0.0]),
        np.array([1.0, 0.0]),
        np.array([-1.0, 0.0]),
        np.array([0.0, 1.0]),
        np.array([0.0, -1.0]),
    ]

    best = None

    for start in starts:
        try:
            sol = root(equations, start)
            theta, eta = sol.x

            pp = tilted_probs(p_q, x, theta, eta)

            if np.any(~np.isfinite(pp)):
                continue

            mu, var = moments_from_probs(pp, x)
            residual = np.array([mu - target_mean, var - target_var], dtype=float)
            residual_norm = float(np.linalg.norm(residual))

            candidate = {
                "success": bool(sol.success and residual_norm < 1e-6),
                "theta": float(theta),
                "eta": float(eta),
                "p_p": pp,
                "matched_mean": float(mu),
                "matched_var": float(var),
                "tilt_residual_mean": float(residual[0]),
                "tilt_residual_var": float(residual[1]),
                "tilt_residual_norm": residual_norm,
                "message": str(sol.message),
            }

            if best is None or candidate["tilt_residual_norm"] < best["tilt_residual_norm"]:
                best = candidate

        except Exception:
            continue

    if best is None:
        return {
            "success": False,
            "theta": np.nan,
            "eta": np.nan,
            "p_p": np.full_like(p_q, np.nan),
            "matched_mean": np.nan,
            "matched_var": np.nan,
            "tilt_residual_mean": np.nan,
            "tilt_residual_var": np.nan,
            "tilt_residual_norm": np.nan,
            "message": "solver_exception_all_starts",
        }

    if not best["success"]:
        best["message"] = "tilt_not_converged_" + best["message"]

    return best


def _safe_weighted_average(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    valid = (
        np.isfinite(values)
        & np.isfinite(weights)
        & (weights > 0)
    )

    if not valid.any():
        return np.nan

    values = values[valid]
    weights = weights[valid]

    return float(np.sum(values * weights) / np.sum(weights))


def empirical_crash_probability(K, p, K_thr):
    K = np.asarray(K, dtype=float)
    p = np.asarray(p, dtype=float)

    if not np.isfinite(K_thr) or K_thr <= 0:
        return np.nan

    valid = np.isfinite(K) & np.isfinite(p) & (p >= 0) & (K > 0)

    K = K[valid]
    p = p[valid]

    if len(K) == 0 or np.sum(p) <= 0:
        return np.nan

    p = p / np.sum(p)

    return float(np.sum(p[K <= K_thr]))


def q_crash_with_pareto(
    K,
    p_q,
    F,
    K_thr,
    pareto_row=None,
    use_pareto_tail_adjustment: bool = True,
):
    """
    Risk-neutral crash probability.

    If Pareto parameters are available and the crash threshold lies inside
    the fitted Pareto tail, use:
        pi_u * (u / ell_thr)^alpha,
    where:
        ell_thr = F / K_thr.

    Otherwise, use empirical projected RND mass:
        sum_{K_i <= K_thr} p_i^Q.
    """
    empirical = empirical_crash_probability(K, p_q, K_thr)

    if not use_pareto_tail_adjustment:
        return empirical, "empirical_projected_rnd_pareto_disabled"

    if pareto_row is None:
        return empirical, "empirical_projected_rnd_no_pareto"

    if not np.isfinite(F) or not np.isfinite(K_thr) or F <= 0 or K_thr <= 0:
        return empirical, "empirical_projected_rnd_invalid_threshold"

    status = pareto_row.get("pareto_status", None)
    alpha = pareto_row.get("pareto_alpha", np.nan)
    u = pareto_row.get("pareto_u", np.nan)
    tail_mass = pareto_row.get("pareto_tail_mass", np.nan)

    if status not in {"ok", "ok_alpha_clipped"}:
        return empirical, f"empirical_projected_rnd_pareto_status_{status}"

    valid_pareto = (
        np.isfinite(alpha)
        and np.isfinite(u)
        and np.isfinite(tail_mass)
        and alpha > 0
        and u > 0
        and tail_mass >= 0
    )

    if not valid_pareto:
        return empirical, "empirical_projected_rnd_invalid_pareto_params"

    ell_thr = F / K_thr

    if ell_thr >= u:
        q_pareto = float(tail_mass * (u / ell_thr) ** alpha)
        return q_pareto, "pareto_tail_survival"

    return empirical, "empirical_projected_rnd_threshold_outside_pareto_tail"


def p_crash_with_pareto(
    K,
    p_p,
    F,
    K_thr,
    pareto_row=None,
    use_pareto_tail_adjustment=True,
):
    """
    Physical crash probability using the SAME Pareto threshold u
    used for the risk-neutral tail.

    The Pareto tail index is re-estimated using physical masses p_P.
    """

    empirical = empirical_crash_probability(K, p_p, K_thr)

    if not use_pareto_tail_adjustment:
        return empirical, "empirical_physical_pareto_disabled", np.nan, np.nan

    if pareto_row is None:
        return empirical, "empirical_physical_no_pareto", np.nan, np.nan

    status = pareto_row.get("pareto_status", None)
    u = pareto_row.get("pareto_u", np.nan)

    if status not in {"ok", "ok_alpha_clipped"}:
        return (
            empirical,
            f"empirical_physical_pareto_status_{status}",
            np.nan,
            np.nan,
        )

    if (
        not np.isfinite(u)
        or u <= 0
        or not np.isfinite(F)
        or F <= 0
        or not np.isfinite(K_thr)
        or K_thr <= 0
    ):
        return empirical, "empirical_physical_invalid_pareto", np.nan, np.nan

    K = np.asarray(K, dtype=float)
    p_p = np.asarray(p_p, dtype=float)

    valid = (
        np.isfinite(K)
        & np.isfinite(p_p)
        & (K > 0)
        & (p_p >= 0)
    )

    K = K[valid]
    p_p = p_p[valid]

    if len(K) == 0 or np.sum(p_p) <= 0:
        return empirical, "empirical_physical_invalid_grid", np.nan, np.nan

    p_p = p_p / np.sum(p_p)

    L = F / K
    ell_thr = F / K_thr

    if ell_thr < u:
        return (
            empirical,
            "empirical_physical_threshold_outside_pareto_tail",
            np.nan,
            np.nan,
        )

    tail = np.isfinite(L) & (L >= u)

    if tail.sum() < 2:
        return empirical, "empirical_physical_insufficient_tail", np.nan, np.nan

    pi_u_p = float(np.sum(p_p[tail]))

    if not np.isfinite(pi_u_p) or pi_u_p <= 0:
        return empirical, "empirical_physical_zero_tail_mass", np.nan, np.nan

    weights = p_p[tail] / pi_u_p

    denominator = float(
        np.sum(
            weights
            * np.log(L[tail] / u)
        )
    )

    if not np.isfinite(denominator) or denominator <= 0:
        return empirical, "empirical_physical_invalid_alpha", np.nan, pi_u_p

    alpha_p = 1.0 / denominator

    P_crash = pi_u_p * (u / ell_thr) ** alpha_p
    P_crash = float(np.clip(P_crash, 0.0, 1.0))

    return P_crash, "pareto_physical_tail_survival", alpha_p, pi_u_p


def data_adaptive_gate_bandwidth(dte_values):
    """
    Data-derived maturity bandwidth.

    If multiple maturities exist on a date, use the median positive spacing
    of the observed DTE grid. This adapts to the actual expiry calendar.
    """
    d = np.asarray(dte_values, dtype=float)
    d = np.sort(d[np.isfinite(d)])

    if len(d) <= 1:
        return np.inf

    unique = np.unique(d)

    if len(unique) <= 1:
        return np.inf

    spacings = np.diff(unique)
    spacings = spacings[np.isfinite(spacings) & (spacings > 0)]

    if len(spacings) == 0:
        return np.inf

    return float(np.nanmedian(spacings))


def gaussian_maturity_weights(dte_values, target_dte):
    d = np.asarray(dte_values, dtype=float)
    h = data_adaptive_gate_bandwidth(d)

    if not np.isfinite(h) or h <= 0:
        w = np.ones_like(d, dtype=float)
    else:
        w = np.exp(-0.5 * ((d - target_dte) / h) ** 2)

    w = np.where(np.isfinite(w), w, 0.0)

    if np.sum(w) <= 0:
        w = np.ones_like(d, dtype=float)

    return w / np.sum(w), h


def add_maturity_gates(term_df, target_dte=TARGET_DTE):
    if term_df.empty:
        return term_df.copy()

    out_parts = []

    for (ticker, date), g in term_df.groupby(["ticker", "date"], dropna=False):
        g = g.sort_values("dte").copy()

        weights, h = gaussian_maturity_weights(
            g["dte"].to_numpy(dtype=float),
            target_dte,
        )

        q_vals = g["Q_crash"].to_numpy(dtype=float)
        p_vals = g["P_crash"].to_numpy(dtype=float)

        valid_q = np.isfinite(q_vals) & np.isfinite(weights)
        valid_p = np.isfinite(p_vals) & np.isfinite(weights)

        if valid_q.sum() == 0 or np.sum(weights[valid_q]) <= 0:
            gate_q = np.nan
        else:
            gate_q = float(
                np.sum(weights[valid_q] * q_vals[valid_q])
                / np.sum(weights[valid_q])
            )

        if valid_p.sum() == 0 or np.sum(weights[valid_p]) <= 0:
            gate_p = np.nan
        else:
            gate_p = float(
                np.sum(weights[valid_p] * p_vals[valid_p])
                / np.sum(weights[valid_p])
            )

        if np.isfinite(gate_q) and np.isfinite(gate_p):
            gate_premium = float(
                np.log(
                    (gate_q + CRASH_PROB_EPS)
                    / (gate_p + CRASH_PROB_EPS)
                )
            )
        else:
            gate_premium = np.nan

        g["Gate_Q"] = gate_q
        g["Gate_P"] = gate_p
        g["Gate_premium"] = gate_premium

        g["gate_target_dte"] = target_dte
        g["gate_bandwidth_dte"] = h
        g["gate_weight"] = weights

        out_parts.append(g)

    return pd.concat(out_parts, ignore_index=True)


def build_forced_daily_output_from_rnd(
    rnd,
    hist,
    strict_term=None,
    target_dte=20,
    z_sigma_crash: float = Z_SIGMA_CRASH,
):
    """
    Produce one output row for every ticker-date intersection.

    Priority:
        1. strict_entropy_tilt
           Use strict maturity-gated crash premium if available.

        2. fallback_historical_normal
           If strict tilt failed, compute Q_crash from RND and approximate
           P_crash using historical normal log-return moments.

        3. unavailable_no_valid_rnd_expiry
           If even Q_crash cannot be computed, still keep the ticker-date row
           but mark output_available = False.
    """

    rnd = rnd.copy()
    hist = hist.copy()

    required_rnd = [
        "ticker",
        "date",
        "exdate",
        "K",
        "dte",
        "F",
        "K_thr",
        "alpha_dyn",
        "atm_iv",
        "risk_neutral_pdf_truncated",
        "integration_weight",
    ]

    required_hist = [
        "ticker",
        "date",
        "mu_daily",
        "var_daily",
        "spot_contract",
    ]

    missing_rnd = [c for c in required_rnd if c not in rnd.columns]
    missing_hist = [c for c in required_hist if c not in hist.columns]

    if missing_rnd:
        raise ValueError(f"RND data missing required columns for forced output: {missing_rnd}")

    if missing_hist:
        raise ValueError(f"Historical moments missing required columns for forced output: {missing_hist}")

    rnd["ticker"] = rnd["ticker"].astype(str).str.upper().str.strip()
    hist["ticker"] = hist["ticker"].astype(str).str.upper().str.strip()

    rnd["date"] = pd.to_datetime(rnd["date"], errors="coerce").dt.normalize()
    rnd["exdate"] = pd.to_datetime(rnd["exdate"], errors="coerce").dt.normalize()
    hist["date"] = pd.to_datetime(hist["date"], errors="coerce").dt.normalize()

    rnd_dates = rnd[["ticker", "date"]].dropna().drop_duplicates()
    hist_dates = hist[["ticker", "date"]].dropna().drop_duplicates()

    date_grid = (
        rnd_dates
        .merge(hist_dates, on=["ticker", "date"], how="inner")
        .drop_duplicates()
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )

    hist_small = (
        hist[["ticker", "date", "mu_daily", "var_daily", "spot_contract",]]
        .drop_duplicates(["ticker", "date"])
        .copy()
    )

    date_grid = date_grid.merge(
        hist_small,
        on=["ticker", "date"],
        how="left",
    )

    strict_daily = pd.DataFrame()

    if strict_term is not None and len(strict_term) > 0:
        s = strict_term.copy()
        s["ticker"] = s["ticker"].astype(str).str.upper().str.strip()
        s["date"] = pd.to_datetime(s["date"], errors="coerce").dt.normalize()

        if "Gate_premium" in s.columns:
            strict_candidates = s[s["Gate_premium"].notna()].copy()
        else:
            strict_candidates = pd.DataFrame()

        if len(strict_candidates) > 0:
            agg_spec = {
                "Gate_Q": "first",
                "Gate_P": "first",
                "Gate_premium": "first",
                "gate_target_dte": "first",
                "gate_bandwidth_dte": "first",
                "dte": "count",
            }

            if "n_support_points" in strict_candidates.columns:
                agg_spec["n_support_points"] = "sum"

            strict_daily = (
                strict_candidates
                .sort_values(["ticker", "date", "dte"])
                .groupby(["ticker", "date"], as_index=False)
                .agg(agg_spec)
                .rename(columns={
                    "Gate_Q": "Q_crash",
                    "Gate_P": "P_crash",
                    "Gate_premium": "crash_premium",
                    "dte": "n_expiries_used",
                })
            )

            strict_daily["mean_dte_used"] = (
                strict_candidates
                .groupby(["ticker", "date"])["dte"]
                .mean()
                .to_numpy()
            )

            strict_daily["estimation_status"] = "strict_entropy_tilt"
            strict_daily["fallback_used"] = False

    fallback_rows = []

    hist_lookup = hist_small.set_index(["ticker", "date"])

    for (ticker, date), g_date in rnd.groupby(["ticker", "date"], sort=True):
        if (ticker, date) not in hist_lookup.index:
            continue

        h = hist_lookup.loc[(ticker, date)]

        if isinstance(h, pd.DataFrame):
            h = h.iloc[0]

        mu_daily = float(h["mu_daily"])
        var_daily = float(h["var_daily"])

        expiry_rows = []

        for exdate, g_slice in g_date.groupby("exdate", sort=True):
            dte = float(np.nanmedian(pd.to_numeric(g_slice["dte"], errors="coerce")))
            F = float(np.nanmedian(pd.to_numeric(g_slice["F"], errors="coerce")))
            atm_iv = float(np.nanmedian(pd.to_numeric(g_slice["atm_iv"], errors="coerce")))
            calendar_days = (
                (pd.Timestamp(exdate) - pd.Timestamp(date)).days
                if pd.notna(exdate) and pd.notna(date)
                else np.nan
            )

            tau = (
                calendar_days / 365.0
                if np.isfinite(calendar_days) and calendar_days > 0
                else np.nan
            )

            K_thr, alpha_dyn = compute_dynamic_crash_threshold(
                F=F,
                tau=tau,
                atm_iv=atm_iv,
                z_sigma=z_sigma_crash,
            )

            if not np.isfinite(K_thr) or K_thr <= 0:
                K_thr = float(np.nanmedian(pd.to_numeric(g_slice["K_thr"], errors="coerce")))
                alpha_dyn = float(np.nanmedian(pd.to_numeric(g_slice["alpha_dyn"], errors="coerce")))

            g_slice = g_slice.copy()
            g_slice["K_thr"] = K_thr
            g_slice["alpha_dyn"] = alpha_dyn

            Q_crash = compute_q_crash_from_rnd_slice(g_slice)

            spot_contract = float(h["spot_contract"])

            P_crash = compute_physical_crash_prob_historical_normal(
                alpha_dyn=alpha_dyn,
                F=F,
                spot_contract=spot_contract,
                dte=dte,
                mu_daily=mu_daily,
                var_daily=var_daily,
            )

            if np.isfinite(Q_crash) and np.isfinite(P_crash):
                crash_premium = float(
                    np.log(
                        (Q_crash + CRASH_PROB_EPS)
                        / (P_crash + CRASH_PROB_EPS)
                    )
                )

                status = "fallback_historical_normal"
                fallback_used = True
            else:
                continue

            expiry_rows.append({
                "ticker": ticker,
                "date": date,
                "exdate": exdate,
                "dte": dte,
                "F": F,
                "K_thr": K_thr,
                "alpha_dyn": alpha_dyn,
                "atm_iv": atm_iv,
                "Q_crash": Q_crash,
                "P_crash": P_crash,
                "crash_premium": crash_premium,
                "estimation_status": status,
                "fallback_used": fallback_used,
            })

        expiry_df = pd.DataFrame(expiry_rows)

        if expiry_df.empty:
            fallback_rows.append({
                "ticker": ticker,
                "date": date,
                "Q_crash": np.nan,
                "P_crash": np.nan,
                "crash_premium": np.nan,
                "gate_target_dte": target_dte,
                "gate_bandwidth_dte": np.nan,
                "n_expiries_used": 0,
                "mean_dte_used": np.nan,
                "estimation_status": "unavailable_no_valid_rnd_expiry",
                "fallback_used": True,
            })
            continue

        weights = maturity_gate_weights(
            expiry_df["dte"].to_numpy(dtype=float),
            target_dte=target_dte,
        )

        gate_q = _safe_weighted_average(
            expiry_df["Q_crash"],
            weights,
        )

        gate_p = _safe_weighted_average(
            expiry_df["P_crash"],
            weights,
        )

        if np.isfinite(gate_q) and np.isfinite(gate_p):
            gate_premium = float(
                np.log(
                    (gate_q + CRASH_PROB_EPS)
                    / (gate_p + CRASH_PROB_EPS)
                )
            )
        else:
            gate_premium = np.nan

        fallback_rows.append({
            "ticker": ticker,
            "date": date,
            "Q_crash": gate_q,
            "P_crash": gate_p,
            "crash_premium": gate_premium,
            "gate_target_dte": target_dte,
            "gate_bandwidth_dte": np.nan,
            "n_expiries_used": int(len(expiry_df)),
            "mean_dte_used": float(
                np.nanmean(expiry_df["dte"])
            ),
            "estimation_status": "fallback_historical_normal",
            "fallback_used": True,
        })

    fallback_daily = pd.DataFrame(fallback_rows)

    forced = date_grid[["ticker", "date", "mu_daily", "var_daily"]].copy()

    if len(strict_daily) > 0:
        forced = forced.merge(
            strict_daily,
            on=["ticker", "date"],
            how="left",
        )
    else:
        forced["Q_crash"] = np.nan
        forced["P_crash"] = np.nan
        forced["crash_premium"] = np.nan
        forced["gate_target_dte"] = np.nan
        forced["gate_bandwidth_dte"] = np.nan
        forced["n_expiries_used"] = np.nan
        forced["mean_dte_used"] = np.nan
        forced["estimation_status"] = np.nan
        forced["fallback_used"] = np.nan

    if len(fallback_daily) > 0:
        fallback_daily = fallback_daily.rename(columns={
            "Q_crash": "Q_crash_fallback",
            "P_crash": "P_crash_fallback",
            "crash_premium": "crash_premium_fallback",
            "gate_target_dte": "gate_target_dte_fallback",
            "gate_bandwidth_dte": "gate_bandwidth_dte_fallback",
            "n_expiries_used": "n_expiries_used_fallback",
            "mean_dte_used": "mean_dte_used_fallback",
            "estimation_status": "estimation_status_fallback",
            "fallback_used": "fallback_used_fallback",
        })

        forced = forced.merge(
            fallback_daily,
            on=["ticker", "date"],
            how="left",
        )

        use_fallback = forced["crash_premium"].isna()

        fill_pairs = [
            ("Q_crash", "Q_crash_fallback"),
            ("P_crash", "P_crash_fallback"),
            ("crash_premium", "crash_premium_fallback"),
            ("gate_target_dte", "gate_target_dte_fallback"),
            ("gate_bandwidth_dte", "gate_bandwidth_dte_fallback"),
            ("n_expiries_used", "n_expiries_used_fallback"),
            ("mean_dte_used", "mean_dte_used_fallback"),
            ("estimation_status", "estimation_status_fallback"),
            ("fallback_used", "fallback_used_fallback"),
        ]

        for main_col, fallback_col in fill_pairs:
            if fallback_col in forced.columns:
                forced.loc[use_fallback, main_col] = forced.loc[use_fallback, fallback_col]

        helper_cols = [c for c in forced.columns if c.endswith("_fallback")]
        forced = forced.drop(columns=helper_cols)

    forced["forced_output"] = True

    forced["output_available"] = (
        forced["Q_crash"].notna()
        & forced["P_crash"].notna()
        & forced["crash_premium"].notna()
    )

    valid_premium = (
        forced["Q_crash"].notna()
        & forced["P_crash"].notna()
        & (forced["Q_crash"] >= 0)
        & (forced["P_crash"] >= 0)
    )

    forced.loc[valid_premium, "crash_premium"] = np.log(
        (forced.loc[valid_premium, "Q_crash"] + CRASH_PROB_EPS)
        / (forced.loc[valid_premium, "P_crash"] + CRASH_PROB_EPS)
    )

    forced.loc[
        forced["estimation_status"].isna(),
        "estimation_status"
    ] = "unavailable_no_valid_rnd_expiry"

    forced.loc[
        forced["fallback_used"].isna(),
        "fallback_used"
    ] = True

    return forced.sort_values(["ticker", "date"]).reset_index(drop=True)


def build_slice_metadata(rnd_df, z_sigma_crash=2.0):
    rows = []
    failures = []

    for keys, g in rnd_df.groupby(["ticker", "date", "exdate"], dropna=False):
        prepared, status = prepare_one_rnd_slice(
            g,
            z_sigma_crash=z_sigma_crash,
        )

        if status == "ok":
            rows.append(prepared["meta"])
        else:
            failures.append({
                "ticker": keys[0],
                "date": keys[1],
                "exdate": keys[2],
                "stage": "prepare_one_rnd_slice",
                "reason": status,
            })

    return pd.DataFrame(rows), pd.DataFrame(failures)


def merge_pareto_into_meta(slice_meta, pareto_df):
    meta = slice_meta.copy()

    if pareto_df is None:
        return meta

    p = pareto_df.copy()

    keep_cols = [
        "ticker", "date", "exdate",
        "pareto_status", "pareto_alpha", "pareto_alpha_raw",
        "pareto_u", "pareto_tail_mass", "pareto_tail_points",
        "pareto_q_tail", "pareto_K_threshold", "pareto_logm_threshold",
    ]

    keep_cols = [c for c in keep_cols if c in p.columns]
    p = p[keep_cols].copy()

    return meta.merge(
        p,
        on=["ticker", "date", "exdate"],
        how="left",
    )


def compute_one_slice(
    row,
    rnd_group,
    use_pareto_tail_adjustment: bool = True,
    z_sigma_crash: float = 2.0,
):
    prepared, status = prepare_one_rnd_slice(
        rnd_group,
        z_sigma_crash=z_sigma_crash,
    )

    if status != "ok":
        return None, status

    K = prepared["K"]
    p_q = prepared["p_q"]
    meta = prepared["meta"]

    F = float(meta["F"])

    K_thr = row.get("K_thr", meta.get("K_thr", np.nan))
    K_thr = float(K_thr) if np.isfinite(K_thr) else np.nan

    if not np.isfinite(K_thr) or K_thr <= 0:
        return None, "missing_or_invalid_crash_threshold_inputs"

    target_mean = row.get("mu_hist", np.nan)
    target_var = row.get("var_hist", np.nan)

    if not np.isfinite(target_mean) or not np.isfinite(target_var) or target_var <= 0:
        return None, "missing_or_invalid_historical_moments"

    x = np.log(K / F)

    mu_q, var_q = moments_from_probs(p_q, x)

    tilt = fit_exponential_tilt(
        p_q=p_q,
        x=x,
        target_mean=float(target_mean),
        target_var=float(target_var),
    )

    if not tilt["success"]:
        return None, f"tilt_failed_{tilt['message']}"

    p_p = tilt["p_p"]

    pareto_row = row.to_dict()

    Q_crash, q_source = q_crash_with_pareto(
        K=K,
        p_q=p_q,
        F=F,
        K_thr=K_thr,
        pareto_row=pareto_row,
        use_pareto_tail_adjustment=use_pareto_tail_adjustment,
    )

    P_crash, p_source, alpha_p, tail_mass_p = p_crash_with_pareto(
        K=K,
        p_p=p_p,
        F=F,
        K_thr=K_thr,
        pareto_row=pareto_row,
        use_pareto_tail_adjustment=use_pareto_tail_adjustment,
    )

    if not np.isfinite(Q_crash) or not np.isfinite(P_crash):
        return None, "invalid_crash_probability"

    crash_premium = float(
        np.log((Q_crash + CRASH_PROB_EPS) / (P_crash + CRASH_PROB_EPS))
    )

    ell_thr = F / K_thr

    out = {
        "ticker": row["ticker"],
        "date": row["date"],
        "exdate": row["exdate"],
        "dte": row.get("dte", np.nan),
        "tau": row.get("tau", np.nan),
        "F": F,
        "F_prepaid": row.get("F_prepaid", np.nan),
        "DF": row.get("DF", np.nan),

        "atm_iv": row.get("atm_iv", np.nan),
        "atm_iv_source": row.get("atm_iv_source", ""),
        "K_thr": K_thr,
        "ell_thr": ell_thr,
        "alpha_dyn": row.get("alpha_dyn", np.nan),
        "z_sigma_crash": row.get("z_sigma_crash", np.nan),
        "threshold_source": row.get("threshold_source", ""),

        "Q_crash": Q_crash,
        "Q_crash_source": q_source,
        "P_crash": P_crash,
        "P_crash_source": p_source,
        "pareto_alpha_P": alpha_p,
        "pareto_tail_mass_P": tail_mass_p,
        "crash_premium": crash_premium,

        "theta": tilt["theta"],
        "eta": tilt["eta"],
        "mu_q": mu_q,
        "var_q": var_q,
        "target_mean": target_mean,
        "target_var": target_var,
        "matched_mean": tilt["matched_mean"],
        "matched_var": tilt["matched_var"],
        "tilt_residual_mean": tilt["tilt_residual_mean"],
        "tilt_residual_var": tilt["tilt_residual_var"],
        "tilt_residual_norm": tilt["tilt_residual_norm"],
        "tilt_message": tilt["message"],
        "hist_moment_source": row.get("hist_moment_source", ""),

        "n_support_points": row.get("n_support_points", len(K)),

        "pareto_status": row.get("pareto_status", np.nan),
        "pareto_alpha": row.get("pareto_alpha", np.nan),
        "pareto_alpha_raw": row.get("pareto_alpha_raw", np.nan),
        "pareto_u": row.get("pareto_u", np.nan),
        "pareto_tail_mass": row.get("pareto_tail_mass", np.nan),
        "pareto_tail_points": row.get("pareto_tail_points", np.nan),
        "pareto_q_tail": row.get("pareto_q_tail", np.nan),
        "pareto_K_threshold": row.get("pareto_K_threshold", np.nan),
        "pareto_logm_threshold": row.get("pareto_logm_threshold", np.nan),

        "source_file": row.get("source_file", ""),
    }

    out["Q_downside"] = Q_crash
    out["P_downside"] = P_crash
    out["downside_pricing_wedge"] = crash_premium

    out["digital_downside_price"] = (
        out["DF"] * Q_crash
        if (
            np.isfinite(out["DF"])
            and np.isfinite(Q_crash)
        )
        else np.nan
    )

    return out, "ok"


def run_crash_premium_pipeline(
        rnd_file=RND_FILE,
        rnd_dir=RND_DIR,
        pareto_file=PARETO_FILE,
        hist_moment_file=HIST_MOMENT_FILE,
        tickers=TICKERS,
        output_dir=OUTPUT_DIR,
        target_dte=TARGET_DTE,
        use_pareto_tail_adjustment: bool = False,
        z_sigma_crash: float = 2.0,
    ):
    """
    Run strict expiry-level crash-premium estimation and always create
    a forced daily output.

    Outputs saved inside output_dir:
        strict_crash_premium_term_structure.csv/parquet
        failure_log_pareto_integrated.csv
        slice_metadata.csv
        forced_daily_crash_premium.csv/parquet

    The forced daily file has one row per ticker-date intersection between:
        RND option valuation dates
        historical stock-moment dates
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rnd = load_rnd_data(rnd_file=rnd_file, rnd_dir=rnd_dir)

    required = ["ticker", "date", "exdate", "K"]
    missing = [c for c in required if c not in rnd.columns]

    if missing:
        raise ValueError(f"RND input is missing required columns: {missing}")

    if "F" not in rnd.columns:
        if {"F_prepaid", "DF"}.issubset(rnd.columns):
            rnd["F_prepaid"] = pd.to_numeric(rnd["F_prepaid"], errors="coerce")
            rnd["DF"] = pd.to_numeric(rnd["DF"], errors="coerce")
            rnd["F"] = rnd["F_prepaid"] / rnd["DF"]
        else:
            raise ValueError("RND input must contain F, or both F_prepaid and DF.")

    if tickers is not None:
        if isinstance(tickers, str):
            tickers = [tickers]

        tickers = [str(t).upper() for t in tickers]
        rnd = rnd[rnd["ticker"].astype(str).str.upper().isin(tickers)].copy()

    pareto = load_pareto_summary(pareto_file=pareto_file, rnd_dir=rnd_dir)
    hist = load_historical_moments(hist_moment_file)

    rnd, skipped_missing_stock_date = intersect_rnd_with_historical_moment_dates(
        rnd=rnd,
        hist=hist,
        output_dir=output_dir,
        save_skipped=True,
    )

    if rnd.empty:
        raise RuntimeError(
            "No RND rows remain after ticker-date intersection with historical moments."
        )

    slice_meta, prep_failures = build_slice_metadata(
        rnd,
        z_sigma_crash=z_sigma_crash,
    )

    if slice_meta.empty:
        print("No valid strict RND slice metadata prepared.")
        print("The notebook will still create forced daily outputs.")
        slice_meta = pd.DataFrame()
        prep_failures = pd.DataFrame()

    else:
        slice_meta = merge_pareto_into_meta(slice_meta, pareto)
        slice_meta = attach_historical_moments(slice_meta, hist)

    grouped = {
        keys: g
        for keys, g in rnd.groupby(["ticker", "date", "exdate"], dropna=False)
    }

    rows = []
    failures = []

    if not slice_meta.empty:
        for _, row in slice_meta.iterrows():
            keys = (row["ticker"], row["date"], row["exdate"])
            g = grouped.get(keys, None)

            if g is None:
                failures.append({
                    "ticker": row["ticker"],
                    "date": row["date"],
                    "exdate": row["exdate"],
                    "stage": "lookup_rnd_group",
                    "reason": "missing_group",
                })
                continue

            result, status = compute_one_slice(
                row,
                g,
                use_pareto_tail_adjustment=use_pareto_tail_adjustment,
                z_sigma_crash=z_sigma_crash,
            )

            if status == "ok":
                rows.append(result)

            else:
                failures.append({
                    "ticker": row["ticker"],
                    "date": row["date"],
                    "exdate": row["exdate"],
                    "stage": "compute_one_slice",
                    "reason": status,
                    "threshold_source": row.get("threshold_source", ""),
                    "atm_iv_source": row.get("atm_iv_source", ""),
                    "hist_moment_source": row.get("hist_moment_source", ""),
                })

    term = pd.DataFrame(rows)

    skipped_log = pd.DataFrame()

    if len(skipped_missing_stock_date) > 0:
        skipped_log = (
            skipped_missing_stock_date[["ticker", "date", "exdate"]]
            .drop_duplicates()
            .copy()
        )
        skipped_log["stage"] = "pre_filter_intersection"
        skipped_log["reason"] = "skipped_missing_stock_date"
        skipped_log["threshold_source"] = ""
        skipped_log["atm_iv_source"] = ""
        skipped_log["hist_moment_source"] = "missing"

    failure_log = pd.concat(
        [
            skipped_log,
            prep_failures if isinstance(prep_failures, pd.DataFrame) else pd.DataFrame(),
            pd.DataFrame(failures),
        ],
        ignore_index=True,
    )

    if not term.empty:
        term = add_maturity_gates(term, target_dte=target_dte)
        term = term.sort_values(["ticker", "date", "dte", "exdate"]).reset_index(drop=True)
    else:
        print("No strict successful crash-premium slices.")
        print("Forced daily output will still be created using fallback logic.")

    forced_daily = build_forced_daily_output_from_rnd(
        rnd=rnd,
        hist=hist,
        strict_term=term,
        target_dte=target_dte,
        z_sigma_crash=z_sigma_crash,
    )

    strict_csv = output_dir / "strict_crash_premium_term_structure.csv"
    strict_parquet = output_dir / "strict_crash_premium_term_structure.parquet"

    failure_csv = output_dir / "failure_log_pareto_integrated.csv"
    meta_csv = output_dir / "slice_metadata.csv"

    forced_csv = output_dir / "forced_daily_crash_premium.csv"
    forced_parquet = output_dir / "forced_daily_crash_premium.parquet"

    if not term.empty:
        term.to_csv(strict_csv, index=False)
        term.to_parquet(strict_parquet, index=False)

        for ticker, g in term.groupby("ticker"):
            ticker = str(ticker).upper()

            g.to_csv(
                output_dir / f"{ticker}_strict_crash_premium_term_structure.csv",
                index=False,
            )

            g.to_parquet(
                output_dir / f"{ticker}_strict_crash_premium_term_structure.parquet",
                index=False,
            )

    failure_log.to_csv(failure_csv, index=False)
    slice_meta.to_csv(meta_csv, index=False)

    forced_daily.to_csv(forced_csv, index=False)
    forced_daily.to_parquet(forced_parquet, index=False)

    print("Saved:")
    if not term.empty:
        print(" ", strict_csv)
        print(" ", strict_parquet)

    print(" ", failure_csv)
    print(" ", meta_csv)
    print(" ", forced_csv)
    print(" ", forced_parquet)
    print()

    print("Strict successful expiry-level slices:", len(term))
    print("Failed/skipped expiry-level slices:", len(failure_log))
    print("Forced daily rows:", len(forced_daily))

    if "output_available" in forced_daily.columns:
        print("Forced unavailable rows:", int((~forced_daily["output_available"]).sum()))

    if "estimation_status" in forced_daily.columns:
        print()
        print("Forced daily estimation status:")
        print(forced_daily["estimation_status"].value_counts(dropna=False))

    if not term.empty:
        print()
        print("Strict tickers:", sorted(term["ticker"].astype(str).unique()))

        if "threshold_source" in term.columns:
            print()
            print("Strict threshold sources:")
            print(term["threshold_source"].value_counts(dropna=False))

        if "atm_iv_source" in term.columns:
            print()
            print("Strict ATM IV sources:")
            print(term["atm_iv_source"].value_counts(dropna=False))

        if "Q_crash_source" in term.columns:
            print()
            print("Strict Q_crash sources:")
            print(term["Q_crash_source"].value_counts(dropna=False))

    else:
        print()
        print("No strict term-structure rows were produced.")
        print("Use forced_daily_crash_premium.csv/parquet for the complete daily dataset.")

    return term, failure_log, slice_meta, forced_daily


def _normal_cdf(x):
    return 0.5 * (1.0 + math.erf(float(x) / math.sqrt(2.0)))


def compute_q_crash_from_rnd_slice(g):
    """
    Compute risk-neutral crash probability directly from the RND slice.

    Q_crash = integral_{K <= K_thr} f_Q(K) dK
    """

    g = g.copy()

    required = [
        "K",
        "K_thr",
        "risk_neutral_pdf_truncated",
        "integration_weight",
    ]

    missing = [c for c in required if c not in g.columns]

    if missing:
        return np.nan

    K = pd.to_numeric(g["K"], errors="coerce").to_numpy(dtype=float)
    K_thr = float(np.nanmedian(pd.to_numeric(g["K_thr"], errors="coerce")))
    f = pd.to_numeric(g["risk_neutral_pdf_truncated"], errors="coerce").to_numpy(dtype=float)
    w = pd.to_numeric(g["integration_weight"], errors="coerce").to_numpy(dtype=float)

    valid = (
        np.isfinite(K)
        & np.isfinite(f)
        & np.isfinite(w)
        & np.isfinite(K_thr)
        & (K > 0)
        & (f >= 0)
        & (w > 0)
    )

    if not valid.any():
        return np.nan

    K = K[valid]
    f = f[valid]
    w = w[valid]

    mass = float(np.sum(f * w))

    if not np.isfinite(mass) or mass <= 0:
        return np.nan

    q = float(np.sum(f[K <= K_thr] * w[K <= K_thr]) / mass)

    return min(max(q, 0.0), 1.0)


def compute_physical_crash_prob_historical_normal(
    alpha_dyn,
    F,
    spot_contract,
    dte,
    mu_daily,
    var_daily,
):
    try:
        alpha_dyn = float(alpha_dyn)
        F = float(F)
        spot_contract = float(spot_contract)
        dte = float(dte)
        mu_daily = float(mu_daily)
        var_daily = float(var_daily)
    except Exception:
        return np.nan

    if (
        not np.isfinite(alpha_dyn)
        or not np.isfinite(F)
        or not np.isfinite(spot_contract)
        or not np.isfinite(dte)
        or not np.isfinite(mu_daily)
        or not np.isfinite(var_daily)
        or alpha_dyn <= 0
        or F <= 0
        or spot_contract <= 0
        or dte <= 0
        or var_daily <= 0
    ):
        return np.nan

    threshold_x = math.log(alpha_dyn)

    horizon_mu_x = (
        mu_daily * dte
        - math.log(F / spot_contract)
    )

    horizon_sd = math.sqrt(var_daily * dte)

    if not np.isfinite(horizon_sd) or horizon_sd <= 0:
        return np.nan

    z = (threshold_x - horizon_mu_x) / horizon_sd

    p = _normal_cdf(z)

    return min(max(float(p), 0.0), 1.0)


def maturity_gate_weights(dte_values, target_dte=20):
    """
    Smooth maturity gate around target_dte.

    The bandwidth is adaptive to the observed expiry grid.
    """

    dte_values = np.asarray(dte_values, dtype=float)

    valid = np.isfinite(dte_values) & (dte_values > 0)

    out = np.full_like(dte_values, np.nan, dtype=float)

    if not valid.any():
        return out

    d = dte_values[valid]
    unique_d = np.sort(np.unique(d))

    if len(unique_d) >= 2:
        spacing = np.diff(unique_d)
        spacing = spacing[spacing > 0]
        bandwidth = float(np.median(spacing)) if len(spacing) else 20.0
    else:
        bandwidth = 20.0

    bandwidth = max(float(bandwidth), 5.0)

    out[valid] = np.exp(-0.5 * ((dte_values[valid] - target_dte) / bandwidth) ** 2)

    weight_sum = np.nansum(out)

    if np.isfinite(weight_sum) and weight_sum > 0:
        out = out / weight_sum

    return out
