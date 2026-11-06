# Downside-risk-premium term structures, directed network and similarity hypergraph

from __future__ import annotations
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@dataclass
class TailRiskConfig:
    crash_dir: str = "options_dir/crash_premium_outputs"
    price_dir: str = "stocks_dir"
    out_dir: str = "tail_risk_outputs"

    # term structure
    maturity_grid: Tuple[int, ...] = (20, 40, 60, 126, 252)
    gamma: float = 2.0
    min_expiries_required: int = 2
    max_carry_forward_days: int = 5

    # premium definition
    premium_col: str = "crash_premium"
    use_q_crash_only: bool = False
    gate_mode: str = "gaussian"          # "gaussian" or "nearest"
    shock_weighting: str = "inverse"     # "inverse" (1/tau) or "equal"
    downside_kappa: float = 0.0

    # directed network
    flow_window: int = 60
    flow_lags: Tuple[int, ...] = (1, 2, 3)
    k_out: int = 10
    min_overlap: int = 40
    use_flow_graph: bool = True

    # hypergraph
    n_clusters: int = 3
    use_hypergraph: bool = True
    random_state: int = 42

    # chronological splits
    train_start: str = "2018-01-01"
    train_end: str = "2020-12-31"
    val_start: str = "2021-01-01"
    val_end: str = "2021-12-31"
    test_start: str = "2022-01-01"
    test_end: str = "2022-12-31"

    n_pca_components: int = 2


def _premium_col(cfg: TailRiskConfig) -> str:
    return "Q_crash" if cfg.use_q_crash_only else cfg.premium_col


def _split_dates(cfg: TailRiskConfig) -> Dict[str, pd.Timestamp]:
    keys = ["train_start", "train_end", "val_start", "val_end", "test_start", "test_end"]
    return {k: pd.Timestamp(getattr(cfg, k)).normalize() for k in keys}


# Term structures
def chain_laplacian(m: int) -> np.ndarray:
    L = np.zeros((m, m))
    for i in range(m):
        if i > 0:
            L[i, i - 1] = -1.0
            L[i, i] += 1.0
        if i < m - 1:
            L[i, i + 1] = -1.0
            L[i, i] += 1.0
    return L


def load_crash_panel(cfg: TailRiskConfig) -> pd.DataFrame:
    # Expiry-level premia
    premium_col = _premium_col(cfg)
    grid = list(cfg.maturity_grid)
    parts = []

    for folder in sorted(Path(cfg.crash_dir).iterdir()):
        strict_path = folder / "strict_crash_premium_term_structure.parquet"
        daily_path = folder / "forced_daily_crash_premium.parquet"
        if not folder.is_dir() or not (strict_path.exists() or daily_path.exists()):
            continue

        strict_dates = set()
        if strict_path.exists():
            s = pd.read_parquet(strict_path)
            s["ticker"] = s["ticker"].astype(str).str.upper().str.strip()
            s["date"] = pd.to_datetime(s["date"], errors="coerce").dt.normalize()
            s["exdate"] = pd.to_datetime(s["exdate"], errors="coerce").dt.normalize()
            s["dte"] = pd.to_numeric(s["dte"], errors="coerce")
            s[premium_col] = pd.to_numeric(s[premium_col], errors="coerce")
            cols = ["ticker", "date", "exdate", "dte"] + [
                c for c in dict.fromkeys([premium_col, "Q_crash", "P_crash", "Gate_Q", "Gate_P", "Gate_premium"])
                if c in s.columns
            ]
            s = s[cols].dropna(subset=["ticker", "date", "dte", premium_col])
            s = s[np.isfinite(s["dte"]) & np.isfinite(s[premium_col])].copy()
            s["crash_input_source"] = "strict_expiry_term_structure"
            if len(s):
                parts.append(s)

            n_exp = s.groupby(["ticker", "date"])["dte"].nunique()
            strict_dates = set(n_exp[n_exp >= cfg.min_expiries_required].index)

        if daily_path.exists():
            f = pd.read_parquet(daily_path)
            f["ticker"] = f["ticker"].astype(str).str.upper().str.strip()
            f["date"] = pd.to_datetime(f["date"], errors="coerce").dt.normalize()
            f[premium_col] = pd.to_numeric(f[premium_col], errors="coerce")
            f = f[f["output_available"].eq(True)].dropna(subset=["ticker", "date", premium_col])
            f = f[np.isfinite(f[premium_col])]

            rows = []
            for row in f.itertuples(index=False):
                if (row.ticker, pd.Timestamp(row.date)) in strict_dates:
                    continue
                for tau in grid:
                    rows.append({
                        "ticker": row.ticker,
                        "date": pd.Timestamp(row.date),
                        "exdate": pd.NaT,
                        "dte": float(tau),
                        premium_col: float(getattr(row, premium_col)),
                        "fallback_used": row.fallback_used,
                        "output_available": row.output_available,
                        "crash_input_source": "forced_daily_flat_curve",
                    })
            if rows:
                parts.append(pd.DataFrame(rows))

    out = pd.concat(parts, ignore_index=True)
    out["exdate"] = pd.to_datetime(out["exdate"], errors="coerce").dt.normalize()
    out = out.dropna(subset=["ticker", "date", "dte", premium_col])
    out = out[np.isfinite(out["dte"]) & np.isfinite(out[premium_col])]
    return out.sort_values(["ticker", "date", "dte"]).reset_index(drop=True)


def interpolate_to_grid(dte: np.ndarray, prem: np.ndarray, grid: np.ndarray) -> np.ndarray:
    # Linear interpolation in maturity, flat beyond the shortest and longest expiry
    order = np.argsort(dte)
    x, y = np.asarray(dte, dtype=float)[order], np.asarray(prem, dtype=float)[order]
    xu, inv = np.unique(x, return_inverse=True)
    yu, cnt = np.zeros_like(xu), np.zeros_like(xu)
    for i, j in enumerate(inv):
        yu[j] += y[i]
        cnt[j] += 1.0
    yu = yu / np.maximum(cnt, 1.0)
    if len(xu) == 1:
        return np.repeat(yu[0], len(grid))
    return np.interp(grid, xu, yu, left=yu[0], right=yu[-1])


def build_term_structures(cfg: TailRiskConfig) -> pd.DataFrame:
    raw = load_crash_panel(cfg)
    premium_col = _premium_col(cfg)
    grid = np.asarray(cfg.maturity_grid, dtype=float)
    A = np.eye(len(grid)) + float(cfg.gamma) * chain_laplacian(len(grid))

    if cfg.use_q_crash_only:
        gate_col = "Gate_Q"
    elif premium_col == "P_crash":
        gate_col = "Gate_P"
    else:
        gate_col = "Gate_premium"

    rows = []
    last_curve: Dict[str, Tuple[pd.Timestamp, np.ndarray]] = {}
    for ticker, g_ticker in raw.groupby("ticker", sort=True):
        for date, g in g_ticker.groupby("date", sort=True):
            g = g.dropna(subset=[premium_col, "dte"])
            dte = g["dte"].to_numpy(dtype=float)
            prem = g[premium_col].to_numpy(dtype=float)
            ok = np.isfinite(dte) & np.isfinite(prem)
            dte, prem = dte[ok], prem[ok]
            source = str(g["crash_input_source"].mode(dropna=True).iloc[0]) if len(g) else "unknown"

            if len(np.unique(dte)) >= cfg.min_expiries_required:
                y_raw = interpolate_to_grid(dte, prem, grid)
                curve_source = f"{source}_interpolated"
                last_curve[ticker] = (date, y_raw.copy())
            elif ticker in last_curve and (date - last_curve[ticker][0]).days <= cfg.max_carry_forward_days:
                y_raw = last_curve[ticker][1].copy()
                curve_source = "carry_forward"
            else:
                y_raw = np.full(len(grid), np.nan)
                curve_source = "missing"

            y_smooth = np.linalg.solve(A, y_raw) if np.isfinite(y_raw).all() else np.full(len(grid), np.nan)
            row = {"ticker": ticker, "date": date, "curve_source": curve_source}

            # 20-day summary g_t: Gaussian gate from notebook 04, or the expiry closest to 20 days
            if cfg.gate_mode == "nearest":
                p_arr = pd.to_numeric(g[premium_col], errors="coerce").to_numpy(dtype=float)
                d_arr = pd.to_numeric(g["dte"], errors="coerce").to_numpy(dtype=float)
                ok_gate = np.isfinite(p_arr) & np.isfinite(d_arr)
                if ok_gate.any():
                    row["gate_premium"] = float(p_arr[ok_gate][np.argmin(np.abs(d_arr[ok_gate] - 20.0))])
            elif gate_col in g.columns:
                vals = pd.to_numeric(g[gate_col], errors="coerce").dropna()
                if len(vals):
                    row["gate_premium"] = float(vals.iloc[0])

            # indicators of how the curve was built (model inputs)
            if "fallback_used" in g.columns:
                row["fallback_used_numeric"] = float(pd.to_numeric(g["fallback_used"], errors="coerce").fillna(0).max())
            if "output_available" in g.columns:
                row["output_available_numeric"] = float(pd.to_numeric(g["output_available"], errors="coerce").fillna(1).min())
            row["is_forced_flat_curve"] = float("forced_daily_flat_curve" in source)
            row["is_strict_curve"] = float("strict_expiry_term_structure" in source)

            for tau, v in zip(cfg.maturity_grid, y_raw):
                row[f"y_raw_{int(tau)}"] = float(v) if np.isfinite(v) else np.nan
            for tau, v in zip(cfg.maturity_grid, y_smooth):
                row[f"y_smooth_{int(tau)}"] = float(v) if np.isfinite(v) else np.nan
            rows.append(row)

    curves = pd.DataFrame(rows).sort_values(["ticker", "date"]).reset_index(drop=True)
    return curves


# Features and targets
def add_shape_features(curves: pd.DataFrame, cfg: TailRiskConfig) -> pd.DataFrame:
    df = curves.copy().sort_values(["ticker", "date"])
    grid = list(cfg.maturity_grid)
    ys = [f"y_smooth_{t}" for t in grid]
    dys = [f"dy_{t}" for t in grid]
    for t in grid:
        df[f"dy_{t}"] = df.groupby("ticker")[f"y_smooth_{t}"].diff()

    if len(grid) == 1:
        # single maturity: level only, no slope or curvature
        for prefix, col in (("", ys[0]), ("d", dys[0])):
            df[f"{prefix}level"] = df[col]
            df[f"{prefix}level_short"] = df[col]
            df[f"{prefix}level_long"] = df[col]
            df[f"{prefix}slope"] = 0.0
            df[f"{prefix}curvature"] = 0.0
        return df

    # curvature
    mid = len(grid) // 2
    t0, tm, t1 = float(grid[0]), float(grid[mid]), float(grid[-1])
    w_short, w_long = (t1 - tm) / (t1 - t0), (tm - t0) / (t1 - t0)

    for prefix, cols in (("", ys), ("d", dys)):
        df[f"{prefix}level"] = df[cols].mean(axis=1)
        df[f"{prefix}level_short"] = df[cols[:min(3, len(grid))]].mean(axis=1)
        df[f"{prefix}level_long"] = df[cols[-min(2, len(grid)):]].mean(axis=1)
        df[f"{prefix}slope"] = df[cols[-1]] - df[cols[0]]
        df[f"{prefix}curvature"] = df[cols[mid]] - (w_short * df[cols[0]] + w_long * df[cols[-1]])
    return df


def add_pca_features(df: pd.DataFrame, cfg: TailRiskConfig) -> pd.DataFrame:
    # PCA scores of the smoothed curve; scaler and PCA are fitted on training dates only
    out = df.copy().sort_values(["date", "ticker"]).reset_index(drop=True)
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    split = _split_dates(cfg)
    cols = [f"y_smooth_{t}" for t in cfg.maturity_grid]

    train = (out["date"] >= split["train_start"]) & (out["date"] <= split["train_end"])
    X_train = out.loc[train, cols].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    n_comp = min(cfg.n_pca_components, len(cols), max(1, X_train.shape[0]))
    scaler = StandardScaler()
    pca = PCA(n_components=n_comp, random_state=cfg.random_state).fit(scaler.fit_transform(X_train))

    X = out[cols].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
    ok = np.isfinite(X).all(axis=1)
    Z = np.full((len(out), n_comp), np.nan)
    Z[ok] = pca.transform(scaler.transform(X[ok]))
    for j in range(n_comp):
        out[f"pca_{j + 1}"] = Z[:, j]
    return out


def add_targets(features: pd.DataFrame, cfg: TailRiskConfig) -> pd.DataFrame:
    parts = []
    for ticker in sorted(set(features["ticker"].unique())):
        px = pd.read_csv(Path(cfg.price_dir) / f"{ticker}_stock.csv")
        px = pd.DataFrame({
            "ticker": ticker,
            "date": pd.to_datetime(px["date"], errors="coerce").dt.normalize(),
            "return_price": pd.to_numeric(px["close_yahoo_split_adjusted"], errors="coerce"),
            "close": pd.to_numeric(px["spot"], errors="coerce"),
        }).dropna()
        parts.append(px[(px["return_price"] > 0) & (px["close"] > 0)].sort_values("date"))

    px = pd.concat(parts, ignore_index=True).sort_values(["ticker", "date"]).reset_index(drop=True)
    px["ret_1d"] = px.groupby("ticker")["return_price"].transform(lambda s: np.log(s).diff())
    px["target_ret_1d"] = px.groupby("ticker")["ret_1d"].shift(-1)
    px["downside_1d"] = np.where(px["target_ret_1d"].notna(),
                                 (px["target_ret_1d"] < -cfg.downside_kappa).astype(float), np.nan)
    cols = ["ticker", "date", "close", "ret_1d", "target_ret_1d", "downside_1d"]
    return features.merge(px[cols], on=["ticker", "date"], how="left")


def add_shocks(df: pd.DataFrame, cfg: TailRiskConfig) -> pd.DataFrame:
    out = df.copy()
    grid = list(cfg.maturity_grid)
    taus = np.asarray(grid, dtype=float)
    if cfg.shock_weighting == "equal":
        w = np.ones(len(taus)) / len(taus)
    else:
        w = (1.0 / taus) / (1.0 / taus).sum()
    dys = [f"dy_{t}" for t in grid]
    out["shock"] = out[dys].to_numpy(dtype=float) @ w
    out["shock_short"] = out[dys[:min(3, len(grid))]].mean(axis=1)
    out["shock_long"] = out[dys[-min(2, len(grid)):]].mean(axis=1)
    return out


def add_split_labels(panel: pd.DataFrame, cfg: TailRiskConfig) -> pd.DataFrame:
    out = panel.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    split = _split_dates(cfg)
    out["split"] = pd.Series(np.nan, index=out.index, dtype=object)
    for name in ("train", "val", "test"):
        mask = (out["date"] >= split[f"{name}_start"]) & (out["date"] <= split[f"{name}_end"])
        out.loc[mask, "split"] = name
    out = out[out["split"].notna()].copy()
    out["split"] = out["split"].astype(str)
    return out.sort_values(["date", "ticker"]).reset_index(drop=True)


# Directed network
def _corr(x: np.ndarray, y: np.ndarray, min_overlap: int) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < min_overlap:
        return np.nan
    x, y = x[m], y[m]
    if np.nanstd(x) < 1e-12 or np.nanstd(y) < 1e-12:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def build_flow_edges(panel: pd.DataFrame, cfg: TailRiskConfig) -> pd.DataFrame:
    """Edge i -> j at date t: signed correlation of shock_i(u) and r_j(u + lag) over the window before t,
    at the lag with the largest absolute correlation."""
    cols = ["date", "sender", "receiver", "weight", "best_lag"]
    if not cfg.use_flow_graph:
        return pd.DataFrame(columns=cols)

    p = panel.sort_values(["date", "ticker"])
    dates = np.array(sorted(p["date"].dropna().unique()))
    tickers = sorted(p["ticker"].dropna().unique())
    shocks = p.pivot(index="date", columns="ticker", values="shock").reindex(index=dates, columns=tickers)
    rets = p.pivot(index="date", columns="ticker", values="ret_1d").reindex(index=dates, columns=tickers)
    W = int(cfg.flow_window)
    lags = tuple(sorted(set(int(k) for k in cfg.flow_lags if k >= 1)))

    edges = []
    for i in range(W, len(dates)):
        s_win, r_win = shocks.iloc[i - W:i], rets.iloc[i - W:i]   # dates strictly before t
        for sender in tickers:
            s = s_win[sender].to_numpy(dtype=float)
            scores = []
            for receiver in tickers:
                if receiver == sender:
                    continue
                r = r_win[receiver].to_numpy(dtype=float)
                best, best_lag = np.nan, np.nan
                for lag in lags:
                    if lag >= len(s):
                        continue
                    c = _corr(s[:-lag], r[lag:], cfg.min_overlap)
                    if not np.isnan(c) and (np.isnan(best) or abs(c) > abs(best)):
                        best, best_lag = c, lag
                if not np.isnan(best):
                    scores.append((receiver, best, best_lag))
            for receiver, w, lag in sorted(scores, key=lambda z: abs(z[1]), reverse=True)[:cfg.k_out]:
                edges.append({"date": pd.Timestamp(dates[i]), "sender": sender, "receiver": receiver,
                              "weight": float(w), "best_lag": int(lag)})
    return pd.DataFrame(edges, columns=cols)


# Hypergraph
SHAPE_COLS = ["level", "level_short", "level_long", "slope", "curvature",
              "dlevel", "dlevel_short", "dlevel_long", "dslope", "dcurvature"]


def build_hyperedges(panel: pd.DataFrame, cfg: TailRiskConfig) -> pd.DataFrame:
    """Daily k-means communities of the shape features.
    Cluster labels are matched to the previous day's communities by solving the assignment problem on
    centroid distances (Hungarian algorithm), so that community ids persist over time.
    """
    cols = ["date", "community_id", "members", "size"]
    feature_cols = [c for c in SHAPE_COLS + [c for c in panel.columns if c.startswith("pca_")] if c in panel.columns]
    if not cfg.use_hypergraph or not feature_cols:
        return pd.DataFrame(columns=cols)

    split = _split_dates(cfg)
    z = panel[["ticker", "date"] + feature_cols].copy()
    train = (z["date"] >= split["train_start"]) & (z["date"] <= split["train_end"])
    scaler = StandardScaler().fit(
        z.loc[train, feature_cols].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    )

    rows = []
    prev_centroids: Optional[np.ndarray] = None
    prev_ids: Optional[np.ndarray] = None
    next_id = 0
    for date in np.array(sorted(z["date"].dropna().unique())):
        day = z[z["date"] == date].copy()
        X = day[feature_cols].to_numpy(dtype=float)
        ok = np.isfinite(X).all(axis=1)
        day["cluster_local"] = np.nan
        centroids = np.empty((0, len(feature_cols)))
        k = min(cfg.n_clusters, int(ok.sum()))
        if k > 0:
            km = KMeans(n_clusters=k, random_state=cfg.random_state, n_init=20)
            day.loc[day.index[ok], "cluster_local"] = km.fit_predict(scaler.transform(X[ok]))
            centroids = km.cluster_centers_

        if prev_centroids is None or len(prev_centroids) == 0 or len(centroids) == 0:
            mapping = {j: next_id + j for j in range(len(centroids))}
            next_id += len(centroids)
        else:
            dist = np.linalg.norm(centroids[:, None, :] - prev_centroids[None, :, :], axis=2)
            r, c = linear_sum_assignment(dist)
            mapping = {int(rr): int(prev_ids[cc]) for rr, cc in zip(r, c)}
            for j in range(len(centroids)):
                if j not in mapping:
                    mapping[j] = next_id
                    next_id += 1

        day["community_id"] = day["cluster_local"].map(mapping)
        prev_centroids = centroids.copy()
        prev_ids = np.array([mapping[j] for j in range(len(centroids))], dtype=int)

        for cid, g in day.groupby("community_id", dropna=True):
            members = sorted(g["ticker"].astype(str).tolist())
            rows.append({"date": pd.Timestamp(date), "community_id": int(cid),
                         "members": "|".join(members), "size": len(members)})
    return pd.DataFrame(rows, columns=cols)


# Pipeline
def run_pipeline(cfg: TailRiskConfig) -> Dict[str, object]:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    panel = add_shape_features(build_term_structures(cfg), cfg)
    panel = add_pca_features(panel, cfg)
    panel = add_targets(panel, cfg)
    panel = add_shocks(panel, cfg)
    panel = add_split_labels(panel, cfg)

    edges = build_flow_edges(panel, cfg)
    hyperedges = build_hyperedges(panel, cfg)

    panel.to_parquet(out_dir / "node_panel.parquet", index=False)
    edges.to_parquet(out_dir / "flow_edges.parquet", index=False)
    hyperedges.to_parquet(out_dir / "hyperedges.parquet", index=False)

    with open(out_dir / "network_config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2, default=str)

    print(f"Network outputs saved to {out_dir}")
    return {"panel": panel, "edges": edges, "hyperedges": hyperedges}
