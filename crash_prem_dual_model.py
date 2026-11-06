"""Temporal multi-task model on the DRP network and hypergraph, and the confidence-weighted strategy.

For each date t the model reads the last L days of DRP features of every stock, encodes them with a GRU,
aggregates the encodings over the directed network (signed attention) and the hypergraph communities,
and predicts
  * the next-day log return (Huber loss on cross-sectionally demeaned returns),
  * the next-day DRP shock (MSE),
  * the probability of a negative next-day return (binary cross-entropy).

The lookback L is chosen on the training period with Ljung-Box tests. The equal-weighted long-short (EWLS)
portfolio and the confidence-weighted strategy (CWS) are evaluated net
"""
from __future__ import annotations

import json
import math
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import chi2
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler


@dataclass
class DualModelConfig:
    panel_path: str = "tail_risk_outputs/node_panel.parquet"
    flow_edges_path: str = "tail_risk_outputs/flow_edges.parquet"
    hyperedges_path: str = "tail_risk_outputs/hyperedges.parquet"
    out_dir: str = "tail_risk_model_outputs"

    # targets
    target_col: str = "target_ret_1d"
    shock_source_col: str = "shock"
    shock_target_col: str = "target_shock_1d"
    shock_target_horizon: int = 1
    downside_col: str = "downside_1d"

    lookback: int = 20
    auto_lookback: bool = True
    lookback_diagnostic_max_lag: int = 252
    lookback_diagnostic_alpha: float = 0.05
    lookback_diagnostic_quantile: float = 0.90
    hidden_dim: int = 96
    temporal_layers: int = 1
    dropout: float = 0.10
    min_history_coverage: float = 0.6
    hyperedge_dropout: float = 0.0

    # training
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 80
    batch_size_dates: int = 16
    early_stopping_patience: int = 12
    grad_clip: float = 1.0
    huber_delta: float = 0.01
    lambda_shock: float = 0.01
    lambda_down: float = 0.01
    use_downside_loss: bool = True

    # portfolios
    top_k: int = 2
    strategy_top_k: Optional[int] = 2
    gross_exposure: float = 1.0
    initial_capital: float = 1_000_000.0
    commission_per_share: float = 0.0035
    min_commission_per_order: float = 0.35
    max_commission_rate: float = 0.01
    target_is_log_return: bool = True

    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Lookback from Ljung-Box tests

def _autocorr(x: np.ndarray, lag: int) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if lag <= 0 or n <= lag + 1:
        return np.nan
    x = x - np.nanmean(x)
    denom = np.dot(x, x)
    if not np.isfinite(denom) or denom <= 1e-12:
        return np.nan
    return float(np.dot(x[:-lag], x[lag:]) / denom)


def ljung_box_memory(x: np.ndarray, max_lag: int, alpha: float) -> int:
    # Largest lag at which the Ljung-Box statistic is significant
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    max_lag = int(min(max_lag, max(1, n // 3), n - 2))
    if n < 10 or max_lag < 1:
        return 1

    rhos = np.nan_to_num(np.array([_autocorr(x, h) for h in range(1, max_lag + 1)]), nan=0.0, posinf=0.0, neginf=0.0)
    significant, running = [], 0.0
    for lag in range(1, max_lag + 1):
        running += rhos[lag - 1] ** 2 / max(n - lag, 1)
        p_value = 1.0 - chi2.cdf(n * (n + 2) * running, df=lag)
        if np.isfinite(p_value) and p_value < alpha:
            significant.append(lag)
    return int(max(significant)) if significant else 1


def estimate_lookback(panel, feature_cols, max_lag, alpha, quantile, fallback):
    # Memory of x, |x| and x^2 per stock and feature on the training period; the cross-stock quantile is taken for each feature and the lookback is the largest of these
    train = panel.loc[panel["split"].eq("train")]
    if train.empty:
        return int(fallback), pd.DataFrame()

    rows = []
    for feature in feature_cols:
        memories = []
        for ticker, g in train[["ticker", "date", feature]].dropna().groupby("ticker"):
            x = g.sort_values("date")[feature].to_numpy(dtype=float)
            if len(x) < max(15, min(10, max_lag // 3)):
                continue
            m = [ljung_box_memory(v, max_lag, alpha) for v in (x, np.abs(x), x**2)]
            memories.append(max(m))
            rows.append({"ticker": ticker, "feature": feature, "memory_linear": m[0],
                         "memory_abs": m[1], "memory_squared": m[2], "memory": max(m)})
        if memories:
            rows.append({"ticker": "quantile", "feature": feature, "memory": float(np.quantile(memories, quantile))})

    diag = pd.DataFrame(rows)
    if diag.empty or not diag["ticker"].eq("quantile").any():
        return int(fallback), diag
    estimated = int(np.ceil(np.nanmax(diag.loc[diag["ticker"].eq("quantile"), "memory"].to_numpy(dtype=float))))
    return int(max(1, min(max_lag, estimated))), diag



class TailRiskGraphDataset:
    # Features [D, N, F], targets [D, N] and the graphs of each date
    FEATURES = {
        "level", "level_short", "level_long", "slope", "curvature",
        "dlevel", "dslope", "dcurvature",
        "shock", "shock_short", "shock_long",
        "gate_premium",
        "fallback_used_numeric", "output_available_numeric", "is_forced_flat_curve", "is_strict_curve",
    }

    def __init__(self, cfg: DualModelConfig):
        self.cfg = cfg
        self.panel = pd.read_parquet(cfg.panel_path)
        self.flow_edges = pd.read_parquet(cfg.flow_edges_path)
        self.hyperedges = pd.read_parquet(cfg.hyperedges_path)
        for df in (self.panel, self.flow_edges, self.hyperedges):
            df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        self.panel = self.panel.sort_values(["date", "ticker"]).reset_index(drop=True)

        self._add_shock_target()

        self.dates = [pd.Timestamp(d).normalize() for d in sorted(self.panel["date"].dropna().unique())]
        self.tickers = sorted(self.panel["ticker"].dropna().astype(str).unique())
        self.date_to_idx = {d: i for i, d in enumerate(self.dates)}
        self.ticker_to_idx = {t: i for i, t in enumerate(self.tickers)}
        self.idx_to_ticker = {i: t for t, i in self.ticker_to_idx.items()}
        self.close_lookup = (
            self.panel[["date", "ticker", "close"]].dropna()
            .assign(date=lambda x: pd.to_datetime(x["date"]).dt.normalize(), ticker=lambda x: x["ticker"].astype(str))
            .set_index(["date", "ticker"])["close"].to_dict()
        )

        self.feature_cols = self._feature_cols(self.panel)
        self.date_split = {
            pd.Timestamp(r.date).normalize(): str(r.split)
            for r in self.panel[["date", "split"]].dropna().drop_duplicates().sort_values("date").itertuples(index=False)
        }
        self._estimate_lookback()
        self._fit_scaler()
        self._build_tensors()
        self._build_graph_cache()
        self._build_samples()

    def _feature_cols(self, panel: pd.DataFrame) -> List[str]:
        skip = {"ticker", "date", "split", "curve_source", "close", "ret_1d",
                self.cfg.target_col, self.cfg.shock_target_col, self.cfg.downside_col}
        cols = []
        for c in panel.columns:
            if c in skip:
                continue
            if c.startswith(("y_smooth_", "dy_")) or c in self.FEATURES:
                cols.append(c)
            elif c.startswith("pca_"):
                nums = re.findall(r"\d+", c)
                if (int(nums[-1]) if nums else 999) <= 2:
                    cols.append(c)
        return sorted(cols)

    def _add_shock_target(self) -> None:
        # s_{t+h}: the shock h trading days ahead in the panel's date index
        src, dst, h = self.cfg.shock_source_col, self.cfg.shock_target_col, int(self.cfg.shock_target_horizon)
        if dst in self.panel.columns:
            return
        p = self.panel.sort_values(["ticker", "date"]).copy()
        rank = {d: i for i, d in enumerate(sorted(p["date"].dropna().unique()))}
        p["_rank"] = p["date"].map(rank)
        p[dst] = p.groupby("ticker", group_keys=False)[src].shift(-h)
        p["_next_rank"] = p.groupby("ticker", group_keys=False)["_rank"].shift(-h)
        p.loc[~p["_next_rank"].eq(p["_rank"] + h), dst] = np.nan
        self.panel = p.drop(columns=["_rank", "_next_rank"]).sort_values(["date", "ticker"]).reset_index(drop=True)

    def _estimate_lookback(self) -> None:
        fallback = int(max(1, self.cfg.lookback))
        self.lookback, self.lookback_diagnostics = fallback, pd.DataFrame()
        if not self.cfg.auto_lookback:
            return
        self.lookback, self.lookback_diagnostics = estimate_lookback(
            self.panel, self.feature_cols, self.cfg.lookback_diagnostic_max_lag,
            self.cfg.lookback_diagnostic_alpha, self.cfg.lookback_diagnostic_quantile, fallback,
        )
        # keep at least two training samples
        max_safe = max(1, int(self.panel.loc[self.panel["split"].eq("train"), "date"].nunique()) - 2)
        if self.lookback > max_safe:
            self.lookback = max(1, min(fallback, max_safe))

    def _fit_scaler(self) -> None:
        X = self.panel.loc[self.panel["split"].eq("train"), self.feature_cols].replace([np.inf, -np.inf], np.nan)
        means = X.mean(axis=0).fillna(0.0)
        self.feature_fill_values = means.to_dict()
        self.scaler = StandardScaler().fit(X.fillna(means).to_numpy(dtype=float))
        self.scaler.mean_ = np.where(np.isfinite(self.scaler.mean_), self.scaler.mean_, 0.0)
        self.scaler.scale_ = np.where(np.isfinite(self.scaler.scale_) & (self.scaler.scale_ > 1e-12), self.scaler.scale_, 1.0)

    def _build_tensors(self) -> None:
        D, N, Fdim = len(self.dates), len(self.tickers), len(self.feature_cols)
        X = np.zeros((D, N, Fdim), dtype=np.float32)
        target = np.full((D, N), np.nan, dtype=np.float32)
        shock = np.full((D, N), np.nan, dtype=np.float32)
        downside = np.full((D, N), np.nan, dtype=np.float32)
        feat_valid = np.zeros((D, N), dtype=bool)
        target_valid = np.zeros((D, N), dtype=bool)

        tmp = self.panel.copy()
        tmp[self.feature_cols] = tmp[self.feature_cols].replace([np.inf, -np.inf], np.nan).fillna(self.feature_fill_values)
        arr = self.scaler.transform(tmp[self.feature_cols].to_numpy(dtype=float))
        tmp[self.feature_cols] = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        for row in tmp.itertuples(index=False):
            d, n = self.date_to_idx[row.date], self.ticker_to_idx[str(row.ticker)]
            vec = np.nan_to_num(np.asarray([getattr(row, c) for c in self.feature_cols], dtype=np.float32),
                                nan=0.0, posinf=0.0, neginf=0.0)
            X[d, n, :] = vec
            feat_valid[d, n] = np.isfinite(vec).all()

            y = getattr(row, self.cfg.target_col, np.nan)
            s = getattr(row, self.cfg.shock_target_col, np.nan)
            dn = getattr(row, self.cfg.downside_col, np.nan)
            target[d, n] = np.float32(y) if pd.notna(y) else np.nan
            shock[d, n] = np.float32(s) if pd.notna(s) else np.nan
            downside[d, n] = np.float32(dn) if pd.notna(dn) else np.nan
            target_valid[d, n] = pd.notna(y)

        self.X, self.target, self.shock_target, self.downside = X, target, shock, downside
        self.feat_valid, self.target_valid = feat_valid, target_valid

    def _build_graph_cache(self) -> None:
        self.hyperedges_by_date: Dict[int, List[np.ndarray]] = {}
        for row in self.hyperedges.itertuples(index=False):
            if row.date not in self.date_to_idx:
                continue
            members = [self.ticker_to_idx[t] for t in str(row.members).split("|") if t in self.ticker_to_idx]
            if members:
                self.hyperedges_by_date.setdefault(self.date_to_idx[row.date], []).append(
                    np.asarray(sorted(set(members)), dtype=np.int64))

        edges: Dict[int, List[Tuple[int, int, float]]] = {}
        for row in self.flow_edges.itertuples(index=False):
            if (row.date not in self.date_to_idx or row.sender not in self.ticker_to_idx
                    or row.receiver not in self.ticker_to_idx):
                continue
            edges.setdefault(self.date_to_idx[row.date], []).append(
                (self.ticker_to_idx[row.sender], self.ticker_to_idx[row.receiver], float(row.weight)))
        self.flow_edges_by_date = {
            d: (np.asarray([t[0] for t in e], dtype=np.int64), np.asarray([t[1] for t in e], dtype=np.int64),
                np.asarray([t[2] for t in e], dtype=np.float32))
            for d, e in edges.items()
        }

    def _build_samples(self) -> None:
        self.sample_indices = {"train": [], "val": [], "test": []}
        for d, date in enumerate(self.dates):
            split = self.date_split.get(pd.Timestamp(date).normalize())
            if split in self.sample_indices and d >= self.lookback - 1 and self.target_valid[d].sum() > 0:
                self.sample_indices[split].append(d)
        print("Samples:", {k: len(v) for k, v in self.sample_indices.items()}, "| lookback:", self.lookback)
        if not self.sample_indices["train"]:
            raise ValueError("No training samples after the lookback filter.")

    def iter_date_batches(self, split: str, shuffle: bool) -> List[List[int]]:
        idx = list(self.sample_indices[split])
        if shuffle:
            random.shuffle(idx)
        B = max(1, self.cfg.batch_size_dates)
        return [idx[i:i + B] for i in range(0, len(idx), B)]

    def get_batch(self, date_indices: Sequence[int], device: str) -> Dict[str, object]:
        L = self.lookback
        xs, feat_masks, target, target_mask = [], [], [], []
        shock, shock_mask, downside, downside_mask = [], [], [], []
        hypergraphs, flow_graphs, dates = [], [], []

        for d in date_indices:
            xs.append(np.transpose(self.X[d - L + 1:d + 1], (1, 0, 2)))          # [N, L, F]
            # an asset is used if its features exist today and on enough of the window
            mask = (self.feat_valid[d - L + 1:d + 1].mean(axis=0) >= self.cfg.min_history_coverage) & self.feat_valid[d]
            feat_masks.append(mask.astype(np.float32))
            target.append(self.target[d])
            target_mask.append((self.target_valid[d] & mask).astype(np.float32))
            shock.append(self.shock_target[d])
            shock_mask.append((np.isfinite(self.shock_target[d]) & mask).astype(np.float32))
            downside.append(self.downside[d])
            downside_mask.append((np.isfinite(self.downside[d]) & mask).astype(np.float32))
            dates.append(self.dates[d])

            hypergraphs.append([torch.as_tensor(h, dtype=torch.long, device=device)
                                for h in self.hyperedges_by_date.get(d, [])])
            if d in self.flow_edges_by_date:
                s, r, w = self.flow_edges_by_date[d]
                flow_graphs.append((torch.as_tensor(s, dtype=torch.long, device=device),
                                    torch.as_tensor(r, dtype=torch.long, device=device),
                                    torch.as_tensor(w, dtype=torch.float32, device=device)))
            else:
                flow_graphs.append((torch.empty(0, dtype=torch.long, device=device),
                                    torch.empty(0, dtype=torch.long, device=device),
                                    torch.empty(0, dtype=torch.float32, device=device)))

        def t(a):
            return torch.as_tensor(np.stack(a), dtype=torch.float32, device=device)

        return {"x": t(xs), "feat_mask": t(feat_masks), "target": t(target), "target_mask": t(target_mask),
                "shock_target": t(shock), "shock_mask": t(shock_mask), "downside": t(downside),
                "downside_mask": t(downside_mask), "hypergraphs": hypergraphs, "flow_graphs": flow_graphs,
                "dates": dates}


# Model

class TemporalEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, layers: int, dropout: float):
        super().__init__()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=layers,
                          batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, L, Fdim = x.shape
        h, _ = self.gru(x.reshape(B * N, L, Fdim))
        return self.out_norm(h[:, -1, :]).reshape(B, N, -1)


class HypergraphLayer(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward_one(self, h: torch.Tensor, hyperedges: List[torch.Tensor]) -> torch.Tensor:
        if not hyperedges:
            return h
        add = torch.zeros_like(h)
        counts = torch.zeros(h.shape[0], 1, device=h.device)
        for members in hyperedges:
            if members.numel() == 0:
                continue
            hm = h[members]
            a = torch.softmax(self.score(torch.tanh(hm)).squeeze(-1), dim=0)
            g = self.dropout(self.proj(torch.sum(a.unsqueeze(-1) * hm, dim=0)))
            add[members] += g.unsqueeze(0)
            counts[members] += 1.0
        return self.norm(h.clone() + add / counts.clamp_min(1.0))

    def forward(self, h: torch.Tensor, hypergraphs: List[List[torch.Tensor]]) -> torch.Tensor:
        return torch.stack([self.forward_one(h[b], hypergraphs[b]) for b in range(h.shape[0])], dim=0)


class DirectedFlowLayer(nn.Module):
    # Attention over incoming edges j -> i

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.src_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.dst_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.msg_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.attn_vec = nn.Parameter(torch.empty(hidden_dim))
        self.weight_bias = nn.Parameter(torch.tensor(1.0))
        self.norm = nn.LayerNorm(hidden_dim)
        nn.init.xavier_uniform_(self.src_proj.weight)
        nn.init.xavier_uniform_(self.dst_proj.weight)
        nn.init.xavier_uniform_(self.msg_proj.weight)
        nn.init.normal_(self.attn_vec, mean=0.0, std=0.02)

    def forward_one(self, h: torch.Tensor, graph: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        senders, receivers, weights = graph
        if senders.numel() == 0:
            return h
        raw = F.leaky_relu((self.src_proj(h[senders]) + self.dst_proj(h[receivers])) * self.attn_vec,
                           negative_slope=0.2).sum(dim=-1)
        raw = raw + self.weight_bias * torch.abs(weights)
        sign = torch.sign(weights)

        msg = torch.zeros_like(h)
        for r in torch.unique(receivers):
            mask = receivers == r
            a = torch.softmax(raw[mask], dim=0)
            m = self.msg_proj(h[senders[mask]]) * sign[mask].unsqueeze(-1)
            msg[r] = torch.sum(a.unsqueeze(-1) * m, dim=0)
        return self.norm(h + msg)

    def forward(self, h: torch.Tensor, graphs: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        return torch.stack([self.forward_one(h[b], graphs[b]) for b in range(h.shape[0])], dim=0)


class TailRiskDualModel(nn.Module):
    def __init__(self, input_dim: int, cfg: DualModelConfig):
        super().__init__()
        H = cfg.hidden_dim
        self.temporal = TemporalEncoder(input_dim, H, cfg.temporal_layers, cfg.dropout)
        self.hyper = HypergraphLayer(H, cfg.hyperedge_dropout)
        self.flow = DirectedFlowLayer(H)
        self.return_head = nn.Sequential(nn.Linear(H, H), nn.ReLU(), nn.Dropout(cfg.dropout), nn.Linear(H, 1))

        # hypergraph correction f_c, initialised at zero so training starts from the network prediction
        self.hyper_correction_head = nn.Sequential(nn.Linear(H, H // 2), nn.ReLU(), nn.Linear(H // 2, 1))
        nn.init.zeros_(self.hyper_correction_head[-1].weight)
        nn.init.zeros_(self.hyper_correction_head[-1].bias)
        self.hyper_correction_logit = nn.Parameter(torch.tensor(-2.0))

        self.shock_head = nn.Sequential(nn.Linear(H, H // 2), nn.ReLU(), nn.Linear(H // 2, 1))
        self.downside_head = nn.Sequential(nn.Linear(H, H // 2), nn.ReLU(), nn.Linear(H // 2, 1))

    def forward(self, x, feat_mask, hypergraphs, flow_graphs) -> Dict[str, torch.Tensor]:
        h0 = self.temporal(x) * feat_mask.unsqueeze(-1)
        h_flow = self.flow(h0, flow_graphs)
        h_hyper = self.hyper(h0, hypergraphs)

        # r_hat = f_r(h_flow) + sigmoid(chi) f_c(h_comm - h_temp)
        pred_ret = (self.return_head(h_flow).squeeze(-1)
                    + torch.sigmoid(self.hyper_correction_logit) * self.hyper_correction_head(h_hyper - h0).squeeze(-1))
        return {
            "pred_ret": pred_ret,
            "pred_shock": self.shock_head(h_flow).squeeze(-1),
            "pred_down": self.downside_head(h_flow).squeeze(-1),
        }


# Losses

def masked_cs_huber(pred, target, mask, delta):
    # Huber loss on returns demeaned across assets of each date
    valid = mask > 0.5
    losses = []
    for b in range(pred.shape[0]):
        m = valid[b]
        if m.sum() < 2:
            continue
        p, y = pred[b, m], target[b, m]
        losses.append(F.huber_loss(p - p.mean(), y - y.mean(), delta=delta, reduction="mean"))
    return torch.stack(losses).mean() if losses else pred.new_tensor(0.0)


def masked_mse(pred, target, mask):
    m = mask > 0.5
    return F.mse_loss(pred[m], target[m], reduction="mean") if m.sum() > 0 else pred.new_tensor(0.0)


def masked_bce_logits(pred, target, mask):
    m = mask > 0.5
    return F.binary_cross_entropy_with_logits(pred[m], target[m], reduction="mean") if m.sum() > 0 else pred.new_tensor(0.0)


def compute_total_loss(out, batch, cfg: DualModelConfig):
    loss_ret = masked_cs_huber(out["pred_ret"], batch["target"], batch["target_mask"], cfg.huber_delta)
    loss_shock = masked_mse(out["pred_shock"], batch["shock_target"], batch["shock_mask"])
    loss_down = (masked_bce_logits(out["pred_down"], batch["downside"], batch["downside_mask"])
                 if cfg.use_downside_loss else out["pred_ret"].new_tensor(0.0))
    return loss_ret + cfg.lambda_shock * loss_shock + cfg.lambda_down * loss_down


# Portfolios and evaluation

def compute_ibkr_cost_return(prev_weights, current_weights, prices, cfg) -> Tuple[float, float]:
    # Per-share commission ($0.0035, min $0.35, max 1% of trade value) on the change in weights.
    turnover, cost = 0.0, 0.0
    for ticker in set(prev_weights) | set(current_weights):
        delta = current_weights.get(ticker, 0.0) - prev_weights.get(ticker, 0.0)
        trade_value = abs(delta) * cfg.initial_capital
        turnover += abs(delta)
        if trade_value <= 0:
            continue
        price = prices.get(ticker, np.nan)
        if not np.isfinite(price) or price <= 0:
            cost += 0.0005 * trade_value
            continue
        shares = trade_value / price
        commission = max(cfg.commission_per_share * shares, cfg.min_commission_per_order)
        cost += min(commission, cfg.max_commission_rate * trade_value)
    return turnover, cost / cfg.initial_capital


def drift_weights(prev_weights, held_returns, log_returns=True):
    # Yesterday's weights after one day of returns: w_i (1 + r_i) / (1 + sum_j w_j r_j)
    if not prev_weights:
        return {}
    simple = {}
    for ticker, r in held_returns.items():
        r = float(r)
        if np.isfinite(r):
            simple[ticker] = float(np.expm1(r)) if log_returns else r
    growth = 1.0 + sum(w * simple.get(t, 0.0) for t, w in prev_weights.items())
    if not np.isfinite(growth) or growth <= 1e-12:
        return dict(prev_weights)
    return {t: w * (1.0 + simple.get(t, 0.0)) / growth for t, w in prev_weights.items()}


def _sharpe(r) -> float:
    r = np.asarray(r, dtype=float)
    std = float(np.std(r, ddof=1)) if len(r) >= 2 else np.nan
    return float(np.sqrt(252.0) * np.mean(r) / std) if len(r) and np.isfinite(std) and std > 1e-12 else np.nan


def _max_drawdown(r) -> float:
    # Largest fall of wealth exp(cumsum r) from its running peak
    r = np.asarray(r, dtype=float)
    if not len(r):
        return np.nan
    wealth = np.exp(np.cumsum(r))
    return float(np.min(wealth / np.maximum.accumulate(wealth) - 1.0))


def long_short_returns(pred_df: pd.DataFrame, top_k: int, cfg: DualModelConfig) -> pd.DataFrame:
    # Daily gross and net returns of the equal-weighted long-short portfolio (EWLS)
    rows, prev_w, prev_r = [], {}, {}
    for date, g in pred_df.groupby("date"):
        g = g.drop_duplicates(subset=["ticker"]).sort_values("prediction", ascending=False)
        k = min(top_k, max(1, len(g) // 2))
        if len(g) < 2 * k or k <= 0:
            continue
        w = {t: (cfg.gross_exposure / 2.0) / k for t in g.head(k)["ticker"]}
        w.update({t: -(cfg.gross_exposure / 2.0) / k for t in g.tail(k)["ticker"]})
        targets = dict(zip(g["ticker"], g["target"].astype(float)))
        r_gross = float(sum(wt * targets[t] for t, wt in w.items() if t in targets))
        prices = g[["ticker", "close"]].dropna().set_index("ticker")["close"].astype(float).to_dict()
        _, cost = compute_ibkr_cost_return(drift_weights(prev_w, prev_r, cfg.target_is_log_return), w, prices, cfg)
        prev_w, prev_r = w, {t: targets[t] for t in w if t in targets}
        rows.append({"date": date, "gross": r_gross, "net": r_gross - cost})
    return pd.DataFrame(rows, columns=["date", "gross", "net"])


def evaluate_predictions(pred_df: pd.DataFrame, top_k: int, cfg: DualModelConfig):
    # MSE, MAE and the Sharpe ratios and maximum drawdown of EWLS
    empty = {"mse": np.nan, "mae": np.nan, "ls_sharpe": np.nan, "ls_sharpe_gross": np.nan, "ls_max_drawdown": np.nan}
    if pred_df.empty:
        return empty
    df = pred_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["date", "ticker", "prediction", "target"]).copy()
    if df.empty:
        return empty
    df["ticker"] = df["ticker"].astype(str)
    y, p = df["target"].to_numpy(dtype=float), df["prediction"].to_numpy(dtype=float)
    daily = long_short_returns(df, top_k, cfg)
    return {"mse": float(np.mean((p - y) ** 2)), "mae": float(np.mean(np.abs(p - y))),
            "ls_sharpe": _sharpe(daily["net"]), "ls_sharpe_gross": _sharpe(daily["gross"]),
            "ls_max_drawdown": _max_drawdown(daily["net"])}


def choose_downside_threshold(val_df: pd.DataFrame) -> float:
    # Probability threshold maximising balanced accuracy on the validation set
    d = val_df[["pred_down_prob", "target_downside"]].replace([np.inf, -np.inf], np.nan).dropna()
    if d.empty:
        return 0.5
    y = d["target_downside"].astype(int).to_numpy()
    p = d["pred_down_prob"].astype(float).to_numpy()
    if len(np.unique(y)) < 2:
        return 0.5
    thresholds = np.linspace(0.05, 0.95, 181)
    scores = np.array([balanced_accuracy_score(y, (p >= th).astype(int)) for th in thresholds])
    tied = np.where(np.isclose(scores, np.nanmax(scores), rtol=0.0, atol=1e-12))[0]
    return float(thresholds[min(tied, key=lambda i: abs(float(thresholds[i]) - 0.5))])


def downside_f1(pred_df: pd.DataFrame, threshold: float) -> Dict[str, float]:
    d = pred_df[["pred_down_prob", "target_downside"]].replace([np.inf, -np.inf], np.nan).dropna()
    if d.empty:
        return {"f1": np.nan}
    y = d["target_downside"].astype(int).to_numpy()
    yhat = (d["pred_down_prob"].astype(float).to_numpy() >= float(threshold)).astype(int)
    return {"f1": float(f1_score(y, yhat, zero_division=0))}


def build_confidence_weighted_strategy(pred_df: pd.DataFrame, cfg: DualModelConfig) -> pd.DataFrame:
    # CWS
    cols = ["date", "ticker", "split", "target", "prediction", "pred_shock", "pred_down_prob", "portfolio_side",
            "direction_confidence", "shock_stability", "confidence_score", "weight", "position_return",
            "strategy_return_gross", "turnover", "transaction_cost", "strategy_return"]
    if pred_df.empty:
        return pd.DataFrame(columns=cols)
    df = pred_df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["date", "ticker", "split", "prediction", "target", "pred_shock", "pred_down_prob"]).copy()
    if df.empty:
        return pd.DataFrame(columns=cols)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["ticker"] = df["ticker"].astype(str)
    train_abs = df.loc[df["split"].eq("train"), "pred_shock"].abs()
    if train_abs.empty:
        raise ValueError("Training predictions are needed for the shock threshold.")
    shock_threshold = float(train_abs.quantile(0.90))
    if not np.isfinite(shock_threshold):
        raise ValueError("Cannot estimate the training-period shock threshold.")

    top_k = cfg.strategy_top_k if cfg.strategy_top_k is not None else cfg.top_k
    rows, prev_w, prev_r = [], {}, {}
    for (_, split), day in df.groupby(["date", "split"], sort=True):
        day = day.drop_duplicates("ticker").sort_values(["prediction", "ticker"], ascending=[False, True])
        k = min(int(top_k), len(day) // 2)
        if k < 1:
            continue
        longs, shorts = day.head(k).copy(), day.tail(k).copy()
        longs["portfolio_side"], shorts["portfolio_side"] = 1.0, -1.0
        sel = pd.concat([longs, shorts], ignore_index=True)

        p = sel["pred_down_prob"].clip(0, 1)
        sel["direction_confidence"] = np.where(sel["portfolio_side"].eq(1), 1.0 - p, p)
        sel["shock_stability"] = np.where(sel["pred_shock"].abs() > shock_threshold, 0.5, 1.0)
        sel["confidence_score"] = sel["direction_confidence"] * sel["shock_stability"]
        sel["weight"] = 0.0
        for side in (+1, -1):
            mask = sel["portfolio_side"].eq(side)
            c = sel.loc[mask, "confidence_score"].to_numpy(dtype=float)
            if not np.isfinite(c).all() or c.sum() <= 1e-12:
                c = np.ones(len(c))
            sel.loc[mask, "weight"] = side * (cfg.gross_exposure / 2.0) * c / c.sum()

        sel["position_return"] = sel["weight"] * sel["target"]
        gross = float(sel["position_return"].sum())
        current = dict(zip(sel["ticker"], sel["weight"]))
        prices = day.dropna(subset=["close"]).set_index("ticker")["close"].to_dict()
        # trade from yesterday's drifted positions to today's targets
        turnover, cost = compute_ibkr_cost_return(
            drift_weights(prev_w.get(split, {}), prev_r.get(split, {}), cfg.target_is_log_return),
            current, prices, cfg)
        prev_w[split] = current
        prev_r[split] = dict(zip(sel["ticker"], sel["target"]))

        sel["strategy_return_gross"] = gross
        sel["turnover"] = turnover
        sel["transaction_cost"] = cost
        sel["strategy_return"] = gross - cost
        rows.append(sel)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=cols)


def evaluate_strategy(strategy_df: pd.DataFrame) -> Dict[str, float]:
    # Annualised Sharpe ratio and max drawdown
    test = strategy_df.loc[strategy_df["split"].eq("test"), ["date", "split", "strategy_return"]]
    r = test.drop_duplicates().sort_values("date")["strategy_return"].to_numpy(dtype=float)
    r = r[np.isfinite(r)]
    return {"sharpe": _sharpe(r), "max_drawdown": _max_drawdown(r)}


def validation_mse(pred_df: pd.DataFrame) -> float:
    # Mean squared error of the return forecasts, used for early stopping
    if pred_df.empty:
        return math.inf
    df = pred_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["date", "ticker", "prediction", "target"])
    if df.empty:
        return math.inf
    return float(np.mean((df["prediction"].to_numpy(dtype=float) - df["target"].to_numpy(dtype=float)) ** 2))


# Training
@torch.no_grad()
def predict_split(model, data: TailRiskGraphDataset, split: str, device: str) -> pd.DataFrame:
    model.eval()
    rows = []
    for batch_dates in data.iter_date_batches(split=split, shuffle=False):
        batch = data.get_batch(batch_dates, device=device)
        out = model(batch["x"], batch["feat_mask"], batch["hypergraphs"], batch["flow_graphs"])
        pred_ret = out["pred_ret"].cpu().numpy()
        pred_shock = out["pred_shock"].cpu().numpy()
        pred_down = 1.0 / (1.0 + np.exp(-np.clip(out["pred_down"].cpu().numpy(), -50, 50)))
        target = batch["target"].cpu().numpy()
        downside = batch["downside"].cpu().numpy()
        mask = batch["target_mask"].cpu().numpy() > 0.5

        for b, date in enumerate(batch["dates"]):
            for n in np.where(mask[b])[0]:
                pr, tg = float(pred_ret[b, n]), float(target[b, n])
                if not (np.isfinite(pr) and np.isfinite(tg)):
                    continue
                ticker = data.idx_to_ticker[int(n)]
                close = data.close_lookup.get((pd.Timestamp(date).normalize(), ticker), np.nan)
                rows.append({
                    "date": pd.Timestamp(date), "ticker": ticker, "prediction": pr, "target": tg,
                    "pred_shock": float(pred_shock[b, n]), "pred_down_prob": float(pred_down[b, n]),
                    "target_downside": float(downside[b, n]) if np.isfinite(downside[b, n]) else np.nan,
                    "split": split, "close": float(close) if np.isfinite(close) else np.nan,
                })
    return pd.DataFrame(rows)


def train_model(cfg: DualModelConfig) -> Dict[str, object]:
    set_seed(cfg.seed)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = TailRiskGraphDataset(cfg)
    model = TailRiskDualModel(len(data.feature_cols), cfg).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # early stopping on validation MSE
    best_val, best_state, patience = math.inf, None, cfg.early_stopping_patience
    for _ in range(cfg.epochs):
        model.train()
        for batch_dates in data.iter_date_batches(split="train", shuffle=True):
            batch = data.get_batch(batch_dates, device=cfg.device)
            out = model(batch["x"], batch["feat_mask"], batch["hypergraphs"], batch["flow_graphs"])
            loss = compute_total_loss(out, batch, cfg)
            optimizer.zero_grad()
            loss.backward()
            if cfg.grad_clip is not None and cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

        val_mse = validation_mse(predict_split(model, data, "val", cfg.device))
        val_mse = val_mse if np.isfinite(val_mse) else math.inf
        if val_mse < best_val:
            best_val, patience = val_mse, cfg.early_stopping_patience
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience -= 1
            if patience <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    preds = {s: predict_split(model, data, s, cfg.device) for s in ("train", "val", "test")}
    pred_all = pd.concat(preds.values(), ignore_index=True)

    threshold = choose_downside_threshold(preds["val"])
    downside_metrics = {"threshold": threshold, "test": downside_f1(preds["test"], threshold)}
    metrics = {"test": evaluate_predictions(preds["test"], cfg.top_k, cfg)}
    strategy_df = build_confidence_weighted_strategy(pred_all, cfg)
    strategy_metrics = {"test": evaluate_strategy(strategy_df)}

    pred_all.to_parquet(out_dir / "predictions.parquet", index=False)
    strategy_df.to_parquet(out_dir / "confidence_weighted_strategy.parquet", index=False)
    data.lookback_diagnostics.to_csv(out_dir / "lookback_diagnostics.csv", index=False)
    for name, obj in (("metrics.json", metrics), ("confidence_weighted_strategy_metrics.json", strategy_metrics),
                      ("downside_classification_metrics.json", downside_metrics),
                      ("config.json", {**asdict(cfg), "effective_lookback": data.lookback})):
        with open(out_dir / name, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)
    torch.save({"state_dict": model.state_dict(), "feature_cols": data.feature_cols, "tickers": data.tickers,
                "config": asdict(cfg), "effective_lookback": data.lookback}, out_dir / "dual_model.pt")

    print(f"Saved model outputs to {out_dir} (lookback {data.lookback})")
    return {"model": model, "predictions": pred_all, "metrics": metrics, "strategy": strategy_df,
            "strategy_metrics": strategy_metrics, "downside_metrics": downside_metrics,
            "effective_lookback": data.lookback}
