# Full model (A0) and ablations A1-A12
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, fields, replace
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from crash_prem_dual_model import DualModelConfig, evaluate_predictions, train_model
from crash_prem_network import TailRiskConfig, run_pipeline

OUT_ROOT = Path("ablation_outputs")
CRASH_DIR = "options_dir/crash_premium_outputs"

BASE_NETWORK_CFG = TailRiskConfig(
    crash_dir=CRASH_DIR,
    price_dir="stocks_dir",
    out_dir="ablation_outputs/A0_full/network",
    maturity_grid=(20, 40, 60, 126, 252),
    gamma=2.0,
    flow_window=60,
    flow_lags=(1, 2, 3),
    k_out=10,
    min_overlap=40,
    use_flow_graph=True,
    n_clusters=3,
    use_hypergraph=True,
    random_state=42,
    downside_kappa=0.0,
    train_start="2018-01-01",
    train_end="2020-12-31",
    val_start="2021-01-01",
    val_end="2021-12-31",
    test_start="2022-01-01",
    test_end="2022-12-31",
    n_pca_components=2,
)

BASE_MODEL_CFG = DualModelConfig(
    lookback=20,
    auto_lookback=True,
    lookback_diagnostic_max_lag=252,
    lookback_diagnostic_alpha=0.05,
    lookback_diagnostic_quantile=0.90,
    min_history_coverage=0.6,
    hidden_dim=96,
    temporal_layers=1,
    dropout=0.10,
    hyperedge_dropout=0.0,
    lr=1e-3,
    weight_decay=1e-4,
    epochs=80,
    batch_size_dates=16,
    early_stopping_patience=12,
    grad_clip=1.0,
    huber_delta=0.01,
    lambda_shock=0.01,
    lambda_down=0.01,
    use_downside_loss=True,
    top_k=2,
    strategy_top_k=2,
    gross_exposure=1.0,
    initial_capital=1_000_000.0,
    commission_per_share=0.0035,
    min_commission_per_order=0.35,
    max_commission_rate=0.01,
    seed=42,
)

# each specification changes one component of A0
ABLATIONS: Dict[str, dict] = {
    "A0_full": {"label": "A0 Proposed model"},
    "A1_no_pareto_tail": {"label": "A1 No Pareto tail",
                          "crash_dir": "options_dir/crash_premium_outputs_no_pareto"},
    "A2_q_crash_only": {"label": "A2 Risk-neutral (Q) only", "network": {"use_q_crash_only": True}},
    "A3_no_maturity_gate_nearest_expiry": {"label": "A3 Nearest-expiry premium",
                                           "network": {"gate_mode": "nearest"}},
    "A4_no_term_structure_20d_only": {"label": "A4 20-day maturity only",
                                      "network": {"maturity_grid": (20,), "n_pca_components": 1}},
    "A5_no_directed_flow_graph": {"label": "A5 No directed network", "network": {"use_flow_graph": False}},
    "A6_no_hypergraph": {"label": "A6 No hypergraph", "network": {"use_hypergraph": False}},
    "A7_no_graph_no_hypergraph": {"label": "A7 No network or hypergraph",
                                  "network": {"use_flow_graph": False, "use_hypergraph": False}},
    "A8_equal_maturity_weighting": {"label": "A8 Equal maturity weighting",
                                    "network": {"shock_weighting": "equal"}},
    "A9_iv_spline_smoothing": {"label": "A9 IV spline smoothing",
                               "crash_dir": "options_dir/crash_premium_outputs_iv_spline"},
    "A10_no_downside_auxiliary_loss": {"label": "A10 No downside auxiliary loss",
                                       "model": {"lambda_down": 0.0, "use_downside_loss": False}},
    "A11_no_shock_auxiliary_loss": {"label": "A11 No shock auxiliary loss", "model": {"lambda_shock": 0.0}},
    "A12_p_crash_only": {"label": "A12 Physical proxy (P) only",
                         "network": {"premium_col": "P_crash", "use_q_crash_only": False}},
}
TABLE4_ORDER = ["A0_full", "A1_no_pareto_tail", "A2_q_crash_only", "A12_p_crash_only", "A9_iv_spline_smoothing",
                "A3_no_maturity_gate_nearest_expiry", "A4_no_term_structure_20d_only", "A8_equal_maturity_weighting",
                "A5_no_directed_flow_graph", "A6_no_hypergraph", "A7_no_graph_no_hypergraph",
                "A10_no_downside_auxiliary_loss", "A11_no_shock_auxiliary_loss"]

METRICS = ["test_mse", "test_mae", "test_down_f1", "test_ls_sharpe_net", "test_ls_sharpe_gross",
           "test_ls_max_drawdown", "test_strategy_sharpe", "test_strategy_max_drawdown"]


def model_dir_for(name: str, seed: int) -> Path:
    root = OUT_ROOT / name
    return root / "model" if seed == BASE_MODEL_CFG.seed else root / f"model_seed{seed}"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def network_is_current(network_dir: Path, net_cfg: TailRiskConfig) -> bool:
    files = ["node_panel.parquet", "flow_edges.parquet", "hyperedges.parquet", "network_config.json"]
    if not all((network_dir / f).exists() for f in files):
        return False
    saved = read_json(network_dir / "network_config.json")
    return json.loads(json.dumps(saved, default=str)) == json.loads(json.dumps(asdict(net_cfg), default=str))


def run_one(name: str, spec: dict, seed: int, rebuild_network: bool) -> None:
    network_dir = OUT_ROOT / name / "network"
    net_cfg = replace(BASE_NETWORK_CFG, crash_dir=spec.get("crash_dir", CRASH_DIR), out_dir=str(network_dir),
                      **spec.get("network", {}))
    model_cfg = replace(BASE_MODEL_CFG,
                        panel_path=str(network_dir / "node_panel.parquet"),
                        flow_edges_path=str(network_dir / "flow_edges.parquet"),
                        hyperedges_path=str(network_dir / "hyperedges.parquet"),
                        out_dir=str(model_dir_for(name, seed)), seed=seed, **spec.get("model", {}))

    print(f"=== {name} | seed {seed} ===")
    if rebuild_network or not network_is_current(network_dir, net_cfg):
        run_pipeline(net_cfg)
    train_model(model_cfg)


def collect(name: str, seed: int) -> dict:
    """Test metrics of one trained model, read from the files written by train_model."""
    model_dir = model_dir_for(name, seed)
    pred = read_json(model_dir / "metrics.json")["test"]
    strat = read_json(model_dir / "confidence_weighted_strategy_metrics.json")["test"]
    saved_cfg = read_json(model_dir / "config.json")
    cfg = DualModelConfig(**{k: v for k, v in saved_cfg.items() if k in {f.name for f in fields(DualModelConfig)}})
    # EWLS drawdown from the saved test predictions
    p = pd.read_parquet(model_dir / "predictions.parquet")
    ls = evaluate_predictions(p[p["split"].eq("test")], cfg.top_k, cfg)
    return {
        "ablation": name,
        "seed": seed,
        "test_mse": pred["mse"],
        "test_mae": pred["mae"],
        "test_down_f1": read_json(model_dir / "downside_classification_metrics.json")["test"]["f1"],
        "test_ls_sharpe_net": pred["ls_sharpe"],
        "test_ls_sharpe_gross": pred["ls_sharpe_gross"],
        "test_ls_max_drawdown": ls["ls_max_drawdown"],
        "test_strategy_sharpe": strat["sharpe"],
        "test_strategy_max_drawdown": strat["max_drawdown"],
    }


def aggregate_over_seeds(summary: pd.DataFrame) -> pd.DataFrame:
    # Median and interquartile range of each metric across seeds
    rows = []
    for name, g in summary.groupby("ablation", sort=False):
        row = {"ablation": name, "n_seeds": int(g["seed"].nunique())}
        for m in METRICS:
            x = pd.to_numeric(g[m], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
            q1, med, q3 = np.percentile(x, [25, 50, 75]) if len(x) else (np.nan,) * 3
            row.update({f"{m}_median": med, f"{m}_iqr": q3 - q1})
        rows.append(row)
    return pd.DataFrame(rows)


def fmt(agg_row: pd.Series, metric: str, scale: float, decimals: int) -> str:
    med, iqr = agg_row[f"{metric}_median"], agg_row[f"{metric}_iqr"]
    if pd.isna(med):
        return ""
    return f"{med * scale:.{decimals}f} ({iqr * scale:.{decimals}f})"


def paper_tables(agg: pd.DataFrame):
    by_name = agg.set_index("ablation")

    table2 = pd.DataFrame()
    if "A0_full" in by_name.index:
        a0 = by_name.loc["A0_full"]
        table2 = pd.DataFrame([
            ("Net EWLS Sharpe", fmt(a0, "test_ls_sharpe_net", 1, 3)),
            ("Gross EWLS Sharpe", fmt(a0, "test_ls_sharpe_gross", 1, 3)),
            ("Net CWS Sharpe", fmt(a0, "test_strategy_sharpe", 1, 3)),
            ("Maximum drawdown, net EWLS (%)", fmt(a0, "test_ls_max_drawdown", 100, 3)),
            ("Maximum drawdown, net CWS (%)", fmt(a0, "test_strategy_max_drawdown", 100, 3)),
            ("MSE (x1e-4)", fmt(a0, "test_mse", 1e4, 3)),
            ("MAE (x1e-2)", fmt(a0, "test_mae", 1e2, 4)),
            ("F1", fmt(a0, "test_down_f1", 1, 3)),
        ], columns=["Metric", "Median (IQR)"])

    rows = []
    for name in [n for n in TABLE4_ORDER if n in by_name.index]:
        r = by_name.loc[name]
        rows.append({"Specification": ABLATIONS[name]["label"],
                     "MSE (x1e-4)": fmt(r, "test_mse", 1e4, 3),
                     "MAE (x1e-3)": fmt(r, "test_mae", 1e3, 3),
                     "F1": fmt(r, "test_down_f1", 1, 3),
                     "Net LS Sharpe": fmt(r, "test_ls_sharpe_net", 1, 3),
                     "Net CWS Sharpe": fmt(r, "test_strategy_sharpe", 1, 3)})
    return table2, pd.DataFrame(rows)


def run_all(only: Optional[Sequence[str]], seeds: Sequence[int], overwrite: bool) -> pd.DataFrame:
    unknown = set(only or []) - set(ABLATIONS)
    if unknown:
        raise ValueError(f"Unknown specification(s): {sorted(unknown)}")
    names = list(ABLATIONS) if not only else [n for n in ABLATIONS if n in set(only)]

    rows = []
    for name in names:
        network_built = False
        for seed in seeds:
            done = (model_dir_for(name, seed) / "metrics.json").exists()
            if done and not overwrite:
                print(f"Using saved outputs of {name} seed {seed} (use --overwrite to retrain).")
            else:
                # the network does not depend on the model seed
                run_one(name, ABLATIONS[name], seed, rebuild_network=overwrite and not network_built)
                network_built = True
            rows.append(collect(name, seed))

    summary = pd.DataFrame(rows)
    agg = aggregate_over_seeds(summary)
    table2, table4 = paper_tables(agg)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_ROOT / "ablation_summary_by_seed.csv", index=False)
    agg.to_csv(OUT_ROOT / "ablation_summary_seed_agg.csv", index=False)
    table2.to_csv(OUT_ROOT / "table2_proposed_model.csv", index=False)
    table4.to_csv(OUT_ROOT / "table4_ablations.csv", index=False)
    print(table2.to_string(index=False))
    print(table4.to_string(index=False))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full model and the ablations over several seeds.")
    parser.add_argument("--only", nargs="*", default=None, help="run only these specifications")
    parser.add_argument("--seeds", nargs="*", type=int, default=[BASE_MODEL_CFG.seed])
    parser.add_argument("--overwrite", action="store_true", help="retrain models that already have saved outputs")
    args = parser.parse_args()
    run_all(args.only, [int(s) for s in args.seeds], args.overwrite)


if __name__ == "__main__":
    main()
