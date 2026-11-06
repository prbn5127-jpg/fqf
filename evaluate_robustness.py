# Trading implementation checks
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ANN = 252.0


def sharpe(r) -> float:
    r = np.asarray(r, float)
    r = r[np.isfinite(r)]
    if len(r) < 2 or np.std(r, ddof=1) < 1e-15:
        return np.nan
    return float(np.sqrt(ANN) * r.mean() / r.std(ddof=1))


def strategy_daily(strategy_df: pd.DataFrame, split: str = "test") -> pd.DataFrame:
    s = strategy_df[strategy_df["split"].astype(str).eq(split)]
    d = s[["date", "strategy_return", "strategy_return_gross", "turnover", "transaction_cost"]]
    d = d.drop_duplicates("date").sort_values("date").reset_index(drop=True)
    d["date"] = pd.to_datetime(d["date"])
    return d


def lagged_execution_daily(strategy_df: pd.DataFrame, pred: pd.DataFrame, split: str = "test") -> pd.DataFrame:
    """Positions formed at t earn r_{t+2} (the next date's target) instead of r_{t+1}; costs are unchanged."""
    s = strategy_df[strategy_df["split"].astype(str).eq(split)].copy()
    p = pred[pred["split"].astype(str).eq(split)].copy()
    s["date"] = pd.to_datetime(s["date"])
    p["date"] = pd.to_datetime(p["date"])
    dates = np.array(sorted(p["date"].unique()))
    next_date = dict(zip(dates[:-1], dates[1:]))
    target = p.set_index(["date", "ticker"])["target"].to_dict()

    s["next_date"] = s["date"].map(next_date)
    s = s.dropna(subset=["next_date"])
    s["target_next"] = [target.get((d, t), np.nan) for d, t in zip(s["next_date"], s["ticker"].astype(str))]
    s["pos_ret"] = s["weight"] * s["target_next"]
    daily = s.groupby("date").agg(gross=("pos_ret", lambda x: x.sum(min_count=1)),
                                  cost=("transaction_cost", "first"),
                                  turnover=("turnover", "first")).reset_index()
    daily["strategy_return"] = daily["gross"] - daily["cost"]
    return daily.dropna(subset=["strategy_return"])


def seed_sharpes(model_dir: Path, extra_cost_bp: float) -> list:
    # Baseline, one-day-lag and extra-cost Sharpe ratios of one trained model
    pred = pd.read_parquet(model_dir / "predictions.parquet")
    strat = pd.read_parquet(model_dir / "confidence_weighted_strategy.parquet")
    for df in (pred, strat):
        df["date"] = pd.to_datetime(df["date"])
        df["ticker"] = df["ticker"].astype(str)

    daily = strategy_daily(strat)
    lagged = lagged_execution_daily(strat, pred)
    extra = daily["strategy_return"] - extra_cost_bp * 1e-4 * daily["turnover"]
    return [sharpe(daily["strategy_return"]), sharpe(lagged["strategy_return"]), sharpe(extra)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="ablation_outputs")
    ap.add_argument("--reference", default="A0_full")
    ap.add_argument("--seeds", nargs="*", type=int, default=[42],
                    help="model seeds; with several seeds the table reports median (IQR) across seeds")
    ap.add_argument("--extra-cost-bp", type=float, default=5.0)
    ap.add_argument("--out", default="robustness_outputs")
    args = ap.parse_args()

    run_dir = Path(args.root) / args.reference
    labels = ["Baseline", "One-day execution lag", f"Additional {args.extra_cost_bp:g} bp one-way cost"]
    per_seed = pd.DataFrame(
        [seed_sharpes(run_dir / ("model" if s == 42 else f"model_seed{s}"), args.extra_cost_bp) for s in args.seeds],
        index=args.seeds, columns=labels,
    )

    # Median and interquartile range 
    q1, med, q3 = (per_seed.quantile(q) for q in (0.25, 0.50, 0.75))
    table = pd.DataFrame({"specification": labels, "net_strategy_sharpe_median": med.values,
                          "net_strategy_sharpe_iqr": (q3 - q1).values})
    table["change_from_baseline_pct"] = 100.0 * (table["net_strategy_sharpe_median"] / med["Baseline"] - 1.0)
    table.loc[0, "change_from_baseline_pct"] = np.nan

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(out / "table5_by_seed.csv", index_label="seed")
    table.to_csv(out / "table5_robustness.csv", index=False)
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
