# downside-risk premium term structure of one stock on two dates.

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MATURITIES = np.array([20, 40, 60, 126, 252])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="AMZN")
    ap.add_argument("--dates", nargs="+", default=["2019-10-02", "2022-02-02"])
    ap.add_argument("--crash-dir", default="options_dir/crash_premium_outputs")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    df = pd.read_parquet(Path(a.crash_dir) / a.ticker / "strict_crash_premium_term_structure.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df = df[np.isfinite(df["dte"]) & np.isfinite(df["crash_premium"])]

    span = df.groupby("date")["dte"].agg(["min", "max", "nunique"])
    eligible = span[(span["min"] <= MATURITIES.min()) & (span["max"] >= MATURITIES.max()) & (span["nunique"] >= 2)].index
    if len(eligible) == 0:
        raise ValueError("No date has expiries spanning 20 to 252 business days.")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for target in pd.to_datetime(a.dates):
        date = min(eligible, key=lambda d: abs((d - target).days))
        curve = df[df["date"] == date].groupby("dte")["crash_premium"].mean()
        drp = np.interp(MATURITIES, curve.index.to_numpy(), curve.to_numpy())
        ax.plot(MATURITIES, drp, marker="o", linewidth=2, label=str(date.date()))
        print(f"{date.date()} (requested {target.date()}): " + ", ".join(f"{m}d {v:+.3f}" for m, v in zip(MATURITIES, drp)))

    ax.axhline(0, color="gray", linestyle="--")
    ax.set_xticks(MATURITIES)
    ax.set_xlabel("Maturity (business days)")
    ax.set_ylabel("Log downside probability ratio")
    ax.set_title(f"{a.ticker}: Downside-Risk Premium Term Structure")
    ax.legend(title="Observation date")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    out = a.out or f"figure3_{a.ticker}.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
