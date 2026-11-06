"""risk-neutral distribution, physical-distribution proxy and historical returns.

All three are distributions of x = log(S_T / F_{t,T}) for one stock, date and expiry:
  * Q: Breeden-Litzenberger masses from notebook 03 (the same masses notebook 04 uses).
  * P: exponential tilt of Q matching mu_hist = d mu_daily - log(F/S_t) and var_hist = d var_daily, with the
    daily moments from the 252 trading days ending at t-1 (as in notebook 04).
  * Historical: the same 252 daily log returns rescaled to the d-day horizon,
    x_j = d mu_daily + sqrt(d) (r_j - mu_daily) - log(F/S_t),
    so their mean and variance are exactly the targets matched by P.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.optimize import root
from scipy.stats import gaussian_kde

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOOKBACK_DAYS = 252
MIN_PERIODS = 126
TINY = np.finfo(float).tiny


def trapezoid_weights(v):
    v = np.asarray(v, dtype=float)
    w = np.empty_like(v)
    w[0], w[-1] = (v[1] - v[0]) / 2, (v[-1] - v[-2]) / 2
    w[1:-1] = (v[2:] - v[:-2]) / 2
    return w


def load_rnd_slice(options_dir, ticker, date, exdate):
    # Q on the strike grid
    rnd = pd.read_parquet(Path(options_dir) / f"{ticker}_breeden_litzenberger_rnd_density.parquet")
    rnd["date"] = pd.to_datetime(rnd["date"]).dt.normalize()
    rnd["exdate"] = pd.to_datetime(rnd["exdate"]).dt.normalize()
    g = rnd[(rnd["date"] == date) & (rnd["exdate"] == exdate)].dropna(subset=["K", "F"])
    if g.empty:
        raise ValueError(f"No density for {ticker} {date.date()} {exdate.date()}")
    g = g[(g["K"] > 0) & (g["F"] > 0)].sort_values("K").drop_duplicates(subset=["K"], keep="last")

    K = g["K"].to_numpy(dtype=float)
    p = g["risk_neutral_pdf_projected"].to_numpy(dtype=float) * g["integration_weight"].to_numpy(dtype=float)
    ok = np.isfinite(K) & np.isfinite(p) & (K > 0) & (p >= 0)
    K, p = K[ok], p[ok]
    F = float(np.nanmedian(g["F"]))
    return K, p / p.sum(), F, float(np.nanmedian(g["dte"])), float(np.nanmedian(g["K_thr"]))


def daily_returns(stock_dir, ticker, date):
    # The 252 daily log returns ending at t-1 and the contract-basis spot at t
    px = pd.read_csv(Path(stock_dir) / f"{ticker}_stock.csv")
    px["date"] = pd.to_datetime(px["date"]).dt.normalize()
    px = px.sort_values("date")
    r = np.log(px["close_yahoo_split_adjusted"] / px["close_yahoo_split_adjusted"].shift(1))
    r = r[px["date"] < date].dropna().to_numpy()[-LOOKBACK_DAYS:]
    if len(r) < MIN_PERIODS:
        raise ValueError(f"Only {len(r)} daily returns before {date.date()}")
    spot = float(px.loc[px["date"] == date, "spot"].iloc[0])
    return r, spot


def tilted_probs(p_q, x, theta, eta):
    logw = np.log(np.maximum(p_q, TINY)) + theta * x + eta * x**2
    w = np.exp(logw - np.max(logw))
    return w / w.sum()


def fit_tilt(p_q, x, target_mean, target_var):
    if not (x.min() <= target_mean <= x.max()):
        raise ValueError("Target mean outside the strike range; notebook 04 drops this slice.")

    def equations(par):
        pp = tilted_probs(p_q, x, *par)
        mu = np.sum(pp * x)
        return np.array([mu - target_mean, np.sum(pp * (x - mu) ** 2) - target_var])

    best = None
    for start in ([0.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]):
        sol = root(equations, np.array(start))
        resid = float(np.linalg.norm(equations(sol.x)))
        if best is None or resid < best[2]:
            best = (sol.x, sol.success, resid)
    if not (best[1] and best[2] < 1e-6):
        raise RuntimeError("Tilt did not converge; notebook 04 drops this slice.")
    return tilted_probs(p_q, x, *best[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="AMZN")
    ap.add_argument("--date", default="2020-08-14")
    ap.add_argument("--exdate", default="2020-09-11")
    ap.add_argument("--options-dir", default="options_dir")
    ap.add_argument("--stock-dir", default="stocks_dir")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    date, exdate = pd.Timestamp(a.date), pd.Timestamp(a.exdate)

    K, p_q, F, d, K_thr = load_rnd_slice(a.options_dir, a.ticker, date, exdate)
    x = np.log(K / F)
    r, spot = daily_returns(a.stock_dir, a.ticker, date)
    mu_daily, var_daily = r.mean(), r.var(ddof=1)
    shift = np.log(F / spot)
    mu_hist, var_hist = d * mu_daily - shift, d * var_daily

    p_p = fit_tilt(p_q, x, mu_hist, var_hist)
    x_hist = d * mu_daily + np.sqrt(d) * (r - mu_daily) - shift
    x_thr = np.log(K_thr / F)

    def moments(p):
        mu = np.sum(p * x)
        return mu, np.sum(p * (x - mu) ** 2)

    print(f"{a.ticker} {date.date()} -> {exdate.date()}, d = {d:g} business days")
    print(f"{'':12s}{'mean':>10s}{'variance':>12s}")
    for name, (m, v) in [("Q", moments(p_q)), ("P", moments(p_p)), ("Historical", (x_hist.mean(), x_hist.var(ddof=1)))]:
        print(f"{name:12s}{m:10.5f}{v:12.6f}")

    # Q and P masses shown as densities in x
    lo, hi = min(x.min(), x_hist.min()), max(x.max(), x_hist.max())
    grid = np.linspace(lo, hi, 600)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(x, p_q / trapezoid_weights(x), lw=2, label="Risk-neutral $Q$")
    ax.plot(x, p_p / trapezoid_weights(x), lw=2, label="Physical proxy $P$")
    ax.plot(grid, gaussian_kde(x_hist)(grid), lw=1.8, color="tab:green",
            label=f"Historical returns (past 252 days, scaled to {d:g} days)")
    ax.axvline(x_thr, color="black", ls="--", lw=1.0, label="Downside-event threshold")
    ax.set_xlabel(r"$x=\log(S_T/F_{t,T})$")
    ax.set_ylabel("Density")
    ax.set_title(f"{a.ticker}: valuation {date.date()}, expiry {exdate.date()}")
    ax.legend(fontsize=8, frameon=False)
    ax.set_xlim(lo, hi)
    fig.tight_layout()
    out = a.out or f"figure2_{a.ticker}.png"
    fig.savefig(out, dpi=200)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
