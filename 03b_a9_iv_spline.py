# Ablation A9: implied-volatility spline smoothing.
# Repeats steps 3 and 4 for the eight stocks with the 
# 2D local-linear smoother replaced by a 1D cubic smoothing spline of total implied variance in log-moneyness,
# fitted separately for each expiry; all other settings are unchanged.

from pathlib import Path

import pandas as pd

import a9_helpers_03 as h3
import a9_helpers_04 as h4

TICKERS = ["AMZN", "GOOGL", "JNJ", "JPM", "MSFT", "PG", "TSLA", "V"]
REPAIRED_DIR = Path("options_dir")
SPLINE_DIR = Path("options_dir_iv_spline")
CRASH_DIR = Path("options_dir") / "crash_premium_outputs_iv_spline"
SPLINE_SMOOTHING_MULTIPLIER = 1.0
Q_TAIL = 0.10

assert h3.SMOOTHING_SUCCESS_STATUS == "smoothed_1d_spline_per_expiry"

pareto_parts = []
for ticker in TICKERS:
    h3.run_spline_iv_smoothing_for_ticker(
        ticker=ticker,
        repaired_options_dir=REPAIRED_DIR,
        output_dir=SPLINE_DIR,
        smoothing_multiplier=SPLINE_SMOOTHING_MULTIPLIER,
        skip_existing=False,
    )
    h3.run_rnd_extraction_one_ticker(ticker=ticker, options_dir=SPLINE_DIR, output_dir=SPLINE_DIR,
                                     save_density_csv=False)
    rnd_path = SPLINE_DIR / f"{ticker}_breeden_litzenberger_rnd_density.parquet"
    rnd = pd.read_parquet(rnd_path)
    if rnd.empty:
        raise RuntimeError(f"{ticker}: no usable spline-smoothed density slices")
    pareto_parts.append(h3.run_pareto_tail_fit_for_q(rnd_df=rnd, output_dir=SPLINE_DIR, q_tail=Q_TAIL))

    h4.run_crash_premium_pipeline(
        rnd_file=rnd_path,
        rnd_dir=SPLINE_DIR,
        pareto_file=SPLINE_DIR / f"{ticker}_pareto_tail_adjustment_summary.parquet",
        hist_moment_file=h4.HIST_MOMENT_FILE,
        tickers=[ticker],
        output_dir=CRASH_DIR / ticker,
        target_dte=h4.TARGET_DTE,
        use_pareto_tail_adjustment=True,
        z_sigma_crash=h4.Z_SIGMA_CRASH,
    )

combined = pd.concat(pareto_parts, ignore_index=True)
combined.to_parquet(SPLINE_DIR / "pareto_tail_adjustment_summary.parquet", index=False)
combined.to_csv(SPLINE_DIR / "pareto_tail_adjustment_summary.csv", index=False)
print("A9 premia saved to", CRASH_DIR)
