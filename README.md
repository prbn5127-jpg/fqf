# Code for "Option-Implied Downside-Risk Premium for Equity Trading"

Run the steps in this order.

| Step | File | Produces |
|---|---|---|
| 1 | `01_process_options.py` | stock prices, dividends, Treasury curve, option base tables; American-to-European conversion shares |
| 2 | `02_repair_arbitrage.py` | arbitrage-repaired call and put quotes |
| 3 | `03_smooth_rnd_pareto.py` | smoothed surfaces, risk-neutral densities, Pareto tails; Table 1, Figure 4 (`figure4a_GOOGL.png`, `figure4b_GOOGL.png`) |
| 4 | `04_downside_risk_premium.py` | downside-risk premium per stock (main run and A1 without the Pareto tail) |
| 4b | `03b_a9_iv_spline.py` (with `a9_helpers_03.py`, `a9_helpers_04.py`) | steps 3 and 4 with spline smoothing of implied volatility, for A9 |
| 5 | `plot_q_p_historical.py`, `plot_term_structure.py` | Figures 2 and 3 |
| 6 | `run_ablations.py` | network/hypergraph (`crash_prem_network.py`) and model (`crash_prem_dual_model.py`) for A0 and the ablations A1–A12; Tables 2 and 4 |
| 7 | `evaluate_robustness.py` | Table 5 |
| 8 | `05_cumulative_returns.ipynb` | Figures 5 and 6 |
| 9 | `06_network_figures.ipynb` | Figures 7 and 8, network numbers in Section 6.6 |


Example:

```
python 01_process_options.py
python 02_repair_arbitrage.py
python 03_smooth_rnd_pareto.py
python 04_downside_risk_premium.py
python 03b_a9_iv_spline.py
python plot_q_p_historical.py --ticker AMZN --date 2020-08-14 --exdate 2020-09-11
python plot_term_structure.py --ticker AMZN --dates 2019-10-02 2022-02-02
python run_ablations.py --seeds 42 1 2 3
python evaluate_robustness.py --seeds 42 1 2 3
```

Tables 2, 4 and 5 report the median (IQR) across the four seeds. `run_ablations.py` reuses saved results and
only trains models whose outputs are missing; add `--overwrite` to retrain.

A9 repeats steps 3 and 4 with the 2D local-linear smoother replaced by a 1D cubic smoothing spline of total
implied variance in log-moneyness, fitted separately for each expiry. `a9_helpers_03.py` and `a9_helpers_04.py`
contain the functions used for this run.

Input data: WRDS OptionMetrics files in `output_per_ticker/<TICKER>.csv`. Stock prices and dividends are
downloaded from Yahoo Finance and Treasury yields from FRED in step 1.

Packages: numpy, pandas 2.x, scipy, scikit-learn, matplotlib, torch, pyarrow, yfinance, arbitragerepair (with
cvxopt). `03b_a9_iv_spline.py` needs pandas 2.x: pandas 3 drops the grouping columns inside `groupby().apply`.
