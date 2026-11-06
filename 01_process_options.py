# 01 Option data preparation
#
# Builds the option base table for each stock: contract-basis spot price, trailing dividend yield,
# maturity-matched Treasury rate, forward price and discount factor (ACT/365), and European-equivalent
# option prices obtained by repricing each quote at the vendor implied volatility.

# %%
from pathlib import Path
from time import sleep

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

TICKERS = ["AMZN", "GOOGL", "JNJ", "JPM", "MSFT", "PG", "TSLA", "V"]

OPTION_DIR = Path("output_per_ticker")
STOCK_DIR = Path("stocks_dir")
OUT_DIR = Path("options_dir")
OUT_DIR.mkdir(parents=True, exist_ok=True)

STRIKE_DIVISOR = 1000.0
MIN_DTE_BUS, MAX_DTE_BUS = 5, 252

RATE_COLS = ["DGS1MO", "DGS3MO", "DGS6MO", "DGS1"]
RATE_TENORS = np.array([1 / 12, 3 / 12, 6 / 12, 1.0])

# %%
# Sample period
dates = [pd.read_csv(OPTION_DIR / f"{t}.csv", usecols=["date"])["date"] for t in TICKERS]
dates = pd.to_datetime(pd.concat(dates), errors="coerce").dropna()
option_min_date, option_max_date = dates.min().normalize(), dates.max().normalize()
print("Option dates:", option_min_date.date(), "to", option_max_date.date())

# %% [markdown]
# ## Stock prices and dividends
#
# Yahoo closes are split-adjusted, while WRDS strikes are on the original contract basis, so the close is
# multiplied by all split ratios that occur after the valuation date to recover the contract-basis spot.

# %%
def future_split_factors(ticker, dates):
    splits = yf.Ticker(ticker).splits
    dates = pd.to_datetime(dates, errors="coerce").dt.normalize()
    if splits is None or len(splits) == 0:
        return pd.Series(1.0, index=dates.index)

    split_dates = pd.to_datetime(splits.index)
    if split_dates.tz is not None:
        split_dates = split_dates.tz_localize(None)
    ratios = pd.Series(splits.values, index=split_dates.normalize())
    ratios = pd.to_numeric(ratios, errors="coerce").dropna()
    ratios = ratios[ratios > 0]

    # On the split date itself prices are already post-split.
    return pd.Series(
        [float(ratios[ratios.index > d].prod()) if pd.notna(d) else np.nan for d in dates],
        index=dates.index,
    )


def download_stock_prices(tickers, start, end, out_dir=STOCK_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)
    for t in tickers:
        px = yf.download(t, start=start, end=end, auto_adjust=False, progress=False, threads=False)
        if isinstance(px.columns, pd.MultiIndex):
            close = px[("Close", t)] if ("Close", t) in px.columns else px.xs("Close", axis=1, level=0).iloc[:, 0]
        else:
            close = px["Close"]

        out = close.rename("close_yahoo_split_adjusted").reset_index()
        out.columns = ["date", "close_yahoo_split_adjusted"]
        out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
        out["close_yahoo_split_adjusted"] = pd.to_numeric(out["close_yahoo_split_adjusted"], errors="coerce")
        out["ticker"] = t
        out["split_unadjust_factor"] = future_split_factors(t, out["date"])
        out["spot"] = out["close_yahoo_split_adjusted"] * out["split_unadjust_factor"]

        out = out.dropna(subset=["date", "spot"])
        out = out[out["spot"] > 0]
        out = out[["date", "ticker", "spot", "close_yahoo_split_adjusted", "split_unadjust_factor"]]
        out.sort_values("date").reset_index(drop=True).to_csv(out_dir / f"{t}_stock.csv", index=False)
        print(t, len(out), "rows")


def download_dividends(tickers, start, end, out_csv="dividends_for_option_tickers.csv"):
    rows = []
    for t in tickers:
        for attempt in range(1, 4):
            try:
                s = yf.Ticker(t).dividends
                break
            except Exception:
                if attempt == 3:
                    raise
                sleep(2 * attempt)
        if s is None or len(s) == 0:
            continue
        s.index = pd.to_datetime(s.index).tz_localize(None)
        s = s.loc[start:end]
        if len(s):
            df = s.rename("dividend").to_frame()
            df.index.name = "ex_date"
            rows.append(df.reset_index().assign(ticker=t))

    div = pd.concat(rows, ignore_index=True)
    div["dividend"] = pd.to_numeric(div["dividend"], errors="coerce")
    div["ex_date"] = pd.to_datetime(div["ex_date"], errors="coerce")
    div = div.dropna(subset=["ex_date", "dividend", "ticker"])
    div = div[div["dividend"] > 0].sort_values(["ticker", "ex_date"]).reset_index(drop=True)
    div.to_csv(out_csv, index=False)
    return div

# %% [markdown]
# ## Treasury curve (FRED)

# %%
def download_fred_rate(series_id, start, end):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    df = pd.read_csv(url)
    df.columns = ["date", "rate_percent"]
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df["rate_percent"] = pd.to_numeric(df["rate_percent"], errors="coerce")
    df = df.dropna()
    df = df[(df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))]
    df[series_id] = df["rate_percent"] / 100.0
    return df[["date", series_id]].sort_values("date")


def load_treasury_curve(start, end):
    curve = pd.DataFrame({"date": pd.date_range(start=start, end=end, freq="D")})
    for sid in RATE_COLS:
        curve = curve.merge(download_fred_rate(sid, start, end), on="date", how="left")
    # weekends and holidays take the last available curve
    curve[RATE_COLS] = curve[RATE_COLS].ffill()
    return curve

# %% [markdown]
# ## Maturities, rates, dividends, forwards
#
# Business days (`dte_bus`) for sample selection and the maturity grid
# and ACT/365
# (`tau_cal_365`) for rates, discounting, forwards and option pricing.

# %%
def add_maturities(df):
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce").dt.normalize()
    d["exdate"] = pd.to_datetime(d["exdate"], errors="coerce").dt.normalize()
    valid = d["date"].notna() & d["exdate"].notna() & (d["exdate"] >= d["date"])

    d["tau_cal_365"] = np.nan
    d.loc[valid, "tau_cal_365"] = (d.loc[valid, "exdate"] - d.loc[valid, "date"]).dt.days / 365.0

    dte_bus = np.full(len(d), np.nan)
    pos = np.flatnonzero(valid.to_numpy())
    if len(pos):
        dte_bus[pos] = np.busday_count(
            d.iloc[pos]["date"].to_numpy(dtype="datetime64[D]"),
            d.iloc[pos]["exdate"].to_numpy(dtype="datetime64[D]"),
        )
    d["dte_bus"] = dte_bus
    return d


def add_treasury_rate(opt, curve):
    out = opt.copy()
    out["_row"] = np.arange(len(out))
    out = out.sort_values("date").reset_index(drop=True)
    out = pd.merge_asof(out, curve[["date"] + RATE_COLS].sort_values("date"), on="date", direction="backward")

    rates = out[RATE_COLS].to_numpy(dtype=float)
    tau = out["tau_cal_365"].to_numpy(dtype=float)
    r = np.full(len(out), np.nan)
    for idx in out.groupby("date").indices.values():
        ok = np.isfinite(rates[idx[0]])
        use = idx[np.isfinite(tau[idx]) & (tau[idx] > 0)]
        if ok.any() and len(use):
            # np.interp keeps the end rates flat outside the 1M-1Y range
            r[use] = np.interp(tau[use], RATE_TENORS[ok], rates[idx[0]][ok])

    out["r"] = r
    out = out.sort_values("_row").drop(columns=["_row"] + RATE_COLS).reset_index(drop=True)
    return out


def trailing_dividend_yield(opt, div_csv):
    # Continuous yield equivalent to the trailing 12-month dividend yield (past dividends only)
    div = pd.read_csv(div_csv)
    div["ex_date"] = pd.to_datetime(div["ex_date"], errors="coerce").dt.normalize()
    div = div.dropna(subset=["ticker", "dividend", "ex_date"])
    div = div[div["dividend"] > 0].sort_values(["ticker", "ex_date"]).reset_index(drop=True)
    div["cum_div"] = div.groupby("ticker")["dividend"].cumsum()
    cum_div = div[["ticker", "ex_date", "cum_div"]].sort_values(["ex_date", "ticker"], kind="mergesort")

    grid = opt.groupby(["ticker", "date"], as_index=False)["spot"].median().sort_values(["ticker", "date"])
    if div.empty:
        grid["trail_div_12m"] = 0.0
    else:
        dates = grid[["ticker", "date"]].copy()
        dates["date_lag"] = dates["date"] - pd.offsets.BDay(252)
        now = dates.sort_values(["date", "ticker"], kind="mergesort").reset_index(drop=True)
        lag = dates.sort_values(["date_lag", "ticker"], kind="mergesort").reset_index(drop=True)
        m_now = pd.merge_asof(now, cum_div, left_on="date", right_on="ex_date", by="ticker")
        m_lag = pd.merge_asof(lag, cum_div, left_on="date_lag", right_on="ex_date", by="ticker")
        trail = now[["ticker", "date"]].copy()
        trail["trail_div_12m"] = m_now["cum_div"].fillna(0.0).to_numpy() - m_lag["cum_div"].fillna(0.0).to_numpy()
        grid = grid.merge(trail, on=["ticker", "date"], how="left")

    y = grid["trail_div_12m"].fillna(0.0) / grid["spot"]
    if not ((y >= 0) & (y < 1)).all():
        raise ValueError("Dividend yield outside [0, 1); check dividends and splits.")
    grid["q"] = -np.log1p(-y)
    return opt.merge(grid[["ticker", "date", "q"]], on=["ticker", "date"], how="left")


def black_price(F, K, tau, DF, sigma, cp_flag):
    # Black (1976) price on the forward
    F, K, tau, DF, sigma = (np.asarray(v, dtype=float) for v in (F, K, tau, DF, sigma))
    cp_flag = np.asarray(cp_flag).astype(str)
    out = np.full(len(F), np.nan)
    ok = (np.isfinite(F) & np.isfinite(K) & np.isfinite(tau) & np.isfinite(DF) & np.isfinite(sigma)
          & (F > 0) & (K > 0) & (tau > 0) & (DF > 0) & (sigma > 0))
    if not ok.any():
        return out

    F, K, tau, DF, sigma, cp = F[ok], K[ok], tau[ok], DF[ok], sigma[ok], cp_flag[ok]
    vol = sigma * np.sqrt(tau)
    d1 = (np.log(F / K) + 0.5 * sigma**2 * tau) / vol
    d2 = d1 - vol
    call = DF * (F * norm.cdf(d1) - K * norm.cdf(d2))
    put = DF * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
    out[ok] = np.where(cp == "C", call, np.where(cp == "P", put, np.nan))
    return out

# %% [markdown]
# ## Option base table
#
# Quote filters: positive bid, ask not below bid, 5-252 business days to expiry. American quotes are
# converted to European-equivalent prices by repricing at the vendor implied volatility; the original
# half-spread is kept around the European midpoint.

# %%
def build_option_base(ticker, curve, div_csv="dividends_for_option_tickers.csv"):
    opt = pd.read_csv(OPTION_DIR / f"{ticker}.csv")
    opt["date"] = pd.to_datetime(opt["date"], errors="coerce").dt.normalize()
    opt["exdate"] = pd.to_datetime(opt["exdate"], errors="coerce").dt.normalize()
    opt["cp_flag"] = opt["cp_flag"].astype(str).str.upper().str.strip()
    opt = opt[opt["ticker"].astype(str).str.upper().str.strip() == ticker].copy()
    opt["ticker"] = ticker

    stk = pd.read_csv(STOCK_DIR / f"{ticker}_stock.csv", usecols=["date", "spot"])
    stk["date"] = pd.to_datetime(stk["date"], errors="coerce").dt.normalize()
    stk["spot"] = pd.to_numeric(stk["spot"], errors="coerce")
    stk = stk.dropna()
    stk = stk[stk["spot"] > 0].drop_duplicates("date").assign(ticker=ticker)

    opt = opt.merge(stk, on=["date", "ticker"], how="left")
    if opt["spot"].isna().mean() > 0.01:
        raise ValueError(f"{ticker}: more than 1% of option rows have no spot price")

    opt["bid"] = pd.to_numeric(opt["best_bid"], errors="coerce")
    opt["ask"] = pd.to_numeric(opt["best_offer"], errors="coerce")
    opt = opt[
        opt["cp_flag"].isin(["C", "P"])
        & np.isfinite(opt["spot"]) & (opt["spot"] > 0)
        & np.isfinite(opt["bid"]) & np.isfinite(opt["ask"])
        & (opt["bid"] > 0) & (opt["ask"] >= opt["bid"])
    ].copy()

    opt["K"] = pd.to_numeric(opt["strike_price"], errors="coerce") / STRIKE_DIVISOR
    opt["mid_american"] = 0.5 * (opt["bid"] + opt["ask"])
    opt["half_spread"] = 0.5 * (opt["ask"] - opt["bid"])
    opt["iv_vendor"] = pd.to_numeric(opt["impl_volatility"], errors="coerce")
    opt.loc[~(np.isfinite(opt["iv_vendor"]) & (opt["iv_vendor"] > 0)), "iv_vendor"] = np.nan
    opt = opt[np.isfinite(opt["K"]) & (opt["K"] > 0) & (opt["mid_american"] > 0)].copy()

    # dividends, rates, forward
    opt = opt.dropna(subset=["date", "exdate", "ticker", "spot"])
    opt = trailing_dividend_yield(opt, div_csv)
    opt = add_maturities(opt)
    opt = add_treasury_rate(opt, curve)
    opt["DF"] = np.exp(-opt["r"] * opt["tau_cal_365"])
    opt["F"] = opt["spot"] * np.exp((opt["r"] - opt["q"]) * opt["tau_cal_365"])

    opt = opt[(opt["dte_bus"] >= MIN_DTE_BUS) & (opt["dte_bus"] <= MAX_DTE_BUS)]
    opt = opt[(opt["tau_cal_365"] > 0) & np.isfinite(opt["F"]) & (opt["F"] > 0)].copy()

    # European-equivalent prices
    opt["mid_european"] = black_price(
        opt["F"], opt["K"], opt["tau_cal_365"], opt["DF"], opt["iv_vendor"], opt["cp_flag"]
    )
    opt["bid_european"] = np.maximum(opt["mid_european"] - opt["half_spread"], np.finfo(float).eps)
    opt["ask_european"] = np.maximum(opt["mid_european"] + opt["half_spread"], opt["bid_european"])

    # share of repricing differences that fall inside the observed half-spread
    gap = opt["mid_american"] - opt["mid_european"]
    ok = np.isfinite(gap)
    conversion = (
        pd.DataFrame({"cp_flag": opt.loc[ok, "cp_flag"], "within": gap[ok].abs() <= opt.loc[ok, "half_spread"]})
        .groupby("cp_flag")["within"].agg(n="size", share_within_half_spread="mean")
        .reset_index()
        .assign(ticker=ticker)
    )
    conversion.to_csv(OUT_DIR / f"{ticker}_american_european_conversion.csv", index=False)

    opt = opt[
        np.isfinite(opt["mid_european"]) & (opt["mid_european"] > 0)
        & np.isfinite(opt["bid_european"]) & np.isfinite(opt["ask_european"])
        & (opt["ask_european"] >= opt["bid_european"])
    ].copy()
    opt["bid"], opt["ask"] = opt["bid_european"], opt["ask_european"]
    opt["price_basis"] = "european_equivalent_vendor_iv"
    opt.to_parquet(OUT_DIR / f"{ticker}_option_base.parquet", index=False)
    print(f"{ticker}: {len(opt):,} quotes")
    return conversion

# %%
download_stock_prices(
    TICKERS,
    start=(option_min_date - pd.Timedelta(days=10)).strftime("%Y-%m-%d"),
    end=(option_max_date + pd.Timedelta(days=10)).strftime("%Y-%m-%d"),
)

# 420 extra days so that the first trailing 12-month window is complete
download_dividends(
    TICKERS,
    start=(option_min_date - pd.Timedelta(days=420)).strftime("%Y-%m-%d"),
    end=(option_max_date + pd.Timedelta(days=5)).strftime("%Y-%m-%d"),
)

curve = load_treasury_curve(
    start=(option_min_date - pd.Timedelta(days=10)).strftime("%Y-%m-%d"),
    end=(option_max_date + pd.Timedelta(days=5)).strftime("%Y-%m-%d"),
)

# %%
conversion = pd.concat([build_option_base(t, curve) for t in TICKERS], ignore_index=True)

# pooled over stocks
pooled = conversion.assign(k=conversion["n"] * conversion["share_within_half_spread"]).groupby("cp_flag")[["k", "n"]].sum()
print((pooled["k"] / pooled["n"]).rename("share within half-spread"))
