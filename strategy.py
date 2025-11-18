import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import json
import os
from datetime import datetime

# --------------------------
# USER OPTION: Toggle local caching
# --------------------------
USE_LOCAL_DATA = True   # <-- Set to False to re-download everything

# --------------------------
# Suppress warnings
# --------------------------
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

# --------------------------
# 1. Define Universe & Parameters
# --------------------------

TICKER_UNIVERSE = [
    # Technology
    'MSFT', 'AAPL', 'GOOGL', 'AMZN', 'NVDA', 'META', 'AVGO', 'ADBE', 'CSCO', 
    'CRM', 'AMD', 'INTC', 'QCOM', 'TXN', 'ORCL', 'IBM', 'NOW', 'PYPL', 'PANW', 
    'SNPS', 'CDNS', 'MU', 'KLAC', 'ADI', 'LRCX', 'AMAT',
    
    # Health Care
    'JNJ', 'LLY', 'UNH', 'MRK', 'PFE', 'TMO', 'ABBV', 'DHR', 'AMGN', 'MDT', 
    'GILD', 'ISRG', 'CVS', 'CI', 'BMY', 'ZTS', 'BSX', 'SYK', 'ABT', 'ELV',
    
    # Financials
    'JPM', 'BRK-B', 'V', 'MA', 'BAC', 'WFC', 'GS', 'MS', 'BLK', 'AXP', 
    'SPGI', 'C', 'CB', 'PNC', 'SCHW', 'AON', 'MMC', 'TROW', 'COF', 'USB',
    
    # Consumer Discretionary
    'TSLA', 'HD', 'COST', 'MCD', 'NFLX', 'DIS', 'LOW', 'SBUX', 'TGT', 
    'BKNG', 'NKE', 'CMG', 'F', 'GM', 'MAR', 'ORLY', 'ROST', 'YUM', 'LULU',
    
    # Consumer Staples
    'PG', 'PEP', 'KO', 'WMT', 'MO', 'PM', 'CL', 'KMB', 'MDLZ', 'GIS', 
    'KHC', 'EL', 'MNST', 'TGT', 'STZ', 'KR', 'ADM', 'CPB',
    
    # Industrials
    'CAT', 'UPS', 'UNP', 'BA', 'HON', 'LMT', 'GE', 'DE', 'RTX', 'WM', 
    'ETN', 'FDX', 'CSX', 'ITW', 'EMR', 'PCAR', 'GD', 'NOC', 'DAL',
    
    # Energy
    'XOM', 'CVX', 'SHEL', 'COP', 'EOG', 'SLB', 'MPC', 'PSX', 'VLO', 'PXD', 
    'KMI', 'WMB', 'OKE', 'HAL',
    
    # Materials
    'LIN', 'SHW', 'APD', 'ECL', 'NUE', 'FCX', 'DD', 'LYB', 'PPG',
    
    # Real Estate
    'AMT', 'PLD', 'CCI', 'EQIX', 'PSA', 'SPG', 'O', 'WY',
    
    # Utilities
    'NEE', 'DUK', 'SO', 'D', 'AEP', 'SRE', 'EXC', 'NGG', 'PEG'
]

BENCHMARK_TICKER = "^GSPC"

START_DATE = "2015-01-01"
END_DATE   = "2025-01-01"

N_STOCKS_TO_LONG = 10
REBALANCE_FREQ = "6M" # Semi-annual rebalancing

# --------------------------
# 2. PRICE DATA CACHING
# --------------------------

def load_or_download_prices(tickers, start, end, filename="price_data.csv"):
    if USE_LOCAL_DATA and os.path.exists(filename):
        print(f"Loading price data from {filename} ...")
        return pd.read_csv(filename, index_col=0, parse_dates=True)

    print("Downloading fresh price data from Yahoo...")
    data = yf.download(tickers, start=start, end=end, progress=False)["Close"]
    data = data.ffill().bfill()

    data.to_csv(filename)
    print(f"Saved price data to {filename}")

    return data

# --------------------------
# 3. FUNDAMENTAL DATA CACHING
# --------------------------

def get_fundamental_factors(tickers):
    factors = {}
    print(f"Fetching fundamental data for {len(tickers)} tickers...")
    
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info

            pe = info.get("trailingPE")
            pb = info.get("priceToBook")
            ev_ebitda = info.get("enterpriseToEbitda")
            roe = info.get("returnOnEquity")
            de = info.get("debtToEquity")

            if all([pe, pb, ev_ebitda, roe, de]):
                factors[ticker] = {
                    "P/E": pe,
                    "P/B": pb,
                    "EV/EBITDA": ev_ebitda,
                    "ROE": roe,
                    "Debt/Equity": de
                }

        except Exception as e:
            print(f"  Error fetching fundamentals for {ticker}: {e}")

    return pd.DataFrame.from_dict(factors, orient="index")


def load_or_download_fundamentals(tickers, filename="fundamentals.json"):
    if USE_LOCAL_DATA and os.path.exists(filename):
        print(f"Loading fundamentals from {filename} ...")
        with open(filename, "r") as f:
            data = json.load(f)
        return pd.DataFrame.from_dict(data, orient="index")

    print("Downloading fresh fundamental data...")
    df = get_fundamental_factors(tickers)

    with open(filename, "w") as f:
        json.dump(df.to_dict(orient="index"), f)

    print(f"Saved fundamentals to {filename}")
    return df

# --------------------------
# 4. Momentum factors
# --------------------------

def get_momentum_factors(price_data, rebalance_date):
    rebalance_date = pd.to_datetime(rebalance_date)
    prices_to_date = price_data.loc[:rebalance_date]

    latest = prices_to_date.iloc[-1]

    if len(prices_to_date) >= 252:
        mom_12m = latest / prices_to_date.iloc[-252] - 1
    else:
        mom_12m = pd.Series(0, index=latest.index)

    if len(prices_to_date) >= 63:
        mom_3m = latest / prices_to_date.iloc[-63] - 1
    else:
        mom_3m = pd.Series(0, index=latest.index)

    return pd.DataFrame({"Mom_12M": mom_12m, "Mom_3M": mom_3m})

# --------------------------
# 5. Ranking logic
# --------------------------

def calculate_volatility(price_data, window=252):
    """
    Robust annualised volatility per ticker.

    - price_data: DataFrame of prices indexed by date, columns = tickers
    - window: lookback window for rolling vol (days)

    Returns a pd.Series indexed by tickers. Any ticker with insufficient
    history or all-NaN prices will get a large vol (999) so it is filtered out.
    """
    # Ensure a DataFrame (handle Series or single-column)
    if isinstance(price_data, pd.Series):
        price_data = price_data.to_frame()

    # Compute returns; allow columns that are entirely NaN to stay as NaN
    returns = price_data.pct_change()

    # If there are no valid returns at all, return high vol for all tickers
    if returns.dropna(how="all").empty:
        return pd.Series(999.0, index=price_data.columns)

    # Prepare output series
    vols = pd.Series(index=price_data.columns, dtype=float)

    # For each ticker compute vol robustly
    for col in price_data.columns:
        col_ret = returns[col].dropna()
        if col_ret.empty:
            vols[col] = 999.0
            continue

        # If we have at least `window` observations use the last `window`
        if len(col_ret) >= window:
            sample = col_ret.iloc[-window:]
            vol = sample.std(ddof=0) * np.sqrt(252)   # population std (ddof=0) or ddof=1 as you prefer
        else:
            # fallback: use sample std of all available history
            vol = col_ret.std(ddof=0) * np.sqrt(252)

        # guard against NaN
        vols[col] = float(vol) if pd.notna(vol) else 999.0

    # Final safety: replace any remaining NaNs with large vol
    vols = vols.fillna(999.0)

    return vols


def rank_stocks(fundamental_data, momentum_data, price_data, rebal_date):
    df = pd.concat([fundamental_data, momentum_data], axis=1).dropna()

    df = df[(df["Mom_12M"] > 0) & (df["Mom_3M"] > 0)]

    # Add volatility filter (lower vol = safer stocks)
    price_to_date = price_data.loc[:rebal_date]
    vol = calculate_volatility(price_to_date)   # you'll pass price_data into rank_stocks

    # Attach volatility
    df["Vol"] = vol.reindex(df.index).fillna(999.0)
    df["RiskAdj_Momentum"] = (df["Mom_12M"] + df["Mom_3M"]) / df["Vol"]


    # Remove extremely volatile stocks

    if df.empty:
        return pd.Series(dtype=float)

    df["Rank_PE"] = df["P/E"].rank()
    df["Rank_PB"] = df["P/B"].rank()
    df["Rank_EV_EBITDA"] = df["EV/EBITDA"].rank()
    df["Rank_ROE"] = df["ROE"].rank(ascending=False)
    df["Rank_DE"] = df["Debt/Equity"].rank()
    df["Rank_RiskAdjMom"] = df["RiskAdj_Momentum"].rank(ascending=False)


    df["Composite_Rank"] = df[
        ["Rank_PE", "Rank_PB", "Rank_EV_EBITDA", "Rank_ROE", "Rank_DE", "Rank_RiskAdjMom"]
    ].mean(axis=1)

    return df.sort_values("Composite_Rank").index

# --------------------------
# 6. Backtest
# --------------------------

def run_backtest(tickers, start_date, end_date):
    print("Starting backtest...")

    hist_start = (pd.to_datetime(start_date) - pd.DateOffset(years=1)).strftime("%Y-%m-%d")

    prices = load_or_download_prices(tickers, hist_start, end_date)
    fundamentals = load_or_download_fundamentals(tickers)

    rebalance_dates = pd.date_range(start=start_date, end=end_date, freq=REBALANCE_FREQ)

    portfolio_returns = []

    for i, rebal_date in enumerate(rebalance_dates):
        print(f"\n--- Rebalancing {rebal_date.date()} ---")

        momentum = get_momentum_factors(prices, rebal_date)
        selected = rank_stocks(fundamentals, momentum, prices, rebal_date)[:N_STOCKS_TO_LONG]

        print("Selected stocks:", list(selected))

        period_start = rebal_date
        period_end = (
            rebalance_dates[i + 1] if i + 1 < len(rebalance_dates) else pd.to_datetime(end_date)
        )
        period_prices = prices.loc[period_start:period_end]

        if len(selected) == 0:
            daily_ret = pd.Series(0, index=period_prices.index[1:])
        else:
            rets = period_prices[selected].pct_change().dropna()
            daily_ret = rets.mean(axis=1)

        portfolio_returns.append(daily_ret)

    return pd.concat(portfolio_returns)

# --------------------------
# 7. Performance metrics
# --------------------------

def calculate_metrics(returns):
    if returns.empty:
        return 0, 0, 0

    days = 252
    total_return = (1 + returns).prod() - 1
    years = (returns.index[-1] - returns.index[0]).days / 365.25

    annualized_return = (1 + total_return) ** (1 / years) - 1
    annualized_vol = returns.std() * np.sqrt(days)
    sharpe = annualized_return / annualized_vol if annualized_vol > 0 else 0

    return annualized_return, annualized_vol, sharpe

def calculate_max_drawdown(returns):
    """
    Returns the maximum drawdown of a return series.
    """
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    drawdown = (cum - peak) / peak
    max_dd = drawdown.min()
    return max_dd

def calculate_var(returns, confidence=0.95):
    """
    Computes historical Value at Risk (VaR) at given confidence level.
    VaR is the loss percentile.
    """
    return returns.quantile(1 - confidence)

def calculate_cvar(returns, confidence=0.95):
    cutoff = returns.quantile(1 - confidence)
    return returns[returns <= cutoff].mean()

# --------------------------
# 8. Plot
# --------------------------

def plot_performance(strategy_returns, benchmark_returns):
    strat_ann, strat_vol, strat_sharpe = calculate_metrics(strategy_returns)
    bench_ann, bench_vol, bench_sharpe = calculate_metrics(benchmark_returns)
    max_dd = calculate_max_drawdown(strategy_returns)
    bench_max_dd = calculate_max_drawdown(benchmark_returns)
    var95 = calculate_var(strategy_returns, 0.95)
    bench_var95 = calculate_var(benchmark_returns, 0.95)
    cvar = calculate_cvar(strategy_returns, 0.95)
    bench_cvar = calculate_cvar(benchmark_returns, 0.95)

    print("\n---------------------------")
    print("       Performance")
    print("---------------------------")
    print(f"Annual Return:   {strat_ann:.2%} | Benchmark: {bench_ann:.2%}")
    print(f"Annual Vol:      {strat_vol:.2%} | Benchmark: {bench_vol:.2%}")
    print(f"Sharpe Ratio:    {strat_sharpe:.2f} | Benchmark: {bench_sharpe:.2f}")
    print(f"Max Drawdown: {max_dd:.2%} | Benchmark: {bench_max_dd:.2%}")
    print(f"95% VaR (daily): {var95:.2%} | Benchmark: {bench_var95:.2%}")
    print(f"95% CVaR (Expected daily Shortfall): {cvar:.2%} | Benchmark: {bench_cvar:.2%}")

    plt.figure(figsize=(12, 7))
    plt.style.use("dark_background")

    plt.plot((1 + strategy_returns).cumprod(), label="Strategy", linewidth=2)
    plt.plot((1 + benchmark_returns).cumprod(), label="S&P 500", linestyle="--", linewidth=2)

    plt.title("Backtest Performance")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig("backtest_performance.png", dpi=300)
    plt.show()

# --------------------------
# 9. Main
# --------------------------

if __name__ == "__main__":
    strategy_returns = run_backtest(TICKER_UNIVERSE, START_DATE, END_DATE)

    benchmark_data = load_or_download_prices(
        [BENCHMARK_TICKER], START_DATE, END_DATE, filename="benchmark_prices.csv"
    )
    benchmark_returns = benchmark_data.pct_change().dropna().squeeze()

    combined = pd.DataFrame({
        "Strategy": strategy_returns,
        "Benchmark": benchmark_returns
    }).dropna()

    plot_performance(combined["Strategy"], combined["Benchmark"])
