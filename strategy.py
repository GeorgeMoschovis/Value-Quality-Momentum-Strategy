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

def rank_stocks(fundamental_data, momentum_data):
    df = pd.concat([fundamental_data, momentum_data], axis=1).dropna()

    df = df[(df["Mom_12M"] > 0) & (df["Mom_3M"] > 0)]

    if df.empty:
        return pd.Series(dtype=float)

    df["Rank_PE"] = df["P/E"].rank()
    df["Rank_PB"] = df["P/B"].rank()
    df["Rank_EV_EBITDA"] = df["EV/EBITDA"].rank()
    df["Rank_ROE"] = df["ROE"].rank(ascending=False)
    df["Rank_DE"] = df["Debt/Equity"].rank()

    df["Composite_Rank"] = df[
        ["Rank_PE", "Rank_PB", "Rank_EV_EBITDA", "Rank_ROE", "Rank_DE"]
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
        selected = rank_stocks(fundamentals, momentum)[:N_STOCKS_TO_LONG]

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

# --------------------------
# 8. Plot
# --------------------------

def plot_performance(strategy_returns, benchmark_returns):
    strat_ann, strat_vol, strat_sharpe = calculate_metrics(strategy_returns)
    bench_ann, bench_vol, bench_sharpe = calculate_metrics(benchmark_returns)

    print("\n---------------------------")
    print("       Performance")
    print("---------------------------")
    print(f"Annual Return:   {strat_ann:.2%} | Benchmark: {bench_ann:.2%}")
    print(f"Annual Vol:      {strat_vol:.2%} | Benchmark: {bench_vol:.2%}")
    print(f"Sharpe Ratio:    {strat_sharpe:.2f} | Benchmark: {bench_sharpe:.2f}")

    plt.figure(figsize=(12, 7))
    plt.style.use("dark_background")

    plt.plot((1 + strategy_returns).cumprod(), label="Strategy", linewidth=2)
    plt.plot((1 + benchmark_returns).cumprod(), label="S&P 500", linestyle="--", linewidth=2)

    plt.title("Backtest Performance")
    plt.legend()
    plt.grid(alpha=0.3)
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
