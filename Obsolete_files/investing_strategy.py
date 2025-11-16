import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from datetime import datetime

# Suppress warnings, especially from yfinance
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

# --- 1. Define Universe & Backtest Parameters ---

# Using a smaller, diverse list of S&P 500 stocks for this basic example.
# In a full version, you'd dynamically pull the top 300.

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

"""
TICKER_UNIVERSE = [
    'MSFT', 'AAPL', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'BRK-B', 'JPM',
    'JNJ', 'XOM', 'WMT', 'PG', 'COST', 'CVX', 'LLY', 'HD', 'MRK', 'PEP',
    'AVGO', 'KO', 'ADBE', 'CSCO', 'CRM', 'MCD', 'PFE', 'NFLX', 'TMO',
    'AMD', 'ABNB', 'DIS', 'CAT', 'UPS', 'UNP', 'LOW'
]
"""
# Benchmark for comparison
BENCHMARK_TICKER = '^GSPC' # S&P 500

# Backtest period
START_DATE = '2015-01-01'
END_DATE = '2025-01-01'

# Portfolio parameters
N_STOCKS_TO_LONG = 10 # Simplified from 15-20
REBALANCE_FREQ = 'Y' # Yearly rebalancing

# --- 2. Data Fetching Functions ---

def get_price_data(tickers, start, end):
    """Downloads daily adjusted close prices for all tickers."""
    try:
        data = yf.download(tickers, start=start, end=end, progress=False)['Close']
        if data.empty:
            print(f"Error: No price data downloaded for tickers. Check list and dates.")
            return None
        # Forward-fill missing values, then back-fill
        data = data.ffill().bfill()
        return data
    except Exception as e:
        print(f"Error downloading price data: {e}")
        return None

def get_fundamental_factors(tickers):
    """
    Fetches key fundamental factors for a list of tickers.
    Uses yfinance .info, which can be slow and unreliable (some data may be missing).
    """
    factors = {}
    print(f"Fetching fundamental data for {len(tickers)} tickers...")
    for ticker in tickers:
        try:
            stock_info = yf.Ticker(ticker).info
            
            # --- Strategy Factors ---
            # Value Factors (Lower is better)
            pe = stock_info.get('trailingPE')
            pb = stock_info.get('priceToBook')
            ev_ebitda = stock_info.get('enterpriseToEbitda')
            
            # Quality Factors (ROE: Higher, D/E: Lower)
            roe = stock_info.get('returnOnEquity')
            de = stock_info.get('debtToEquity')
            
            # Ensure we have all factors, use None if missing
            if all([pe, pb, ev_ebitda, roe, de]):
                factors[ticker] = {
                    'P/E': pe,
                    'P/B': pb,
                    'EV/EBITDA': ev_ebitda,
                    'ROE': roe,
                    'Debt/Equity': de
                }
            else:
                print(f"  Skipping {ticker}: Missing one or more fundamental factors.")

        except Exception as e:
            print(f"  Could not fetch .info for {ticker}: {e}")
            
    print("Fundamental data fetch complete.")
    return pd.DataFrame.from_dict(factors, orient='index')

def get_momentum_factors(price_data, rebalance_date):
    """Calculates 3-month and 12-month momentum up to the rebalance date."""
    
    # Ensure rebalance_date is a Timestamp
    rebalance_date = pd.to_datetime(rebalance_date)
    
    # Get data up to the rebalance date
    prices_to_date = price_data.loc[:rebalance_date]
    if prices_to_date.empty:
        return pd.DataFrame(index=price_data.columns, columns=['Mom_12M', 'Mom_3M'])
        
    # Get prices from ~12 months ago and ~3 months ago
    # Use iloc[-1] for latest, -252 for ~1 year, -63 for ~3 months
    latest_prices = prices_to_date.iloc[-1]
    
    # 12-Month Momentum
    if len(prices_to_date) >= 252:
        prices_12m_ago = prices_to_date.iloc[-252]
        mom_12m = (latest_prices / prices_12m_ago) - 1
    else:
        # Not enough data, set momentum to 0
        mom_12m = pd.Series(0, index=latest_prices.index)

    # 3-Month Momentum
    if len(prices_to_date) >= 63:
        prices_3m_ago = prices_to_date.iloc[-63]
        mom_3m = (latest_prices / prices_3m_ago) - 1
    else:
        # Not enough data, set momentum to 0
        mom_3m = pd.Series(0, index=latest_prices.index)
        
    mom_df = pd.DataFrame({'Mom_12M': mom_12m, 'Mom_3M': mom_3m})
    return mom_df

# --- 3. Strategy & Backtesting Logic ---

def rank_stocks(fundamental_data, momentum_data):
    """
    Ranks stocks based on combined factors.
    1. Filter by Momentum
    2. Rank by Value and Quality
    3. Combine ranks
    """
    
    # Combine factor dataframes
    all_factors = pd.concat([fundamental_data, momentum_data], axis=1)
    all_factors = all_factors.dropna()
    
    if all_factors.empty:
        print("No stocks with complete factor data.")
        return pd.Series(dtype=float) # Return empty series

    # --- Momentum Filter ---
    # Only select stocks with positive 3-12 month momentum
    # Simplified: require *both* 3M and 12M to be positive
    strategy_stocks = all_factors[
        (all_factors['Mom_12M'] > 0) & (all_factors['Mom_3M'] > 0)
    ].copy()
    
    if strategy_stocks.empty:
        print("No stocks passed momentum filter.")
        return pd.Series(dtype=float) # Return empty series

    # --- Factor Ranking ---
    # Rank factors. `ascending=True` means lower is better (e.g., P/E)
    strategy_stocks['Rank_PE'] = strategy_stocks['P/E'].rank(ascending=True)
    strategy_stocks['Rank_PB'] = strategy_stocks['P/B'].rank(ascending=True)
    strategy_stocks['Rank_EV_EBITDA'] = strategy_stocks['EV/EBITDA'].rank(ascending=True)
    
    # `ascending=False` means higher is better (e.g., ROE)
    strategy_stocks['Rank_ROE'] = strategy_stocks['ROE'].rank(ascending=False)
    # `ascending=True` means lower is better (e.g., D/E)
    strategy_stocks['Rank_DE'] = strategy_stocks['Debt/Equity'].rank(ascending=True)
    
    # --- Composite Score ---
    # Simple average of all ranks
    rank_columns = ['Rank_PE', 'Rank_PB', 'Rank_EV_EBITDA', 'Rank_ROE', 'Rank_DE']
    strategy_stocks['Composite_Rank'] = strategy_stocks[rank_columns].mean(axis=1)
    
    # Sort by the final composite rank (lower is better)
    strategy_stocks = strategy_stocks.sort_values('Composite_Rank', ascending=True)
    
    return strategy_stocks.index

def run_backtest(tickers, start_date, end_date):
    """
    Runs the quarterly rebalancing backtest.
    """
    print("Starting backtest...")
    
    # Get all price data for the period (plus 1 year prior for momentum calcs)
    hist_start_date = (pd.to_datetime(start_date) - pd.DateOffset(years=1)).strftime('%Y-%m-%d')
    all_price_data = get_price_data(tickers, hist_start_date, end_date)
    
    if all_price_data is None:
        print("Backtest failed: Could not get price data.")
        return None

    # Get rebalancing dates
    rebalance_dates = pd.date_range(start=start_date, end=end_date, freq=REBALANCE_FREQ)
    
    portfolio_returns = []
    current_holdings = []

    for i, rebal_date in enumerate(rebalance_dates):
        print(f"\n--- Rebalancing on {rebal_date.date()} ---")
        
        # 1. Get Factors
        # In a real system, you'd query data *as of* rebal_date.
        # Here, we simplify by fetching latest .info (a limitation of yfinance)
        fundamental_data = get_fundamental_factors(tickers)
        momentum_data = get_momentum_factors(all_price_data, rebal_date)
        
        # 2. Rank Stocks
        ranked_tickers = rank_stocks(fundamental_data, momentum_data)
        
        # 3. Select Portfolio
        current_holdings = ranked_tickers[:N_STOCKS_TO_LONG].tolist()
        
        if not current_holdings:
            print("No stocks selected. Holding cash (0% return) for this period.")
            # Set holdings to empty to trigger cash return logic
            
        print(f"Selected Stocks: {current_holdings}")

        # 4. Calculate Returns for the holding period (until next rebalance)
        period_start = rebal_date
        period_end = rebalance_dates[i+1] if i+1 < len(rebalance_dates) else pd.to_datetime(end_date)
        
        # Get price data for just this holding period
        period_price_data = all_price_data.loc[period_start:period_end]
        
        if not current_holdings:
            # Hold cash
            daily_returns = pd.Series(0.0, index=period_price_data.index[1:])
        else:
            # Calculate returns for selected stocks
            holding_returns = period_price_data[current_holdings].pct_change().dropna()
            
            # Assume equal weight portfolio
            daily_returns = holding_returns.mean(axis=1)
            
        portfolio_returns.append(daily_returns)

    print("\nBacktest complete.")
    # Combine all period returns into one series
    if not portfolio_returns:
        print("No returns were generated.")
        return pd.Series(dtype=float)
        
    strategy_returns = pd.concat(portfolio_returns)
    return strategy_returns

# --- 4. Performance Analysis & Visualization ---

def calculate_metrics(returns):
    """Calculates key performance metrics."""
    if returns.empty:
        return 0, 0, 0
        
    trading_days = 252
    
    # Calculate total return
    total_return = (1 + returns).prod() - 1
    
    # Annualized Return
    # Need to calculate time in years
    years = (returns.index[-1] - returns.index[0]).days / 365.25
    if years == 0:
        annualized_return = 0
    else:
        annualized_return = (1 + total_return) ** (1/years) - 1
    
    # Annualized Volatility
    annualized_volatility = returns.std() * np.sqrt(trading_days)
    
    # Sharpe Ratio (assuming 0 risk-free rate)
    if annualized_volatility == 0:
        sharpe_ratio = 0
    else:
        sharpe_ratio = annualized_return / annualized_volatility
        
    return annualized_return, annualized_volatility, sharpe_ratio

def plot_performance(strategy_returns, benchmark_returns):
    """Plots strategy vs. benchmark and prints metrics."""

    # --- Metrics ---
    print("\n--- Performance Metrics ---")
    strat_ann_ret, strat_ann_vol, strat_sharpe = calculate_metrics(strategy_returns)
    bench_ann_ret, bench_ann_vol, bench_sharpe = calculate_metrics(benchmark_returns)
    
    print("\n            | My Strategy | Benchmark")
    print("--------------------------------------")
    print(f"Ann. Return  | {strat_ann_ret:>10.2%} | {bench_ann_ret:>10.2%}")
    print(f"Ann. Vol     | {strat_ann_vol:>10.2%} | {bench_ann_vol:>10.2%}")
    print(f"Sharpe Ratio | {strat_sharpe:>10.2f} | {bench_sharpe:>10.2f}")
    
    # --- Plotting ---
    plt.figure(figsize=(12, 7))

    # Set dark background style
    plt.style.use('dark_background')

    # Strategy Cumulative Returns
    strategy_cum_returns = (1 + strategy_returns).cumprod()
    plt.plot(strategy_cum_returns, label='My Strategy', color='royalblue', linewidth=2)
    
    # Benchmark Cumulative Returns
    benchmark_cum_returns = (1 + benchmark_returns).cumprod()
    plt.plot(benchmark_cum_returns, label=f'Benchmark ({BENCHMARK_TICKER})', color='grey', linestyle='--', linewidth=2)

    plt.title(f'Strategy Backtest: Value + Quality + Momentum ({START_DATE} to {END_DATE})', fontsize=16)
    plt.ylabel('Cumulative Returns', fontsize=12)
    plt.xlabel('Date', fontsize=12)
    plt.legend(loc='upper left', fontsize=11)
    plt.grid(True, axis='y', alpha=0.2)  # Only horizontal lines with reduced opacity
    plt.tight_layout()
    
    # Save the plot
    plot_filename = 'strategy_performance_plot.png'
    plt.savefig(plot_filename)
    print(f"\nPerformance plot saved as '{plot_filename}'")
    plt.show()

    


# --- 5. Main Execution ---

if __name__ == "__main__":
    # 1. Run the strategy backtest
    strategy_returns = run_backtest(TICKER_UNIVERSE, START_DATE, END_DATE)
    
    if strategy_returns is not None and not strategy_returns.empty:
        # 2. Get benchmark data
        benchmark_data = get_price_data([BENCHMARK_TICKER], START_DATE, END_DATE)
        
        if benchmark_data is not None:
            benchmark_returns = benchmark_data.pct_change().dropna().squeeze()
            
            # Align strategy and benchmark returns (in case of missing days)
            combined_data = pd.DataFrame({'Strategy': strategy_returns, 'Benchmark': benchmark_returns}).dropna()
            
            # 3. Plot and print results
            plot_performance(combined_data['Strategy'], combined_data['Benchmark'])
        else:
            print("Could not download benchmark data to compare.")
    else:
        print("Backtest did not produce any returns. Exiting.")