"""
Data Utilities for Financial Data Processing
===========================================

This module provides functions for downloading and processing financial data
from various sources including Yahoo Finance and yahooquery.
"""

import numpy as np
import pandas as pd
import datetime as dt
import yfinance as yf
from yahooquery import Ticker
import warnings

warnings.filterwarnings('ignore')


def download_real_data(tickers, start_date, end_date, progress=False):
    """
    Download real financial data using yahooquery with yfinance fallback
    
    Parameters:
    -----------
    tickers : list or str
        Stock ticker symbols to download
    start_date : datetime
        Start date for data download
    end_date : datetime 
        End date for data download
    progress : bool
        Show download progress (default False)
        
    Returns:
    --------
    pandas.DataFrame
        Historical price data with columns for OHLCV
    """
    print(f"Downloading data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")
    
    try:
        # Try yahooquery first (more reliable)
        print("Using yahooquery for reliable data access...")
        ticker_obj = Ticker(tickers)
        
        hist_data = ticker_obj.history(start=start_date, end=end_date, interval='1d')
        
        if not hist_data.empty:
            # Reset index to get symbol as column
            hist_data = hist_data.reset_index()
            
            # Pivot to get multi-level columns like yfinance format
            pivot_data = hist_data.pivot(index='date', columns='symbol')
            
            print(f"Successfully downloaded {len(pivot_data)} trading days")
            print(f"Symbols: {list(pivot_data.columns.get_level_values(1).unique())}")
            
            return pivot_data
        else:
            raise ValueError("No data returned from yahooquery")
            
    except Exception as e:
        print(f"⚠ Error with yahooquery: {e}")
        print("Falling back to yfinance...")
        
        try:
            # Fallback to yfinance
            data = yf.download(tickers, start=start_date, end=end_date, progress=progress)
            if not data.empty:
                print(f"Fallback successful: {len(data)} trading days")
                return data
            else:
                raise ValueError("No data returned from yfinance")
                
        except Exception as e2:
            print(f"Both data sources failed: {e2}")
            return pd.DataFrame()


def process_returns_data(price_data, return_type='log'):
    """
    Process price data to calculate returns
    
    Parameters:
    -----------
    price_data : pandas.DataFrame
        Price data with multi-level columns or simple columns
    return_type : str
        Type of returns to calculate ('log' or 'simple')
        
    Returns:
    --------
    pandas.DataFrame
        Returns data
    """
    # Handle multi-level columns from yahooquery
    if price_data.columns.nlevels > 1:
        # Extract close prices from multi-level columns
        if 'close' in price_data.columns.get_level_values(0):
            close_prices = price_data['close']  # yahooquery uses lowercase
        elif 'Close' in price_data.columns.get_level_values(0):
            close_prices = price_data['Close']  # yfinance format
        else:
            # Try to find any price column
            price_cols = [col for col in price_data.columns.get_level_values(0) 
                         if col.lower() in ['close', 'adj close', 'adjclose']]
            if price_cols:
                close_prices = price_data[price_cols[0]]
            else:
                raise ValueError("No price columns found in data")
    else:
        # Simple column structure
        if 'Close' in price_data.columns:
            close_prices = price_data['Close']
        elif 'Adj Close' in price_data.columns:
            close_prices = price_data['Adj Close']
        else:
            # Assume all columns are price data
            close_prices = price_data
    
    # Calculate returns
    if return_type == 'log':
        returns = np.log(close_prices / close_prices.shift(1)).dropna()
    else:  # simple returns
        returns = (close_prices / close_prices.shift(1) - 1).dropna()
    
    print(f"✓ Calculated {return_type} returns for {len(returns.columns)} assets")
    print(f"✓ Returns data shape: {returns.shape}")
    
    return returns


def save_portfolio_data(returns_data, prices_data, portfolio_name, data_dir="processed_data"):
    """
    Save portfolio data in multiple formats for reuse
    
    Parameters:
    -----------
    returns_data : pandas.DataFrame
        Returns data
    prices_data : pandas.DataFrame  
        Price data
    portfolio_name : str
        Name of portfolio (e.g., 'high_risk', 'low_risk')
    data_dir : str
        Directory to save data
    """
    import os
    
    # Create directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Save returns data
    returns_csv_path = f"{data_dir}/{portfolio_name}_returns.csv"
    returns_pkl_path = f"{data_dir}/{portfolio_name}_returns.pkl"
    
    returns_data.to_csv(returns_csv_path)
    returns_data.to_pickle(returns_pkl_path)
    
    # Save prices data  
    prices_csv_path = f"{data_dir}/{portfolio_name}_prices.csv"
    prices_pkl_path = f"{data_dir}/{portfolio_name}_prices.pkl"
    
    if prices_data.columns.nlevels > 1:
        # For multi-level columns, save the close prices
        close_prices = prices_data['close'] if 'close' in prices_data.columns.get_level_values(0) else prices_data['Close']
        close_prices.to_csv(prices_csv_path)
        close_prices.to_pickle(prices_pkl_path)
    else:
        prices_data.to_csv(prices_csv_path)
        prices_data.to_pickle(prices_pkl_path)
    
    print(f"✓ Saved {portfolio_name} portfolio data:")
    print(f"  - {returns_csv_path}")
    print(f"  - {returns_pkl_path}")
    print(f"  - {prices_csv_path}")
    print(f"  - {prices_pkl_path}")


def load_portfolio_data(portfolio_name, data_dir="processed_data", data_type="returns"):
    """
    Load saved portfolio data
    
    Parameters:
    -----------
    portfolio_name : str
        Name of portfolio (e.g., 'high_risk', 'low_risk')
    data_dir : str
        Directory containing saved data
    data_type : str
        Type of data to load ('returns' or 'prices')
        
    Returns:
    --------
    pandas.DataFrame
        Loaded data
    """
    import os
    
    pkl_path = f"{data_dir}/{portfolio_name}_{data_type}.pkl"
    csv_path = f"{data_dir}/{portfolio_name}_{data_type}.csv"
    
    try:
        # Try pickle first (preserves data types and index)
        if os.path.exists(pkl_path):
            data = pd.read_pickle(pkl_path)
            print(f"✓ Loaded {portfolio_name} {data_type} from pickle")
            return data
        elif os.path.exists(csv_path):
            data = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            print(f"✓ Loaded {portfolio_name} {data_type} from CSV")
            return data
        else:
            raise FileNotFoundError(f"No data found for {portfolio_name} {data_type}")
            
    except Exception as e:
        print(f"✗ Error loading {portfolio_name} {data_type}: {e}")
        return pd.DataFrame()


def get_date_ranges(days_back=365*2):
    """
    Get start and end dates for data download
    
    Parameters:
    -----------
    days_back : int
        Number of days back from today (default 2 years)
        
    Returns:
    --------
    tuple
        (start_date, end_date) as datetime objects
    """
    end_date = dt.datetime.today()
    start_date = end_date - dt.timedelta(days=days_back)
    
    return start_date, end_date


# Default stock lists for different risk profiles
DEFAULT_HIGH_RISK_TICKERS = ['TSLA', 'NVDA', 'AMZN', 'COIN', 'ARKK', 'PLTR', 'MSTR', 'AMD']
DEFAULT_LOW_RISK_TICKERS = ['JNJ', 'PG', 'KO', 'VZ', 'XOM', 'TLT', 'VTI', 'BRK-B']

# Large diversified stock list for analysis
LARGE_STOCK_LIST = [
    # Technology
    'AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA', 'META', 'NFLX', 'AMD', 'CRM', 'ADBE',
    # Finance  
    'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP', 'BLK', 'SCHW', 'USB',
    # Healthcare
    'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'TMO', 'ABT', 'LLY', 'MDT', 'BMY',
    # Consumer
    'AMZN', 'WMT', 'PG', 'KO', 'PEP', 'MCD', 'COST', 'NKE', 'SBUX', 'TGT',
    # Industrial
    'BA', 'CAT', 'GE', 'MMM', 'HON', 'LMT', 'UPS', 'RTX', 'DE', 'EMR',
    # Energy
    'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'PSX', 'VLO', 'MPC', 'KMI', 'OXY'
]
