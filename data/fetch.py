import time
import numpy as np
import pandas as pd
import yfinance as yf

def download_ticker_with_retry(ticker, start, end, max_retries=5, delay=2):
    """
    Downloads historical data for a given ticker with automatic retry logic on failures.
    """
    for i in range(max_retries):
        try:
            df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
            if not df.empty:
                return df
        except Exception as e:
            print(f"Error downloading {ticker} (attempt {i+1}): {e}")
        time.sleep(delay)
    raise RuntimeError(f"Failed to download data for {ticker} after {max_retries} attempts.")

def download_market_data(start_date='2010-01-01', end_date='2024-12-31'):
    """
    Downloads SPY and VIX data, calculates daily log returns for SPY, 
    5-day VIX percentage changes, and returns the combined DataFrame.
    """
    print(f"Downloading SPY from {start_date} to {end_date}...")
    spy_df = download_ticker_with_retry('SPY', start_date, end_date)[['Close']]
    spy = spy_df.copy()
    spy.columns = ['Close']
    spy['log_return'] = np.log(spy['Close'] / spy['Close'].shift(1))

    print(f"Downloading ^VIX from {start_date} to {end_date}...")
    vix_df = download_ticker_with_retry('^VIX', start_date, end_date)[['Close']]
    vix = vix_df.copy()
    vix.columns = ['vix_close']
    vix['vix_change_5d'] = vix['vix_close'].pct_change(5)

    # Join the datasets
    merged = spy.join(vix, how='inner').dropna()
    return merged
