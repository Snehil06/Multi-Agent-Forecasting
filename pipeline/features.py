import numpy as np
import pandas as pd

# Feature lists
MOM = [
    'lag_1', 'lag_2', 'lag_3', 'lag_5', 'lag_10', 
    'rolling_mean_5', 'rolling_mean_21', 'rolling_std_5', 
    'rsi_14', 'macd_signal'
]

VOL = [
    'bb_width', 'rolling_std_5', 'rolling_std_21', 
    'vix_level', 'vix_change_5d'
]

def _rsi(s, p=14):
    d = s.diff()
    g = d.clip(lower=0)
    l = -d.clip(upper=0)
    ag = g.ewm(com=p - 1, min_periods=p).mean()
    al = l.ewm(com=p - 1, min_periods=p).mean()
    return 100 - 100 / (1 + ag / al.replace(0, np.nan))

def _macd(s):
    line = s.ewm(span=12, adjust=False).mean() - s.ewm(span=26, adjust=False).mean()
    return line.ewm(span=9, adjust=False).mean()

def _boll(s, w=20):
    m = s.rolling(w).mean()
    sd = s.rolling(w).std()
    return ((m + 2 * sd) - (m - 2 * sd)) / m

def build_feature_matrix(df):
    """
    Constructs the feature matrix from raw SPY + VIX data.
    All features (except target log_return) are shifted by 1 period (lagged)
    to strictly prevent lookahead leakage.
    """
    c = df['Close'].shift(1)
    
    cols = {
        'rsi_14': _rsi(c), 
        'macd_signal': _macd(c), 
        'bb_width': _boll(c)
    }
    
    for lag in (1, 2, 3, 5, 10):
        cols[f'lag_{lag}'] = df['log_return'].shift(lag)
        
    for w in (5, 21):
        cols[f'rolling_mean_{w}'] = df['log_return'].shift(1).rolling(w).mean()
        cols[f'rolling_std_{w}'] = df['log_return'].shift(1).rolling(w).std()
        
    cols['vix_level'] = df['vix_close'].shift(1)
    cols['vix_change_5d'] = df['vix_change_5d'].shift(1)
    
    # Target (unshifted)
    cols['log_return'] = df['log_return']
    
    return pd.DataFrame(cols).dropna()
