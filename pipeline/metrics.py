import numpy as np
import pandas as pd

def sharpe_ratio(returns, annualization=252):
    """
    Calculates the annualized Sharpe Ratio of a returns series.
    """
    returns = np.asarray(returns, float)
    if returns.std() == 0: 
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(annualization))

def max_drawdown(equity_curve):
    """
    Calculates the maximum drawdown from an equity curve.
    """
    curve = np.asarray(equity_curve, float)
    peak = np.maximum.accumulate(curve)
    drawdown = (peak - curve) / peak
    return float(drawdown.max())

def directional_accuracy(y_true, y_pred):
    """
    Calculates percentage of daily predictions matching actual sign (+/-).
    """
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)))

def information_ratio(strategy, benchmark, annualization=252):
    """
    Calculates the Information Ratio of strategy returns versus benchmark returns.
    """
    active = np.asarray(strategy) - np.asarray(benchmark)
    if active.std() == 0: 
        return 0.0
    return float(active.mean() / active.std() * np.sqrt(annualization))

def metrics_table(y_true, y_pred, benchmark=None, label='Model'):
    """
    Builds a summary DataFrame for performance metrics.
    """
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    position = np.sign(y_pred)
    strategy_returns = position * y_true
    equity_curve = 10_000 * np.exp(np.cumsum(strategy_returns))
    
    row = {
        'Label': label,
        'MAE': round(float(np.mean(np.abs(y_true - y_pred))), 6),
        'Directional Accuracy': round(directional_accuracy(y_true, y_pred), 4),
        'Sharpe Ratio': round(sharpe_ratio(strategy_returns), 3),
        'Max Drawdown': round(max_drawdown(equity_curve), 4),
    }
    
    if benchmark is not None:
        row['Information Ratio'] = round(information_ratio(strategy_returns, benchmark), 3)
        
    return pd.DataFrame([row])
