import sys
from pathlib import Path
from dateutil.relativedelta import relativedelta
import pandas as pd
import numpy as np

# Ensure project root is in the path for absolute imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from data.fetch import download_market_data
from pipeline.features import build_feature_matrix
from pipeline.agents import TrendAgent, MomentumAgent, VolatilityAgent, SequenceAgent
from pipeline.aggregator import HedgeAggregator
from pipeline.metrics import metrics_table

def walk_forward_backtest(agents, aggregator, feature_df, initial_train_years=3, step_months=3):
    """
    Simulates walk-forward rolling training and backtesting.
    For each step (e.g. 3 months), retrains all agents on historical data 
    and predicts forward returns, adjusting Hedge Aggregator weights step-by-step.
    """
    aggregator.reset()
    names = [a.name for a in agents]
    start = feature_df.index[0]
    train_end = start + relativedelta(years=initial_train_years) - relativedelta(days=1)
    records = []
    fold = 0

    while True:
        test_start = train_end + relativedelta(days=1)
        test_end = min(train_end + relativedelta(months=step_months), feature_df.index[-1])
        if test_start > feature_df.index[-1]:
            break
        train_df = feature_df.loc[:train_end]
        test_df = feature_df.loc[test_start:test_end]
        if len(train_df) < 50 or len(test_df) == 0:
            break

        print(f"--- FOLD {fold}: Training from {train_df.index[0].date()} to {train_df.index[-1].date()}, Testing to {test_end.date()} ---")
        for agent in agents:
            print(f"Fitting {agent.name}...")
            agent.fit(train_df)
            
        print("Generating predictions...")
        preds_all = {name: agent.predict(test_df) for name, agent in zip(names, agents)}

        print("Updating Hedge aggregator weights day-by-day...")
        for i, (date, r) in enumerate(test_df.iterrows()):
            actual = r['log_return']
            step_preds = [float(preds_all[name][i]) for name in names]
            ensemble = aggregator.aggregate(step_preds)
            aggregator.update(step_preds, actual)
            
            rec = {
                'date': date, 
                'actual': actual, 
                'ensemble_pred': ensemble, 
                'fold': fold
            }
            for name, p in zip(names, step_preds):
                rec[f'{name}_pred'] = p
                if name == 'SequenceAgent':
                    rec['sequence_pred'] = p
            for name, w in zip(names, aggregator.weights):
                rec[f'{name}_weight'] = w
                if name == 'SequenceAgent':
                    rec['sequence_weight'] = w
            records.append(rec)

        train_end += relativedelta(months=step_months)
        fold += 1

    return pd.DataFrame(records).set_index('date')


if __name__ == "__main__":
    RESULTS_DIR = PROJECT_ROOT / "results"
    RESULTS_DIR.mkdir(exist_ok=True)
    
    # 1. Fetch raw SPY + VIX data
    raw_df = download_market_data()
    
    # 2. Build feature matrix
    print("Building feature matrix...")
    feature_df = build_feature_matrix(raw_df)
    
    # 3. Initialize agents and aggregator
    print("Initializing agents...")
    agents = [
        TrendAgent(), 
        MomentumAgent(), 
        VolatilityAgent(), 
        SequenceAgent(epochs=5)
    ]
    aggregator = HedgeAggregator(n=len(agents), eta=0.01, alpha=0.0, loss_mode='mse')
    
    # 4. Run walk-forward backtest
    print("Starting walk-forward backtest...")
    results = walk_forward_backtest(agents, aggregator, feature_df)
    results.to_csv(RESULTS_DIR / 'backtest_results.csv')
    print(f"Results saved to {RESULTS_DIR / 'backtest_results.csv'}")

    # 5. Compute metrics report
    print("Evaluating agent and ensemble metrics...")
    bh = results['actual'].values
    metrics = []
    
    # Metrics for individual agents
    for name in [a.name for a in agents]:
        pred_col = f'{name}_pred'
        if name == 'SequenceAgent':
            pred_col = 'sequence_pred'
        if pred_col in results.columns:
            metrics.append(metrics_table(results['actual'].values, results[pred_col].values, benchmark=bh, label=name))

    # Metrics for the Hedge Ensemble
    ensemble_df = metrics_table(results['actual'].values, results['ensemble_pred'].values, benchmark=bh, label='Hedge Ensemble')
    metrics.append(ensemble_df)

    # Metrics for Equal Weight strategy
    pred_cols = [c for c in results.columns if c.endswith('_pred') and c not in {'ensemble_pred', 'sequence_pred'}]
    if 'sequence_pred' in results.columns:
        pred_cols.append('sequence_pred')
    eq = results[pred_cols].mean(axis=1).values
    metrics.append(metrics_table(results['actual'].values, eq, benchmark=bh, label='Equal Weight'))

    # Metrics for Buy & Hold benchmark
    metrics.append(metrics_table(results['actual'].values, np.abs(results['actual'].values), benchmark=bh, label='Buy & Hold'))

    # Combine and save report
    metrics_df = pd.concat(metrics, ignore_index=True)
    metrics_df = metrics_df.sort_values('Sharpe Ratio', ascending=False).reset_index(drop=True)
    metrics_df.to_csv(RESULTS_DIR / 'metrics_report.csv', index=False)
    print(f"Metrics report saved to {RESULTS_DIR / 'metrics_report.csv'}")
    
    print("\nGenerated backtest results preview:")
    print(results.head())
    print("\nGenerated metrics summary:")
    print(metrics_df)
