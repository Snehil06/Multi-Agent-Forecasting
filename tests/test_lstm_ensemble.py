import pandas as pd
from pathlib import Path

p = Path('results/backtest_results.csv')
if not p.exists():
    raise SystemExit('backtest_results.csv missing')

df = pd.read_csv(p, index_col=0, parse_dates=True)
required = ['sequence_pred', 'sequence_weight', 'ensemble_pred']
missing = [c for c in required if c not in df.columns]
assert not missing, f'Missing columns: {missing}'
print('Verified columns:', sorted([c for c in df.columns if 'sequence' in c or c == 'ensemble_pred']))
