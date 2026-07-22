# Multi-Agent Financial Forecasting

A game-theoretically aggregated multi-agent system that forecasts daily stock returns and honestly backtests them, built to demonstrate a robust, leak-free data engineering pipeline.

### Architecture
The system generates daily directional signals using four distinct models, which are then dynamically weighted by a meta-model and rigorously evaluated:

`[Trend, Momentum, Volatility, Equal Weight] → Hedge Referee (Ensemble) → Backtest Engine → Streamlit Dashboard`

### Results
*Note: As expected in highly efficient markets, directional accuracy remains close to 50% across all models. The focus of this architecture is on mathematically principled aggregation and strict avoidance of lookahead bias.*

| Label | Sharpe Ratio | Directional Accuracy |
| :--- | :--- | :--- |
| Buy & Hold | 0.786 | 0.5551 |
| TrendAgent | 0.447 | 0.5277 |
| Hedge Ensemble | 0.173 | 0.5147 |
| VolatilityAgent | 0.168 | 0.5174 |
| Equal Weight | -0.003 | 0.5057 |
| MomentumAgent | -0.077 | 0.5020 |

### Setup
To run the interactive dashboard locally, clone this repository and execute the following commands in your terminal:

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

### What I learned
I learned that because stock markets are highly efficient and hard to predict, the real value of this project wasn't about trying to magically beat the market, but rather about building a clean, leak-free data pipeline and designing a system that can dynamically adapt to sudden market shocks.
