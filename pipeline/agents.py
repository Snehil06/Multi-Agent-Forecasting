import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBRegressor
from pipeline.features import MOM, VOL

warnings.filterwarnings("ignore")

class TrendAgent:
    """
    Fits a linear regression model on calendar Fourier terms and time trend.
    """
    name = 'TrendAgent'

    def __init__(self):
        self._m = LinearRegression()
        self._dp = None

    def fit(self, tr):
        self._dp = DeterministicProcess(
            index=tr.index, 
            constant=True, 
            order=1, 
            additional_terms=[CalendarFourier(freq='YE', order=4)], 
            drop=True
        )
        self._m.fit(self._dp.in_sample(), tr['log_return'].values)

    def predict(self, te):
        return self._m.predict(self._dp.out_of_sample(steps=len(te), forecast_index=te.index))


class _XGBAgent:
    """
    Base XGBoost agent class.
    """
    def __init__(self, cols):
        self.cols = cols
        self._m = XGBRegressor(
            n_estimators=200, 
            learning_rate=0.05, 
            max_depth=4, 
            subsample=0.8, 
            colsample_bytree=0.8, 
            random_state=42, 
            verbosity=0
        )

    def fit(self, tr):
        self._m.fit(tr[self.cols], tr['log_return'].values)

    def predict(self, te):
        return self._m.predict(te[self.cols])


class MomentumAgent(_XGBAgent):
    """
    XGBoost agent trained on momentum, lags, and RSI indicators.
    """
    name = 'MomentumAgent'

    def __init__(self):
        super().__init__(MOM)


class VolatilityAgent(_XGBAgent):
    """
    XGBoost agent trained on volatility metrics like Bollinger Width and VIX.
    """
    name = 'VolatilityAgent'

    def __init__(self):
        super().__init__(VOL)


class LSTMNet(nn.Module):
    """
    LSTM architecture for sequence prediction.
    """
    def __init__(self, input_size=1, hidden=64, layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden, 
            num_layers=layers, 
            batch_first=True, 
            dropout=dropout if layers > 1 else 0.0
        )
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


class SequenceAgent:
    """
    LSTM-based agent that learns representations from sequences of log returns.
    """
    name = 'SequenceAgent'

    def __init__(self, window_size=30, hidden=64, epochs=5, batch_size=64, lr=1e-3, patience=5):
        self.window_size = window_size
        self.hidden = hidden
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience
        self._scaler = StandardScaler()
        self._net = None
        self._train_tail = None
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _make_sequences(self, series):
        X, y = [], []
        for i in range(self.window_size, len(series)):
            X.append(series[i - self.window_size:i])
            y.append(series[i])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    def fit(self, tr):
        torch.manual_seed(0)
        np.random.seed(0)
        train_returns = tr['log_return'].values.reshape(-1, 1)
        scaled_train = self._scaler.fit_transform(train_returns).flatten()
        self._train_tail = tr['log_return'].values[-self.window_size:]
        X_np, y_np = self._make_sequences(scaled_train)
        X_t = torch.tensor(X_np).unsqueeze(-1).to(self._device)
        y_t = torch.tensor(y_np).to(self._device)

        val_n = max(1, int(0.1 * len(X_t)))
        X_tr, X_val = X_t[:-val_n], X_t[-val_n:]
        y_tr, y_val = y_t[:-val_n], y_t[-val_n:]
        loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=self.batch_size, shuffle=False)

        self._net = LSTMNet(hidden=self.hidden).to(self._device)
        optimizer = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        best_val, wait, best_state = float('inf'), 0, None
        for _ in range(self.epochs):
            self._net.train()
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(self._net(xb), yb)
                loss.backward()
                optimizer.step()
            self._net.eval()
            with torch.no_grad():
                val_loss = criterion(self._net(X_val), y_val).item()
            if val_loss < best_val - 1e-6:
                best_val, wait = val_loss, 0
                best_state = {k: v.clone() for k, v in self._net.state_dict().items()}
            else:
                wait += 1
                if wait >= self.patience:
                    break
        if best_state:
            self._net.load_state_dict(best_state)

    def predict(self, te):
        test_returns = te['log_return'].values
        full = np.concatenate([self._train_tail, test_returns])
        scaled_full = self._scaler.transform(full.reshape(-1, 1)).flatten()
        Xte, _ = self._make_sequences(scaled_full)
        Xte_t = torch.tensor(Xte).unsqueeze(-1).to(self._device)
        self._net.eval()
        with torch.no_grad():
            preds_scaled = self._net(Xte_t).cpu().numpy()
        return self._scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
