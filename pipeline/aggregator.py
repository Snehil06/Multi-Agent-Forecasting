import numpy as np

class HedgeAggregator:
    """
    Ensemble aggregator using the Hedge (multiplicative weights) algorithm.
    Dynamically adjusts agent weights based on historical forecast errors.
    """
    def __init__(self, n, eta=0.2, alpha=0.05, loss_mode='mse'):
        self.n = n
        self.eta = eta
        self.alpha = alpha
        self.loss_mode = loss_mode
        self.weights = np.ones(n) / n
        self.weight_history = []

    def aggregate(self, p):
        """
        Calculates the weighted average prediction.
        """
        return float(np.dot(self.weights, p))

    def update(self, p, a):
        """
        Updates weights exponentially based on the relative prediction error.
        """
        p = np.array(p, dtype=float)
        if self.loss_mode == 'directional':
            ls = -(np.sign(p) * a)
            lo, hi = ls.min(), ls.max()
            nm = (ls - lo) / (hi - lo + 1e-12)
        else:
            ls = (p - a) ** 2
            nm = ls / (ls.mean() + 1e-12)
            
        self.weights *= np.exp(-self.eta * nm)
        self.weights /= self.weights.sum()
        
        # Apply mixture constant alpha to prevent weights from decaying to absolute zero
        if self.alpha > 0:
            self.weights = (1 - self.alpha) * self.weights + self.alpha / self.n
            self.weights /= self.weights.sum()
            
        self.weight_history.append(self.weights.copy())

    def reset(self):
        """
        Resets weights to equal distribution.
        """
        self.weights = np.ones(self.n) / self.n
        self.weight_history.clear()
