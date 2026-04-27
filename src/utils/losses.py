import torch
import torch.nn as nn

class MAPELoss(nn.Module):
    """
    Mean Absolute Percentage Error Loss.
    Operates on raw space by converting log predictions back.
    """
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, log_pred, log_true):
        # Convert back to raw space
        pred = torch.expm1(log_pred)
        true = torch.expm1(log_true)
        
        # Compute MAPE: |true - pred| / true
        loss = torch.abs(true - pred) / (true + self.eps)
        return torch.mean(loss)

class LogCoshLoss(nn.Module):
    """
    Log-Cosh Loss: log(cosh(x)).
    Smoother and more robust to outliers than MSE.
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred, true):
        x = pred - true
        return torch.mean(torch.log(torch.cosh(x + 1e-12)))

class HybridMCMLoss(nn.Module):
    """
    Hybrid loss combining MAPE (for percentage accuracy) 
    and LogCosh (for log-space stability).
    """
    def __init__(self, mape_weight=0.7, logcosh_weight=0.3):
        super().__init__()
        self.mape = MAPELoss()
        self.logcosh = LogCoshLoss()
        self.mape_weight = mape_weight
        self.logcosh_weight = logcosh_weight

    def forward(self, pred, true):
        l_mape = self.mape(pred, true)
        l_logcosh = self.logcosh(pred, true)
        return self.mape_weight * l_mape + self.logcosh_weight * l_logcosh

class ScaleAwareMAPELoss(nn.Module):
    """
    MAPE loss that weights small values MORE heavily.
    Small costs get 2-3x more weight than large costs.
    """
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, log_pred, log_true):
        pred = torch.expm1(log_pred)
        true = torch.expm1(log_true)
        
        mape = torch.abs(true - pred) / (true + self.eps)
        
        # Scale-aware weighting: small costs get higher weight
        # log10(100) = 2, log10(1M) = 6 -> weights: 3.0 vs 1.0
        scale = torch.log10(true + 10)  # avoid log(0)
        weights = 1.0 / scale.clamp(min=1.0)
        weights = weights / weights.mean()  # normalize
        
        return (mape * weights).mean()

class MasteryLoss(nn.Module):
    """90% Scale-Aware MAPE + 10% LogCosh for stability."""
    def __init__(self):
        super().__init__()
        self.mape = ScaleAwareMAPELoss()
        self.logcosh = LogCoshLoss()

    def forward(self, pred, true):
        return 0.9 * self.mape(pred, true) + 0.1 * self.logcosh(pred, true)
