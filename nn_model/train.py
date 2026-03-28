"""
NEURAL NETWORK — MCM Cost Predictor (Retrained)
================================================
Uses 131 features, MAPE loss, deeper architecture, OneCycleLR.
"""

import sys, os, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import joblib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.dataloader import load_data_split

# ============================================================
# 1. CUSTOM LOSSES
# ============================================================
class MAPELoss(nn.Module):
    """MAPE in raw cost space (works on log1p-transformed predictions)."""
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, y_pred, y_true):
        true_val = torch.expm1(y_true)
        pred_val = torch.expm1(y_pred)
        return torch.mean(torch.abs((true_val - pred_val) / (true_val + self.eps)))

class LogCoshLoss(nn.Module):
    """Smooth loss, less sensitive to outliers than MSE."""
    def forward(self, y_pred, y_true):
        diff = y_pred - y_true
        return torch.mean(torch.log(torch.cosh(diff + 1e-12)))

# ============================================================
# 2. ARCHITECTURE
# ============================================================
class MCMNeuralNet(nn.Module):
    def __init__(self, input_size, hidden_sizes=[512, 256, 128, 64], dropout=0.15):
        super().__init__()
        layers = []
        prev = input_size
        for h in hidden_sizes:
            layers.extend([
                nn.Linear(prev, h),
                nn.LayerNorm(h),
                nn.SiLU(),
                nn.Dropout(dropout),
            ])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# ============================================================
# 3. TRAINING
# ============================================================
def train_nn():
    t0 = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 60)
    print("  NEURAL NETWORK — RETRAINED (131 features, MAPE loss)")
    print(f"  Device: {device}")
    print("=" * 60)

    # 1. Load data
    print("\n[1/5] Loading data...")
    X_train, X_val, X_test, y_train, y_val, y_test, _, _, _ = load_data_split(version='v2')
    
    y_train_log = np.log1p(y_train).astype(np.float32)
    y_val_log = np.log1p(y_val).astype(np.float32)
    y_test_log = np.log1p(y_test).astype(np.float32)
    
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # 2. Scale features
    print("\n[2/5] Scaling features...")
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train).astype(np.float32)
    X_val_sc = scaler.transform(X_val).astype(np.float32)
    X_test_sc = scaler.transform(X_test).astype(np.float32)
    
    scaler_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'nn_scaler.joblib')
    joblib.dump(scaler, scaler_path)

    # 3. DataLoaders
    batch_size = 256
    train_ds = TensorDataset(torch.from_numpy(X_train_sc), torch.from_numpy(y_train_log).view(-1, 1))
    val_ds = TensorDataset(torch.from_numpy(X_val_sc), torch.from_numpy(y_val_log).view(-1, 1))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=0)

    # 4. Model
    print("\n[3/5] Initializing model...")
    model = MCMNeuralNet(X_train.shape[1], hidden_sizes=[512, 256, 128, 64], dropout=0.15).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Architecture: 131 -> 512 -> 256 -> 128 -> 64 -> 1")
    print(f"  Parameters: {param_count:,}")

    mape_loss = MAPELoss()
    logcosh_loss = LogCoshLoss()
    
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    
    epochs = 500
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=3e-4,
        steps_per_epoch=len(train_loader),
        epochs=epochs,
        pct_start=0.1,
        anneal_strategy='cos',
    )

    # 5. Training Loop
    print(f"\n[4/5] Training for {epochs} epochs...")
    best_val = float('inf')
    patience = 0
    max_patience = 60
    best_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'nn_v2.pth')

    for epoch in range(epochs):
        model.train()
        train_mapes = []
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            l_mape = mape_loss(out, by)
            l_lc = logcosh_loss(out, by)
            loss = 0.6 * l_mape + 0.4 * l_lc
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            train_mapes.append(l_mape.item())

        model.eval()
        val_mapes = []
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                val_mapes.append(mape_loss(model(bx), by).item())

        avg_val = np.mean(val_mapes)
        if (epoch + 1) % 25 == 0:
            print(f"  Epoch {epoch+1:4d}/{epochs} | Train MAPE: {np.mean(train_mapes):.4f} | Val MAPE: {avg_val:.4f}")

        if avg_val < best_val:
            best_val = avg_val
            patience = 0
            torch.save(model.state_dict(), best_path)
        else:
            patience += 1
            if patience >= max_patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    # 6. Evaluate
    print(f"\n[5/5] Evaluating on test set...")
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    model.eval()

    all_preds, all_true = [], []
    with torch.no_grad():
        test_ds = TensorDataset(torch.from_numpy(X_test_sc), torch.from_numpy(y_test_log).view(-1, 1))
        for bx, by in DataLoader(test_ds, batch_size=256):
            bx = bx.to(device)
            all_preds.extend(model(bx).cpu().numpy().flatten())
            all_true.extend(by.numpy().flatten())

    preds_raw = np.expm1(np.array(all_preds))
    true_raw = np.expm1(np.array(all_true))

    r2_log = r2_score(all_true, all_preds)
    errors = np.abs(preds_raw - true_raw) / np.maximum(true_raw, 1)
    mape = np.mean(errors) * 100

    print(f"\n{'=' * 60}")
    print(f"  NEURAL NETWORK RESULTS")
    print(f"{'=' * 60}")
    print(f"  R² (log-space):     {r2_log:.6f}")
    print(f"  MAPE (raw-space):   {mape:.2f}%")
    print(f"  Within  1% of DP:   {(errors < 0.01).mean()*100:.1f}%")
    print(f"  Within  5% of DP:   {(errors < 0.05).mean()*100:.1f}%")
    print(f"  Within 10% of DP:   {(errors < 0.10).mean()*100:.1f}%")
    print(f"  Within 20% of DP:   {(errors < 0.20).mean()*100:.1f}%")
    print(f"  Median Error:       {np.median(errors)*100:.2f}%")
    print(f"  95th Percentile:    {np.percentile(errors, 95)*100:.2f}%")
    print(f"  Max Error:          {np.max(errors)*100:.2f}%")

    results = {
        'model': 'NeuralNetwork',
        'features': int(X_train.shape[1]),
        'params': param_count,
        'r2_log': round(float(r2_log), 6),
        'mape_raw': round(float(mape), 2),
        'within_1pct': round(float((errors < 0.01).mean() * 100), 1),
        'within_5pct': round(float((errors < 0.05).mean() * 100), 1),
        'within_10pct': round(float((errors < 0.10).mean() * 100), 1),
        'within_20pct': round(float((errors < 0.20).mean() * 100), 1),
        'median_error': round(float(np.median(errors) * 100), 2),
        'p95_error': round(float(np.percentile(errors, 95) * 100), 2),
        'max_error': round(float(np.max(errors) * 100), 2),
    }
    results_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'nn_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n  Model saved -> {best_path}")
    print(f"  Results saved -> {results_path}")
    print(f"  Total time: {time.time()-t0:.0f}s")

if __name__ == "__main__":
    train_nn()
