"""
REFINED MCM NEURAL NETWORK (ResNet + MAPE Loss)
===============================================
This script implements a deeper Residual Network (ResNet) to better 
capture complex patterns in Matrix Chain Multiplication costs.

Features:
- 6-Layer Residual Architecture (ResNet)
- LayerNorm & Swish Activation (SiLU)
- Custom MAPE Loss (Mean Absolute Percentage Error)
- Cosine Annealing Learning Rate
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import joblib

# Add root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.dataloader import load_data

# ============================================================
# 1. CUSTOM MAPE LOSS
# ============================================================
class MAPELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(MAPELoss, self).__init__()
        self.eps = eps

    def forward(self, y_pred, y_true):
        # We work in raw cost space for MAPE
        # y_true and y_pred here are log-values, so we exponentiate
        true_val = torch.expm1(y_true)
        pred_val = torch.expm1(y_pred)
        
        # MAPE = mean(|true - pred| / true)
        loss = torch.mean(torch.abs((true_val - pred_val) / (true_val + self.eps)))
        return loss

# ============================================================
# 2. RESNET ARCHITECTURE
# ============================================================
class ResBlock(nn.Module):
    def __init__(self, size):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(size, size),
            nn.LayerNorm(size),
            nn.SiLU(), # Swish activation
            nn.Dropout(0.1),
            nn.Linear(size, size),
            nn.LayerNorm(size)
        )
        self.silu = nn.SiLU()

    def forward(self, x):
        residual = x
        out = self.block(x)
        return self.silu(out + residual)

class MCMResNet(nn.Module):
    def __init__(self, input_size, hidden_size=512, num_blocks=6):
        super(MCMResNet, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU()
        )
        
        self.res_blocks = nn.ModuleList([ResBlock(hidden_size) for _ in range(num_blocks)])
        
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.SiLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.input_layer(x)
        for block in self.res_blocks:
            x = block(x)
        return self.output_layer(x)

# ============================================================
# 3. TRAINING LOOP
# ============================================================
def train_refined():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")

    # A. Load & Prepare Data
    print("Loading data...")
    X, y = load_data()
    y_log = np.log1p(y)
    
    X_train, X_temp, y_train, y_temp = train_test_split(X, y_log, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    joblib.dump(scaler, os.path.join(os.path.dirname(__file__), 'scaler_refined.joblib'))

    # B. Dataloaders
    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train).view(-1, 1))
    val_ds = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val).view(-1, 1))
    
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128)

    # C. Model & Optimizer
    model = MCMResNet(X.shape[1]).to(device)
    
    # We use a hybrid loss: MAPE for precision + MSE for overall scale
    mape_criterion = MAPELoss()
    mse_criterion = nn.MSELoss()
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    epochs = 300
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-3, 
        steps_per_epoch=len(train_loader), 
        epochs=epochs
    )

    best_val_mape = float('inf')
    
    print("Starting Refined Training (ResNet + MAPE)...")
    for epoch in range(epochs):
        model.train()
        train_mapes = []
        for b_x, b_y in train_loader:
            b_x, b_y = b_x.to(device), b_y.to(device)
            optimizer.zero_grad()
            
            out = model(b_x)
            
            # Hybrid Loss: 0.8 * MAPE + 0.2 * MSE (on logs)
            loss_mape = mape_criterion(out, b_y)
            loss_mse = mse_criterion(out, b_y)
            loss = 0.8 * loss_mape + 0.2 * loss_mse
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Prevent explosions
            optimizer.step()
            train_mapes.append(loss_mape.item())
        
        scheduler.step()

        # Validation
        model.eval()
        val_mapes = []
        with torch.no_grad():
            for b_x, b_y in val_loader:
                b_x, b_y = b_x.to(device), b_y.to(device)
                v_out = model(b_x)
                v_mape = mape_criterion(v_out, b_y)
                val_mapes.append(v_mape.item())
        
        avg_val_mape = np.mean(val_mapes)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | Val MAPE: {avg_val_mape:.4f}")

        if avg_val_mape < best_val_mape:
            best_val_mape = avg_val_mape
            torch.save(model.state_dict(), os.path.join(os.path.dirname(__file__), 'best_model_refined.pth'))

    print(f"\nRefined Training Complete! Best Val MAPE: {best_val_mape:.4f}")

if __name__ == "__main__":
    train_refined()
