import torch
import torch.optim as optim
import os
import time
import json
import joblib
import numpy as np
from sklearn.metrics import r2_score

from src.models.hybrid_transformer import HybridMCMTransformer
from src.data.loader import create_dataloaders
from src.utils.losses import MasteryLoss
from data.feature_extractor_v3 import extract_features_v3

def train_model():
    # 1. Configuration
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"GPU DETECTED: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        device = torch.device('cpu')
        print("WARNING: CUDA not available! Training will be SLOW on CPU.")
        print("  -> Make sure .venv is activated: .\.venv\Scripts\Activate.ps1")
        print("  -> Then reinstall torch: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        resp = input("Continue on CPU? (y/n): ").strip().lower()
        if resp != 'y':
            return
    
    batch_size = 256
    epochs = 200
    lr = 3e-4
    data_path = 'data/mcm_120000.json'
    
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Please run the data generator first.")
        return

    # 2. Load Data
    print(f"Loading data from {data_path}...")
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    cache_path = 'data/mcm_120k_features_cache.joblib'
    if os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}...")
        X_feats = joblib.load(cache_path)
    else:
        print(f"Extracting features for {len(data):,} samples...")
        # Pre-extract features for speed
        X_feats = []
        for i, item in enumerate(data):
            X_feats.append(extract_features_v3(item['input']))
            if (i+1) % 20000 == 0:
                print(f"  Processed {i+1}/{len(data)}")
        X_feats = np.array(X_feats)
        print(f"Saving features to cache: {cache_path}")
        joblib.dump(X_feats, cache_path)
    
    # 4. Create Datasets
    train_loader, val_loader, test_loader, test_indices = create_dataloaders(data, X_feats, batch_size=batch_size)

    # 5. Model, Loss, Optimizer
    model = HybridMCMTransformer().to(device)
    criterion = MasteryLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # OneCycleLR is great for Transformers
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=lr, 
        steps_per_epoch=len(train_loader), 
        epochs=epochs,
        pct_start=0.15
    )

    # 4. Training Loop
    print(f"\nStarting Hybrid Training on {device}...")
    best_val_loss = float('inf')
    patience = 20
    no_improve = 0
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        t0 = time.time()
        
        for seq, feats, mask, target, greedy_baseline in train_loader:
            seq, feats, mask = seq.to(device), feats.to(device), mask.to(device)
            target, greedy_baseline = target.to(device), greedy_baseline.to(device)
            
            optimizer.zero_grad()
            preds = model(seq, feats, mask, greedy_baseline)
            loss = criterion(preds, target)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for seq, feats, mask, target, greedy_baseline in val_loader:
                seq, feats, mask = seq.to(device), feats.to(device), mask.to(device)
                target, greedy_baseline = target.to(device), greedy_baseline.to(device)
                val_loss += criterion(model(seq, feats, mask, greedy_baseline), target).item()
        
        avg_train = epoch_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | Train Loss: {avg_train:.6f} | Val Loss: {avg_val:.6f} | Time: {time.time()-t0:.1f}s")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            no_improve = 0
            torch.save(model.state_dict(), 'models/hybrid_transformer_best.pth')
            # print(f"  --> Saved Best Model (Val Loss: {best_val_loss:.6f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # 5. Final Evaluation
    print("\n--- Final Evaluation on Test Set ---")
    model.load_state_dict(torch.load('models/hybrid_transformer_best.pth'))
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for seq, feats, mask, target, greedy_baseline in test_loader:
            seq, feats, mask, greedy_baseline = seq.to(device), feats.to(device), mask.to(device), greedy_baseline.to(device)
            p = model(seq, feats, mask, greedy_baseline).cpu().numpy().flatten()
            all_preds.extend(p)
            all_targets.extend(target.numpy().flatten())
            
    preds_raw = np.expm1(np.array(all_preds))
    targets_raw = np.expm1(np.array(all_targets))
    
    mape = np.mean(np.abs(targets_raw - preds_raw) / (targets_raw + 1e-8)) * 100
    r2 = r2_score(all_targets, all_preds)
    
    print(f"Test R2 (Log Space): {r2:.4f}")
    print(f"Test MAPE (Raw Space): {mape:.2f}%")
    
    if mape < 5.0:
        print("SUCCESS: Target MAPE of <5% achieved!")
    else:
        print(f"RESULT: MAPE is {mape:.2f}%. Still room for improvement.")

if __name__ == "__main__":
    train_model()
