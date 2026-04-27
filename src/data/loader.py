import torch
import numpy as np
import json
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Import the v3 feature extractor
from data.feature_extractor_v3 import extract_features_v3

class HybridMCMDataset(Dataset):
    def __init__(self, raw_data, features, greedy_baselines, max_len=51):
        self.raw_data = raw_data
        self.features = torch.FloatTensor(features)
        self.greedy_baselines = torch.FloatTensor(greedy_baselines)
        self.max_len = max_len

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        item = self.raw_data[idx]
        dims = item['input']
        cost = item['output']
        
        # 1. Sequence Data (Log-scaled dimensions — using log1p to match target scale)
        seq = np.log1p(np.array(dims, dtype=float)).reshape(-1, 1)
        padded_seq = np.zeros((self.max_len, 1))
        padded_seq[:len(seq)] = seq
        
        # Padding mask (True for padded positions)
        mask = np.zeros(self.max_len, dtype=bool)
        mask[len(seq):] = True
        
        # 2. Target (Log-scaled cost)
        target = np.log1p(cost)
        
        return (torch.FloatTensor(padded_seq), 
                self.features[idx], 
                torch.BoolTensor(mask), 
                torch.FloatTensor([target]),
                self.greedy_baselines[idx].unsqueeze(0))

def create_dataloaders(data, X_feats, batch_size=256):
    # Split indices
    indices = np.arange(len(data))
    train_idx, test_idx = train_test_split(indices, test_size=0.15, random_state=42)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.15, random_state=42)
    
    # Scale features based on TRAIN set only
    scaler = StandardScaler()
    scaler.fit(X_feats[train_idx])
    X_feats_scaled = scaler.transform(X_feats)
    
    # Save scaler for inference
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/hybrid_scaler.joblib')
    
    # Extract unscaled greedy baseline (feature index 47: log_greedy_min_all)
    greedy_baselines = X_feats[:, 47]
    
    # Create Dataset objects
    train_ds = HybridMCMDataset([data[i] for i in train_idx], X_feats_scaled[train_idx], greedy_baselines[train_idx])
    val_ds = HybridMCMDataset([data[i] for i in val_idx], X_feats_scaled[val_idx], greedy_baselines[val_idx])
    test_ds = HybridMCMDataset([data[i] for i in test_idx], X_feats_scaled[test_idx], greedy_baselines[test_idx])
    
    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=0)
    
    # Return loaders and test indices for evaluation
    return train_loader, val_loader, test_loader, test_idx

def get_dataloaders(data_path, batch_size=256, version='v3'):
    print(f"Loading data from {data_path}...")
    with open(data_path) as f:
        data = json.load(f)
    
    print(f"Extracting features for {len(data):,} samples...")
    # Pre-extract features for speed
    X_feats = []
    for i, item in enumerate(data):
        X_feats.append(extract_features_v3(item['input']))
        if (i+1) % 20000 == 0:
            print(f"  Processed {i+1}/{len(data)}")
    
    X_feats = np.array(X_feats)
    return create_dataloaders(data, X_feats, batch_size)
