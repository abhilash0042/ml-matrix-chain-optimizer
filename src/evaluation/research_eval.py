import os
import sys
import json
import time
import torch
import numpy as np
import joblib
import xgboost as xgb
from typing import List, Tuple

# Add project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.models.pointer_mcm import PointerMCMNet, compute_cost_from_splits
from src.training.train_trees import predict_xgb_ensemble, predict_rf_ensemble
from src.data.feature_extractor import extract_features_v4
from src.data.pointer_features import extract_pointer_features, pad_features
from src.data.generator import (
    greedy_cost_left_to_right, 
    greedy_cost_right_to_left, 
    greedy_cost_min_first, 
    greedy_cost_balanced,
    mcm_dp
)

def generate_distribution(dist_type: str, n: int, count: int) -> List[List[int]]:
    chains = []
    for _ in range(count):
        if dist_type == 'uniform':
            dims = np.random.randint(5, 500, size=n+1).tolist()
        elif dist_type == 'spiky':
            dims = [(np.random.randint(5, 50) if i % 2 == 0 else np.random.randint(500, 1000)) for i in range(n + 1)]
        elif dist_type == 'bottleneck':
            dims = np.random.randint(500, 1000, size=n+1).tolist()
            dims[np.random.randint(1, n)] = np.random.randint(1, 5)
        elif dist_type == 'monotone':
            start, step = np.random.randint(5, 100), np.random.randint(10, 50)
            if np.random.random() > 0.5: dims = [start + i*step for i in range(n+1)]
            else: dims = [start + (n-i)*step for i in range(n+1)]
        else: dims = np.random.randint(5, 100, size=n+1).tolist()
        chains.append(dims)
    return chains

def run_comparative_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 Starting Comparative Study: Neural vs. Tree-Based Models")
    
    ptr_model = PointerMCMNet(input_dim=8, d_model=128).to(device)
    if os.path.exists("models/pointer_best.pth"):
        ptr_model.load_state_dict(torch.load("models/pointer_best.pth", map_location=device, weights_only=True))
        ptr_model.eval()
    
    xgb_models = {}
    if os.path.exists("models/xgb_direct_v4.json"):
        xgb_models['direct'] = xgb.Booster()
        xgb_models['ratio'] = xgb.Booster()
        xgb_models['direct'].load_model("models/xgb_direct_v4.json")
        xgb_models['ratio'].load_model("models/xgb_ratio_v4.json")

    rf_models = joblib.load("models/rf_ensemble_v4.joblib") if os.path.exists("models/rf_ensemble_v4.joblib") else None

    distributions = ['uniform', 'spiky', 'bottleneck', 'monotone']
    n_test, chain_len = 500, 50
    dist_results = {}

    for d_type in distributions:
        print(f"\n📊 Testing Distribution: {d_type.upper()}")
        chains = generate_distribution(d_type, chain_len, n_test)
        metrics = {'Pointer': {'errors': [], 'valid': 0}, 'XGBoost': {'errors': [], 'valid': 0}, 'RF': {'errors': [], 'valid': 0}}
        
        for dims in chains:
            true_opt = mcm_dp(dims)
            g_min = min(greedy_cost_left_to_right(dims), greedy_cost_right_to_left(dims), greedy_cost_min_first(dims), greedy_cost_balanced(dims))
            
            if ptr_model:
                p_feats = extract_pointer_features(dims)
                padded, mask = pad_features(p_feats, chain_len + 1)
                with torch.no_grad():
                    pred_splits, _ = ptr_model.predict(torch.FloatTensor(padded).unsqueeze(0).to(device), torch.BoolTensor(mask).unsqueeze(0).to(device), torch.LongTensor([len(dims)-1]).to(device))
                    cost = compute_cost_from_splits(dims, pred_splits[0])
                metrics['Pointer']['errors'].append(abs(cost - true_opt) / (true_opt + 1e-9) * 100)
                metrics['Pointer']['valid'] += 1 if cost >= true_opt - 1 else 0
            
            tree_feat = np.array(extract_features_v4(dims)).reshape(1, -1)
            if xgb_models:
                cost = predict_xgb_ensemble(xgb_models, tree_feat, np.array([g_min]))[0]
                metrics['XGBoost']['errors'].append(abs(cost - true_opt) / (true_opt + 1e-9) * 100)
                metrics['XGBoost']['valid'] += 1 if cost >= true_opt - 1 else 0
            if rf_models:
                cost = predict_rf_ensemble(rf_models, tree_feat, np.array([g_min]))[0]
                metrics['RF']['errors'].append(abs(cost - true_opt) / (true_opt + 1e-9) * 100)
                metrics['RF']['valid'] += 1 if cost >= true_opt - 1 else 0

        summary = {m: {'mape': np.mean(metrics[m]['errors']), 'valid_rate': (metrics[m]['valid']/n_test)*100} for m in metrics if metrics[m]['errors']}
        dist_results[d_type] = summary
        for m, s in summary.items(): print(f"   {m:<10} | MAPE: {s['mape']:>7.4f}% | Validity: {s['valid_rate']:>6.1f}%")

if __name__ == "__main__":
    run_comparative_benchmark()
