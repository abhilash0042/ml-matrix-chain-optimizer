import torch
import numpy as np
from torch.utils.data import DataLoader
from src.models.pointer_mcm import PointerMCMNet, compute_cost_from_splits
from src.data.pointer_loader import PointerMCMDataset
import joblib
import os

def evaluate_model(model_path, dataset_path, n_limit=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n===========================================================")
    print(f"  POINTER NETWORK FINAL EVALUATION")
    print(f"  Model: {model_path}")
    print(f"  Device: {device}")
    print(f"===========================================================\n")
    
    # 1. Load Data
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset {dataset_path} not found.")
        return

    print("Loading test dataset...")
    data = joblib.load(dataset_path)
    samples = data['samples']
    features = data['features']
    splits = data['splits']
    
    # Use the same split logic as training
    indices = [i for i, s in enumerate(samples) if len(s['dims'])-1 <= n_limit]
    # The last 15% was the test set
    test_start = int(0.85 * len(indices))
    test_indices = indices[test_start:]
    
    test_samples = [samples[i] for i in test_indices]
    test_features = features[test_indices]
    test_splits = splits[test_indices]
    
    dataset = PointerMCMDataset(test_samples, test_features, test_splits)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # 2. Load Model
    model = PointerMCMNet(input_dim=8, hidden_dim=256).to(device)
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found.")
        return
        
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    
    # 3. Evaluation Loop
    total_mape = 0
    count = 0
    exact_matches = 0
    
    print(f"Running inference on {len(test_samples)} samples...")
    with torch.no_grad():
        for i, (feat_batch, target_splits_batch, target_cost_batch) in enumerate(loader):
            feat_batch = feat_batch.to(device)
            
            # Predict splits for the whole batch
            for b in range(feat_batch.shape[0]):
                sample_idx = count
                n = test_samples[sample_idx]['dims_count'] - 1
                dims = test_samples[sample_idx]['dims']
                
                # Predict splits
                pred_splits = model.predict(feat_batch[b:b+1], n)
                
                # Compute cost from these splits
                pred_cost = compute_cost_from_splits(0, n-1, dims, pred_splits)
                true_cost = target_cost_batch[b].item()
                
                mape = abs(pred_cost - true_cost) / (true_cost + 1e-9)
                total_mape += mape
                count += 1
                
                if mape < 0.001: # < 0.1% error
                    exact_matches += 1
                
                if count % 500 == 0:
                    print(f"  Processed {count}/{len(test_indices)} | Current Avg MAPE: {total_mape/count*100:.4f}%")

    final_mape = (total_mape / count) * 100
    exact_rate = (exact_matches / count) * 100
    
    print("\n" + "="*40)
    print(f"  FINAL TEST RESULTS")
    print(f"  Average MAPE:    {final_mape:.4f}%")
    print(f"  Exact Match:     {exact_rate:.2f}%")
    print("="*40)
    
    if final_mape < 1.0:
        print("\nSUCCESS: Model is extremely accurate!")
    else:
        print("\nModel needs more training.")

def test_real_world(model_path, custom_dims):
    """Test the model on a specific set of dimensions."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PointerMCMNet(input_dim=8, hidden_dim=256).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    n = len(custom_dims) - 1
    
    # Feature engineering for these dims
    from src.pointer_features import get_matrix_features
    feat = get_matrix_features(custom_dims)
    feat_tensor = torch.FloatTensor(feat).unsqueeze(0).to(device)
    
    with torch.no_grad():
        pred_splits = model.predict(feat_tensor, n)
        pred_cost = compute_cost_from_splits(0, n-1, custom_dims, pred_splits)
        
    print(f"\nReal World Test for dims: {custom_dims}")
    print(f"Predicted Cost: {pred_cost:,}")
    return pred_cost

if __name__ == "__main__":
    MODEL_FILE = "models/pointer_stage4.pth"
    # The evaluation script needs the JSON to get sample details
    JSON_FILE = "data/mcm_120000.json"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n===========================================================")
    print(f"  POINTER NETWORK FINAL EVALUATION")
    print(f"  Model: {MODEL_FILE}")
    print(f"  Device: {device}")
    print(f"===========================================================\n")

    # Load the same way as training
    from src.data.pointer_loader import create_pointer_dataloaders
    
    print("Loading test dataset (this may take a minute)...")
    _, _, test_loader, test_idx = create_pointer_dataloaders(JSON_FILE, batch_size=32, max_chain_len=50)
    
    # We need the actual data to get dims for cost calculation
    import json
    with open(JSON_FILE, 'r') as f:
        all_samples = json.load(f)
    test_samples = [all_samples[i] for i in test_idx]
    
    # Load Model
    model = PointerMCMNet(input_dim=8, d_model=128).to(device)
    model.load_state_dict(torch.load(MODEL_FILE, map_location=device, weights_only=True))
    model.eval()
    
    # Evaluation Loop
    total_mape = 0
    count = 0
    exact_matches = 0
    
    print(f"Running inference on {len(test_idx)} samples...")
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            # batch: seq_features, padding_mask, split_targets, split_mask, cost_target, actual_n, raw_dims
            feat_batch = batch[0].to(device)
            mask_batch = batch[1].to(device)
            actual_n_batch = batch[5].to(device)
            raw_dims_batch = batch[6]
            
            # Predict splits for the whole batch at once (much faster!)
            batch_pred_splits, _ = model.predict(feat_batch, mask_batch, actual_n_batch)
            
            for b in range(feat_batch.shape[0]):
                if count >= len(test_idx): break
                
                dims = raw_dims_batch[b]
                pred_splits = batch_pred_splits[b]
                
                # compute_cost_from_splits in pointer_mcm.py uses (dims, splits)
                from src.models.pointer_mcm import compute_cost_from_splits
                pred_cost = compute_cost_from_splits(dims, pred_splits)
                
                true_cost = test_samples[count]['output']
                
                mape = abs(pred_cost - true_cost) / (true_cost + 1e-9)
                total_mape += mape
                count += 1
                
                if mape < 0.001: exact_matches += 1
                
                if count % 1000 == 0:
                    print(f"  Processed {count}/{len(test_idx)} | Current Avg MAPE: {total_mape/count*100:.4f}%")

    final_mape = (total_mape / count) * 100
    exact_rate = (exact_matches / count) * 100
    
    print("\n" + "="*40)
    print(f"  FINAL TEST RESULTS")
    print(f"  Average MAPE:    {final_mape:.4f}%")
    print(f"  Exact Match:     {exact_rate:.2f}%")
    print("="*40)
