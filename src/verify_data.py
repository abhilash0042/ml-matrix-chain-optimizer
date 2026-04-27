import json
import numpy as np
import os
import sys
from tqdm import tqdm

# Add project root to path
sys.path.append(os.getcwd())

try:
    from data.feature_extractor_v3 import extract_features_v3, TOTAL_FEATURES_V3
except ImportError as e:
    print(f"Error: Could not import feature_extractor_v3. {e}")
    print("Ensure you are running from the project root and all dependencies (numpy, scipy) are installed.")
    sys.exit(1)

def verify_dataset(file_path, num_samples=1000):
    print(f"--- Verifying Dataset: {file_path} ---")
    
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return

    total_size = len(data)
    print(f"Total samples in dataset: {total_size:,}")
    
    # Take a slice for verification if requested
    verify_slice = data[:num_samples] if num_samples < total_size else data
    print(f"Verifying {len(verify_slice):,} samples...")

    stats = {
        'invalid_feature_count': 0,
        'nans': 0,
        'infs': 0,
        'zero_outputs': 0,
        'negative_outputs': 0,
        'max_cost': 0,
        'min_cost': float('inf'),
        'feature_dims': set()
    }

    all_costs = []
    
    for i, sample in enumerate(tqdm(verify_slice, desc="Processing")):
        dims = sample.get('input', [])
        target = sample.get('output', 0)
        
        if target <= 0:
            if target == 0: stats['zero_outputs'] += 1
            else: stats['negative_outputs'] += 1
        
        all_costs.append(target)
        stats['max_cost'] = max(stats['max_cost'], target)
        stats['min_cost'] = min(stats['min_cost'], target)
        
        try:
            features = extract_features_v3(dims)
            feats_arr = np.array(features)
            
            # Check dimensions
            stats['feature_dims'].add(len(features))
            if len(features) != TOTAL_FEATURES_V3:
                stats['invalid_feature_count'] += 1
            
            # Check for NaNs/Infs
            if np.isnan(feats_arr).any():
                stats['nans'] += 1
            if not np.isfinite(feats_arr).all():
                stats['infs'] += 1
                
        except Exception as e:
            print(f"\nError processing sample {i}: {e}")
            continue

    print("\n--- Verification Results ---")
    print(f"Feature Dimensions found: {stats['feature_dims']} (Expected: {TOTAL_FEATURES_V3})")
    
    if stats['invalid_feature_count'] == 0 and len(stats['feature_dims']) == 1 and TOTAL_FEATURES_V3 in stats['feature_dims']:
        print("✅ Feature Dimensions: All samples have correct count (177).")
    else:
        print(f"❌ Feature Dimensions: {stats['invalid_feature_count']} samples have incorrect counts!")

    if stats['nans'] == 0:
        print("✅ NaNs: No NaNs found in features.")
    else:
        print(f"❌ NaNs: Found NaNs in {stats['nans']} samples!")

    if stats['infs'] == 0:
        print("✅ Infs: No Infinite values found in features.")
    else:
        print(f"❌ Infs: Found Infinite values in {stats['infs']} samples!")

    if stats['zero_outputs'] == 0 and stats['negative_outputs'] == 0:
        print("✅ Target Costs: All targets are positive.")
    else:
        print(f"⚠️ Target Costs: Found {stats['zero_outputs']} zeros and {stats['negative_outputs']} negatives.")

    print(f"\nCost Statistics (Optimal):")
    print(f"- Min Cost: {stats['min_cost']:,}")
    print(f"- Max Cost: {stats['max_cost']:,}")
    print(f"- Mean Cost: {np.mean(all_costs):,.2f}")
    print(f"- Median Cost: {np.median(all_costs):,.2f}")

    if stats['nans'] == 0 and stats['infs'] == 0 and stats['invalid_feature_count'] == 0:
        print("\nOVERALL STATUS: ✅ DATASET IS PERFECT FOR TRAINING.")
    else:
        print("\nOVERALL STATUS: ❌ DATASET HAS ISSUES. DO NOT TRAIN YET.")

if __name__ == "__main__":
    dataset_path = 'data/mcm_100000.json'
    # Check 5000 samples for a robust check
    verify_dataset(dataset_path, num_samples=5000)
