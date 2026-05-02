"""
Tree-Based Models for Matrix Chain Multiplication — Research Pipeline
=====================================================================
Trains XGBoost and Random Forest using the advanced v4 feature set
(213 features) and DUAL-TARGET training (direct log1p + ratio to greedy).

Features sample weighting to focus on hard spiky/bottleneck chains.

Usage:
    python -m src.train_tree_models
"""

import json
import os
import sys
import time
import math
import numpy as np
import joblib

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.feature_extractor_v4 import extract_features_v4, TOTAL_FEATURES_V4
from data.feature_extractor_v3 import greedy_cost_left_to_right, greedy_cost_right_to_left, greedy_cost_min_first, greedy_cost_balanced

# ═══════════════════════════════════════════════════════════════════════
#   DATA LOADING & ENGINEERING
# ═══════════════════════════════════════════════════════════════════════

def get_greedy_min(dims):
    """Get the minimum greedy heuristic cost."""
    return min(
        greedy_cost_left_to_right(dims),
        greedy_cost_right_to_left(dims),
        greedy_cost_min_first(dims),
        greedy_cost_balanced(dims)
    )

def load_dataset_v4(json_path):
    """Load dataset, extract 213 features, engineer dual targets and weights."""
    print(f"\n📂 Loading dataset: {json_path}")
    with open(json_path, 'r') as f:
        raw = json.load(f)

    n_samples = len(raw)
    print(f"   {n_samples:,} samples. Extracting {TOTAL_FEATURES_V4} features...")

    X = np.zeros((n_samples, TOTAL_FEATURES_V4), dtype=np.float32)
    y_raw = np.zeros(n_samples, dtype=np.float64)
    y_log1p = np.zeros(n_samples, dtype=np.float32)
    y_ratio = np.zeros(n_samples, dtype=np.float32)
    weights = np.ones(n_samples, dtype=np.float32)
    
    dims_list = []
    greedy_mins = np.zeros(n_samples, dtype=np.float64)
    spreads = np.zeros(n_samples, dtype=np.float32)

    t0 = time.time()
    for i, sample in enumerate(raw):
        dims = sample['input']
        cost = sample['output']
        
        # 1. Features
        feats = extract_features_v4(dims)
        X[i] = feats
        
        # 2. Raw cost and standard log target
        y_raw[i] = cost
        y_log1p[i] = math.log1p(cost)
        
        # 3. Ratio target: log(optimal / greedy_min)
        g_min = get_greedy_min(dims) + 1e-9
        greedy_mins[i] = g_min
        
        # Ratio is typically <= 1.0. We learn log(ratio).
        # We cap ratio at 1.0 because optimal <= greedy.
        ratio = min(1.0, cost / g_min) 
        y_ratio[i] = math.log(ratio + 1e-9)
        
        # 4. Spread (proxy for difficulty)
        # feat 48 is greedy_spread in v3/v4
        spreads[i] = feats[49] if len(feats) > 49 else 0.0
        
        dims_list.append(dims)

        if (i + 1) % 10000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n_samples - i - 1) / rate
            print(f"   [{i+1:>7,}/{n_samples:,}] {elapsed:>4.0f}s elapsed, ~{eta:.0f}s remaining")

    # Clean NaNs
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # 5. Compute Sample Weights
    # Harder patterns (high spread) get up to 5x weight
    med_spread = np.median(spreads) + 1e-9
    weights = 1.0 + 2.0 * np.clip(spreads / med_spread, 0, 5)
    
    elapsed = time.time() - t0
    print(f"   ✅ Feature engineering complete in {elapsed:.0f}s")
    print(f"   Targets: log1p range [{y_log1p.min():.2f}, {y_log1p.max():.2f}]")
    print(f"            ratio range [{y_ratio.min():.2f}, {y_ratio.max():.2f}]")
    print(f"   Weights: range [{weights.min():.2f}, {weights.max():.2f}] (mean={weights.mean():.2f})")

    return {
        'X': X,
        'y_raw': y_raw,
        'y_log1p': y_log1p,
        'y_ratio': y_ratio,
        'weights': weights,
        'greedy_mins': greedy_mins,
        'dims_list': np.array(dims_list, dtype=object)
    }


def split_data(data, test_frac=0.15, val_frac=0.10):
    """Stratified random split."""
    n = len(data['X'])
    idx = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(idx)

    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    n_train = n - n_test - n_val

    splits = {}
    for name, indices in [('train', idx[:n_train]), 
                          ('val', idx[n_train:n_train + n_val]), 
                          ('test', idx[n_train + n_val:])]:
        splits[name] = {k: v[indices] for k, v in data.items()}
        
    return splits


# ═══════════════════════════════════════════════════════════════════════
#   XGBOOST DUAL-TARGET TRAINING
# ═══════════════════════════════════════════════════════════════════════

def train_xgb_target(X_tr, y_tr, w_tr, X_va, y_va, target_name):
    """Train XGBoost on a specific target."""
    import xgboost as xgb
    
    print(f"\n   [XGBoost] Training target: {target_name}...")
    
    dtrain = xgb.DMatrix(X_tr, label=y_tr, weight=w_tr)
    dval   = xgb.DMatrix(X_va, label=y_va)
    
    params = {
        'objective':         'reg:squarederror',
        'eval_metric':       'mae',
        'max_depth':         14,         # Deep trees for complex interactions
        'learning_rate':     0.02,       # Slow learning
        'subsample':         0.75,
        'colsample_bytree':  0.6,
        'min_child_weight':  3,
        'reg_alpha':         0.1,
        'reg_lambda':        2.0,
        'gamma':             0.05,
        'tree_method':       'hist',
        'seed':              42,
        'verbosity':         0,
    }

    model = xgb.train(
        params, dtrain,
        num_boost_round=5000,
        evals=[(dval, 'val')],
        early_stopping_rounds=150,
        verbose_eval=500,
    )
    
    print(f"   ✓ {target_name} done (best iter: {model.best_iteration}, val_mae: {model.best_score:.6f})")
    return model


def predict_xgb_ensemble(models, X, greedy_mins):
    """Ensemble prediction from both targets."""
    import xgboost as xgb
    dmat = xgb.DMatrix(X)
    
    # 1. Direct log1p prediction
    pred_log1p = models['direct'].predict(dmat)
    pred_cost_direct = np.clip(np.expm1(pred_log1p), 0, None)
    
    # 2. Ratio prediction
    pred_ratio_log = models['ratio'].predict(dmat)
    # Undo log transform and cap ratio at 1.0 (optimal <= greedy)
    pred_ratio = np.clip(np.exp(pred_ratio_log), 0, 1.0)
    pred_cost_ratio = greedy_mins * pred_ratio
    
    # 3. Ensemble (average)
    # The ratio prediction tends to be much more stable
    pred_final = 0.3 * pred_cost_direct + 0.7 * pred_cost_ratio
    
    return pred_final


def train_xgboost_pipeline(data_splits):
    """Train both XGBoost targets."""
    try:
        import xgboost as xgb
    except ImportError:
        print("\n❌ XGBoost not installed. Run: pip install xgboost")
        return None
        
    print("\n" + "▓"*65)
    print("  🌲 TRAINING XGBOOST (DUAL-TARGET)")
    print("▓"*65)
    
    tr = data_splits['train']
    va = data_splits['val']
    
    t0 = time.time()
    
    models = {}
    # Target 1: Direct log1p
    models['direct'] = train_xgb_target(
        tr['X'], tr['y_log1p'], tr['weights'], 
        va['X'], va['y_log1p'], "Direct log1p(cost)"
    )
    
    # Target 2: Ratio log(cost/greedy)
    models['ratio'] = train_xgb_target(
        tr['X'], tr['y_ratio'], tr['weights'], 
        va['X'], va['y_ratio'], "Ratio log(cost/greedy)"
    )
    
    elapsed = time.time() - t0
    print(f"\n✅ XGBoost pipeline completed in {elapsed:.0f}s")
    
    return models


# ═══════════════════════════════════════════════════════════════════════
#   RANDOM FOREST DUAL-TARGET TRAINING
# ═══════════════════════════════════════════════════════════════════════

def train_rf_pipeline(data_splits):
    """Train Random Forest targets."""
    from sklearn.ensemble import RandomForestRegressor
    
    print("\n" + "▓"*65)
    print("  🌲 TRAINING RANDOM FOREST (DUAL-TARGET)")
    print("▓"*65)
    
    tr = data_splits['train']
    
    # For RF, we merge train+val for more data
    X_full = np.vstack([tr['X'], data_splits['val']['X']])
    y_log1p_full = np.concatenate([tr['y_log1p'], data_splits['val']['y_log1p']])
    y_ratio_full = np.concatenate([tr['y_ratio'], data_splits['val']['y_ratio']])
    w_full = np.concatenate([tr['weights'], data_splits['val']['weights']])
    
    params = {
        'n_estimators':      1000,
        'max_depth':         30,
        'min_samples_leaf':  2,
        'max_features':      0.4,
        'n_jobs':            -1,
        'random_state':      42,
    }
    
    t0 = time.time()
    models = {}
    
    print(f"\n   [RandomForest] Training target: Direct log1p(cost)...")
    models['direct'] = RandomForestRegressor(**params)
    # sklearn RF doesn't fully support sample_weight well in all edge cases, but we use it
    models['direct'].fit(X_full, y_log1p_full, sample_weight=w_full)
    
    print(f"   [RandomForest] Training target: Ratio log(cost/greedy)...")
    models['ratio'] = RandomForestRegressor(**params)
    models['ratio'].fit(X_full, y_ratio_full, sample_weight=w_full)
    
    elapsed = time.time() - t0
    print(f"\n✅ RandomForest pipeline completed in {elapsed:.0f}s")
    
    return models


def predict_rf_ensemble(models, X, greedy_mins):
    """Ensemble prediction for Random Forest."""
    pred_log1p = models['direct'].predict(X)
    pred_cost_direct = np.clip(np.expm1(pred_log1p), 0, None)
    
    pred_ratio_log = models['ratio'].predict(X)
    pred_ratio = np.clip(np.exp(pred_ratio_log), 0, 1.0)
    pred_cost_ratio = greedy_mins * pred_ratio
    
    # Weight ratio higher as it's typically more accurate
    return 0.3 * pred_cost_direct + 0.7 * pred_cost_ratio


# ═══════════════════════════════════════════════════════════════════════
#   MAIN RUNNER
# ═══════════════════════════════════════════════════════════════════════

def main():
    DATA_PATH  = "data/mcm_120000.json"
    CACHE_PATH = "data/tree_features_v4_cache.npz"
    MODEL_DIR  = "models"
    os.makedirs(MODEL_DIR, exist_ok=True)

    # ── 1. Load / Cache Data ─────────────────────────────────────────
    if os.path.exists(CACHE_PATH):
        print(f"\n📦 Loading cached V4 features from {CACHE_PATH}...")
        cached = np.load(CACHE_PATH, allow_pickle=True)
        data = {k: cached[k] for k in cached.files}
        print(f"   ✅ Loaded {data['X'].shape[0]:,} samples")
    else:
        data = load_dataset_v4(DATA_PATH)
        print(f"\n💾 Caching features to {CACHE_PATH}...")
        np.savez_compressed(CACHE_PATH, **data)
        print("   ✅ Cached.")

    # ── 2. Split Data ────────────────────────────────────────────────
    splits = split_data(data)
    print(f"\n   Split sizes — Train: {len(splits['train']['X']):,} | Val: {len(splits['val']['X']):,} | Test: {len(splits['test']['X']):,}")

    # ── 3. Train XGBoost ─────────────────────────────────────────────
    xgb_models = train_xgboost_pipeline(splits)
    if xgb_models:
        import xgboost as xgb
        # Save models
        xgb_models['direct'].save_model(os.path.join(MODEL_DIR, "xgb_direct_v4.json"))
        xgb_models['ratio'].save_model(os.path.join(MODEL_DIR, "xgb_ratio_v4.json"))
        print(f"   💾 Saved XGBoost models.")

        # Test inference to verify
        y_pred = predict_xgb_ensemble(xgb_models, splits['test']['X'], splits['test']['greedy_mins'])
        y_true = splits['test']['y_raw']
        mape = np.mean(np.abs(y_true - y_pred) / (np.abs(y_true) + 1e-9)) * 100
        print(f"   📊 XGBoost Quick Test MAPE: {mape:.4f}%")

    # ── 4. Train Random Forest ───────────────────────────────────────
    rf_models = train_rf_pipeline(splits)
    if rf_models:
        joblib.dump(rf_models, os.path.join(MODEL_DIR, "rf_ensemble_v4.joblib"))
        print(f"   💾 Saved RandomForest models.")
        
        y_pred = predict_rf_ensemble(rf_models, splits['test']['X'], splits['test']['greedy_mins'])
        y_true = splits['test']['y_raw']
        mape = np.mean(np.abs(y_true - y_pred) / (np.abs(y_true) + 1e-9)) * 100
        print(f"   📊 RandomForest Quick Test MAPE: {mape:.4f}%")

    print("\n✅ Training Phase Complete!")
    print("   Next step: Run evaluate_tree_models.py to generate paper tables.")


if __name__ == "__main__":
    main()
