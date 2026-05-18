"""
FULL MODEL EVALUATION v2 — Enhanced with F1, R2, p-values
==========================================================
Evaluates: Pointer Network, GNN, XGBoost, Random Forest
Metrics: MAPE, MAE, RMSE, R2, F1, Precision, Recall, p-values, correlations
"""
import os, sys, torch, numpy as np, time, joblib, xgboost as xgb, json
from scipy import stats as scipy_stats
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.pointer_mcm import PointerMCMNet, compute_cost_from_splits
from src.models.gnn_mcm import GraphMCMNet
from src.data.pointer_features import extract_pointer_features, pad_features
from src.data.gnn_loader import precompute_graph, collate_gnn_batch
from src.data.feature_extractor import extract_features_v4
from src.data.generator import (mcm_dp, greedy_cost_left_to_right, greedy_cost_right_to_left,
                                 greedy_cost_min_first, greedy_cost_balanced)
from src.training.train_trees import predict_xgb_ensemble, predict_rf_ensemble, get_greedy_min

device = torch.device("cpu")

# ============================================================
# Chain Generation
# ============================================================
def generate_chains(seed=42):
    np.random.seed(seed)
    buckets = [(5, 10), (11, 20), (21, 30), (31, 40), (41, 50)]
    distributions = ['uniform', 'spiky', 'bottleneck', 'monotone']
    chains = []
    for lo, hi in buckets:
        for dist in distributions:
            for _ in range(25):
                n = np.random.randint(lo, hi + 1)
                if dist == 'uniform':
                    dims = np.random.randint(10, 500, size=n+1).tolist()
                elif dist == 'spiky':
                    dims = [(np.random.randint(5, 50) if i % 2 == 0 else np.random.randint(200, 800)) for i in range(n + 1)]
                elif dist == 'bottleneck':
                    dims = np.random.randint(50, 300, size=n+1).tolist()
                    dims[np.random.randint(1, n)] = np.random.randint(1, 5)
                elif dist == 'monotone':
                    start = np.random.randint(10, 100)
                    step = np.random.randint(5, 30)
                    dims = [start + i * step for i in range(n + 1)]
                chains.append({'dims': dims, 'n': n, 'bucket': f"{lo}-{hi}", 'dist': dist, 'dp_cost': mcm_dp(dims)})
    return chains

# ============================================================
# Model Evaluation
# ============================================================
def eval_pointer(model, chain):
    dims = chain['dims']
    pf = extract_pointer_features(dims)
    pad, mask = pad_features(pf, 51)
    with torch.no_grad():
        ps, _ = model.predict(torch.FloatTensor(pad).unsqueeze(0), torch.BoolTensor(mask).unsqueeze(0), torch.LongTensor([chain['n']]))
        return compute_cost_from_splits(dims, ps[0])

def eval_gnn(model, chain):
    dims, opt = chain['dims'], chain['dp_cost']
    g = precompute_graph({'input': dims, 'output': opt})
    nf, ei, bi, _, _, _ = collate_gnn_batch([g])
    with torch.no_grad():
        ps, _ = model.predict(nf, ei, bi)
        return compute_cost_from_splits(dims, ps[0])

def eval_xgb(models, chain):
    feat = np.array(extract_features_v4(chain['dims']), dtype=np.float32).reshape(1, -1)
    g_min = np.array([get_greedy_min(chain['dims'])])
    return predict_xgb_ensemble(models, feat, g_min)[0]

def eval_rf(models, chain):
    feat = np.array(extract_features_v4(chain['dims']), dtype=np.float32).reshape(1, -1)
    g_min = np.array([get_greedy_min(chain['dims'])])
    return predict_rf_ensemble(models, feat, g_min)[0]

# ============================================================
# Comprehensive Metrics
# ============================================================
def compute_full_metrics(preds, trues, structural=False):
    preds = np.array(preds, dtype=float)
    trues = np.array(trues, dtype=float)
    
    # Error metrics
    errors_pct = np.abs(preds - trues) / np.maximum(trues, 1) * 100
    errors_abs = np.abs(preds - trues)
    
    # Basic metrics
    mape = float(np.mean(errors_pct))
    median_err = float(np.median(errors_pct))
    p95_err = float(np.percentile(errors_pct, 95))
    max_err = float(np.max(errors_pct))
    mae = float(mean_absolute_error(trues, preds))
    rmse = float(np.sqrt(mean_squared_error(trues, preds)))
    
    # R-squared (on raw costs and log-transformed costs)
    r2_raw = float(r2_score(trues, preds)) if len(set(trues)) > 1 else 0.0
    log_trues = np.log1p(np.maximum(trues, 0))
    log_preds = np.log1p(np.maximum(preds, 0))
    r2_log = float(r2_score(log_trues, log_preds)) if len(set(log_trues)) > 1 else 0.0
    
    # Correlation metrics
    spearman_r, spearman_p = scipy_stats.spearmanr(trues, preds)
    pearson_r, pearson_p = scipy_stats.pearsonr(trues, preds)
    
    # Validity
    valid_count = int(np.sum(preds >= trues - 1))
    valid_rate = valid_count / len(trues) * 100
    
    # Threshold accuracy
    within_1 = float(np.mean(errors_pct < 1) * 100)
    within_5 = float(np.mean(errors_pct < 5) * 100)
    within_10 = float(np.mean(errors_pct < 10) * 100)
    within_20 = float(np.mean(errors_pct < 20) * 100)
    
    result = {
        'MAPE': mape,
        'Median_Err': median_err,
        'P95_Err': p95_err,
        'Max_Err': max_err,
        'MAE': mae,
        'RMSE': rmse,
        'R2_Raw': r2_raw,
        'R2_Log': r2_log,
        'Spearman_r': float(spearman_r),
        'Spearman_p': float(spearman_p),
        'Pearson_r': float(pearson_r),
        'Pearson_p': float(pearson_p),
        'Valid_Rate': valid_rate,
        'Within_1pct': within_1,
        'Within_5pct': within_5,
        'Within_10pct': within_10,
        'Within_20pct': within_20,
        'Count': len(trues),
    }
    
    if structural:
        exact_count = int(np.sum(errors_pct < 0.001))
        result['Exact_Match'] = exact_count / len(trues) * 100
        
        # F1-Score: treat each prediction as binary (correct within 1% = positive)
        tp = int(np.sum(errors_pct < 1))      # True positive: accurate prediction
        fp = int(np.sum(errors_pct >= 1))      # False positive: inaccurate
        fn = 0                                  # False negative: 0 since we always predict
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0  # recall = 1 since fn=0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        result['Precision'] = precision * 100
        result['Recall'] = recall * 100
        result['F1_Score'] = f1 * 100
        
        # Stricter F1: within 0.1%
        tp_strict = int(np.sum(errors_pct < 0.1))
        fp_strict = int(np.sum(errors_pct >= 0.1))
        prec_strict = tp_strict / (tp_strict + fp_strict) if (tp_strict + fp_strict) > 0 else 0.0
        f1_strict = 2 * prec_strict * 1.0 / (prec_strict + 1.0) if (prec_strict + 1.0) > 0 else 0.0
        result['F1_Strict'] = f1_strict * 100
    
    return result

# ============================================================
# Main
# ============================================================
def main():
    print("=" * 70)
    print("  FULL MODEL EVALUATION v2 (n=5-50, 500 chains, enhanced metrics)")
    print("=" * 70)
    
    # Load models
    ptr = PointerMCMNet(input_dim=8, d_model=128).to(device)
    ptr.load_state_dict(torch.load("models/pointer_best.pth", map_location=device, weights_only=True))
    ptr.eval()
    print("[OK] Pointer Network")

    gnn = GraphMCMNet(d_model=128, num_layers=6, dropout=0.1, max_n=50).to(device)
    ckpt = torch.load("models/gnn_checkpoint.pth", map_location=device, weights_only=True)
    gnn.load_state_dict(ckpt['model_state_dict'])
    gnn.eval()
    print("[OK] GNN")

    xgb_m = {'direct': xgb.Booster(), 'ratio': xgb.Booster()}
    xgb_m['direct'].load_model("models/xgb_direct_145.json")
    xgb_m['ratio'].load_model("models/xgb_ratio_145.json")
    print("[OK] XGBoost")

    rf_m = joblib.load("models/rf_145.joblib")
    print("[OK] Random Forest")

    # Generate chains
    chains = generate_chains()
    print(f"\nTest chains: {len(chains)}, n={min(c['n'] for c in chains)}-{max(c['n'] for c in chains)}")
    trues = [c['dp_cost'] for c in chains]

    # Evaluate all models
    all_preds = {}
    eval_fns = [
        ('Pointer', lambda c: eval_pointer(ptr, c), True),
        ('GNN', lambda c: eval_gnn(gnn, c), True),
        ('XGBoost', lambda c: eval_xgb(xgb_m, c), False),
        ('RF', lambda c: eval_rf(rf_m, c), False),
    ]

    all_results = {}
    
    for name, fn, structural in eval_fns:
        print(f"\nEvaluating {name}...")
        t0 = time.time()
        preds = []
        for chain in chains:
            try:
                preds.append(fn(chain))
            except:
                preds.append(chain['dp_cost'] * 2)
        dt = time.time() - t0
        
        all_preds[name] = preds
        
        # Overall
        overall = compute_full_metrics(preds, trues, structural)
        overall['Time_s'] = dt
        
        # By bucket
        by_bucket = {}
        for bucket in ['5-10', '11-20', '21-30', '31-40', '41-50']:
            idx = [i for i, c in enumerate(chains) if c['bucket'] == bucket]
            by_bucket[bucket] = compute_full_metrics([preds[i] for i in idx], [trues[i] for i in idx], structural)
        
        # By distribution
        by_dist = {}
        for dist in ['uniform', 'spiky', 'bottleneck', 'monotone']:
            idx = [i for i, c in enumerate(chains) if c['dist'] == dist]
            by_dist[dist] = compute_full_metrics([preds[i] for i in idx], [trues[i] for i in idx], structural)
        
        all_results[name] = {'overall': overall, 'by_bucket': by_bucket, 'by_dist': by_dist}
        
        print(f"  MAPE: {overall['MAPE']:.4f}% | R2: {overall['R2_Log']:.6f} | Valid: {overall['Valid_Rate']:.1f}% | {dt:.1f}s")

    # Statistical significance tests (Wilcoxon signed-rank)
    print("\n" + "=" * 70)
    print("STATISTICAL SIGNIFICANCE (Wilcoxon Signed-Rank Test)")
    print("=" * 70)
    
    sig_tests = {}
    model_names = list(all_preds.keys())
    for i in range(len(model_names)):
        for j in range(i+1, len(model_names)):
            m1, m2 = model_names[i], model_names[j]
            errs1 = np.abs(np.array(all_preds[m1]) - np.array(trues)) / np.maximum(np.array(trues), 1)
            errs2 = np.abs(np.array(all_preds[m2]) - np.array(trues)) / np.maximum(np.array(trues), 1)
            
            # Wilcoxon test
            try:
                stat, p_val = scipy_stats.wilcoxon(errs1, errs2, alternative='two-sided')
            except:
                stat, p_val = 0, 1.0
            
            # Cohen's d effect size
            diff = errs1 - errs2
            cohens_d = float(np.mean(diff) / (np.std(diff) + 1e-9))
            
            key = f"{m1}_vs_{m2}"
            sig_tests[key] = {'W_stat': float(stat), 'p_value': float(p_val), 'cohens_d': cohens_d}
            
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            print(f"  {m1} vs {m2}: W={stat:.0f}, p={p_val:.2e} ({sig}), Cohen's d={cohens_d:.4f}")
    
    all_results['significance_tests'] = sig_tests

    # ============================================================
    # Print Full Tables
    # ============================================================
    print("\n" + "=" * 100)
    print("COMPREHENSIVE RESULTS TABLE")
    print("=" * 100)
    
    for name, r in all_results.items():
        if name == 'significance_tests':
            continue
        o = r['overall']
        print(f"\n{'='*50}")
        print(f"  {name}")
        print(f"{'='*50}")
        print(f"  MAPE:              {o['MAPE']:.4f}%")
        print(f"  Median Error:      {o['Median_Err']:.4f}%")
        print(f"  P95 Error:         {o['P95_Err']:.4f}%")
        print(f"  Max Error:         {o['Max_Err']:.4f}%")
        print(f"  MAE (raw cost):    {o['MAE']:,.0f}")
        print(f"  RMSE (raw cost):   {o['RMSE']:,.0f}")
        print(f"  R2 (raw):          {o['R2_Raw']:.6f}")
        print(f"  R2 (log1p):        {o['R2_Log']:.6f}")
        print(f"  Spearman rho:      {o['Spearman_r']:.6f} (p={o['Spearman_p']:.2e})")
        print(f"  Pearson r:         {o['Pearson_r']:.6f} (p={o['Pearson_p']:.2e})")
        print(f"  Valid Rate:        {o['Valid_Rate']:.2f}%")
        print(f"  Within 1%:         {o['Within_1pct']:.1f}%")
        print(f"  Within 5%:         {o['Within_5pct']:.1f}%")
        print(f"  Within 10%:        {o['Within_10pct']:.1f}%")
        print(f"  Within 20%:        {o['Within_20pct']:.1f}%")
        if 'Exact_Match' in o:
            print(f"  Exact Match:       {o['Exact_Match']:.2f}%")
            print(f"  Precision (<1%):   {o['Precision']:.2f}%")
            print(f"  Recall (<1%):      {o['Recall']:.2f}%")
            print(f"  F1-Score (<1%):    {o['F1_Score']:.2f}%")
            print(f"  F1-Strict (<0.1%): {o['F1_Strict']:.2f}%")
        print(f"  Eval Time:         {o['Time_s']:.1f}s")

    # Save JSON
    json_path = os.path.join(os.path.dirname(__file__), 'full_eval_results_v2.json')
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nJSON saved: {json_path}")

if __name__ == '__main__':
    main()
