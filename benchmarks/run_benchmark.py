"""
UNIFIED BENCHMARK — Compare All Models vs Exact DP
===================================================
Loads all 4 trained models, evaluates on:
1. Held-out test set (same split as training)
2. Brand new randomly generated chains (unseen data)
3. Classic textbook examples + edge cases

Outputs: JSON results + Markdown report
"""

import sys, os, json, time, random, math
import numpy as np
import torch
import joblib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.dataloader import load_data_split, extract_features_v2
from sklearn.preprocessing import StandardScaler

# ============================================================
# DP Solver
# ============================================================
def mcm_dp(dims):
    n = len(dims) - 1
    m = [[0] * (n + 1) for _ in range(n + 1)]
    for length in range(2, n + 1):
        for i in range(1, n - length + 2):
            j = i + length - 1
            m[i][j] = float('inf')
            for k in range(i, j):
                q = m[i][k] + m[k+1][j] + dims[i-1] * dims[k] * dims[j]
                if q < m[i][j]:
                    m[i][j] = q
    return m[1][n]

# ============================================================
# Model Loaders
# ============================================================
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

def load_rf():
    """Load Random Forest model."""
    path = os.path.join(MODELS_DIR, 'rf_v2.pkl')
    if not os.path.exists(path):
        return None
    return joblib.load(path)

def load_xgb():
    """Load XGBoost model."""
    path = os.path.join(MODELS_DIR, 'xgb_v2.joblib')
    if not os.path.exists(path):
        return None
    return joblib.load(path)

def load_nn():
    """Load Neural Network model."""
    # Need to import the architecture
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'nn_model'))
    from train import MCMNeuralNet
    
    model_path = os.path.join(MODELS_DIR, 'nn_v2.pth')
    scaler_path = os.path.join(MODELS_DIR, 'nn_scaler.joblib')
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None, None
    
    model = MCMNeuralNet(131)
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    model.eval()
    scaler = joblib.load(scaler_path)
    return model, scaler

def load_resnet():
    """Load ResNet model."""
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'resnet_model'))
    from train import MCMResNet
    
    model_path = os.path.join(MODELS_DIR, 'resnet_v2.pth')
    scaler_path = os.path.join(MODELS_DIR, 'resnet_scaler.joblib')
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None, None
    
    model = MCMResNet(131)
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    model.eval()
    scaler = joblib.load(scaler_path)
    return model, scaler

# ============================================================
# Prediction Functions
# ============================================================
def predict_rf(rf_model, X):
    if rf_model is None:
        return None
    preds_log = rf_model.predict(X)
    return np.expm1(preds_log)

def predict_xgb(xgb_model, X):
    if xgb_model is None:
        return None
    preds_log = xgb_model.predict(X)
    return np.expm1(preds_log)

def predict_nn(nn_model, nn_scaler, X):
    if nn_model is None:
        return None
    X_sc = nn_scaler.transform(X).astype(np.float32)
    with torch.no_grad():
        preds = nn_model(torch.from_numpy(X_sc)).numpy().flatten()
    return np.expm1(preds)

def predict_resnet(resnet_model, resnet_scaler, X):
    if resnet_model is None:
        return None
    X_sc = resnet_scaler.transform(X).astype(np.float32)
    with torch.no_grad():
        preds = resnet_model(torch.from_numpy(X_sc)).numpy().flatten()
    return np.expm1(preds)

def predict_single(dims, rf, xgb, nn, nn_sc, resnet, resnet_sc):
    """Predict cost for a single chain using all models."""
    feats = np.array([extract_features_v2(dims)])
    dp_cost = mcm_dp(dims)
    
    results = {'dp_cost': dp_cost, 'dims': dims, 'n': len(dims) - 1}
    
    for name, pred_fn in [
        ('rf', lambda: predict_rf(rf, feats)),
        ('xgb', lambda: predict_xgb(xgb, feats)),
        ('nn', lambda: predict_nn(nn, nn_sc, feats)),
        ('resnet', lambda: predict_resnet(resnet, resnet_sc, feats)),
    ]:
        pred = pred_fn()
        if pred is not None:
            pred_val = float(max(0, pred[0]))
            error = abs(pred_val - dp_cost) / max(dp_cost, 1)
            results[name] = {'pred': round(pred_val), 'error_pct': round(error * 100, 2)}
        else:
            results[name] = {'pred': None, 'error_pct': None}
    
    return results

# ============================================================
# Evaluation
# ============================================================
def evaluate_model(y_true, y_pred, label):
    """Compute comprehensive metrics."""
    errors = np.abs(y_pred - y_true) / np.maximum(y_true, 1)
    return {
        'model': label,
        'mape': round(float(np.mean(errors) * 100), 2),
        'median_error': round(float(np.median(errors) * 100), 2),
        'p95_error': round(float(np.percentile(errors, 95) * 100), 2),
        'max_error': round(float(np.max(errors) * 100), 2),
        'within_1pct': round(float((errors < 0.01).mean() * 100), 1),
        'within_5pct': round(float((errors < 0.05).mean() * 100), 1),
        'within_10pct': round(float((errors < 0.10).mean() * 100), 1),
        'within_20pct': round(float((errors < 0.20).mean() * 100), 1),
    }

# ============================================================
# Main Benchmark
# ============================================================
def run_benchmark():
    print("=" * 70)
    print("  UNIFIED BENCHMARK — All Models vs Exact DP")
    print("=" * 70)
    
    # 1. Load all models
    print("\n[1/4] Loading models...")
    rf = load_rf()
    xgb = load_xgb()
    nn_model, nn_scaler = load_nn()
    resnet_model, resnet_scaler = load_resnet()
    
    loaded = []
    if rf is not None: loaded.append("RandomForest")
    if xgb is not None: loaded.append("XGBoost")
    if nn_model is not None: loaded.append("NeuralNetwork")
    if resnet_model is not None: loaded.append("ResNet")
    print(f"  Loaded: {', '.join(loaded)} ({len(loaded)}/4)")
    
    if not loaded:
        print("ERROR: No models found! Train models first.")
        return

    # 2. Evaluate on test set
    print("\n[2/4] Evaluating on held-out test set...")
    X_train, X_val, X_test, y_train, y_val, y_test, _, _, dims_test = load_data_split(version='v2')
    
    test_results = {}
    for name, pred_fn in [
        ('RandomForest', lambda: predict_rf(rf, X_test)),
        ('XGBoost', lambda: predict_xgb(xgb, X_test)),
        ('NeuralNetwork', lambda: predict_nn(nn_model, nn_scaler, X_test)),
        ('ResNet', lambda: predict_resnet(resnet_model, resnet_scaler, X_test)),
    ]:
        preds = pred_fn()
        if preds is not None:
            preds = np.maximum(preds, 0)
            metrics = evaluate_model(y_test, preds, name)
            test_results[name] = metrics
            print(f"  {name:15s} | MAPE: {metrics['mape']:7.2f}% | "
                  f"Within 5%: {metrics['within_5pct']:5.1f}% | "
                  f"Within 10%: {metrics['within_10pct']:5.1f}% | "
                  f"Median: {metrics['median_error']:6.2f}%")

    # 3. Evaluate on unseen data
    print("\n[3/4] Evaluating on 200 brand-new random chains...")
    random.seed(999)  # separate seed from training
    unseen_results_list = []
    
    for i in range(200):
        n = random.randint(3, 50)
        pattern = random.choice(['random', 'bottleneck', 'spiky', 'uniform'])
        
        if pattern == 'random':
            dims = [random.randint(1, 500) for _ in range(n + 1)]
        elif pattern == 'bottleneck':
            dims = [random.randint(50, 500) for _ in range(n + 1)]
            dims[random.randint(0, n)] = random.randint(1, 5)
        elif pattern == 'spiky':
            dims = [(random.randint(200, 500) if j % 2 == 0 else random.randint(1, 20)) for j in range(n + 1)]
        else:
            base = random.randint(10, 300)
            dims = [max(1, base + random.randint(-20, 20)) for _ in range(n + 1)]
        
        r = predict_single(dims, rf, xgb, nn_model, nn_scaler, resnet_model, resnet_scaler)
        unseen_results_list.append(r)

    unseen_results = {}
    for name in loaded:
        key = name.lower().replace('network', '').replace('neural', 'nn')
        key_map = {'randomforest': 'rf', 'xgboost': 'xgb', 'nn': 'nn', 'resnet': 'resnet'}
        k = key_map.get(name.lower(), name.lower())
        
        preds = []
        trues = []
        for r in unseen_results_list:
            if r[k]['pred'] is not None:
                preds.append(r[k]['pred'])
                trues.append(r['dp_cost'])
        
        if preds:
            metrics = evaluate_model(np.array(trues), np.array(preds), name)
            unseen_results[name] = metrics
            print(f"  {name:15s} | MAPE: {metrics['mape']:7.2f}% | "
                  f"Within 5%: {metrics['within_5pct']:5.1f}% | "
                  f"Within 10%: {metrics['within_10pct']:5.1f}% | "
                  f"Median: {metrics['median_error']:6.2f}%")

    # 4. Classic test cases
    print("\n[4/4] Classic test cases...")
    classic_cases = [
        ("Textbook (n=3)", [10, 30, 5, 60]),
        ("Medium (n=5)", [40, 20, 30, 10, 30, 50]),
        ("Increasing (n=6)", [5, 10, 20, 30, 40, 50, 60]),
        ("Bottleneck (n=5)", [100, 200, 1, 300, 100, 200]),
        ("Spiky (n=6)", [500, 2, 400, 3, 300, 2, 500]),
        ("Large uniform (n=10)", [100]*11),
        ("Random large (n=15)", [random.randint(1, 500) for _ in range(16)]),
    ]
    
    classic_results = []
    print(f"\n  {'Case':<25} {'DP Cost':>12} ", end="")
    for name in loaded:
        print(f"{'|':>3} {name[:6]:>8} {'Err%':>7}", end="")
    print()
    print("  " + "-" * (40 + len(loaded) * 20))
    
    for case_name, dims in classic_cases:
        r = predict_single(dims, rf, xgb, nn_model, nn_scaler, resnet_model, resnet_scaler)
        classic_results.append({'case': case_name, **r})
        
        print(f"  {case_name:<25} {r['dp_cost']:>12,} ", end="")
        key_map = {'RandomForest': 'rf', 'XGBoost': 'xgb', 'NeuralNetwork': 'nn', 'ResNet': 'resnet'}
        for name in loaded:
            k = key_map[name]
            if r[k]['pred'] is not None:
                print(f"| {r[k]['pred']:>8,} {r[k]['error_pct']:>6.1f}%", end="")
            else:
                print(f"|      N/A    N/A", end="")
        print()

    # ============================================================
    # Generate Report
    # ============================================================
    report_lines = []
    report_lines.append("# MCM Model Benchmark Report")
    report_lines.append(f"\n**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"**Dataset**: 50,000 samples (dims 1-500, chains 3-50)")
    report_lines.append(f"**Models Evaluated**: {', '.join(loaded)}")
    
    report_lines.append("\n## 1. Test Set Results (Held-Out)")
    report_lines.append(f"\n| Model | MAPE | Median Err | 95th Pctl | Within 5% | Within 10% | Within 20% |")
    report_lines.append(f"|:------|-----:|-----------:|----------:|----------:|-----------:|-----------:|")
    
    best_mape = float('inf')
    best_model = ""
    for name, m in test_results.items():
        report_lines.append(f"| {name} | {m['mape']:.2f}% | {m['median_error']:.2f}% | {m['p95_error']:.2f}% | {m['within_5pct']:.1f}% | {m['within_10pct']:.1f}% | {m['within_20pct']:.1f}% |")
        if m['mape'] < best_mape:
            best_mape = m['mape']
            best_model = name
    
    report_lines.append(f"\n**🏆 Best Model (Test Set): {best_model}** — MAPE: {best_mape:.2f}%")
    
    report_lines.append("\n## 2. Unseen Data Results (200 New Chains)")
    report_lines.append(f"\n| Model | MAPE | Median Err | 95th Pctl | Within 5% | Within 10% | Within 20% |")
    report_lines.append(f"|:------|-----:|-----------:|----------:|----------:|-----------:|-----------:|")
    
    best_unseen_mape = float('inf')
    best_unseen = ""
    for name, m in unseen_results.items():
        report_lines.append(f"| {name} | {m['mape']:.2f}% | {m['median_error']:.2f}% | {m['p95_error']:.2f}% | {m['within_5pct']:.1f}% | {m['within_10pct']:.1f}% | {m['within_20pct']:.1f}% |")
        if m['mape'] < best_unseen_mape:
            best_unseen_mape = m['mape']
            best_unseen = name
    
    report_lines.append(f"\n**🏆 Best Model (Unseen): {best_unseen}** — MAPE: {best_unseen_mape:.2f}%")
    
    report_lines.append("\n## 3. Classic Test Cases")
    report_lines.append(f"\n| Case | DP Cost | " + " | ".join(loaded) + " |")
    report_lines.append(f"|:-----|--------:|" + "|".join(["---------:" for _ in loaded]) + "|")
    
    key_map_report = {'RandomForest': 'rf', 'XGBoost': 'xgb', 'NeuralNetwork': 'nn', 'ResNet': 'resnet'}
    for cr in classic_results:
        row = f"| {cr['case']} | {cr['dp_cost']:,} |"
        for name in loaded:
            k = key_map_report[name]
            if cr[k]['pred'] is not None:
                row += f" {cr[k]['pred']:,} ({cr[k]['error_pct']:.1f}%) |"
            else:
                row += " N/A |"
        report_lines.append(row)
    
    report_lines.append(f"\n## 4. Verdict")
    report_lines.append(f"\n**Champion Model: {best_model}**")
    report_lines.append(f"- Test MAPE: {best_mape:.2f}%")
    if best_unseen in unseen_results:
        report_lines.append(f"- Unseen MAPE: {unseen_results[best_unseen]['mape']:.2f}%")
    report_lines.append(f"\nAll models trained on 50K samples with 131 enhanced features (30 engineered + 51 raw dims + 50 pairwise products).")
    report_lines.append(f"Target: log1p(DP_cost). Evaluation in raw cost space vs exact DP solution.")
    
    report_path = os.path.join(os.path.dirname(__file__), 'benchmark_report.md')
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    print(f"\n  Report saved -> {report_path}")
    
    # Save raw results
    all_results = {
        'test_results': test_results,
        'unseen_results': unseen_results,
        'classic_results': classic_results,
        'best_model_test': best_model,
        'best_model_unseen': best_unseen,
    }
    json_path = os.path.join(os.path.dirname(__file__), 'benchmark_results.json')
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"  JSON saved -> {json_path}")

if __name__ == "__main__":
    run_benchmark()
