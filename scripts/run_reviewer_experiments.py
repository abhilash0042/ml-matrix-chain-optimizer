"""
Run all reviewer-requested experiments in one script.
Outputs: scratch/reviewer_experiment_results.json

Experiments:
1. Greedy heuristic baselines (MAPE, validity) on 500 test chains
2. Latency table (ms/sample) at n=10, 25, 50 for all models
3. Bootstrap 95% CIs for MAPE of all models
"""
import os
import sys
import time
import json
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.generator import (
    mcm_dp,
    greedy_cost_left_to_right,
    greedy_cost_right_to_left,
    greedy_cost_min_first,
    greedy_cost_balanced,
)
from src.models.pointer_mcm import PointerMCMNet, compute_cost_from_splits
from src.data.feature_extractor import extract_features_v4
from src.data.pointer_features import extract_pointer_features, pad_features

# ─── Test chain generation (same as research_eval.py) ───
def generate_test_chains(seed=42, n_per_dist=125, n_range=(5, 50)):
    """Generate the standard 500-chain test set."""
    rng = np.random.RandomState(seed)
    chains = []
    for dist in ['uniform', 'spiky', 'bottleneck', 'monotone']:
        for _ in range(n_per_dist):
            n = rng.randint(n_range[0], n_range[1] + 1)
            if dist == 'uniform':
                dims = rng.randint(5, 500, size=n + 1).tolist()
            elif dist == 'spiky':
                dims = [(rng.randint(5, 50) if i % 2 == 0 else rng.randint(500, 1000)) for i in range(n + 1)]
            elif dist == 'bottleneck':
                dims = rng.randint(500, 1000, size=n + 1).tolist()
                dims[rng.randint(1, n)] = rng.randint(1, 5)
            elif dist == 'monotone':
                start, step = rng.randint(5, 100), rng.randint(10, 50)
                if rng.random() > 0.5:
                    dims = [start + i * step for i in range(n + 1)]
                else:
                    dims = [start + (n - i) * step for i in range(n + 1)]
            chains.append({'dims': dims, 'dist': dist, 'n': n})
    return chains


def generate_fixed_length_chains(n, count=100, seed=99):
    """Generate chains of a specific length for latency testing."""
    rng = np.random.RandomState(seed)
    return [rng.randint(5, 500, size=n + 1).tolist() for _ in range(count)]


# ════════════════════════════════════════════════════════════
#  EXPERIMENT 1: Greedy heuristic baselines
# ════════════════════════════════════════════════════════════
def run_greedy_baselines(chains):
    print("\n" + "=" * 60)
    print("  EXPERIMENT 1: Greedy Heuristic Baselines")
    print("=" * 60)

    heuristics = {
        'Left-to-Right': greedy_cost_left_to_right,
        'Right-to-Left': greedy_cost_right_to_left,
        'Minimum-First': greedy_cost_min_first,
        'Balanced':      greedy_cost_balanced,
    }

    results = {}
    for name, func in heuristics.items():
        errors = []
        valid = 0
        for c in chains:
            dp_cost = mcm_dp(c['dims'])
            h_cost = func(c['dims'])
            err = abs(h_cost - dp_cost) / (dp_cost + 1e-9) * 100
            errors.append(err)
            if h_cost >= dp_cost - 1:
                valid += 1

        errors = np.array(errors)
        results[name] = {
            'mape': float(np.mean(errors)),
            'median_err': float(np.median(errors)),
            'p95_err': float(np.percentile(errors, 95)),
            'max_err': float(np.max(errors)),
            'validity': float(valid / len(chains) * 100),
            'within_1pct': float(np.mean(errors <= 1.0) * 100),
            'within_5pct': float(np.mean(errors <= 5.0) * 100),
            'within_10pct': float(np.mean(errors <= 10.0) * 100),
        }
        print(f"  {name:<16} | MAPE: {results[name]['mape']:>8.3f}% | "
              f"Median: {results[name]['median_err']:>8.3f}% | "
              f"Valid: {results[name]['validity']:>6.1f}%")

    # Best-of-4 greedy (oracle)
    errors_best = []
    valid_best = 0
    for c in chains:
        dp_cost = mcm_dp(c['dims'])
        best = min(func(c['dims']) for func in heuristics.values())
        err = abs(best - dp_cost) / (dp_cost + 1e-9) * 100
        errors_best.append(err)
        if best >= dp_cost - 1:
            valid_best += 1

    errors_best = np.array(errors_best)
    results['Best-of-4 Greedy'] = {
        'mape': float(np.mean(errors_best)),
        'median_err': float(np.median(errors_best)),
        'p95_err': float(np.percentile(errors_best, 95)),
        'max_err': float(np.max(errors_best)),
        'validity': float(valid_best / len(chains) * 100),
        'within_1pct': float(np.mean(errors_best <= 1.0) * 100),
        'within_5pct': float(np.mean(errors_best <= 5.0) * 100),
        'within_10pct': float(np.mean(errors_best <= 10.0) * 100),
    }
    print(f"  {'Best-of-4':<16} | MAPE: {results['Best-of-4 Greedy']['mape']:>8.3f}% | "
          f"Median: {results['Best-of-4 Greedy']['median_err']:>8.3f}% | "
          f"Valid: {results['Best-of-4 Greedy']['validity']:>6.1f}%")

    return results


# ════════════════════════════════════════════════════════════
#  EXPERIMENT 2: Latency measurements
# ════════════════════════════════════════════════════════════
def run_latency_measurements():
    print("\n" + "=" * 60)
    print("  EXPERIMENT 2: Latency Measurements (ms/sample)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}
    warmup_runs = 10
    measure_runs = 100

    for n in [10, 25, 50]:
        chains = generate_fixed_length_chains(n)
        print(f"\n  --- n = {n} ---")

        # DP baseline
        times = []
        for _ in range(warmup_runs):
            mcm_dp(chains[0])
        for ch in chains[:measure_runs]:
            t0 = time.perf_counter()
            mcm_dp(ch)
            times.append((time.perf_counter() - t0) * 1000)
        dp_mean, dp_std = np.mean(times), np.std(times)
        results[f'DP_n{n}'] = {'mean_ms': float(dp_mean), 'std_ms': float(dp_std)}
        print(f"  DP          : {dp_mean:.3f} ± {dp_std:.3f} ms")

        # Greedy (best-of-4)
        times = []
        for _ in range(warmup_runs):
            for func in [greedy_cost_left_to_right, greedy_cost_right_to_left,
                         greedy_cost_min_first, greedy_cost_balanced]:
                func(chains[0])
        for ch in chains[:measure_runs]:
            t0 = time.perf_counter()
            for func in [greedy_cost_left_to_right, greedy_cost_right_to_left,
                         greedy_cost_min_first, greedy_cost_balanced]:
                func(ch)
            times.append((time.perf_counter() - t0) * 1000)
        g_mean, g_std = np.mean(times), np.std(times)
        results[f'Greedy_n{n}'] = {'mean_ms': float(g_mean), 'std_ms': float(g_std)}
        print(f"  Greedy(4)   : {g_mean:.3f} ± {g_std:.3f} ms")

        # GNN
        try:
            from src.models.gnn_mcm import GraphMCMNet
            from src.data.gnn_loader import precompute_graph, collate_gnn_batch

            gnn_model = GraphMCMNet(d_model=128, num_layers=6, dropout=0.1, max_n=50).to(device)
            gnn_path = None
            for p in ["models/gnn_best.pth", "models/gnn_checkpoint.pth"]:
                if os.path.exists(p):
                    gnn_path = p
                    break
            if gnn_path:
                ckpt = torch.load(gnn_path, map_location=device, weights_only=True)
                if 'model_state_dict' in ckpt:
                    gnn_model.load_state_dict(ckpt['model_state_dict'])
                else:
                    gnn_model.load_state_dict(ckpt)
                gnn_model.eval()

                # warmup
                for _ in range(min(warmup_runs, 3)):
                    g_sample = precompute_graph({'input': chains[0], 'output': mcm_dp(chains[0])})
                    nf, ei, bi, _, _, _ = collate_gnn_batch([g_sample])
                    nf, ei = nf.to(device), ei.to(device)
                    bi['actual_lengths'] = bi['actual_lengths'].to(device)
                    bi['root_indices'] = bi['root_indices'].to(device)
                    for L in bi['split_parent_idx']:
                        bi['split_parent_idx'][L] = bi['split_parent_idx'][L].to(device)
                        bi['split_left_idx'][L] = bi['split_left_idx'][L].to(device)
                        bi['split_right_idx'][L] = bi['split_right_idx'][L].to(device)
                        bi['split_valid'][L] = bi['split_valid'][L].to(device)
                    with torch.no_grad():
                        gnn_model.predict(nf, ei, bi)

                times = []
                for ch in chains[:measure_runs]:
                    dp_cost = mcm_dp(ch)
                    t0 = time.perf_counter()
                    g_sample = precompute_graph({'input': ch, 'output': dp_cost})
                    nf, ei, bi, _, _, _ = collate_gnn_batch([g_sample])
                    nf, ei = nf.to(device), ei.to(device)
                    bi['actual_lengths'] = bi['actual_lengths'].to(device)
                    bi['root_indices'] = bi['root_indices'].to(device)
                    for L in bi['split_parent_idx']:
                        bi['split_parent_idx'][L] = bi['split_parent_idx'][L].to(device)
                        bi['split_left_idx'][L] = bi['split_left_idx'][L].to(device)
                        bi['split_right_idx'][L] = bi['split_right_idx'][L].to(device)
                        bi['split_valid'][L] = bi['split_valid'][L].to(device)
                    with torch.no_grad():
                        gnn_model.predict(nf, ei, bi)
                    times.append((time.perf_counter() - t0) * 1000)
                gnn_mean, gnn_std = np.mean(times), np.std(times)
                results[f'GNN_n{n}'] = {'mean_ms': float(gnn_mean), 'std_ms': float(gnn_std)}
                print(f"  GNN         : {gnn_mean:.3f} ± {gnn_std:.3f} ms")
            else:
                print("  GNN         : [no checkpoint]")
        except Exception as e:
            print(f"  GNN         : [error: {e}]")

        # Pointer Network
        try:
            ptr_model = PointerMCMNet(input_dim=8, d_model=128).to(device)
            if os.path.exists("models/pointer_best.pth"):
                ptr_model.load_state_dict(torch.load("models/pointer_best.pth",
                                                      map_location=device, weights_only=True))
                ptr_model.eval()

                max_len = 51
                # warmup
                for _ in range(min(warmup_runs, 3)):
                    p_feats = extract_pointer_features(chains[0])
                    padded, mask = pad_features(p_feats, max_len)
                    with torch.no_grad():
                        ptr_model.predict(
                            torch.FloatTensor(padded).unsqueeze(0).to(device),
                            torch.BoolTensor(mask).unsqueeze(0).to(device),
                            torch.LongTensor([len(chains[0]) - 1]).to(device))

                times = []
                for ch in chains[:measure_runs]:
                    t0 = time.perf_counter()
                    p_feats = extract_pointer_features(ch)
                    padded, mask = pad_features(p_feats, max_len)
                    with torch.no_grad():
                        ptr_model.predict(
                            torch.FloatTensor(padded).unsqueeze(0).to(device),
                            torch.BoolTensor(mask).unsqueeze(0).to(device),
                            torch.LongTensor([len(ch) - 1]).to(device))
                    times.append((time.perf_counter() - t0) * 1000)
                ptr_mean, ptr_std = np.mean(times), np.std(times)
                results[f'Pointer_n{n}'] = {'mean_ms': float(ptr_mean), 'std_ms': float(ptr_std)}
                print(f"  Pointer     : {ptr_mean:.3f} ± {ptr_std:.3f} ms")
            else:
                print("  Pointer     : [no checkpoint]")
        except Exception as e:
            print(f"  Pointer     : [error: {e}]")

        # XGBoost
        try:
            import xgboost as xgb
            from src.training.train_trees import predict_xgb_ensemble
            if os.path.exists("models/xgb_direct_v4.json"):
                xgb_models = {
                    'direct': xgb.Booster(model_file="models/xgb_direct_v4.json"),
                    'ratio':  xgb.Booster(model_file="models/xgb_ratio_v4.json"),
                }
                # warmup
                for _ in range(warmup_runs):
                    feat = np.array(extract_features_v4(chains[0])).reshape(1, -1)
                    g_min = min(func(chains[0]) for func in [greedy_cost_left_to_right,
                                greedy_cost_right_to_left, greedy_cost_min_first,
                                greedy_cost_balanced])
                    predict_xgb_ensemble(xgb_models, feat, np.array([g_min]))

                times = []
                for ch in chains[:measure_runs]:
                    t0 = time.perf_counter()
                    feat = np.array(extract_features_v4(ch)).reshape(1, -1)
                    g_min = min(func(ch) for func in [greedy_cost_left_to_right,
                                greedy_cost_right_to_left, greedy_cost_min_first,
                                greedy_cost_balanced])
                    predict_xgb_ensemble(xgb_models, feat, np.array([g_min]))
                    times.append((time.perf_counter() - t0) * 1000)
                xgb_mean, xgb_std = np.mean(times), np.std(times)
                results[f'XGBoost_n{n}'] = {'mean_ms': float(xgb_mean), 'std_ms': float(xgb_std)}
                print(f"  XGBoost     : {xgb_mean:.3f} ± {xgb_std:.3f} ms")
            else:
                print("  XGBoost     : [no checkpoint]")
        except Exception as e:
            print(f"  XGBoost     : [error: {e}]")

    return results


# ════════════════════════════════════════════════════════════
#  EXPERIMENT 3: Bootstrap confidence intervals
# ════════════════════════════════════════════════════════════
def run_bootstrap_ci(chains, n_bootstrap=10000, seed=42):
    print("\n" + "=" * 60)
    print("  EXPERIMENT 3: Bootstrap 95% CIs for MAPE")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.RandomState(seed)

    # Collect per-chain errors for all models
    model_errors = {}

    # DP ground truth
    dp_costs = {}
    greedy_mins = {}
    for c in chains:
        dp_costs[id(c)] = mcm_dp(c['dims'])
        greedy_mins[id(c)] = min(
            greedy_cost_left_to_right(c['dims']),
            greedy_cost_right_to_left(c['dims']),
            greedy_cost_min_first(c['dims']),
            greedy_cost_balanced(c['dims']))

    # GNN errors
    try:
        from src.models.gnn_mcm import GraphMCMNet
        from src.data.gnn_loader import precompute_graph, collate_gnn_batch

        gnn_model = GraphMCMNet(d_model=128, num_layers=6, dropout=0.1, max_n=50).to(device)
        gnn_path = None
        for p in ["models/gnn_best.pth", "models/gnn_checkpoint.pth"]:
            if os.path.exists(p):
                gnn_path = p
                break
        if gnn_path:
            ckpt = torch.load(gnn_path, map_location=device, weights_only=True)
            if 'model_state_dict' in ckpt:
                gnn_model.load_state_dict(ckpt['model_state_dict'])
            else:
                gnn_model.load_state_dict(ckpt)
            gnn_model.eval()

            errs = []
            for c in chains:
                dp_cost = dp_costs[id(c)]
                g_sample = precompute_graph({'input': c['dims'], 'output': dp_cost})
                nf, ei, bi, _, _, _ = collate_gnn_batch([g_sample])
                nf, ei = nf.to(device), ei.to(device)
                bi['actual_lengths'] = bi['actual_lengths'].to(device)
                bi['root_indices'] = bi['root_indices'].to(device)
                for L in bi['split_parent_idx']:
                    bi['split_parent_idx'][L] = bi['split_parent_idx'][L].to(device)
                    bi['split_left_idx'][L] = bi['split_left_idx'][L].to(device)
                    bi['split_right_idx'][L] = bi['split_right_idx'][L].to(device)
                    bi['split_valid'][L] = bi['split_valid'][L].to(device)
                with torch.no_grad():
                    pred_splits, _ = gnn_model.predict(nf, ei, bi)
                    cost = compute_cost_from_splits(c['dims'], pred_splits[0])
                errs.append(abs(cost - dp_cost) / (dp_cost + 1e-9) * 100)
            model_errors['GNN'] = np.array(errs)
            print(f"  GNN per-chain errors collected: {len(errs)} chains")
    except Exception as e:
        print(f"  GNN: [error: {e}]")

    # Pointer errors
    try:
        ptr_model = PointerMCMNet(input_dim=8, d_model=128).to(device)
        if os.path.exists("models/pointer_best.pth"):
            ptr_model.load_state_dict(torch.load("models/pointer_best.pth",
                                                  map_location=device, weights_only=True))
            ptr_model.eval()
            errs = []
            max_len = 51
            for c in chains:
                dp_cost = dp_costs[id(c)]
                p_feats = extract_pointer_features(c['dims'])
                padded, mask = pad_features(p_feats, max_len)
                with torch.no_grad():
                    pred_splits, _ = ptr_model.predict(
                        torch.FloatTensor(padded).unsqueeze(0).to(device),
                        torch.BoolTensor(mask).unsqueeze(0).to(device),
                        torch.LongTensor([len(c['dims']) - 1]).to(device))
                    cost = compute_cost_from_splits(c['dims'], pred_splits[0])
                errs.append(abs(cost - dp_cost) / (dp_cost + 1e-9) * 100)
            model_errors['Pointer'] = np.array(errs)
            print(f"  Pointer per-chain errors collected: {len(errs)} chains")
    except Exception as e:
        print(f"  Pointer: [error: {e}]")

    # XGBoost errors
    try:
        import xgboost as xgb
        from src.training.train_trees import predict_xgb_ensemble
        if os.path.exists("models/xgb_direct_v4.json"):
            xgb_models = {
                'direct': xgb.Booster(model_file="models/xgb_direct_v4.json"),
                'ratio':  xgb.Booster(model_file="models/xgb_ratio_v4.json"),
            }
            errs = []
            for c in chains:
                dp_cost = dp_costs[id(c)]
                feat = np.array(extract_features_v4(c['dims'])).reshape(1, -1)
                cost = predict_xgb_ensemble(xgb_models, feat, np.array([greedy_mins[id(c)]]))[0]
                errs.append(abs(cost - dp_cost) / (dp_cost + 1e-9) * 100)
            model_errors['XGBoost'] = np.array(errs)
            print(f"  XGBoost per-chain errors collected: {len(errs)} chains")
    except Exception as e:
        print(f"  XGBoost: [error: {e}]")

    # RF errors
    try:
        import joblib
        from src.training.train_trees import predict_rf_ensemble
        if os.path.exists("models/rf_ensemble_v4.joblib"):
            rf_models = joblib.load("models/rf_ensemble_v4.joblib")
            errs = []
            for c in chains:
                dp_cost = dp_costs[id(c)]
                feat = np.array(extract_features_v4(c['dims'])).reshape(1, -1)
                cost = predict_rf_ensemble(rf_models, feat, np.array([greedy_mins[id(c)]]))[0]
                errs.append(abs(cost - dp_cost) / (dp_cost + 1e-9) * 100)
            model_errors['RF'] = np.array(errs)
            print(f"  RF per-chain errors collected: {len(errs)} chains")
    except Exception as e:
        print(f"  RF: [error: {e}]")

    # Bootstrap
    results = {}
    for name, errs in model_errors.items():
        boot_mapes = []
        for _ in range(n_bootstrap):
            idx = rng.randint(0, len(errs), size=len(errs))
            boot_mapes.append(np.mean(errs[idx]))
        boot_mapes = np.array(boot_mapes)
        lo, hi = np.percentile(boot_mapes, [2.5, 97.5])
        results[name] = {
            'mape': float(np.mean(errs)),
            'ci_lo': float(lo),
            'ci_hi': float(hi),
        }
        print(f"  {name:<12} MAPE: {np.mean(errs):.4f}% [{lo:.4f}, {hi:.4f}]")

    return results


# ════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("Generating 500-chain test set (seed=42)...")
    chains = generate_test_chains()
    print(f"  Generated {len(chains)} chains")

    all_results = {}

    # Exp 1
    all_results['greedy_baselines'] = run_greedy_baselines(chains)

    # Exp 2
    all_results['latency'] = run_latency_measurements()

    # Exp 3
    all_results['bootstrap_ci'] = run_bootstrap_ci(chains)

    # Save
    out_path = os.path.join('scratch', 'reviewer_experiment_results.json')
    os.makedirs('scratch', exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n{'=' * 60}")
    print(f"  All results saved to {out_path}")
    print(f"{'=' * 60}")
