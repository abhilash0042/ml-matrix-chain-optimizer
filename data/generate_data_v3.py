"""
MCM Dataset Generator v3 — Research-Level
==========================================
Generates 100,000 samples with:
- Stratified chain lengths (balanced short/medium/long/very-long)
- Better-balanced pattern distribution (more hard cases)
- 7 pattern types including new "mixed/hybrid" chains
- Exact DP labels (ground truth)

Author: Research rebuild for <5% MAPE target
"""

import json
import random
import numpy as np
import os
import time
import sys


def mcm_dp(p):
    """Exact DP solution for Matrix Chain Multiplication. O(n^3)."""
    n = len(p) - 1
    m = [[0] * (n + 1) for _ in range(n + 1)]
    for length in range(2, n + 1):
        for i in range(1, n - length + 2):
            j = i + length - 1
            m[i][j] = float('inf')
            for k in range(i, j):
                q = m[i][k] + m[k + 1][j] + p[i - 1] * p[k] * p[j]
                if q < m[i][j]:
                    m[i][j] = q
    return m[1][n]


# ─── Chain Pattern Generators ───────────────────────────────────────────

def generate_random_chain(n, min_dim=1, max_dim=500):
    """Fully random dimensions."""
    return [random.randint(min_dim, max_dim) for _ in range(n + 1)]


def generate_increasing_chain(n, min_dim=1, max_dim=500):
    """Monotonically increasing dimensions."""
    start = random.randint(min_dim, max_dim // 3)
    end = random.randint(max_dim // 2, max_dim)
    dims = np.linspace(start, end, n + 1).astype(int).tolist()
    return [max(1, d) for d in dims]


def generate_decreasing_chain(n, min_dim=1, max_dim=500):
    """Monotonically decreasing dimensions."""
    start = random.randint(max_dim // 2, max_dim)
    end = random.randint(min_dim, max_dim // 3)
    dims = np.linspace(start, end, n + 1).astype(int).tolist()
    return [max(1, d) for d in dims]


def generate_bottleneck_chain(n, min_dim=1, max_dim=500):
    """Has very small dimensions creating bottlenecks."""
    dims = [random.randint(50, max_dim) for _ in range(n + 1)]
    num_bottlenecks = random.randint(1, min(3, n))
    for _ in range(num_bottlenecks):
        idx = random.randint(0, n)
        dims[idx] = random.randint(1, 10)
    return dims


def generate_uniform_chain(n, min_dim=1, max_dim=500):
    """All dimensions are similar (near-uniform)."""
    base = random.randint(min_dim, max_dim)
    noise = max(1, base // 10)
    return [max(1, base + random.randint(-noise, noise)) for _ in range(n + 1)]


def generate_spiky_chain(n, min_dim=1, max_dim=500):
    """Alternating small and large dimensions."""
    small = random.randint(1, 20)
    large = random.randint(200, max_dim)
    dims = []
    for i in range(n + 1):
        if i % 2 == 0:
            dims.append(large + random.randint(-20, 20))
        else:
            dims.append(small + random.randint(-min(small - 1, 5), 5))
    return [max(1, d) for d in dims]


def generate_mixed_chain(n, min_dim=1, max_dim=500):
    """
    Mixed/Hybrid chain: combines two different patterns.
    This creates realistic scenarios that don't fit neatly into one category.
    """
    mid = (n + 1) // 2
    pattern = random.choice(['inc_bot', 'dec_spike', 'rand_uniform', 'spike_dec'])

    if pattern == 'inc_bot':
        left = generate_increasing_chain(mid, min_dim, max_dim)
        right = generate_bottleneck_chain(n - mid, min_dim, max_dim)
    elif pattern == 'dec_spike':
        left = generate_decreasing_chain(mid, min_dim, max_dim)
        right = generate_spiky_chain(n - mid, min_dim, max_dim)
    elif pattern == 'rand_uniform':
        left = generate_random_chain(mid, min_dim, max_dim)
        right = generate_uniform_chain(n - mid, min_dim, max_dim)
    else:  # spike_dec
        left = generate_spiky_chain(mid, min_dim, max_dim)
        right = generate_decreasing_chain(n - mid, min_dim, max_dim)

    # Merge at boundary (skip duplicate dimension at junction)
    return left + right[1:]


def generate_tiny_chain(n, min_dim=1, max_dim=20):
    """Very small dimensions — teaches model precision at low values."""
    return [random.randint(min_dim, max_dim) for _ in range(n + 1)]


def get_stratified_chain_length():
    """
    Stratified sampling for chain lengths.
    Ensures balanced representation across difficulty levels.
    """
    r = random.random()
    if r < 0.25:       # 25% short
        return random.randint(3, 10)
    elif r < 0.60:     # 35% medium
        return random.randint(11, 25)
    elif r < 0.85:     # 25% long
        return random.randint(26, 40)
    else:              # 15% very long
        return random.randint(41, 50)


def generate_dataset_v3(num_samples=120000, min_dim=1, max_dim=1000):
    """
    Generate research-level MCM dataset with balanced patterns.

    Distribution:
    - Random:     25%
    - Bottleneck: 18%
    - Spiky:      18%
    - Tiny:       13%  (NEW)
    - Increasing:  8%
    - Decreasing:  8%
    - Uniform:     5%
    - Mixed:       5%
    """
    dataset = []

    # Weighted generator selection
    generators = [
        (generate_random_chain, 25),
        (generate_bottleneck_chain, 18),
        (generate_spiky_chain, 18),
        (generate_tiny_chain, 13),
        (generate_increasing_chain, 8),
        (generate_decreasing_chain, 8),
        (generate_uniform_chain, 5),
        (generate_mixed_chain, 5),
    ]

    gen_pool = []
    for gen, weight in generators:
        gen_pool.extend([gen] * weight)

    print(f"Generating {num_samples:,} research-level MCM samples...")
    print(f"  Dims: {min_dim}–{max_dim}, Chains: 3–50 (stratified)")
    print(f"  Patterns: Random(25%) Bottleneck(18%) Spiky(18%) Tiny(13%) "
          f"Inc(8%) Dec(8%) Uniform(5%) Mixed(5%)")
    print()

    t0 = time.time()

    for i in range(num_samples):
        n = get_stratified_chain_length()
        gen = random.choice(gen_pool)
        dims = gen(n, min_dim, max_dim)

        cost = mcm_dp(dims)
        dataset.append({
            "input": dims,
            "output": float(cost)
        })

        if (i + 1) % 5000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (num_samples - i - 1) / rate
            print(f"  [{i + 1:>7,}/{num_samples:,}] "
                  f"{elapsed:>6.0f}s elapsed, ~{eta:.0f}s remaining "
                  f"({rate:.0f} samples/s)")

    # Save
    out_path = os.path.join(os.path.dirname(__file__), f'mcm_{num_samples}.json')
    with open(out_path, "w") as f:
        json.dump(dataset, f)

    total = time.time() - t0
    costs = [s["output"] for s in dataset]
    lengths = [len(s["input"]) - 1 for s in dataset]

    print(f"\n{'=' * 60}")
    print(f"Dataset saved: {out_path}")
    print(f"  Samples:     {len(dataset):,}")
    print(f"  Total time:  {total:.0f}s")
    print(f"  Cost range:  {min(costs):,.0f} → {max(costs):,.0f}")
    print(f"  Median cost: {np.median(costs):,.0f}")
    print(f"  Chain sizes: {min(lengths)}–{max(lengths)} "
          f"(mean={np.mean(lengths):.1f})")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    n = 120000
    if len(sys.argv) > 1:
        n = int(sys.argv[1])
    generate_dataset_v3(num_samples=n)
