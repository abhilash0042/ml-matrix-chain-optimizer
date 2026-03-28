"""
Generate MCM Training Dataset v2
=================================
Generates 50,000 samples with:
- Chain lengths: 3 to 50 matrices
- Dimensions: 1 to 500 (wider range for harder cases)
- Balanced distribution across chain lengths
- Mix of patterns: random, increasing, decreasing, bottleneck, uniform, spiky
"""

import json
import random
import numpy as np
import os
import time

def mcm_dp(p):
    """Exact DP solution for Matrix Chain Multiplication."""
    n = len(p) - 1
    m = [[0] * (n + 1) for _ in range(n + 1)]
    for length in range(2, n + 1):
        for i in range(1, n - length + 2):
            j = i + length - 1
            m[i][j] = float('inf')
            for k in range(i, j):
                q = m[i][k] + m[k+1][j] + p[i-1] * p[k] * p[j]
                if q < m[i][j]:
                    m[i][j] = q
    return m[1][n]

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
    """Has a very small dimension creating a bottleneck."""
    dims = [random.randint(50, max_dim) for _ in range(n + 1)]
    # Insert 1-3 bottleneck points (very small dims)
    num_bottlenecks = random.randint(1, min(3, n))
    for _ in range(num_bottlenecks):
        idx = random.randint(0, n)
        dims[idx] = random.randint(1, 5)
    return dims

def generate_uniform_chain(n, min_dim=1, max_dim=500):
    """All dimensions are similar."""
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
            dims.append(small + random.randint(-min(small-1, 5), 5))
    return [max(1, d) for d in dims]

def generate_dataset(num_samples=50000, min_n=3, max_n=50, min_dim=1, max_dim=500):
    """Generate diverse MCM dataset."""
    dataset = []
    generators = [
        (generate_random_chain, 0.50),       # 50% random
        (generate_bottleneck_chain, 0.15),   # 15% bottleneck
        (generate_spiky_chain, 0.10),        # 10% spiky
        (generate_increasing_chain, 0.10),   # 10% increasing
        (generate_decreasing_chain, 0.10),   # 10% decreasing
        (generate_uniform_chain, 0.05),      # 5% uniform
    ]
    
    # Build weighted list of generators
    gen_choices = []
    for gen, weight in generators:
        gen_choices.extend([gen] * int(weight * 100))
    
    print(f"Generating {num_samples} samples (dims {min_dim}-{max_dim}, chains {min_n}-{max_n})...")
    t0 = time.time()
    
    for i in range(num_samples):
        # Balanced chain lengths
        n = random.randint(min_n, max_n)
        
        # Pick generator
        gen = random.choice(gen_choices)
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
            print(f"  Generated {i + 1:,}/{num_samples:,} samples "
                  f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")
    
    out_path = os.path.join(os.path.dirname(__file__), f'mcm_{num_samples}.json')
    with open(out_path, "w") as f:
        json.dump(dataset, f)
    
    total = time.time() - t0
    costs = [s["output"] for s in dataset]
    print(f"\nDataset saved to {out_path}")
    print(f"  Total time: {total:.0f}s")
    print(f"  Cost range: {min(costs):,.0f} to {max(costs):,.0f}")
    print(f"  Median cost: {np.median(costs):,.0f}")

if __name__ == "__main__":
    generate_dataset(num_samples=50000, min_n=3, max_n=50, min_dim=1, max_dim=500)
