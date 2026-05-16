import random
import numpy as np
import json
import os

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

def greedy_cost_left_to_right(dims):
    if len(dims) <= 2: return 0
    total_cost = 0
    current_rows, current_cols = dims[0], dims[1]
    for i in range(2, len(dims)):
        next_cols = dims[i]
        total_cost += current_rows * current_cols * next_cols
        current_cols = next_cols
    return total_cost

def greedy_cost_right_to_left(dims):
    if len(dims) <= 2: return 0
    total_cost = 0
    current_rows, current_cols = dims[-2], dims[-1]
    for i in range(len(dims) - 3, -1, -1):
        next_rows = dims[i]
        total_cost += next_rows * current_rows * current_cols
        current_rows = next_rows
    return total_cost

def greedy_cost_min_first(dims):
    if len(dims) <= 2: return 0
    d = list(dims)
    total_cost = 0
    while len(d) > 2:
        best_k = 1
        best_cost = d[0] * d[1] * d[2]
        for k in range(1, len(d) - 1):
            c = d[k - 1] * d[k] * d[k + 1]
            if d[k] < d[best_k] or (d[k] == d[best_k] and c < best_cost):
                best_k = k
                best_cost = c
        total_cost += best_cost
        d.pop(best_k)
    return total_cost

def greedy_cost_balanced(dims):
    n = len(dims) - 1
    if n <= 0: return 0
    memo = {}
    def _solve(i, j):
        if i == j: return 0
        if (i, j) in memo: return memo[(i, j)]
        k = (i + j) // 2
        cost = _solve(i, k) + _solve(k + 1, j) + dims[i - 1] * dims[k] * dims[j]
        memo[(i, j)] = cost
        return cost
    return _solve(1, n)

# --- Pattern Generators (consolidated from generate_data_v3.py) ---

def generate_random_chain(n, min_dim=1, max_dim=500):
    return [random.randint(min_dim, max_dim) for _ in range(n + 1)]

def generate_bottleneck_chain(n, min_dim=1, max_dim=500):
    dims = [random.randint(50, max_dim) for _ in range(n + 1)]
    for _ in range(random.randint(1, min(3, n))):
        dims[random.randint(0, n)] = random.randint(1, 10)
    return dims

def generate_spiky_chain(n, min_dim=1, max_dim=500):
    small, large = random.randint(1, 20), random.randint(200, max_dim)
    return [max(1, (large if i % 2 == 0 else small) + random.randint(-5, 5)) for i in range(n + 1)]

def generate_increasing_chain(n, min_dim=1, max_dim=500):
    start, end = random.randint(min_dim, max_dim // 3), random.randint(max_dim // 2, max_dim)
    return [max(1, d) for d in np.linspace(start, end, n + 1).astype(int).tolist()]
