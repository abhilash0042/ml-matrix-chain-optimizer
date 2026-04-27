"""
Enrich existing MCM dataset with optimal split tables.
Reads mcm_120000.json and adds the full DP split table to each sample.

Output format per sample:
{
    "input": [40, 20, 30, 10, 30],
    "output": 26000,
    "n": 4,
    "splits": [[0,0,0,0,0], [0,0,1,1,3], [0,0,0,2,3], [0,0,0,0,3], [0,0,0,0,0]]
}

splits[i][j] = optimal split point k (1-indexed matrix index) for sub-chain (i,j).
Only valid for 1 <= i < j <= n. Other entries are 0.
"""

import json
import time
import sys
import os


def mcm_dp_with_splits(p):
    """
    DP solver returning both minimum cost AND the full split table.

    Args:
        p: list of n+1 dimension values (0-indexed)

    Returns:
        cost: minimum multiplication cost
        s: split table as list-of-lists (1-indexed), s[i][j] = optimal split k
    """
    n = len(p) - 1
    m = [[0] * (n + 1) for _ in range(n + 1)]
    s = [[0] * (n + 1) for _ in range(n + 1)]

    for length in range(2, n + 1):
        for i in range(1, n - length + 2):
            j = i + length - 1
            m[i][j] = float('inf')
            for k in range(i, j):
                q = m[i][k] + m[k + 1][j] + p[i - 1] * p[k] * p[j]
                if q < m[i][j]:
                    m[i][j] = q
                    s[i][j] = k

    return m[1][n], s


def enrich_dataset(input_path, output_path):
    """Add split tables to an existing MCM dataset."""
    print(f"Loading dataset from {input_path}...")
    with open(input_path, 'r') as f:
        data = json.load(f)

    print(f"Enriching {len(data):,} samples with split tables...")
    t0 = time.time()

    for i, sample in enumerate(data):
        dims = sample['input']
        n = len(dims) - 1
        cost, s = mcm_dp_with_splits(dims)

        # Verify cost matches
        assert abs(cost - sample['output']) < 1, \
            f"Sample {i}: cost mismatch {cost} vs {sample['output']}"

        sample['n'] = n
        sample['splits'] = s  # (n+1) x (n+1) list-of-lists

        if (i + 1) % 10000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(data) - i - 1) / rate
            print(f"  [{i + 1:>7,}/{len(data):,}] "
                  f"{elapsed:>6.0f}s elapsed, ~{eta:.0f}s remaining "
                  f"({rate:.0f} samples/s)")

    # Save enriched dataset
    print(f"Saving enriched dataset to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(data, f)

    total = time.time() - t0
    print(f"Done in {total:.0f}s. Enriched {len(data):,} samples.")


if __name__ == '__main__':
    input_file = os.path.join(os.path.dirname(__file__), 'mcm_120000.json')
    output_file = os.path.join(os.path.dirname(__file__), 'mcm_120000_splits.json')

    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]

    enrich_dataset(input_file, output_file)
