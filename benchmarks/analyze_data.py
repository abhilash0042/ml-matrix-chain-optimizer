import json
import collections
import numpy as np

d = json.load(open('data/mcm_50000.json'))
ns = [len(x['input'])-1 for x in d]
dims_lens = [len(x['input']) for x in d]
costs = [x['output'] for x in d]

print(f"Total samples: {len(d)}")
print(f"n (matrices) range: {min(ns)} to {max(ns)}")
print(f"Max dims length: {max(dims_lens)}")
print(f"Cost range: {min(costs):,.0f} to {max(costs):,.0f}")
print(f"Median cost: {np.median(costs):,.0f}")

# Check dimension value range
all_dims = []
for x in d:
    all_dims.extend(x['input'])
print(f"Dimension value range: {min(all_dims)} to {max(all_dims)}")

# Distribution by n
c = collections.Counter(ns)
print("\nDistribution by n:")
for n in sorted(c.keys()):
    print(f"  n={n}: {c[n]} samples")
