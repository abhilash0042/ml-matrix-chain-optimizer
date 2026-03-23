import json
import numpy as np

def load_data():
    with open("data/mcm_10000.json") as f:
        data = json.load(f)

    X = []
    y = []

    for item in data:
        dims = item["input"]   # ✅ FIXED

        X.append([
            len(dims),
            min(dims),
            max(dims),
            max(dims) - min(dims),
            sum(dims) / len(dims),
            np.std(dims),
            np.log(len(dims)),
            np.log(max(dims))
        ])

        y.append(item["output"])   # ✅ target

    return np.array(X), np.array(y)