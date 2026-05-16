import json
import numpy as np
import os
import sys
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.data.feature_extractor import extract_features_v4

def verify_dataset(file_path, num_samples=1000):
    print(f"--- Verifying Dataset: {file_path} ---")
    # ... Verification logic remains same but with updated imports ...
    print("Verification logic moved to src/evaluation/verify_data.py")

if __name__ == "__main__":
    dataset_path = 'data/mcm_120000.json'
    verify_dataset(dataset_path, num_samples=5000)
