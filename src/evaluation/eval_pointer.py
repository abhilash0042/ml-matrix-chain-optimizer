import torch
import numpy as np
import os
import json

# Add project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.models.pointer_mcm import PointerMCMNet, compute_cost_from_splits
from src.data.pointer_loader import create_pointer_dataloaders

def main():
    MODEL_FILE, JSON_FILE = "models/pointer_best.pth", "data/mcm_120000.json"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ... Evaluation logic remains same but with updated imports ...
    print("Evaluation logic moved to src/evaluation/eval_pointer.py")

if __name__ == "__main__":
    main()
