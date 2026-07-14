import torch
checkpoint = torch.load('models/gnn_best.pth', map_location='cpu', weights_only=True)
# best.pth usually only has the state_dict, but let's check
if isinstance(checkpoint, dict) and 'model_state_dict' not in checkpoint:
    print("Best model is a state_dict")
else:
    print(f"Stage: {checkpoint.get('stage')}")
    print(f"Epoch: {checkpoint.get('epoch')}")
