import torch
checkpoint = torch.load('models/gnn_checkpoint.pth', map_location='cpu', weights_only=False)
print(f"Stage: {checkpoint.get('stage')}")
print(f"Epoch: {checkpoint.get('epoch')}")
print(f"Best Val Accuracy: {checkpoint.get('best_val_accuracy')}")
