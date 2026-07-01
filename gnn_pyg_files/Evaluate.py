import torch

ckpt = torch.load("cone_action_gnn.pt", map_location="cpu", weights_only=False)

