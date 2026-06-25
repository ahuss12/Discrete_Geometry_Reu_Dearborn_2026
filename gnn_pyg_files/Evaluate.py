#will be used for comparison against greedy methods like det min sum, random etc
import torch

ckpt = torch.load("cone_action_gnn.pt", map_location="cpu", weights_only=False)

print(type(ckpt))        # dict
print(ckpt.keys())       # dict_keys(['model', 'args'])
print(ckpt["args"])      # the argparse config you trained with

sd = ckpt["model"]       # the state_dict: OrderedDict[str, Tensor]
for name, tensor in sd.items():
    print(name, tuple(tensor.shape))