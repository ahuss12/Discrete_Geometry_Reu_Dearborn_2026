from utils import *
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, HeteroConv, RGATConv

padding_level = 7

class RGAT(nn.Module): 
    def __init__(self, in_channels = padding_level, hidden_channels = 64, out_channels = padding_level, 
                num_relations = 2, attention_mechanism = "within-relation", dropout = 0.3, num_layers = 4):

        super().__init__()
        self.convs = nn.ModuleList()  
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            in_num = hidden_channels
            out_num = hidden_channels

            if i == 0: in_num = in_channels
            elif i == num_layers-1: out_num = out_channels 

            conv = RGATConv(in_num, out_num, num_relations, attention_mechanism = attention_mechanism, 
                            dropout = dropout)
            self.convs.append(conv)

            if i < num_layers - 1: 
                norm = nn.LayerNorm(out_num)
                self.norms.append(norm)

    def forward(self, x, edge_index, edge_type):
        for i, conv in enumerate(self.convs):
            residual = x
            x = conv(x, edge_index, edge_type)
              if i < len(self.convs) -1: 
                x = self.norms[i](x)
                x = F.gelu(x)
            x = x + residual if i != 0 and i != len(self.convs)-1 else x
        return x

## gives an embedding of the entire graph based on the node-level embedding output from GNN
def graphEmbed(embeddings: Dict[str, Tensor]) -> Tensor:
    return sum(t.sum(dim = 0) for t in embeddings.values()) ## note: worse subdivisions will actually have a larger sum. take into consideration. 

## value head takes in graph-level embedding vector of shape (in_channels,) and outputs a scalar. 
class valueHead(nn.Module): 
    def __init__(self, in_channels: int = padding_level, hidden_channels: int = 32):
        super().__init__()
        self.norm = nn.LayerNorm(in_channels)
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, 1)

    def forward(self, x):
        x = self.norm(x)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x.squeeze(-1)

## outputs a probability distribution over the possible actions
class policyHead(nn.Module): 
    def __init__(self, in_channels: int = 2 * padding_level, hidden_channels: int = 128):
        super().__init__()
        self.norm = nn.LayerNorm(in_channels)
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, 1)

    ## Z are the embeddings of the lattice-action nodes in the graph. g is the graph embedding. 
    def forward(self, Z, g):
        x = torch.cat([Z, g.unsqueeze(0).expand(Z.size(0), -1)], dim = -1)
        x = self.norm(x)
        x = F.gelu(self.fc1(x))
        logits = self.fc2(x).squeeze(-1)
        return F.log_softmax(logits, dim = 0)
