from utils import *
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, HeteroConv, RGATConv

padding_level = 10

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
        



        

