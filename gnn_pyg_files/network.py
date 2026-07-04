from Cone import Cone
from CGLGraph import CGLGraph
from utils import validActionMask, _batch_vectors

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, GATv2Conv, GraphConv
from torch_geometric.utils import softmax


EMBEDDING_SIZE = 64 ## default embedding length

## Our GNN uses GATv2Conv layers + ReLU activation. 
class GAT(nn.Module):
    def __init__(self, hidden_channels: int = 128, out_channels = EMBEDDING_SIZE, num_layers = 4, dropout: float = 0.05):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList(nn.LayerNorm(hidden_channels) for _ in range(num_layers - 1))

        for i in range(num_layers):
            in_num = (-1, -1)
            out_num = hidden_channels
            has_residual = True
            if i == 0: has_residual = False
            if i == num_layers - 1:
                out_num = out_channels
                has_residual = False

            ## layer for relation without edge attributes
            def plain():
                return GATv2Conv(in_num, out_num, dropout = dropout, add_self_loops = False, residual = has_residual)
            
            ## layer for relation with edge attributes
            def attr():
                return GATv2Conv(in_num, out_num, edge_dim = 1, dropout = dropout, add_self_loops = False, residual = has_residual)

            conv = HeteroConv({
                ('cone', 'has', 'generator'): plain(),
                ('generator', 'of', 'cone'): plain(),
                ('cone', 'contains', 'lattice'): plain(),
                ('lattice', 'in', 'cone'): plain(),
                ('generator', 'reaches', 'lattice'): attr(),
                ('lattice', 'reached_by', 'generator'): attr(),
                ('cone', 'adjacent', 'cone'):  plain(),
                ('virtual', 'overview', 'cone'):  plain(),
                ('cone', 'summary', 'virtual'): plain(),
            }, aggr = 'sum')
            self.convs.append(conv)
        
    def forward(self, x_dict, edge_index_dict, edge_attr_dict = None):
        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict, edge_attr_dict = edge_attr_dict)
            if i < len(self.convs) - 1:
                x_dict = {k: F.relu(self.norms[i](v)) for k, v in x_dict.items()}
        return x_dict

## GraphConv GNN for A/B testing
class Graph(nn.Module):
    def __init__(self, hidden_channels: int = 128, out_channels = EMBEDDING_SIZE, num_layers = 4, dropout: float = 0.05):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList(nn.LayerNorm(hidden_channels) for _ in range(num_layers - 1))
        self.dropouts = nn.ModuleList(nn.Dropout(dropout) for _ in range(num_layers - 1))

        for i in range(num_layers):
            in_num = (-1, -1)
            out_num = hidden_channels
            if i == num_layers - 1: out_num = out_channels

            conv = HeteroConv({
                ('cone', 'has', 'generator'): GraphConv(in_num, out_num),
                ('generator', 'of', 'cone'): GraphConv(in_num, out_num),
                ('cone', 'contains', 'lattice'): GraphConv(in_num, out_num),
                ('lattice', 'in', 'cone'): GraphConv(in_num, out_num),
                ('generator', 'reaches', 'lattice'): GraphConv(in_num, out_num),
                ('lattice', 'reached_by', 'generator'): GraphConv(in_num, out_num),
                ('cone', 'adjacent', 'cone'):  GraphConv(in_num, out_num),
                ('virtual', 'overview', 'cone'):  GraphConv(in_num, out_num),
                ('cone', 'summary', 'virtual'): GraphConv(in_num, out_num),
            }, aggr = 'sum')
            self.convs.append(conv)

    def forward(self, x_dict, edge_index_dict, edge_weight_dict=None):
        for i, conv in enumerate(self.convs):
            residual = x_dict
            x_dict = conv(x_dict, edge_index_dict, edge_weight_dict)
            if i < len(self.convs) - 1:
                x_dict = {k: self.dropouts[i](F.relu(self.norms[i](v))) for k, v in x_dict.items()}
                if i != 0:
                    x_dict = {k: v + residual[k] for k, v in x_dict.items()}
        return x_dict

class valueHead(nn.Module): 
    """ Takes in the graph-level embedding vector and outputs a learned scalar valuation. 
    LayerNorm -> FC Layer -> ReLU -> FC Layer -> Scalar Output

    Shape: 
     - g: (EMBEDDING_SIZE,) graph-level embedding vector
     - returns: value (scalar)

    """

    def __init__(self, in_channels: int = EMBEDDING_SIZE, hidden_channels: int = 32):
        super().__init__()
        self.norm = nn.LayerNorm(in_channels)
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, 1)

    def forward(self, g):
        g = self.norm(g)
        g = F.relu(self.fc1(g))
        g = self.fc2(g)
        return g.squeeze(-1)

## outputs logits describing the policy. These logits can later be passed through softmax to produce a probability distribution over the nodes. 
## Concatenates node and graph embeddings (z_1, g), (z_2, g), ..., (z_n, g) then passes through an MLP one-by-one to get logit output. 
class policyHead(nn.Module): 
    def __init__(self, in_channels: int = 2 * EMBEDDING_SIZE, hidden_channels: int = 128):
        super().__init__()
        self.norm = nn.LayerNorm(in_channels)
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, 1)

    ## Z are the embeddings of the lattice-action nodes in the graph. g is the graph embedding. 
    ## assumes g has already been broadcasted to the correct shape to match Z. 
    def forward(self, Z, g):        
        x = torch.cat([Z, g], dim = -1)
        x = self.norm(x)
        x = F.relu(self.fc1(x))
        return self.fc2(x).squeeze(-1)

class network(nn.Module):
    def __init__(self, metadata, hidden: int = 128, embedding_size: int = EMBEDDING_SIZE, num_layers: int = 4, dropout: float = 0.05, layer_type: str = "GraphConv"):
        super().__init__()
        if layer_type == "GAT":
            self.gnn = GAT(hidden_channels = hidden, out_channels = embedding_size, num_layers = num_layers, dropout = dropout)
        elif layer_type == "GraphConv":
            self.gnn = Graph(hidden_channels = hidden, out_channels = embedding_size, num_layers = num_layers, dropout = dropout)
        else:
            raise ValueError("not accepted layer type")

        self.value_head = valueHead(in_channels = embedding_size)
        self.policy_head = policyHead(in_channels = 2 * embedding_size)


    ## forward pass, returning Dict{log_p, value, action_batch} where log_p is the log of action probabilities, value is the output from value head, 
    ## and action_batch are the batch numbers of the available actions. Helps the user interpret the other values it got out of the output. 
    ## requires global node to be added to each graph in data before a forward pass. 
    def forward(self, data) -> dict:
        edge_attr_dict = {
            ('generator', 'reaches', 'lattice'):    data['generator','reaches','lattice'].edge_attr,
            ('lattice', 'reached_by', 'generator'): data['lattice','reached_by','generator'].edge_attr,
        } #get relevant edge attributes
        x_dict = self.gnn(data.x_dict, data.edge_index_dict, edge_attr_dict) ## network output
        batch_dict, B = _batch_vectors(data) ## batch identification information, number of batches

        action_mask = validActionMask(data) ## boolean mask denoting which lattice points are still available actions. Shape (N_lattice,) where N_lattice is the number of lattice points in the whole graph
        action_batch = batch_dict['lattice'][action_mask] ## vector that denotes the batch number of each available action. Shape (A,) where A is the number of available actions.
 
        g = x_dict['virtual'] ## graph level embeddings, one per batch. Shape (B, EMBEDDING_SIZE) 
        g_per_action = g[action_batch] ## projects: for each available action, the row entry is the corresponding graph embedding for that action's graph. Shape (A, EMBEDDING_SIZE). 

        Z = x_dict['lattice'][action_mask] ## keep only the lattice nodes that are still available. Shape (A, EMBEDDING_SIZE). 
        logits = self.policy_head(Z, g_per_action)
        value = self.value_head(g)

        log_p = torch.log(softmax(logits, action_batch) + 1e-12) ## add small number to avoid -inf

        return {"log_p": log_p, "value": value, "action_batch": action_batch}

def main():
    torch.manual_seed(0)

    graph = CGLGraph()
    c1 = graph.addConeNode(Cone(((1, 0, 1), (0, 1, 1), (0, 0, 7))))  # mult 7, singular
    c2 = graph.addConeNode(Cone(((1, 0, 1), (0, 1, 1), (1, 1, 0))))
    graph.addAdjacentEdge(c1, c2)

    meta = graph.metadata()
    print("node types:", meta[0])          # expect ['cone','lattice','global']

    net = network(meta, hidden=128, embedding_size=EMBEDDING_SIZE, num_layers=4)
    net(graph)
    net.eval()

    with torch.inference_mode():
        out = net(graph)

    log_p, value, action_batch = out["log_p"], out["value"], out["action_batch"]
    A = action_batch.numel()
    mask = validActionMask(graph)

    print(f"valid actions A = {A}")
    print(f"log_p shape      {tuple(log_p.shape)}   expect ({A},)")
    print(f"value shape      {tuple(value.shape)}   expect (1,)")
    print(f"action_batch     {action_batch.tolist()}  (all 0 for a single graph)")

    # --- correctness assertions ---
    assert A > 0, "no valid actions: pick a singular cone with nonempty extraneous set"
    assert log_p.shape == (A,)
    assert action_batch.shape == (A,)
    assert mask.sum().item() == A, "mask / action_batch count mismatch"

    # policy is a valid distribution per graph: exp(log_p) sums to 1 within each batch group
    p = log_p.exp()
    for b in action_batch.unique():
        s = p[action_batch == b].sum().item()
        assert abs(s - 1.0) < 1e-5, f"group {int(b)} sums to {s}, not 1"

    print("OK: shapes, mask alignment, and per-graph normalization all pass.")


if __name__ == "__main__":
    main()