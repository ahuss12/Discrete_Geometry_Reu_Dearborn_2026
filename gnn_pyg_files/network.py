from coneEnvironment import *
from graph import *

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, GraphConv, to_hetero
from torch_geometric.utils import softmax


EMBEDDING_SIZE = 7 ## what length we want our embedding vectors (for nodes) to be. 

## Our GNN uses GraphConv layers + GeLU activation. 
class GNN(nn.Module):
    """ The main body of our GNN. Produces node-level embeddings through multiple layers of GraphConv with GeLU activation. 
    Maps each node to a vector of size (1, EMBEDDING_SIZE). Layers progress GraphConv -> LayerNorm -> GeLU -> Dropout, except for the final layer. 

    NOTE: this is written as a homogenous module. The intended usage is for it to be passed through to_hetero() to convert to a heterogenous module. 

    Shape (homogeneous form):
        - x:          (N, feature_size)   N is the number of nodes
        - edge_index: (2, E)    E is the number of (directed) edges
        - returns:    (N, EMBEDDING_SIZE)   

    Shape (after to_hetero, intended usage):
        - x_dict:          {node_type: (N_t, F_t)}
        - edge_index_dict: {(src, rel, dst): (2, E_r)}
        - returns:         {node_type: (N_t, out_channels)}
    """

    def __init__(self, hidden_channels = 64, out_channels = EMBEDDING_SIZE, num_layers = 4, dropout: float = 0.1):
        """
        Args: 
            hidden_channels: hidden width of GraphConv layers
            out_channels: width of embedding
            num_layers: number of GraphConv layers
            dropout: dropout percentage
        """

        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList(nn.LayerNorm(hidden_channels) for _ in range(num_layers - 1))
        self.dropouts = nn.ModuleList(nn.Dropout(dropout) for _ in range(num_layers - 1))

        for i in range(num_layers):
            in_num = (-1, -1)
            out_num = hidden_channels

            if i == num_layers - 1: out_num = out_channels

            conv = GraphConv(in_num, out_num)
            self.convs.append(conv)
        
    def forward(self, x, edge_index):
        """
        Args: 
            - x:          (N, feature_size)  node features
            - edge_index: (2, E)   edges

        Returns:
            - (N, EMBEDDING_SIZE)   node embeddings
        """
        for i, conv in enumerate(self.convs):
            residual = x
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.gelu(self.norms[i](x))
                x = self.dropouts[i](x)
                if i != 0: 
                    x = x + residual
        return x

class valueHead(nn.Module): 
    """ Takes in the graph-level embedding vector and outputs a learned scalar valuation. 
    LayerNorm -> FC Layer -> GeLU -> FC Layer -> Scalar Output

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
        g = F.gelu(self.fc1(g))
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
        x = F.gelu(self.fc1(x))
        return self.fc2(x).squeeze(-1)

## provides a boolean (Tensor) mask of valid actions for the current state.
def validActionMask(data) -> torch.Tensor:
    ei = data['lattice', 'contains', 'cone'].edge_index
    n = data['lattice'].num_nodes
    mask = torch.zeros(n, dtype=torch.bool, device=data['lattice'].x.device)
    mask[ei[0]] = True
    return mask

## returns a dictionary with key = node type, value = vector of length (# nodes of that type) where the entries are the batch that node belongs to. 
## also returns B, the number of batches.
def _batch_vectors(data) -> tuple[dict, int]:
    batch_dict, B = {}, 1
    for node_type in data.x_dict:
        store = data[node_type]
        b = getattr(store, "batch", None) ## vector with batch number of each node of node_type

        ## default to batch = 0 if no batch attribute
        if b is None:
            b = torch.zeros(store.num_nodes, dtype = torch.long, device = store.x.device)
        
        batch_dict[node_type] = b

        if b.numel():
            B = max(B, int(b.max().item()) + 1) ## cautious maximum, sets B to the maximum observed batch number so far

    return batch_dict, B

class network(nn.Module):
    def __init__(self, metadata, hidden: int = 64, embedding_size: int = EMBEDDING_SIZE, num_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        body = GNN(hidden_channels = hidden, out_channels = embedding_size, num_layers = num_layers, dropout = dropout)
        self.gnn = to_hetero(body, metadata, aggr = "sum")
        self.value_head = valueHead(in_channels = embedding_size)
        self.policy_head = policyHead(in_channels = 2 * embedding_size)


    ## forward pass, returning Dict{log_p, value, action_batch} where log_p is the log of action probabilities, value is the output from value head, 
    ## and action_batch are the batch numbers of the available actions. Helps the user interpret the other values it got out of the output. 
    ## requires global node to be added to each graph in data before a forward pass. 
    def forward(self, data) -> dict:
        x_dict = self.gnn(data.x_dict, data.edge_index_dict) ## network output
        batch_dict, B = _batch_vectors(data) ## batch identification information, number of batches

        action_mask = validActionMask(data) ## boolean mask denoting which lattice points are still available actions. Shape (N_lattice,) where N_lattice is the number of lattice points in the whole graph
        action_batch = batch_dict['lattice'][action_mask] ## vector that denotes the batch number of each available action. Shape (A,) where A is the number of available actions.
 
        g = x_dict['graph'] ## graph level embeddings, one per batch. Shape (B, EMBEDDING_SIZE) 
        g_per_action = g[action_batch] ## projects: for each available action, the row entry is the corresponding graph embedding for that action's graph. Shape (A, EMBEDDING_SIZE). 

        Z = x_dict['lattice'][action_mask] ## keep only the lattice nodes that are still available. Shape (A, EMBEDDING_SIZE). 
        logits = self.policy_head(Z, g_per_action)
        value = self.value_head(g)

        log_p = torch.log(softmax(logits, action_batch) + 1e-12) ## add small number to avoid -inf

        return {"log_p": log_p, "value": value, "action_batch": action_batch}

def main():
    torch.manual_seed(0)

    graph = ConeLatticeGraph(dimension=3)
    c1 = graph.addConeNode(Cone(((1, 0, 1), (0, 1, 1), (0, 0, 7))))  # mult 7, singular
    c2 = graph.addConeNode(Cone(((1, 0, 1), (0, 1, 1), (1, 1, 0))))
    graph.addAdjacentEdge(c1, c2)

    # global node must exist BEFORE we read metadata, so to_hetero builds its branch
    graph.addGlobalNode(n=EMBEDDING_SIZE)
    meta = graph.metadata()
    print("node types:", meta[0])          # expect ['cone','lattice','global']

    net = network(meta, hidden=64, embedding_size=EMBEDDING_SIZE, num_layers=4)
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