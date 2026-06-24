## PRIORS IS NOW P! 

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
## from .geometry import CandidateEnumerator, ConeState, GlobalAction, apply_action #geometry.py
## from .graph import build_pyg_graph #graph.py
from utils import Cone, ConeLatticeGraph

latticeId = int

## Node stores (P,N,W,Q) where P is the prior probability from the GNN, N is the current visit count to the node, W is the total observed action-value, 
## and Q is the mean action-value. 
@dataclass
class MCTSNode:
    state: ConeLatticeGraph ## changed to utils.py graph held as state
    prev_action: Optional[latticeId] = None ## latticeId subdivided through to reach this state
    expanded: bool = False
    actions: List[latticeId] = field(default_factory=list) ## index -> latticeId
    P: np.ndarray = field(default_factory = lambda: np.zeros((0,), dtype = np.float64))
    N: np.ndarray = field(default_factory = lambda: np.zeros((0,), dtype = np.int64)) 
    W: np.ndarray = field(default_factory = lambda: np.zeros((0,), dtype = np.float64)) 
    Q: np.ndarray = field(default_factory = lambda: np.zeros((0,), dtype = np.float64)) 
    children: Dict[latticeId, "MCTSNode"] = field(default_factory = dict) ## pair (action, MCTS node with resulting graph)

    ## retrieves N, or 0 if empty. 
    @property
    def visit_count(self) -> int:
        return int(self.N.sum()) if self.N.size else 0

   ## update q-values with q = W/N 
    ##def q_values(self) -> np.ndarray:
    ##    q = np.zeros_like(self.W, dtype=np.float64)
    ##    mask = self.N > 0
    ##    q[mask] = self.W[mask] / self.N[mask]
    ##    return q

## outputs action to take from the initial state, based on the MCTS run. Output is in the form of latticeId for the state graph. 
@dataclass(frozen=True)
class SearchResult:
    actions: List[latticeId] ## changed to latticeID instead of GlobalAction
    visit_counts: np.ndarray ## visit counts of children nodes from current state
    ## visit_probs: np.ndarray
    root_value: float
    temperature: float

    ## finds the index of the best action to take, based on AlphaZero visit count formula. 
    @property
    def best_action(self) -> latticeId:
        if len(self.visit_counts) == 0:
           raise ValueError("no actions in search result")

        counts = self.visit_counts.astype(np.float64)

        if self.temperature <= 1e-8:
            return self.actions[np.argmax(counts)]
        
        counts = counts ** (1.0/self.temperature)
        pi = counts / counts.sum()

        rng = np.random.default_rng()

        return self.actions[int(rng.choice(len(pi), p = pi))]

## Small AlphaZero-style PUCT search for the cone environment.
## cost C(s), the search uses V(s) = -C(s). Each subdivision has reward -1.
class MCTS:
    def __init__(
        self,
        model: torch.nn.Module,
        ## enumerator: CandidateEnumerator,
        *,
        num_simulations: int = 64,
        c_puct: float = 1.5,
        dirichlet_alpha: Optional[float] = None,
        dirichlet_eps: float = 0.25,
        device: Optional[torch.device | str] = None,
    ) -> None:
        self.model = model
        ## self.enumerator = enumerator
        self.num_simulations = int(num_simulations)
        self.c_puct = float(c_puct)
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = float(dirichlet_eps)
        if device is None:
            try:
                device = next(model.parameters()).device
            except StopIteration:
                device = torch.device("cpu")
        self.device = torch.device(device)

    ## Runs the MCTS for num_simulations trials and outputs a SearchResult object (which can itself be called to give the final action choice)
    def run(self, root_state: ConeLatticeGraph, *, temperature: float = 1.0, add_root_noise: bool = False) -> SearchResult:
        root = MCTSNode(state = root_state)
        root_value = self._expand(root)
        if root.P.size == 0:
            return SearchResult([], np.zeros(0, dtype=np.int64), root_value, temperature)

        ## Dirichilet noise
        if add_root_noise and self.dirichlet_alpha is not None and root.P.size > 1:
            noise = np.random.dirichlet([self.dirichlet_alpha] * root.P.size)
            root.P = (1.0 - self.dirichlet_eps) * root.P + self.dirichlet_eps * noise

        for _ in range(self.num_simulations):
            node = root
            path: List[Tuple[MCTSNode, int]] = []

            while True:
                if node.state.isDecomposed():
                    leaf_value = 0.0
                    break
                if not node.expanded:
                    leaf_value = self._expand(node)
                    break
                if node.P.size == 0:
                    leaf_value = 0.0
                    break

                action_index = self._select_action(node)
                path.append((node, action_index))
                if action_index not in node.children:
                    curr_state = node.state.copy()
                    curr_state.subdivide(node.actions[action_index])
                    node.children[action_index] = MCTSNode(state = curr_state, prev_action = node.actions[action_index])

                #    child_state = apply_action(node.state, node.actions[action_index])
                #    node.children[action_index] = MCTSNode(child_state)

                node = node.children[action_index]
                    

            # Backup. The leaf_value is from the leaf state. Crossing one edge
            # back to the parent adds reward -1 for the subdivision action.
            ret = leaf_value
            for parent, action_index in reversed(path):
                ret = -1.0 + ret
                parent.N[action_index] += 1
                parent.W[action_index] += ret
                parent.Q[action_index] = parent.W[action_index] / parent.N[action_index]

        visit_counts = root.N.copy()
        return SearchResult(root.actions, visit_counts, root_value, temperature)

    ## Expands a leaf node, adding outward edges for all possible subdivision actions. Return estimated value of current state. 
    def _expand(self, node: MCTSNode) -> float:
        node.expanded = True
        ##if node.state.isDecomposed():
        ##    node.P = np.zeros(0, dtype = np.float64)
        ##    node.N = np.zeros(0, dtype = np.int64)
        #    node.W = np.zeros(0, dtype = np.float64)
        #    node.Q = np.zeros(0, dtype = np.float64)
        #    return 0.0

        actions, priors, value = self._evaluate_state(node.state)
        node.actions = actions
        node.P = priors
        node.N = np.zeros(len(actions), dtype=np.int64)
        node.W = np.zeros(len(actions), dtype=np.float64)
        node.Q = np.zeros(len(actions), dtype=np.float64)
        node.children = {}

        return value

    ## retrieves next actions, prior probabilities, and estimated value from a forward pass of the GNN. 
    @torch.no_grad()
    def _evaluate_state(self, state: ConeLatticeGraph) -> Tuple[List[latticeId], np.ndarray, float]:
        
        actions = state.listLatticePoints()
        if len(actions) == 0:
            return [], np.zeros(0, dtype=np.float64), 0.0

        self.model.eval()
        data = state.to(self.device)
        priors, value = self.model(data) ## change once we finish policy + value heads
        priors = priors.detach().cpu().numpy().astype(np.float64).reshape(-1)
        value = float(value.detach().cpu().reshape(-1)[0])
        return actions, priors, value



        ##data, actions = build_pyg_graph(state, self.enumerator)
        ##if len(actions) == 0:
        ##   return [], np.zeros(0, dtype=np.float64), 0.0
        ##self.model.eval()
        ##data = data.to(self.device)
        ##out = self.model(data)
        ##logits = out["logits"].detach().cpu()
        ##priors = torch.softmax(logits, dim=0).numpy().astype(np.float64)
        ##priors = priors / max(float(priors.sum()), 1e-12)
        ##cost = float(out["cost"].detach().cpu().view(-1)[0].item())
        ##value = -cost
        ##return actions, priors, value

    ## PUCT action selection for tree traversal
    def _select_action(self, node: MCTSNode) -> latticeId:
        q = node.Q
        total_n = max(1, int(node.N.sum()))
        u = self.c_puct * node.P * np.sqrt(total_n) / (1.0 + node.N)
        scores = q + u
        return int(np.argmax(scores))



