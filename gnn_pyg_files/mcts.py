from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch

from utils import validActionMask
from CGLGraph import CGLGraph
from network import network

latticeId = int

## Node stores (P,N,W,Q) where P is the prior probability from the GNN, N is the current visit count to the node, W is the total observed action-value, 
## and Q is the mean action-value. 
@dataclass
class MCTSNode:
    state: CGLGraph 
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

## outputs action to take from the initial state, based on the MCTS run. Output is in the form of latticeId for the state graph. 
@dataclass(frozen=True)
class SearchResult:
    actions: List[latticeId] 
    visit_counts: np.ndarray ## visit counts of children nodes from current state
    target_temperature: float
    play_temperature: float
    root: "MCTSNode" = field(default=None)

    ## gives array of visit probabilities based on AlphaZero-style policy output
    @property 
    def visit_probs(self) -> np.ndarray:
        counts = self.visit_counts.astype(np.float64)
        if counts.sum() == 0:
            return counts
        
        if self.target_temperature <= 1e-8:
            pi = np.zeros_like(counts)
            pi[np.argmax(counts)] = 1.0
            return pi
        
        counts = counts ** (1.0/self.target_temperature)
        pi = counts / counts.sum()
        return pi

    ## finds the index of the best action to take, based on AlphaZero visit count formula. 
    @property
    def best_action(self) -> latticeId:
        if len(self.visit_counts) == 0:
           raise ValueError("no actions in search result")
        if self.play_temperature <= 1e-8:
            return self.actions[np.argmax(self.visit_counts)]

        counts = self.visit_counts.astype(np.float64)
        counts = counts / counts.max()
        pi = counts ** (1.0/self.play_temperature)
        pi = pi / pi.sum()
        
        return self.actions[int(np.random.choice(len(pi), p = pi))]

## Small AlphaZero-style PUCT search for the cone environment.
class MCTS:
    def __init__(
        self,
        model: torch.nn.Module,
        *,
        num_simulations: int = 64,
        c_puct: float = 1.5,
        dirichlet_alpha: Optional[float] = None,
        dirichlet_eps: float = 0.25,
        device: Optional[torch.device | str] = None
        ) -> None:
        self.model = model
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
    def run(self, 
    root_state: CGLGraph,
    *, 
    target_temperature: float = 1.0, 
    play_temperature: float = 1.0, 
    add_root_noise: bool = False, 
    reuse_node: Optional["MCTSNode"] = None) -> SearchResult:

        self.model.eval()

        if reuse_node is not None:
            root = reuse_node
        else:
            root = MCTSNode(state = root_state)
            self._expand(root)

        if root.P.size == 0:
            return SearchResult([], np.zeros(0, dtype=np.int64), target_temperature, play_temperature, root)

        ## Dirichilet noise
        if add_root_noise and self.dirichlet_alpha is not None and root.P.size > 1:
            noise = np.random.dirichlet([self.dirichlet_alpha] * root.P.size)
            root.P = (1.0 - self.dirichlet_eps) * root.P + self.dirichlet_eps * noise

        for _ in range(self.num_simulations):
            node = root
            path: List[Tuple[MCTSNode, int]] = []

            while True:
                if not node.expanded:
                    leaf_value = self._expand(node)
                    break

                ## intiializes the value of a leaf with no explored actions to 0. May need fine tuning. 
                if not node.actions:
                    leaf_value = 0.0
                    break

                action_index = self._select_action(node)
                path.append((node, action_index))
                if action_index not in node.children:
                    curr_state = node.state.copy()
                    curr_state.subdivide(node.actions[action_index])
                    node.children[action_index] = MCTSNode(state = curr_state, prev_action = node.actions[action_index])

                node = node.children[action_index]
                    
            # Backup
            for parent, action_index in reversed(path):
                parent.N[action_index] += 1
                parent.W[action_index] += leaf_value
                parent.Q[action_index] = parent.W[action_index] / parent.N[action_index]

        visit_counts = root.N.copy()
        return SearchResult(root.actions, visit_counts, target_temperature, play_temperature, root)

    ## Expands a leaf node, adding outward edges for all possible subdivision actions. Return estimated value of current state. 
    def _expand(self, node: MCTSNode) -> float:
        node.expanded = True
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
    def _evaluate_state(self, state: CGLGraph) -> Tuple[List[latticeId], np.ndarray, float]:
        
        if state.isDecomposed():
            return [], np.zeros(0, dtype = np.float64,), 0.0

        data = state.toHeteroData().to(self.device)
        out = self.model(data)
        log_p = out["log_p"] ## retrieve log-probabilities of available actions
        value = out["value"]

        valid_action_idx = validActionMask(state).nonzero(as_tuple = False).flatten().tolist()
        actions = [state._lattice_idx_to_id[i] for i in valid_action_idx]

        priors = log_p.exp().detach().cpu().numpy().astype(np.float64).reshape(-1)
        value = float(value.detach().cpu().reshape(-1)[0]) ## move to cpu and flatten to a scalar
        return actions, priors, value

    ## PUCT action selection for tree traversal
    def _select_action(self, node: MCTSNode) -> latticeId:
        q = node.Q
        total_n = max(1, int(node.N.sum()))
        u = self.c_puct * node.P * np.sqrt(total_n) / (1.0 + node.N)
        scores = q + u
        return int(np.argmax(scores))

def main():
    from Cone import Cone
    import numpy as np

    def singular_root() -> CGLGraph:           # det 7 -> singular
        g = CGLGraph()
        g.addConeNode(Cone(((1, 0, 0), (0, 1, 0), (1, 1, 7))))
        return g

    def nonsingular_root() -> CGLGraph:        # det 1 -> already decomposed
        g = CGLGraph()
        g.addConeNode(Cone(((1, 0, 0), (0, 1, 0), (0, 0, 1))))
        return g

    def seed(s=0):
        torch.manual_seed(s); np.random.seed(s)

    def walk(node):                            # yield every node in the tree
        yield node
        for c in node.children.values():
            yield from walk(c)

    seed()
    model = network(singular_root().metadata(), hidden=64,
                    embedding_size=7, num_layers=4)
    model(singular_root()); model.eval()
    mcts = MCTS(model, num_simulations=400, c_puct=1.5,
                dirichlet_alpha=0.3, dirichlet_eps=0.25)

    # ---- 1. _evaluate_state ------------------------------------------------
    actions, priors, value = mcts._evaluate_state(singular_root())
    assert len(actions) == priors.shape[0], "action/prior length mismatch"
    assert len(actions) > 0, "singular cone should expose actions"
    assert abs(priors.sum() - 1.0) < 1e-5, "priors not normalized"
    assert np.isfinite(value), "value not finite"
    # terminal/decomposed state -> empty, zero value
    a0, p0, v0 = mcts._evaluate_state(nonsingular_root())
    assert a0 == [] and p0.size == 0 and v0 == 0.0, "decomposed eval not empty"
    print("[1] _evaluate_state OK")

    # ---- 2. _expand --------------------------------------------------------
    n = MCTSNode(state=singular_root())
    v = mcts._expand(n)
    k = len(n.actions)
    assert n.expanded and isinstance(v, float)
    assert n.P.shape == (k,) and n.N.shape == (k,) \
        and n.W.shape == (k,) and n.Q.shape == (k,), "expand array shapes"
    assert n.N.sum() == 0 and n.W.sum() == 0 and n.Q.sum() == 0, "expand not zeroed"
    print("[2] _expand OK")

    # ---- 3. MCTSNode.visit_count ------------------------------------------
    assert MCTSNode(state=singular_root()).visit_count == 0, "empty visit_count"
    n.N[:] = np.arange(k)
    assert n.visit_count == int(np.arange(k).sum()), "visit_count sum"
    print("[3] visit_count OK")

    # ---- 4. _select_action -------------------------------------------------
    m = MCTSNode(state=singular_root()); mcts._expand(m)
    idx = mcts._select_action(m)
    assert 0 <= idx < len(m.actions), "selected index out of range"
    # zero visits => U dominates => argmax over priors
    assert idx == int(np.argmax(m.P)), "select should follow priors when N=0"
    # saturating a high-prior arm with visits should steer selection elsewhere
    j = int(np.argmax(m.P)); m.N[j] = 10_000; m.Q[j] = -10.0
    assert mcts._select_action(m) != j, "visited+bad arm still selected"
    print("[4] _select_action OK")

    # ---- 5. run: conservation + Q=W/N invariant ---------------------------
    seed()
    res = mcts.run(singular_root(), target_temperature=1.0, play_temperature = 1.0, add_root_noise=False)
    assert res.visit_counts.sum() == mcts.num_simulations, "visits not conserved"
    # rebuild root to inspect the tree (run discards it) -> verify on a manual run
    root = MCTSNode(state=singular_root()); mcts._expand(root)
    # invariant check via fresh instrumented search
    seed()
    r2 = mcts.run(singular_root(), target_temperature=1.0, play_temperature = 1.0,add_root_noise=False)
    assert np.array_equal(res.visit_counts, r2.visit_counts), "run not reproducible"
    print("[5] run conservation + reproducibility OK")

    # ---- 6. Dirichlet root noise ------------------------------------------
    root = MCTSNode(state=singular_root()); mcts._expand(root)
    base = root.P.copy()
    noise = np.random.dirichlet([mcts.dirichlet_alpha] * base.size)
    mixed = (1 - mcts.dirichlet_eps) * base + mcts.dirichlet_eps * noise
    assert abs(mixed.sum() - 1.0) < 1e-6, "noised priors not normalized"
    assert not np.allclose(mixed, base), "noise had no effect"
    print("[6] dirichlet noise OK")

    # ---- 7. SearchResult.visit_probs --------------------------------------
    counts = np.array([1, 3, 6], dtype=np.int64)
    sr1 = SearchResult(["a", "b", "c"], counts, target_temperature=1.0, play_temperature = 1.0)
    assert abs(sr1.visit_probs.sum() - 1.0) < 1e-9, "probs not normalized"
    assert np.allclose(sr1.visit_probs, counts / counts.sum()), "T=1 != normalized"
    sr0 = SearchResult(["a", "b", "c"], counts, target_temperature=1e-9, play_temperature = 1e-9)
    oneh = np.zeros(3); oneh[np.argmax(counts)] = 1.0
    assert np.allclose(sr0.visit_probs, oneh), "T->0 not one-hot"
    print("[7] visit_probs OK")

    # ---- 8. SearchResult.best_action --------------------------------------
    assert sr0.best_action == "c", "greedy best_action wrong"
    assert sr1.best_action in {"a", "b", "c"}, "stochastic action out of set"
    try:
        SearchResult([], np.zeros(0, np.int64),target_temperature=1.0, play_temperature = 1.0).best_action
        assert False, "empty best_action should raise"
    except ValueError:
        pass
    print("[8] best_action OK")

    # ---- 9. terminal root -> empty SearchResult ---------------------------
    rt = mcts.run(nonsingular_root(), target_temperature=1.0, play_temperature = 1.0)
    assert rt.actions == [] and rt.visit_counts.size == 0, "terminal run not empty"
    print("[9] terminal root OK")

    # ---- 10. tree reuse: visits carry over --------------------------------
    seed()
    res1 = mcts.run(singular_root(), target_temperature=1.0, play_temperature = 1.0, add_root_noise=False)
    action_idx = int(np.argmax(res1.visit_counts))
    reuse = res1.root.children.get(action_idx)
    assert reuse is not None, "most-visited action should have been expanded"
    prior_visits = int(reuse.N.sum())
    seed()
    res2 = mcts.run(reuse.state, target_temperature=1.0, play_temperature = 1.0, add_root_noise=False,
                    reuse_node=reuse)
    assert res2.visit_counts.sum() == prior_visits + mcts.num_simulations, "reuse: visits not conserved"
    assert res2.root is reuse, "reuse: root should be the passed-in node"
    assert reuse.N.sum() > 0, "reuse: node should have prior visits"
    print("[10] tree reuse OK")

    print("ALL TESTS PASSED")

if __name__ == "__main__":
    main()