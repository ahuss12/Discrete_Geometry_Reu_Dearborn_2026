from __future__ import annotations
import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch

from utils import Cone, ConeLatticeGraph

# An action subdivides a *specific* maximal cone through a primitive lattice
# point of its fundamental parallelepiped. A bare latticeId is insufficient for
# ConeLatticeGraph.subdivide(coneId, latticeId), so an action is the pair.
# (Each primitive fpp point lies in exactly one maximal cone, so this is well
# defined; the latticeId alone identifies it for the policy if desired.)
latticeId = int
Action = Tuple[int, int]  # (coneId, latticeId)


# ---------------------------------------------------------------------------
# State helpers built only from utils.ConeLatticeGraph
# ---------------------------------------------------------------------------

## Enumerate every (coneId, latticeId) subdivision action available in the fan,
## in a deterministic order so model priors align positionally.
def available_actions(state: ConeLatticeGraph) -> List[Action]:
    actions: List[Action] = []
    for coneId in sorted(state._cone_id_to_idx):
        # listConeLatticePoints returns coordinates; map each back to its
        # latticeId, which is what ConeLatticeGraph.subdivide expects.
        for coord in state.listConeLatticePoints(coneId):
            latId = state._coord_to_lattice_id[coord]
            actions.append((coneId, latId))
    return actions


## A fan is decomposed iff every maximal cone is nonsingular (mult = 1),
## equivalently iff no primitive lattice point remains in any fundamental
## parallelepiped, i.e. iff there are no available actions.
def is_decomposed(state: ConeLatticeGraph) -> bool:
    return len(available_actions(state)) == 0


@dataclass
class MCTSNode:
    state: ConeLatticeGraph
    expanded: bool = False
    actions: List[Action] = field(default_factory=list)      # index -> (coneId, latticeId)
    P: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float64))  # priors
    N: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.int64))
    W: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float64))
    children: Dict[int, "MCTSNode"] = field(default_factory=dict)  # keyed by action index

    @property
    def visit_count(self) -> int:
        return int(self.N.sum()) if self.N.size else 0

    ## Q = W / N, with unvisited actions held at 0.
    def q_values(self) -> np.ndarray:
        q = np.zeros_like(self.W, dtype=np.float64)
        mask = self.N > 0
        q[mask] = self.W[mask] / self.N[mask]
        return q


@dataclass(frozen=True)
class SearchResult:
    actions: List[Action]
    visit_counts: np.ndarray
    visit_probs: np.ndarray
    root_value: float

    ## Index of the most-visited action (greedy, temperature -> 0).
    @property
    def best_action_index(self) -> int:
        if len(self.visit_counts) == 0:
            raise ValueError("no actions in search result")
        return int(np.argmax(self.visit_counts))

    @property
    def best_action(self) -> Action:
        return self.actions[self.best_action_index]

    ## Sample an action by the AlphaZero visit-count distribution at the given
    ## temperature. temperature <= 0 is greedy.
    def sample_action(self, temperature: float = 1.0) -> Action:
        if len(self.actions) == 0:
            raise ValueError("no actions in search result")
        pi = counts_to_policy(self.visit_counts, temperature=temperature)
        rng = np.random.default_rng()
        return self.actions[int(rng.choice(len(pi), p=pi))]


# ---------------------------------------------------------------------------
# Small AlphaZero-style PUCT search for the cone environment.
# For cost C(s), the search uses value V(s) = -C(s). Each subdivision (edge)
# carries reward -1, so the return measures (negative) number of inserted rays.
# ---------------------------------------------------------------------------
class MCTS:
    def __init__(
        self,
        model: torch.nn.Module,
        *,
        num_simulations: int = 64,
        c_puct: float = 1.5,
        dirichlet_alpha: Optional[float] = None,
        dirichlet_eps: float = 0.25,
        device: Optional[torch.device | str] = None,
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

    def run(
        self,
        root_state: ConeLatticeGraph,
        *,
        temperature: float = 1.0,
        add_root_noise: bool = False,
    ) -> SearchResult:
        root = MCTSNode(state=root_state)
        root_value = self._expand(root)
        if root.P.size == 0:
            return SearchResult([], np.zeros(0, dtype=np.int64),
                                np.zeros(0, dtype=np.float64), root_value)

        # Dirichlet noise at the root for exploration.
        if add_root_noise and self.dirichlet_alpha is not None and root.P.size > 1:
            noise = np.random.dirichlet([self.dirichlet_alpha] * root.P.size)
            root.P = (1.0 - self.dirichlet_eps) * root.P + self.dirichlet_eps * noise

        for _ in range(self.num_simulations):
            node = root
            path: List[Tuple[MCTSNode, int]] = []

            while True:
                if is_decomposed(node.state):
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
                    coneId, latId = node.actions[action_index]
                    child_state = copy.deepcopy(node.state)  # subdivide mutates in place
                    child_state.subdivide(coneId, latId)
                    node.children[action_index] = MCTSNode(state=child_state)

                node = node.children[action_index]

            # Backup. leaf_value is the value at the leaf state. Crossing one
            # edge back to a parent adds reward -1 for that subdivision action.
            ret = leaf_value
            for parent, action_index in reversed(path):
                ret = -1.0 + ret
                parent.N[action_index] += 1
                parent.W[action_index] += ret

        visit_counts = root.N.copy()
        visit_probs = counts_to_policy(visit_counts, temperature=temperature)
        return SearchResult(root.actions, visit_counts, visit_probs, root_value)

    ## Expand a leaf: attach priors/value from the model and initialize stats.
    def _expand(self, node: MCTSNode) -> float:
        node.expanded = True
        if is_decomposed(node.state):
            node.actions = []
            node.P = np.zeros(0, dtype=np.float64)
            node.N = np.zeros(0, dtype=np.int64)
            node.W = np.zeros(0, dtype=np.float64)
            return 0.0

        actions, priors, value = self._evaluate_state(node.state)
        node.actions = actions
        node.P = priors
        node.N = np.zeros(len(actions), dtype=np.int64)
        node.W = np.zeros(len(actions), dtype=np.float64)
        node.children = {}
        return value

    ## One forward pass of the GNN: priors over available actions + value.
    ## NOTE: assumes model(data) -> (priors, value) with priors aligned to
    ## available_actions(state) order. Adjust once the policy/value heads land.
    @torch.no_grad()
    def _evaluate_state(
        self, state: ConeLatticeGraph
    ) -> Tuple[List[Action], np.ndarray, float]:
        actions = available_actions(state)
        if len(actions) == 0:
            return [], np.zeros(0, dtype=np.float64), 0.0

        self.model.eval()
        data = state.to(self.device)
        priors, value = self.model(data)

        priors = torch.as_tensor(priors).detach().cpu().reshape(-1).numpy().astype(np.float64)
        if priors.shape[0] != len(actions):
            # Fall back to uniform if the head's width doesn't match the action set.
            priors = np.ones(len(actions), dtype=np.float64)
        priors = priors / max(float(priors.sum()), 1e-12)
        value = float(torch.as_tensor(value).detach().cpu().reshape(-1)[0].item())
        return actions, priors, value

    ## PUCT: argmax over Q + c_puct * P * sqrt(sum N) / (1 + N).
    def _select_action(self, node: MCTSNode) -> int:
        q = node.q_values()
        total_n = max(1, int(node.N.sum()))
        u = self.c_puct * node.P * np.sqrt(total_n) / (1.0 + node.N)
        scores = q + u
        return int(np.argmax(scores))


def counts_to_policy(counts: np.ndarray, *, temperature: float) -> np.ndarray:
    counts = counts.astype(np.float64)
    if counts.size == 0:
        return counts
    if temperature <= 1e-8:
        out = np.zeros_like(counts, dtype=np.float64)
        out[int(np.argmax(counts))] = 1.0
        return out
    x = counts ** (1.0 / temperature)
    if x.sum() <= 0:
        return np.ones_like(x) / x.size
    return x / x.sum()
