from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from .geometry import CandidateEnumerator, ConeState, GlobalAction, apply_action #geometry.py
from .graph import build_pyg_graph #graph.py


@dataclass
class MCTSNode:
    state: ConeState
    expanded: bool = False
    actions: List[GlobalAction] = field(default_factory=list)
    priors: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float64))
    N: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.int64))
    W: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float64))
    children: Dict[int, "MCTSNode"] = field(default_factory=dict)

    @property
    def visit_count(self) -> int:
        return int(self.N.sum()) if self.N.size else 0

    def q_values(self) -> np.ndarray:
        q = np.zeros_like(self.W, dtype=np.float64)
        mask = self.N > 0
        q[mask] = self.W[mask] / self.N[mask]
        return q


@dataclass(frozen=True)
class SearchResult:
    actions: List[GlobalAction]
    visit_counts: np.ndarray
    visit_probs: np.ndarray
    root_value: float

    @property
    def best_action_index(self) -> int:
        if len(self.visit_counts) == 0:
            raise ValueError("no actions in search result")
        return int(np.argmax(self.visit_counts))

    @property
    def best_action(self) -> GlobalAction:
        return self.actions[self.best_action_index]

 #Small AlphaZero-style PUCT search for the cone environment.
# cost C(s), the search uses V(s) = -C(s). Each subdivision has reward -1.
class MCTS:
   
   

    def __init__(
        self,
        model: torch.nn.Module,
        enumerator: CandidateEnumerator,
        *,
        num_simulations: int = 64,
        c_puct: float = 1.5,
        dirichlet_alpha: Optional[float] = None,
        dirichlet_eps: float = 0.25,
        device: Optional[torch.device | str] = None,
    ) -> None:
        self.model = model
        self.enumerator = enumerator
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

    def run(self, root_state: ConeState, *, temperature: float = 1.0, add_root_noise: bool = False) -> SearchResult:
        root = MCTSNode(root_state)
        root_value = self._expand(root)
        if root.priors.size == 0:
            return SearchResult([], np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64), root_value)

        if add_root_noise and self.dirichlet_alpha is not None and root.priors.size > 1:
            noise = np.random.dirichlet([self.dirichlet_alpha] * root.priors.size)
            root.priors = (1.0 - self.dirichlet_eps) * root.priors + self.dirichlet_eps * noise

        for _ in range(self.num_simulations):
            node = root
            path: List[Tuple[MCTSNode, int]] = []

            while True:
                if node.state.is_terminal:
                    leaf_value = 0.0
                    break
                if not node.expanded:
                    leaf_value = self._expand(node)
                    break
                if node.priors.size == 0:
                    leaf_value = 0.0
                    break

                action_index = self._select_action(node)
                path.append((node, action_index))
                if action_index not in node.children:
                    child_state = apply_action(node.state, node.actions[action_index])
                    node.children[action_index] = MCTSNode(child_state)
                node = node.children[action_index]

            # Backup. The leaf_value is from the leaf state. Crossing one edge
            # back to the parent adds reward -1 for the subdivision action.
            ret = leaf_value
            for parent, action_index in reversed(path):
                ret = -1.0 + ret
                parent.N[action_index] += 1
                parent.W[action_index] += ret

        visit_counts = root.N.copy()
        visit_probs = counts_to_policy(visit_counts, temperature=temperature)
        return SearchResult(root.actions, visit_counts, visit_probs, root_value)

    def _expand(self, node: MCTSNode) -> float:
        node.expanded = True
        if node.state.is_terminal:
            node.actions = []
            node.priors = np.zeros(0, dtype=np.float64)
            node.N = np.zeros(0, dtype=np.int64)
            node.W = np.zeros(0, dtype=np.float64)
            return 0.0

        actions, priors, value = self._evaluate_state(node.state)
        node.actions = actions
        node.priors = priors
        node.N = np.zeros(len(actions), dtype=np.int64)
        node.W = np.zeros(len(actions), dtype=np.float64)
        return value

    @torch.no_grad()
    def _evaluate_state(self, state: ConeState) -> Tuple[List[GlobalAction], np.ndarray, float]:
        data, actions = build_pyg_graph(state, self.enumerator)
        if len(actions) == 0:
            return [], np.zeros(0, dtype=np.float64), 0.0
        self.model.eval()
        data = data.to(self.device)
        out = self.model(data)
        logits = out["logits"].detach().cpu()
        priors = torch.softmax(logits, dim=0).numpy().astype(np.float64)
        priors = priors / max(float(priors.sum()), 1e-12)
        cost = float(out["cost"].detach().cpu().view(-1)[0].item())
        value = -cost
        return actions, priors, value

    def _select_action(self, node: MCTSNode) -> int:
        q = node.q_values()
        total_n = max(1, int(node.N.sum()))
        u = self.c_puct * node.priors * np.sqrt(total_n) / (1.0 + node.N)
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
