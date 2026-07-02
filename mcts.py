from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch

from utils import validActionMask
from CGLGraph import CGLGraph
from baseline import min_sum_steps, graph_signature

latticeId = int


@dataclass
class MCTSNode:
    state: CGLGraph
    prev_action: Optional[latticeId] = None
    expanded: bool = False
    actions: List[latticeId] = field(default_factory=list)
    P: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float64))
    N: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.int64))
    W: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float64))
    Q: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float64))
    children: Dict[int, "MCTSNode"] = field(default_factory=dict)

    @property
    def visit_count(self) -> int:
        return int(self.N.sum()) if self.N.size else 0


@dataclass(frozen=True)
class SearchResult:
    actions: List[latticeId]
    visit_counts: np.ndarray
    temperature: float
    root: Optional[MCTSNode] = None

    @property
    def visit_probs(self) -> np.ndarray:
        counts = self.visit_counts.astype(np.float64)
        if counts.sum() == 0:
            return counts

        if self.temperature <= 1e-8:
            pi = np.zeros_like(counts)
            pi[np.argmax(counts)] = 1.0
            return pi

        counts = counts ** (1.0 / self.temperature)
        pi = counts / counts.sum()
        return pi

    @property
    def best_action(self) -> latticeId:
        if len(self.visit_counts) == 0:
            raise ValueError("no actions in search result")

        pi = self.visit_probs
        if self.temperature <= 1e-8:
            return self.actions[int(np.argmax(pi))]

        return self.actions[int(np.random.choice(len(pi), p=pi))]


class MCTS:
    """PUCT search with normalized min_sum-potential reward backup.

    Potential:
        h(s) = min_sum steps remaining from state s.

    For a search rooted at s_root with B = max(h(s_root), 1), the reward on an
    edge s -> s' is

        r = (h(s) - h(s') - 1) / B.

    This makes a path that solves in T steps have total shaped return

        (h(s_root) - T) / B,

    so tying min_sum gives return 0, beating min_sum gives positive return, and
    being worse gives negative return.  The value network is trained in local
    normalized units, so leaf values are converted into root units by multiplying
    by h(leaf) / B during search.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        num_simulations: int = 64,
        c_puct: float = 1.5,
        dirichlet_alpha: Optional[float] = None,
        dirichlet_eps: float = 0.25,
        device: Optional[torch.device | str] = None,
        mcts_max_depth: int = 0,
    ) -> None:
        self.model = model
        self.num_simulations = int(num_simulations)
        self.c_puct = float(c_puct)
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = float(dirichlet_eps)
        self.mcts_max_depth = int(mcts_max_depth)

        if device is None:
            try:
                device = next(model.parameters()).device
            except StopIteration:
                device = torch.device("cpu")
        self.device = torch.device(device)

        self._h_cache: dict[tuple, int] = {}
        self._root_norm: float = 1.0

    def _h(self, state: CGLGraph) -> int:
        key = graph_signature(state)
        val = self._h_cache.get(key)
        if val is None:
            val = min_sum_steps(state)
            self._h_cache[key] = int(val)
        return int(val)

    def _edge_reward(self, parent_state: CGLGraph, child_state: CGLGraph) -> float:
        hp = self._h(parent_state)
        hc = self._h(child_state)
        return float(hp - hc - 1) / self._root_norm

    def _local_value_to_root_units(self, state: CGLGraph, local_value: float) -> float:
        h = self._h(state)
        if h <= 0:
            return 0.0
        return float(local_value) * (float(h) / self._root_norm)

    def run(
        self,
        root_state: CGLGraph,
        *,
        temperature: float = 1.0,
        add_root_noise: bool = False,
        reuse_node: Optional[MCTSNode] = None,
    ) -> SearchResult:
        self.model.eval()
        self._h_cache = {}
        root_h = self._h(root_state)
        self._root_norm = float(max(root_h, 1))

        if reuse_node is not None:
            # Reusing a node whose Q-values were produced with a different root
            # normalization can be inconsistent.  Training currently passes None,
            # but we keep the API and continue from the provided node if requested.
            root = reuse_node
        else:
            root = MCTSNode(state=root_state)
            self._expand(root)

        if root.P.size == 0:
            return SearchResult([], np.zeros(0, dtype=np.int64), temperature, root)

        if add_root_noise and self.dirichlet_alpha is not None and root.P.size > 1:
            noise = np.random.dirichlet([self.dirichlet_alpha] * root.P.size)
            root.P = (1.0 - self.dirichlet_eps) * root.P + self.dirichlet_eps * noise
            root.P = root.P / max(root.P.sum(), 1e-12)

        for _ in range(self.num_simulations):
            node = root
            depth = 0
            path: List[Tuple[MCTSNode, int, float]] = []

            while True:
                if node.state.isDecomposed():
                    leaf_value = 0.0
                    break

                if self.mcts_max_depth > 0 and depth >= self.mcts_max_depth:
                    leaf_value = self._evaluate_value_only(node.state)
                    break

                if not node.expanded:
                    leaf_value = self._expand(node)
                    break

                if not node.actions:
                    leaf_value = 0.0
                    break

                action_index = self._select_action(node)
                action = node.actions[action_index]

                if action_index not in node.children:
                    child_state = node.state.copy()
                    child_state.subdivide(action)
                    node.children[action_index] = MCTSNode(
                        state=child_state,
                        prev_action=action,
                    )

                child = node.children[action_index]
                reward = self._edge_reward(node.state, child.state)
                path.append((node, action_index, reward))
                node = child
                depth += 1

            # Backup r + V after every individual simulation.
            ret = float(leaf_value)
            for parent, action_index, reward in reversed(path):
                ret = float(reward) + ret
                parent.N[action_index] += 1
                parent.W[action_index] += ret
                parent.Q[action_index] = parent.W[action_index] / parent.N[action_index]

        visit_counts = root.N.copy()
        return SearchResult(root.actions, visit_counts, temperature, root)

    def _expand(self, node: MCTSNode) -> float:
        node.expanded = True
        actions, priors, local_value = self._evaluate_state(node.state)
        node.actions = actions
        node.P = priors
        node.N = np.zeros(len(actions), dtype=np.int64)
        node.W = np.zeros(len(actions), dtype=np.float64)
        node.Q = np.zeros(len(actions), dtype=np.float64)
        node.children = {}
        return self._local_value_to_root_units(node.state, local_value)

    @torch.no_grad()
    def _evaluate_value_only(self, state: CGLGraph) -> float:
        if state.isDecomposed():
            return 0.0
        data = state.toHeteroData().to(self.device)
        out = self.model(data)
        local_value = float(out["value"].detach().cpu().reshape(-1)[0])
        return self._local_value_to_root_units(state, local_value)

    @torch.no_grad()
    def _evaluate_state(self, state: CGLGraph) -> Tuple[List[latticeId], np.ndarray, float]:
        if state.isDecomposed():
            return [], np.zeros(0, dtype=np.float64), 0.0

        data = state.toHeteroData().to(self.device)
        out = self.model(data)
        log_p = out["log_p"]
        value = out["value"]

        valid_action_idx = validActionMask(state).nonzero(as_tuple=False).flatten().tolist()
        actions = [state._lattice_idx_to_id[i] for i in valid_action_idx]

        priors = log_p.exp().detach().cpu().numpy().astype(np.float64).reshape(-1)
        if priors.size and (not np.all(np.isfinite(priors)) or priors.sum() <= 0):
            priors = np.ones_like(priors, dtype=np.float64) / max(len(priors), 1)
        elif priors.size:
            priors = priors / priors.sum()

        value = float(value.detach().cpu().reshape(-1)[0])
        if not np.isfinite(value):
            value = 0.0
        return actions, priors, value

    def _select_action(self, node: MCTSNode) -> int:
        q = node.Q
        total_n = max(1, int(node.N.sum()))
        u = self.c_puct * node.P * np.sqrt(total_n) / (1.0 + node.N)
        scores = q + u
        return int(np.argmax(scores))


def main():
    from Cone import Cone
    from network import network, EMBEDDING_SIZE

    torch.manual_seed(0)
    np.random.seed(0)

    def singular_root() -> CGLGraph:
        g = CGLGraph(dimension=3)
        g.addConeNode(Cone(((1, 0, 0), (0, 1, 0), (1, 1, 7))))
        return g

    def nonsingular_root() -> CGLGraph:
        g = CGLGraph(dimension=3)
        g.addConeNode(Cone(((1, 0, 0), (0, 1, 0), (0, 0, 1))))
        return g

    model = network(singular_root().metadata(), hidden=64, embedding_size=EMBEDDING_SIZE, num_layers=3)
    model(singular_root())
    model.eval()

    mcts = MCTS(model, num_simulations=50, c_puct=1.5, mcts_max_depth=8)
    actions, priors, value = mcts._evaluate_state(singular_root())
    print("actions:", actions)
    print("priors sum:", priors.sum() if priors.size else 0)
    print("local value:", value)
    assert len(actions) == priors.shape[0]
    assert len(actions) == 0 or abs(priors.sum() - 1.0) < 1e-5

    result = mcts.run(singular_root(), temperature=1.0, add_root_noise=False)
    print("visit counts:", result.visit_counts, "total=", int(result.visit_counts.sum()))
    assert result.visit_counts.sum() == mcts.num_simulations

    terminal = mcts.run(nonsingular_root(), temperature=1.0)
    assert terminal.actions == [] and terminal.visit_counts.size == 0

    print("OK: normalized-reward MCTS ran.")


if __name__ == "__main__":
    main()
