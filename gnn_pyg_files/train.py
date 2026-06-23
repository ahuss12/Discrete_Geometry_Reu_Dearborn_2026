from __future__ import annotations

import argparse
import random
from typing import List

import numpy as np
import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader
from tqdm import trange

#my old implementation had separate files for environment.py and graph.py, basically those old imports, can be easiy siwtched to new imports
from .geometry import (
    CandidateEnumerator,
    FundamentalParallelepipedEnumerator,
    GridCandidateEnumerator,
    HybridCandidateEnumerator,
    apply_action,
)
from .graph import build_pyg_graph
from .losses import alphazero_cost_loss
from .mcts import MCTS
from .model import ConeActionGNN
from .random_cones import random_initial_state
from .replay import ReplayBuffer


def make_enumerator(args: argparse.Namespace) -> CandidateEnumerator:
    fpp = FundamentalParallelepipedEnumerator(
        max_dim=7,
        max_candidates=args.max_candidates,
        max_points=args.fpp_max_points,
        strict_interior=not args.include_boundary_actions,
    )
    if args.enumerator == "fpp":
        return fpp
    grid = GridCandidateEnumerator(
        max_dim=7,
        max_candidates=args.max_candidates,
        max_grid_points=args.grid_max_points,
        random_trials=args.grid_random_trials,
        seed=args.seed,
    )
    if args.enumerator == "hybrid":
        return HybridCandidateEnumerator(
            fpp=fpp,
            grid=grid,
            min_candidates=1,
            max_candidates=args.max_candidates,
        )
    return grid


def self_play_episode(
    *,
    model: ConeActionGNN,
    enumerator: CandidateEnumerator,
    rng: random.Random,
    max_steps: int,
    mcts_sims: int,
    det_max: int,
    device: torch.device,
    temperature: float = 1.0,
    c_puct: float = 1.5,
    timeout_penalty: float = 0.0,
) -> List[HeteroData]:
    state = random_initial_state(max_dim=7, det_max=det_max, rng=rng)
    states = []
    policies = []

    for _ in range(max_steps):
        if state.is_terminal:
            break
        search = MCTS(
            model,
            enumerator,
            num_simulations=mcts_sims,
            c_puct=c_puct,
            device=device,
        ).run(state, temperature=temperature)
        if len(search.actions) == 0:
            break
        states.append(state)
        policies.append(torch.tensor(search.visit_probs, dtype=torch.float32))

        # Training uses MCTS visit policy. The environment action can be sampled
        # for exploration or greedy for stability.
        action_index = int(np.random.choice(len(search.actions), p=search.visit_probs))
        state = apply_action(state, search.actions[action_index])

    examples: List[HeteroData] = []
    horizon = len(states)

    # If the rollout stopped because max_steps was reached before terminal, the
    # previous version I was playing with had incorrectly treated the final sampled states as almost
    # solved, which made the value/cost head too optimistic and hurt MCTS.
    # Therefore I added an explicit tail cost to all states from an unfinished rollout.
    tail_cost = 0.0 if state.is_terminal else float(timeout_penalty)

    for t, s in enumerate(states):
        remaining_cost = float(horizon - t) + tail_cost
        data, _ = build_pyg_graph(
            s,
            enumerator,
            target_policy=policies[t],
            target_cost=remaining_cost,
        )
        if data["action"].x.size(0) > 0:
            examples.append(data)
    return examples


def train_one_epoch(
    *,
    model: ConeActionGNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    value_weight: float,
) -> dict:
    model.train()
    total_loss = 0.0
    total_policy = 0.0
    total_value = 0.0
    steps = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        out = model(batch)
        losses = alphazero_cost_loss(out, batch, value_weight=value_weight, model=model)
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += float(losses["loss"].detach().cpu())
        total_policy += float(losses["policy_loss"].detach().cpu())
        total_value += float(losses["value_loss"].detach().cpu())
        steps += 1
    return {
        "loss": total_loss / max(steps, 1),
        "policy_loss": total_policy / max(steps, 1),
        "value_loss": total_value / max(steps, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--epochs-per-iter", type=int, default=1)
    parser.add_argument("--mcts-sims", type=int, default=16)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--timeout-penalty", type=float, default=0.0)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--det-max", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--value-weight", type=float, default=0.25)
    parser.add_argument("--replay-capacity", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save", type=str, default="cone_action_gnn.pt")
    parser.add_argument("--resume", type=str, default="", help="Optional checkpoint to continue training from.")

    # old experiment controls.
    #parser.add_argument("--enumerator", choices=["fpp", "hybrid", "grid"], default="fpp")
    #parser.add_argument("--max-candidates", type=int, default=64)
    #parser.add_argument("--fpp-max-points", type=int, default=20_000)
    #parser.add_argument("--include-boundary-actions", action="store_true")
    #parser.add_argument("--grid-max-points", type=int, default=0)
    #parser.add_argument("--grid-random-trials", type=int, default=0)

    args = parser.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    model = ConeActionGNN(
        hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks,
        dropout=args.dropout,
    ).to(device)

    if args.resume:
        try:
            ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(args.resume, map_location=device)
        state_dict = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
        model.load_state_dict(state_dict)
        print(f"resumed model weights from {args.resume}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    enumerator = make_enumerator(args)
    replay = ReplayBuffer(capacity=args.replay_capacity, seed=args.seed)

    print(
        f"enumerator={args.enumerator} max_candidates={args.max_candidates} "
        f"fpp_max_points={args.fpp_max_points} device={device} "
        f"mcts_sims={args.mcts_sims} c_puct={args.c_puct} "
        f"temperature={args.temperature} timeout_penalty={args.timeout_penalty}"
    )

    for ep in trange(args.episodes, desc="self-play"):
        examples = self_play_episode(
            model=model,
            enumerator=enumerator,
            rng=rng,
            max_steps=args.max_steps,
            mcts_sims=args.mcts_sims,
            det_max=args.det_max,
            device=device,
            temperature=args.temperature,
            c_puct=args.c_puct,
            timeout_penalty=args.timeout_penalty,
        )
        replay.extend(examples)

        if len(replay) >= args.batch_size:
            items = replay.sample(min(len(replay), max(args.batch_size, 4 * args.batch_size)))
            loader = DataLoader(items, batch_size=args.batch_size, shuffle=True)
            metrics = {}
            for _ in range(args.epochs_per_iter):
                metrics = train_one_epoch(
                    model=model,
                    loader=loader,
                    optimizer=optimizer,
                    device=device,
                    value_weight=args.value_weight,
                )
            if (ep + 1) % max(1, args.episodes // 10) == 0:
                print(f"ep={ep+1} replay={len(replay)} metrics={metrics}")

    torch.save({"model": model.state_dict(), "args": vars(args)}, args.save)
    print(f"saved {args.save}")


if __name__ == "__main__":
    main()
