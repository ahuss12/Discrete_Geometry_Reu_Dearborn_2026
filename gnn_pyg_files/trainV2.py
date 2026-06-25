from __future__ import annotations

import argparse
import random
from typing import List

import numpy as np
import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader
from tqdm import trange

from coneEnvironment import Cone
from graph import ConeLatticeGraph, generateRandomCone, DIMENSION
from mctsV2 import MCTS
from replay import ReplayBuffer
from network import *

## Outputs the states visited over the episode, with attached performance targets. 
## NOTE: removed enumerator, added option for dirichlet. removed rng and det_max options. Added
## initial state as a necessary parameter. 
def self_play_episode(
    *,
    model: torch.nn.Module, ## evaluation model
    max_steps: int, ## maximum subdivisions that can be applied before timing out
    mcts_sims: int,
    initial_state: ConeLatticeGraph, 
    device: torch.device,
    temperature: float = 1.0,
    c_puct: float = 1.5,
    timeout_penalty: float = 0.0, 
    dirichlet_alpha: Optional[float] = None,
    dirichlet_eps: float = 0.25,
) -> List[ConeLatticeGraph]:
    states = []
    policies = []
    action_options_history = [] ## latticeIds of the available actions at each time step. 

    state = initial_state
    for _ in range(max_steps):
        if state.isDecomposed():
            break
        
        search = MCTS(
            model = model,
            num_simulations = mcts_sims,
            c_puct = c_puct,
            dirichlet_alpha = dirichlet_alpha,
            dirichlet_eps = dirichlet_eps,
            device = device
        )
        states.append(state.copy())
        result = search.run(state, temperature = temperature, add_root_noise = True)

        policies.append(torch.tensor(result.visit_probs, dtype = torch.float32))
        action_options_history.append(list(result.actions))

        # Training uses MCTS visit policy. The environment action can be sampled
        # for exploration or greedy for stability.
        state.subdivide(result.best_action)
    
    examples: List[HeteroData] = []
    horizon = len(states)

    # If the rollout stopped because max_steps was reached before terminal, the
    # previous version I was playing with had incorrectly treated the final sampled states as almost
    # solved, which made the value/cost head too optimistic and hurt MCTS.
    # Therefore I added an explicit tail cost to all states from an unfinished rollout.
    tail_cost = 0.0 if state.isDecomposed() else float(timeout_penalty)

    for t, state in enumerate(states):
        remaining_cost = float(horizon - t) + tail_cost
        state.addGlobalNode(n = EMBEDDING_SIZE)
        attach_targets(state, target_policy = policies[t], actions = action_options_history[t], target_value = -remaining_cost) ##NOTE: this may need to be changed to +remaining cost
        if len(state.getValidActions()) > 0:
            examples.append(state)
    return examples

    
## attaches the training (policy, value) targets for the GNN to learn. idx order to match policyHead output. 
def attach_targets(graph: ConeLatticeGraph, *, target_policy: np.ndarray, actions: list[int], target_value: float) -> None:
    valid_idx = validActionMask(graph).nonzero(as_tuple=False).flatten().tolist() ##idx of currently valid actions
    mask_order_ids = [graph._lattice_idx_to_id[i] for i in valid_idx]

    ## map latticeId -> its probability from the stored policy
    probs = {a: float(p) for a, p in zip(actions, target_policy)}

    ## check the two action sets agree
    assert set(mask_order_ids) == set(actions), \
        "stored policy actions don't match the graph's current valid actions"

    y_policy = torch.zeros(graph['lattice'].num_nodes, dtype = torch.float)
    for i, a in zip(valid_idx, mask_order_ids):
        y_policy[i] = probs[a]

    graph['lattice'].y_policy = y_policy
    graph.y_value = torch.tensor([target_value], dtype = torch.float)   # shape (1,) -> batches to (B,)

## NOTE: added warning when total_loss/max(steps,1) = 0.0 is returned, as this may seem like a good loss. 
def train_one_epoch(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    value_weight: float
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
        losses = alphazero_cost_loss(out, batch, value_weight=value_weight)
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += float(losses["loss"].detach().cpu())
        total_policy += float(losses["policy_loss"].detach().cpu())
        total_value += float(losses["value_loss"].detach().cpu())
        steps += 1

    if steps == 0:
        raise ValueError(
            "no batches to train on"
        )

    return {
        "loss":        total_loss   / steps,
        "policy_loss": total_policy / steps,
        "value_loss":  total_value  / steps,
    }

## alphazero style loss, out is the output from network.py
## NOTE: got this with claude's help, cross reference with ameen's function.
def alphazero_cost_loss(out: dict, data, *, value_weight: float) -> dict:
    log_p = out["log_p"] # (A,) log-probs over valid actions
    value = out["value"] # (B,) predicted values
    action_batch = out["action_batch"] # (A,) graph/batch id per action

    mask = validActionMask(data)
    y_policy = data['lattice'].y_policy[mask] # (A,), aligned with log_p

    # per-graph cross-entropy: -sum_a y * log_p, then mean over graphs
    ce_per_action = -(y_policy * log_p)                
    B = int(action_batch.max().item()) + 1
    policy_loss = torch.zeros(B, device=log_p.device).scatter_add_(0, action_batch, ce_per_action).mean()

    value_loss = F.mse_loss(value, data.y_value.view(-1))

    loss = policy_loss + value_weight * value_loss
    return {"loss": loss, "policy_loss": policy_loss, "value_loss": value_loss}
    
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
    parser.add_argument("--dirichlet-alpha", type=float, default=None)
    parser.add_argument("--dirichlet-eps", type=float, default=0.25)

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

    ## let the network learn be initialized to the structure of the graph
    dummy_cone = Cone(generateRandomCone(n = EMBEDDING_SIZE, d = args.det_max))
    meta_state = ConeLatticeGraph(dimension = EMBEDDING_SIZE)
    meta_state.addConeNode(dummy_cone)
    meta_state.addGlobalNode(n = EMBEDDING_SIZE)
    model = network(meta_state.metadata(),
        hidden=args.hidden_dim,
        embedding_size=EMBEDDING_SIZE,
        num_layers=args.num_blocks,
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
    replay = ReplayBuffer(capacity=args.replay_capacity, seed=args.seed)

    print(
        f"device={device} mcts_sims={args.mcts_sims} c_puct={args.c_puct} "
        f"temperature={args.temperature} timeout_penalty={args.timeout_penalty} "
    )

    ## generate semirandom training examples
    training_examples = []
    for i in range(args.episodes):
        n = rng.randint(2, DIMENSION)
        d = rng.randint(2, args.det_max)
        g = ConeLatticeGraph(dimension = DIMENSION)
        g.addConeNode(Cone(generateRandomCone(n,d)))
        training_examples.append(g)

    for ep in trange(args.episodes, desc="self-play"):
        examples = self_play_episode(
            model=model,
            max_steps=args.max_steps,
            mcts_sims=args.mcts_sims,
            initial_state=training_examples[ep],
            device=device,
            temperature=args.temperature,
            c_puct=args.c_puct,
            timeout_penalty=args.timeout_penalty,
            dirichlet_alpha=args.dirichlet_alpha,
            dirichlet_eps=args.dirichlet_eps
        )

        replay.extend(examples)

        if len(replay) >= args.batch_size:
            items = replay.sample(min(len(replay), 4 * args.batch_size))
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

