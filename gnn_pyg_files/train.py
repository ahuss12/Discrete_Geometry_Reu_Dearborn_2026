# ===========================================================================================
#  train.py — AlphaZero-style self-play training for cone resolution
# ===========================================================================================
#
#  WHAT IT DOES
#  ------------
#  Trains a GNN (policy + value heads) by self-play with MCTS. Each episode starts from a
#  random singular cone and the agent inserts subdivision rays until the cone is resolved
#  (decomposed into smooth cones) or it times out. The min_sum heuristic and a random policy
#  are run on the same cone as baselines for comparison. Visited states get (policy, value)
#  targets attached and are pushed into a replay buffer; the network trains on minibatches
#  sampled from that buffer.
#
#  USAGE
#  -----
#  Run from this directory (gnn_pyg_files/) so the local imports resolve:
#
#      python train.py                          # train with all defaults
#      python train.py --episodes 500 --device cuda
#      python train.py --resume cone_action_gnn.pt --save run2.pt
#
#  KEY ARGUMENTS (see build_parser() for the full list + defaults)
#  ---------------------------------------------------------------
#    --episodes N            number of self-play episodes to run
#    --mcts-sims N           MCTS simulations per move (higher = stronger, slower)
#    --max-steps N           max subdivisions per episode before timeout
#    --min-dimension/-max    range of cone dimension n sampled per episode
#    --det-min/--det-max     range of cone determinant (multiplicity) sampled per episode
#    --lr, --hidden-dim,     network / optimizer hyperparameters
#      --num-blocks, --dropout, --value-weight, --batch-size
#    --timeout-penalty       penalty added to reward when an episode times out
#    --timeout-multiplier    timeout threshold as a multiple of the baseline step count
#    --early-termination     stop an episode once it exceeds timeout-multiplier * baseline
#    --resolved-reward-type  index (0-5) selecting the reward variant for resolved episodes
#    --timeout-reward-type   index (0-3) selecting the reward variant for timed-out episodes
#    --resume PATH           continue training from an existing checkpoint
#    --save PATH             where to write the final / periodic checkpoint
#    --diag-dir DIR          parent dir for per-run diagnostics (a timestamped subdir is made)
#    --seed N, --device STR  reproducibility / cuda|cpu
#
#  OUTPUTS
#  -------
#    --save checkpoint (.pt)         model weights + args; also saved every 200 episodes
#    <diag-dir>/<timestamp>/         config.json, timing.json, trajectories.jsonl, and the
#                                    diagnostics CSVs/plots produced by Diagnostics.finish().
#  Inspect a run afterwards with read_diagnostics.py / hyp_visualizations.py.
#  Sweep hyperparameters with hyp_sweep.py (it discovers args via build_parser()).
# ===========================================================================================

from __future__ import annotations
import argparse
import json
import random
import time
from datetime import datetime
from typing import List, Optional
from diagnostics import Diagnostics, extract_value
from trajectory_log import TrajectoryLogger
import os

import numpy as np
import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader
from tqdm import tqdm, trange

from utils import generateRandomCone, validActionMask, isPrimitiveNonzero
from Cone import Cone, fanSubdivide
from CGLGraph import CGLGraph, DIMENSION
from mcts import MCTS
from replay import ReplayBuffer
from network import network
from losses import alphazero_cost_loss

Vector = tuple[int, ...]

# ===========================================================================================
#  HELPER FUNCTIONS
# ===========================================================================================

## attaches the training (policy, value) targets for the GNN to learn. idx order to match policyHead output. 
def attach_targets(
    graph: CGLGraph, 
    *, 
    target_policy: np.ndarray, 
    actions: list[int], 
    target_value: float
    ) -> None:

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

## takes in a graph with a single cone, and returns (steps, cones, actions) where steps is the number of steps to resolve, 
## cones are the final cone state, and actions are the subdivisions applied. 
## min-sum ties are broken arbitrarily. 
def min_sum(graph: CGLGraph) -> tuple[int, list["Cone"], list[tuple[int]]]:
    cones = list(graph._cone_objects.values())
    step_count = 0
    actions = []

    ## stores dict (extraneous set points, barycentric lambdas) for each Cone observed. Used for quick lookup
    extraneous_set_cache: dict["Cone", dict[Vector, tuple]] = {}
    ## retrieves (points, lambdas) or adds to dict if not yet there
    def extraneous(cone):
        d = extraneous_set_cache.get(cone)
        if d is None:
            pts, lams = cone.extraneousSet
            d = {p: lam for p, lam in zip(pts, lams) if isPrimitiveNonzero(p)}
            extraneous_set_cache[cone] = d
        return d

    while any(cone.isSingular for cone in cones): 
        data = [(c.multiplicity, extraneous(c)) for c in cones] ## tuple(mult, dict(points, lambdas))
        total = sum(mult for mult, _ in data) ## mult across all cones
        det_sum = {p: total for _, d in data for p in d} ## intiialize all points to total det

        for mult, d in data:
            for point, lambdas in d.items():
                det_sum[point] -= int(mult * (1 - sum(lambdas)))
        
        subdivision_point = min(det_sum, key = det_sum.get)
        cones = fanSubdivide(cones, subdivision_point)
        step_count += 1
        actions.append(subdivision_point)
    return step_count, cones, actions

def random_baseline(graph: CGLGraph, rng: random.Random, max_steps: int = 200) -> tuple[int, bool]:
    cones = list(graph._cone_objects.values())
    step_count = 0

    while any(cone.isSingular for cone in cones):
        if step_count >= max_steps:
            return step_count, False

        candidates = set()
        for cone in cones:
            pts, lams = cone.extraneousSet
            for point in pts:
                if isPrimitiveNonzero(point):
                    candidates.add(point)

        if not candidates:
            return step_count, False

        point = rng.choice(list(candidates))
        cones = fanSubdivide(cones, point)
        step_count += 1

    return step_count, True

## reward function module so we can easily switch our reward function
def reward_function(
    reward_type: int,
    resolved: bool, 
    t: int, 
    state: CGLGraph, 
    agent_steps_taken: int, 
    baseline_steps_taken: int, 
    timeout_penalty: float,
    max_steps: int,
    root_mult: int,
    baseline_left: int,
    baseline_to_go: Optional[list[int,...]] = None, 
    ) -> float:
    if resolved:
        variants = [
            lambda: float(baseline_steps_taken - agent_steps_taken), #0 standard
            lambda: float((baseline_steps_taken - agent_steps_taken)/max(baseline_steps_taken, 1)), #1 normalized by baseline
            lambda: float(-agent_steps_taken), #2 all negative
            lambda: float(baseline_steps_taken - agent_steps_taken)/root_mult, #3 alperen's suggestion
        ]
        reward = variants[reward_type]()
            
    else:
        variants = [
            lambda: -timeout_penalty, #0 standard
            lambda: float((baseline_steps_taken - agent_steps_taken)/max(baseline_steps_taken, 1)) - timeout_penalty, #1 normalized by baseline
            lambda: -timeout_penalty,#2 all negative
            lambda: float(baseline_steps_taken - agent_steps_taken)/root_mult - timeout_penalty #3 alperen's suggestion
        ]
        reward = variants[reward_type]()
    return reward

# ===========================================================================================
#  TRAINING LOOP
# ===========================================================================================

## Outputs the states visited over the episode, with attached performance targets for the loss.
def self_play_episode(
    *,
    model: torch.nn.Module, ## evaluation model
    max_steps: int, ## maximum subdivisions that can be applied before timing out
    mcts_sims: int,
    initial_state: CGLGraph, 
    device: torch.device,
    rng: random.Random,
    temperature: float = 1.0,
    c_puct: float = 1.5,
    timeout_penalty: float = 0.0, 
    dirichlet_alpha: Optional[float] = None,
    dirichlet_eps: float = 0.25,
    diag: Optional["Diagnostics"] = None, # for logging
    traj: Optional["TrajectoryLogger"] = None, # for logging 
    episode_idx: int = -1, # for logging
    timeout_multiplier: float = 2.0,
    early_termination: bool = False,
    reward_type: int
    ) -> List[CGLGraph]:
    states = []
    policies = []
    action_options_history = [] ## latticeIds of the available actions at each time step. 
    baseline_to_go = []         ## min_sum rays-to-finish from each visited state

    ## for data logging
    root_cone = next(iter(initial_state._cone_objects.values()))
    root_n = len(root_cone.rays[0])
    root_mult = int(root_cone.multiplicity)
    resolved = False
    root_rays = next(iter(initial_state._cone_objects.values())).rays   
    subdivision_points = []

    state = initial_state
    baseline_steps_taken, baseline_fan, baseline_actions = min_sum(state)
    random_steps, random_resolved = random_baseline(initial_state, rng, max_steps=max_steps * 3)

    search = MCTS(
            model = model,
            num_simulations = mcts_sims,
            c_puct = c_puct,
            dirichlet_alpha = dirichlet_alpha,
            dirichlet_eps = dirichlet_eps,
            device = device)
    reuse_node = None

    ## run full MCTS from initial state to decomposition. 
    for i in range(max_steps):
        if state.isDecomposed():
            resolved = True
            break
        
        if early_termination and i >= timeout_multiplier * baseline_steps_taken:
            break
        
        result = search.run(state, temperature = temperature, add_root_noise = True, reuse_node = reuse_node)        

        ## log history
        states.append(state.copy())
        policies.append(torch.tensor(result.visit_probs, dtype = torch.float32))
        action_options_history.append(list(result.actions))

        baseline_from_here, _, _ = min_sum(state)
        baseline_to_go.append(baseline_from_here)

        # Training uses MCTS visit policy
        a = result.best_action
        subdivision_points.append(state._lattice_id_to_coord[a])   # log history
        state.subdivide(a)

    if resolved:
        baseline_left = 0
    else:
        baseline_left,_ ,_ = min_sum(state)

    ## capture the terminal fan now
    final_fan = [c.rays for c in state._cone_objects.values()]
    
    examples: List[HeteroData] = []
    agent_steps_taken = len(states) 
    rewards: List[float] = [] ## info for logging

    ## for each observed state, attach training targets
    for t, state in enumerate(states):
        reward = reward_function(
        reward_type,
        resolved, 
        t, 
        state, 
        agent_steps_taken, 
        baseline_steps_taken, 
        timeout_penalty, 
        max_steps, 
        root_mult=root_mult,
        baseline_left=baseline_left,
        baseline_to_go=baseline_to_go)
        rewards.append(reward)

        attach_targets(state, 
        target_policy = policies[t], 
        actions = action_options_history[t], 
        target_value = reward) 

        ## exclude terminal states from training examples
        if len(state.getValidActions()) > 0:
            examples.append(state)

    # log cone subdivision trajectory (agent's action choices)
    if traj is not None:
        traj.log(episode=episode_idx, root_rays=root_rays, points=subdivision_points,
                 final_fan=final_fan, resolved=resolved, agent_rays=agent_steps_taken, 
                 baseline_rays=baseline_steps_taken, baseline_points=baseline_actions,                     
                 baseline_fan=[c.rays for c in baseline_fan])     

    # log cone evaluation metrics
    if diag is not None:
        ## per-episode solution quality (inserted rays vs min_sum baseline)
        diag.log_episode(
            episode=episode_idx, n=root_n, mult=root_mult,
            agent_rays=agent_steps_taken,
            baseline_rays=baseline_steps_taken,
            random_rays=random_steps, random_resolved=random_resolved,
            resolved=resolved or states[-1].isDecomposed()
        )
        ## value-head calibration: predicted value vs reward target over visited states
        if states:
            try:
                was_training = model.training
                model.eval()
                preds: List[float] = []
                with torch.no_grad():
                    loader = DataLoader(states, batch_size = len(states), shuffle = False)
                    for batch in loader:
                        batch = batch.to(device)
                        preds.extend(extract_value(model(batch)).tolist())
                if was_training:
                    model.train()
                diag.log_calibration(episode = episode_idx, predicted = preds, target = rewards)
            except Exception as e:
                diag.warn_calibration_once(e)

    return examples

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
    
def build_parser() -> argparse.ArgumentParser:
    """All train.py CLI args.  Exposed so sweep.py can discover the full set of
    tunable hyperparameters (names, types, defaults) without drifting out of sync."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--epochs-per-iter", type=int, default=1)
    parser.add_argument("--mcts-sims", type=int, default=16)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--timeout-penalty", type=float, default=1.0) ## positive penalty -> negative reward
    parser.add_argument("--timeout-multiplier", type=float, default=2.0) ## how many times the baseline are tolerated before the computation times out
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--det-min", type=int, default=2)
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
    parser.add_argument("--save", type=str, default=None,
                        help="Where to write the checkpoint. Defaults to <run_dir>/model.pt "
                             "so each run keeps its own model alongside its diagnostics.")
    parser.add_argument("--resume", type=str, default="", help="Optional checkpoint to continue training from.")
    parser.add_argument("--dirichlet-alpha", type=float, default=0.3)
    parser.add_argument("--dirichlet-eps", type=float, default=0.25)
    parser.add_argument("--embedding-size", type=int, default=64)
    parser.add_argument("--min-dimension", type=int, default=2)
    parser.add_argument("--max-dimension", type=int, default=4)
    parser.add_argument("--diag-dir", type=str, default="results")
    parser.add_argument("--early-termination", action="store_true")
    parser.add_argument("--reward-type", type=int,default=0)
    parser.add_argument("--layer-type", choices=["GAT", "GraphConv"], default="GraphConv")

    # old experiment controls.
    #parser.add_argument("--enumerator", choices=["fpp", "hybrid", "grid"], default="fpp")
    #parser.add_argument("--max-candidates", type=int, default=64)
    #parser.add_argument("--fpp-max-points", type=int, default=20_000)
    #parser.add_argument("--include-boundary-actions", action="store_true")
    #parser.add_argument("--grid-max-points", type=int, default=0)
    #parser.add_argument("--grid-random-trials", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    ## let the network learn be initialized to the structure of the graph
    dummy_cone = Cone(generateRandomCone(n = args.max_dimension, d = args.det_max, rng = rng))
    meta_state = CGLGraph(dimension = args.max_dimension)
    meta_state.addConeNode(dummy_cone)
    model = network(
        meta_state.metadata(),
        hidden=args.hidden_dim,
        embedding_size=args.embedding_size,
        num_layers=args.num_blocks,
        dropout=args.dropout,
        layer_type=args.layer_type
    ).to(device)

    meta_state = meta_state.to(device)
    model.train()
    model(meta_state)
    model.zero_grad(set_to_none = True)

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

    ## for data visualization
    ## =======================================================================================
    run_dir = os.path.join(args.diag_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    ## default checkpoint lives inside this run's folder so it is never silently
    ## overwritten by a later run and stays co-located with its config/diagnostics.
    save_path = args.save if args.save else os.path.join(run_dir, "model.pt")
    with open(os.path.join(run_dir, "config.json"), "w") as _f:
        json.dump(vars(args), _f, indent=2)
    diag = Diagnostics(outdir=run_dir)
    traj = TrajectoryLogger(os.path.join(run_dir, "trajectories.jsonl"))
    ## =======================================================================================

    args_dict = vars(args)
    width = max(len(k) for k in args_dict)
    print("config:")
    for k in sorted(args_dict):
        print(f"  {k:<{width}} = {args_dict[k]}")

    ## generate semirandom training examples
    training_examples = []
    for i in range(args.episodes):
        n = rng.randint(args.min_dimension, args.max_dimension)
        d = rng.randint(args.det_min, args.det_max)
        g = CGLGraph(dimension = args.max_dimension)
        g.addConeNode(Cone(generateRandomCone(n,d, rng = rng)))
        training_examples.append(g)

    train_start = time.perf_counter()
    for ep in trange(args.episodes, desc="self-play", unit="ep",
                     bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_inv_fmt}{postfix}]"):
        examples = self_play_episode(
            model=model,
            max_steps=args.max_steps,
            mcts_sims=args.mcts_sims,
            initial_state=training_examples[ep],
            device=device,
            rng=rng,
            temperature=args.temperature,
            c_puct=args.c_puct,
            timeout_penalty=args.timeout_penalty,
            dirichlet_alpha=args.dirichlet_alpha,
            dirichlet_eps=args.dirichlet_eps,
            diag=diag,
            traj=traj,
            episode_idx=ep,
            timeout_multiplier = args.timeout_multiplier,
            early_termination=args.early_termination,
            reward_type = args.reward_type
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
            diag.log_train(
                episode=ep,
                loss=metrics["loss"],
                policy_loss=metrics["policy_loss"],
                value_loss=metrics["value_loss"])
            if (ep + 1) % max(1, args.episodes // 20) == 0:
                tqdm.write(f"ep={ep+1} replay={len(replay)} metrics={metrics}")
                
        # ---- periodic checkpoint (NEW) ----  <-- here, inside the loop, outside the batch-size gate
        if (ep + 1) % 200 == 0:
            torch.save({"model": model.state_dict(), "args": vars(args), "episode": ep + 1},
                       save_path)

    ## record wall-clock timing of the self-play/training loop (read by read_diagnostics.py)
    train_elapsed = time.perf_counter() - train_start
    sec_per_episode = train_elapsed / max(1, args.episodes)
    with open(os.path.join(run_dir, "timing.json"), "w") as _f:
        json.dump({"elapsed_seconds": train_elapsed,
                   "episodes": args.episodes,
                   "sec_per_episode": sec_per_episode}, _f, indent=2)
    print(f"elapsed {train_elapsed:.1f}s over {args.episodes} episodes "
          f"({sec_per_episode:.3f} s/episode)")

    ## for data visualization
    ## =======================================================================================
    torch.save({"model": model.state_dict(), "args": vars(args)}, save_path)
    print(f"saved {save_path}")

    traj.close()
    paths = diag.finish()
    paths["trajectories"] = traj.path          # fold the JSONL into the same summary
    print(f"diagnostics written to {run_dir}/:")
    for k, v in paths.items():
        if v is not None:
            print(f"  {k:<18} = {v}")
    ## =======================================================================================

if __name__ == "__main__":
    main()
