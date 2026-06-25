from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor
import torch.nn.functional as F
from torch_geometric.data import HeteroData


def _choose_action_node_type(
    batch: HeteroData,
    out: dict,
    action_node_type: Optional[str],
) -> str:
   

    if action_node_type is not None:
        return action_node_type

    if "action_node_type" in out:
        return out["action_node_type"]

    for candidate in ["fpp", "action"]:
        if candidate in batch.node_types and hasattr(batch[candidate], "target_policy"):
            return candidate

    raise ValueError(
        "Could not infer action node type. Expected one of "
        "'fpp', or 'action' to have target_policy."
    )


def _get_target_value(batch: HeteroData, device: torch.device) -> Tensor:
  

    
       # model outputs value V(s)

    # If the data stores:
       # target_value:
      #      use it directly

     #   target_cost:
     #       convert cost to value by value = -cost

    #This matches MCTS backup where every subdivision has reward -1.
  

    if hasattr(batch, "target_value"):
        return batch.target_value.to(device).view(-1).float()

    if hasattr(batch, "target_cost"):
        return -batch.target_cost.to(device).view(-1).float()

    raise ValueError(
        "Batch must contain either batch.target_value or batch.target_cost."
    )


def alphazero_cost_loss(
    out: dict,
    batch: HeteroData,
    *,
    value_weight: float = 0.25,
    action_node_type: Optional[str] = None,
    model=None,
) -> dict[str, Tensor]:
  
    #AlphaZero-style policy/value loss.
    #Policy loss:
    #cross entropy between MCTS visit distribution and model policy.

    #Value loss:
    #mean squared error between predicted value and target value.

    #Required model outputs:
       # out["action_log_probs"]:
        #    log-probabilities over action nodes

        #out["value"]:
            graph-level value prediction

    #Required batch fields:
     #   batch[action_node_type].target_policy:
      #      MCTS visit distribution over action nodes

       # batch.target_value or batch.target_cost:
        #    graph-level target
    

    if not isinstance(out, dict):
        raise TypeError(
            "Training loss expects model(batch) to return a dict. "
            "For MCTS you can still have model.predict_policy_value(data) "
            "return (priors, value)."
        )

    if "action_log_probs" not in out:
        raise KeyError('Model output must contain out["action_log_probs"].')

    if "value" not in out:
        raise KeyError('Model output must contain out["value"].')

    action_log_probs = out["action_log_probs"].view(-1)
    value_pred = out["value"].view(-1)

    device = action_log_probs.device

    
    # Policy target

    action_type = _choose_action_node_type(
        batch=batch,
        out=out,
        action_node_type=action_node_type,
    )

    if not hasattr(batch[action_type], "target_policy"):
        raise ValueError(
            f'batch["{action_type}"].target_policy is missing.'
        )

    target_policy = batch[action_type].target_policy.to(device).view(-1).float()

    # Some models score only valid action nodes.
    # In that case, the model can return valid_action_mask so the target
    # policy is filtered to the same action set.
    if "valid_action_mask" in out:
        valid_mask = out["valid_action_mask"].to(device).bool()
        target_policy = target_policy[valid_mask]

    if target_policy.numel() != action_log_probs.numel():
        raise ValueError(
            "Policy target size does not match model action output size. "
            f"target_policy has {target_policy.numel()} entries, "
            f"but action_log_probs has {action_log_probs.numel()} entries. "
            "Check action node ordering and valid_action_mask."
        )

    
    # Batch-aware policy normaliz
    # For a single graph, this is just the usual policy loss.
    # For a PyG batch, each graph has its own action distribution.

    if "action_batch" in out:
        action_batch = out["action_batch"].to(device).view(-1)
    elif hasattr(batch[action_type], "batch"):
        action_batch = batch[action_type].batch.to(device).view(-1)
        if "valid_action_mask" in out:
            action_batch = action_batch[out["valid_action_mask"].to(device).bool()]
    else:
        action_batch = torch.zeros(
            action_log_probs.numel(),
            dtype=torch.long,
            device=device,
        )

    num_graphs = int(action_batch.max().item()) + 1 if action_batch.numel() > 0 else 1

    # Normalize target policy per graph for if numerical errors make target_policy sum slightly
    # different from 1 inside a graph
    policy_sums = torch.zeros(
        num_graphs,
        dtype=target_policy.dtype,
        device=device,
    )

    policy_sums.index_add_(0, action_batch, target_policy)
    target_policy = target_policy / policy_sums[action_batch].clamp_min(1e-12)

    policy_loss = -(
        target_policy * action_log_probs
    ).sum() / max(num_graphs, 1)

    
    # Value target

    target_value = _get_target_value(batch, device=device)

    if target_value.numel() != value_pred.numel():
        raise ValueError(
            "Value target size does not match model value output size. "
            f"target_value has {target_value.numel()} entries, "
            f"value_pred has {value_pred.numel()} entries."
        )

    value_loss = F.mse_loss(value_pred, target_value)

    loss = policy_loss + value_weight * value_loss

    return {
        "loss": loss,
        "policy_loss": policy_loss,
        "value_loss": value_loss,
    }
