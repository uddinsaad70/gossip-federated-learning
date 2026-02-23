"""
aggregation.py
--------------
Phase 6: Weighted Aggregation & Model Update

Source: Gossip FL PDF Section 3.6

Steps:
    1. For each received gradient:
           weight_j = reputation_j x quality_j
           (Byzantine: quality=0 -> weight=0 -> excluded automatically)

    2. Own gradient always included:
           weight_own = 1.0

    3. Decompress received gradients (IDCT to spatial domain)

    4. Weighted average:
           aggregated = sum(weight_j x grad_j) / sum(weight_j)

    5. Model update:
           W_new = W_old + learning_rate x aggregated

PDF Section 3.6:
    "The aggregated gradient is a weighted combination where
     weights reflect both reputation and gradient quality,
     effectively excluding Byzantine contributions."
"""

import torch
import numpy as np
from grad_compression import decompress_gradient


def aggregate_and_update(device, received_messages: list,
                          quality_scores: dict,
                          learning_rate: float = 0.1) -> dict:
    """
    Weighted aggregation and model update for one device.

    Parameters
    ----------
    device            : EdgeDevice (has local_gradient and reputation)
    received_messages : list of messages from Phase 4
    quality_scores    : {sender_id: {'quality', 'rep_new', ...}} from Phase 5
    learning_rate     : eta (default 0.1)

    Returns
    -------
    info : {
        'total_weight'    : float,
        'num_contributors': int,
        'num_excluded'    : int  (Byzantine, zero-weight)
        'weights_used'    : {sender_id: weight}
    }
    """
    if device.local_gradient is None:
        return {}

    # Build (weight, gradient) list
    weighted_grads = []
    total_weight   = 0.0
    weights_used   = {}
    num_excluded   = 0

    # Own gradient — weight = 1.0 (always trust self)
    weighted_grads.append((1.0, device.local_gradient))
    total_weight        += 1.0
    weights_used["own"] = 1.0

    # Received gradients
    msg_by_sender = {msg["sender"]: msg for msg in received_messages}

    for sender_id, qinfo in quality_scores.items():
        quality = qinfo["quality"]
        rep     = device.reputation.get(sender_id, 1.0)
        weight  = rep * quality   # PDF Section 3.6

        if weight <= 0:
            num_excluded += 1
            weights_used[sender_id] = 0.0
            continue

        msg = msg_by_sender.get(sender_id)
        if msg is None:
            continue

        # Decompress received gradient (IDCT to spatial domain for update)
        compressed_pkg = {
            name: {
                "data":           tensor,
                "mask":           msg["metadata"][name]["mask"],
                "original_shape": msg["metadata"][name]["original_shape"],
                "Cr":             msg["metadata"][name]["Cr"],
            }
            for name, tensor in msg["gradient"].items()
        }
        recv_gradient = decompress_gradient(compressed_pkg)

        weighted_grads.append((weight, recv_gradient))
        total_weight            += weight
        weights_used[sender_id]  = round(weight, 4)

    if total_weight == 0 or len(weighted_grads) == 0:
        return {}

    # Weighted average
    aggregated = {}
    for name in device.local_gradient.keys():
        agg = torch.zeros_like(device.local_gradient[name])
        for weight, grad in weighted_grads:
            if name in grad:
                agg += weight * grad[name]
        aggregated[name] = agg / total_weight

    # Model update: W_new = W_old + lr x aggregated
    with torch.no_grad():
        for name, param in device.model.named_parameters():
            if name in aggregated:
                param.data += learning_rate * aggregated[name]

    return {
        "total_weight":     round(total_weight, 4),
        "num_contributors": len(weighted_grads),
        "num_excluded":     num_excluded,
        "weights_used":     weights_used,
    }


def run_phase6(devices: list, received: dict,
               all_quality: dict,
               learning_rate: float = 0.1) -> dict:
    """
    Run Phase 6 for all devices.

    Returns
    -------
    agg_info : {device_id: aggregation info dict}
    """
    agg_info = {}
    for d in devices:
        msgs    = received.get(d.id, [])
        quality = all_quality.get(d.id, {})
        agg_info[d.id] = aggregate_and_update(
            d, msgs, quality, learning_rate
        )
    return agg_info
