"""
gossip.py
---------
Phase 4: Gossip Exchange

Source: Gossip FL PDF Section 3.4

প্রতিটা device তার noisy_gradient (Phase 3 output) প্রতিটা
neighbor-এ পাঠায়। Exchange bidirectional:
    D_i → D_j এবং D_j → D_i একই সাথে হয়।

Message format (PDF Section 3.4, Step 1):
    {
        'sender'  : device_id,
        'round'   : t,
        'gradient': {layer_name: noisy_tensor},  ← ∇W_noisy
        'metadata': {
            layer_name: {
                'Cr'            : float,
                'mask'          : np.ndarray,
                'original_shape': tuple,
            }
        }
    }

After exchange, each device holds a list of received messages
from all its neighbors — ready for Phase 5 (Byzantine Detection).
"""

import numpy as np


# ──────────────────────────────────────────
# Step 1: Message Preparation
# PDF Section 3.4, Step 1
# ──────────────────────────────────────────
def prepare_message(device, noisy_gradient: dict, round_num: int) -> dict:
    """
    Packages ∇W_noisy + metadata into a message.
    This is what gets sent to each neighbor.

    PDF example:
        message_from_1 = {
            'sender'  : 1,
            'round'   : 1,
            'gradient': [0.27, 0.03, 0.03, -0.01, ...],
            'metadata': {'compression_ratio': 0.156, 'dct_mask': [...]}
        }
    """
    gradient_data = {}
    metadata      = {}

    for name, pkg in noisy_gradient.items():
        gradient_data[name] = pkg["data"]       # noisy tensor
        metadata[name] = {
            "Cr":             pkg["Cr"],
            "mask":           pkg["mask"],
            "original_shape": pkg["original_shape"],
        }

    return {
        "sender":   device.id,
        "round":    round_num,
        "gradient": gradient_data,
        "metadata": metadata,
    }


# ──────────────────────────────────────────
# Step 2: Bidirectional Exchange
# PDF Section 3.4, Step 2
# ──────────────────────────────────────────
def gossip_exchange(devices: list, noisy_gradients: dict,
                    round_num: int) -> dict:
    """
    Every device sends its message to all neighbors.
    Every device receives messages from all neighbors.

    PDF:
        D1 ↔ D2: Send from 1, Receive from 2
        D1 ↔ D3: Send from 1, Receive from 3
        D1 ↔ D4: Send from 1, Receive from 4

    Parameters
    ----------
    devices         : list of all EdgeDevice objects
    noisy_gradients : {device_id: noisy_gradient}  ← Phase 3 output
    round_num       : current training round

    Returns
    -------
    received : {device_id: [msg_from_neighbor1, msg_from_neighbor2, ...]}
    """
    # Each device prepares its outgoing message
    outbox = {}
    for d in devices:
        if d.id in noisy_gradients:
            outbox[d.id] = prepare_message(d, noisy_gradients[d.id], round_num)

    # Deliver messages — bidirectional
    received = {d.id: [] for d in devices}
    seen     = {d.id: set() for d in devices}   # avoid duplicates

    for d in devices:
        if d.id not in outbox:
            continue
        for neighbor_id in d.neighbors:
            # D sends to neighbor → neighbor receives from D
            if d.id not in seen[neighbor_id] and d.id in outbox:
                received[neighbor_id].append(outbox[d.id])
                seen[neighbor_id].add(d.id)

            # neighbor sends to D → D receives from neighbor
            if neighbor_id not in seen[d.id] and neighbor_id in outbox:
                received[d.id].append(outbox[neighbor_id])
                seen[d.id].add(neighbor_id)

    return received


# ──────────────────────────────────────────
# Traffic Analysis
# PDF Section 3.4
# ──────────────────────────────────────────
def compute_traffic(devices: list, noisy_gradients: dict) -> dict:
    """
    Estimates bytes sent/received per device.
    Only nonzero values are counted (sparse transmission).
    float32 = 4 bytes per value.
    """
    traffic = {}
    for d in devices:
        if d.id not in noisy_gradients:
            continue
        ng = noisy_gradients[d.id]
        nonzero = sum(
            (pkg["data"] != 0).sum().item() for pkg in ng.values()
        )
        bytes_per_msg = nonzero * 4
        n_neighbors   = len(d.neighbors)
        traffic[d.id] = {
            "bytes_per_message": bytes_per_msg,
            "num_neighbors":     n_neighbors,
            "bytes_sent":        bytes_per_msg * n_neighbors,
            "bytes_received":    bytes_per_msg * n_neighbors,
            "total_bytes":       bytes_per_msg * n_neighbors * 2,
        }
    return traffic
