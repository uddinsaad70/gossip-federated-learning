"""
test_step4.py
-------------
Phase 4: Gossip Exchange — সব 20টা device-এর জন্য।
Run with: python test_step4.py
"""

import torch
import copy
from topology import create_devices, build_topology
from data_loader import load_mnist, distribute_iid
from grad_compression import compress_gradient
from privacy import apply_differential_privacy
from gossip import gossip_exchange, compute_traffic
from device import count_parameters

MODEL_SIZE = 101_770


def section(title):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")


def main():
    # ── Phase 0: Setup ──────────────────────────────────
    section("PHASE 0 — Initialization")
    devices    = create_devices()
    G, manager = build_topology(devices)

    W0 = copy.deepcopy(devices[0].model.state_dict())
    for d in devices:
        d.model.load_state_dict(W0)
        d.init_reputation()

    X_train, y_train, X_test, y_test = load_mnist()
    distribute_iid(X_train, y_train, devices)
    print(f"  {len(devices)} devices ready, W0 synced, MNIST distributed.")

    # ── Phase 1 → 2 → 3: সব device ─────────────────────
    section("PHASE 1→2→3 — All 20 Devices")

    noisy_gradients = {}
    for d in devices:
        gradient           = d.local_train(batch_size=32)
        compressed         = compress_gradient(gradient, d, MODEL_SIZE)
        noisy_grad, _      = apply_differential_privacy(compressed)
        noisy_gradients[d.id] = noisy_grad

    print(f"  All {len(devices)} devices completed Phase 1→2→3.")
    print(f"  noisy_gradients ready: {list(noisy_gradients.keys())}")

    # ── Phase 4: Gossip Exchange ─────────────────────────
    section("PHASE 4 — Gossip Exchange (PDF Section 3.4)")

    received = gossip_exchange(devices, noisy_gradients, round_num=1)

    # Per-device exchange summary
    print(f"\n  {'Device':<20} {'Sent to':^30} {'Received from'}")
    print(f"  {'-'*70}")
    for d in devices:
        byz          = " [BYZ]" if d.is_byzantine else ""
        sent_to      = str(d.neighbors)
        recv_from    = [msg["sender"] for msg in received[d.id]]
        print(f"  Device {d.id:2d}{byz:7s}   "
              f"sent → {str(sent_to):<28} "
              f"recv ← {recv_from}")

    # Verify bidirectional exchange
    print(f"\n  Bidirectional exchange verification:")
    errors = 0
    for d in devices:
        recv_senders = {msg["sender"] for msg in received[d.id]}
        for neighbor_id in d.neighbors:
            if neighbor_id not in recv_senders:
                print(f"    [MISSING] Device {d.id} did not receive from {neighbor_id}")
                errors += 1
    if errors == 0:
        print(f"    All devices received from all their neighbors. OK")

    # Message content — Device 1 (PDF Section 3.4 example)
    print(f"\n  Message received by Device 1 (first message):")
    if received[1]:
        msg = received[1][0]
        print(f"    sender          : {msg['sender']}")
        print(f"    round           : {msg['round']}")
        print(f"    gradient layers : {list(msg['gradient'].keys())}")
        for name, tensor in msg["gradient"].items():
            nz     = (tensor != 0).sum().item()
            sample = tensor.detach().numpy().flatten()[:4]
            print(f"      {name:30s} nonzero={nz}  "
                  f"first 4: {[round(float(v),4) for v in sample]}")
        print(f"    metadata (Cr)   : "
              f"{ {k: round(v['Cr'],4) for k,v in msg['metadata'].items()} }")

    # Traffic analysis (PDF Section 3.4)
    section("TRAFFIC ANALYSIS (PDF Section 3.4)")
    traffic = compute_traffic(devices, noisy_gradients)

    print(f"  {'Device':<22} {'Bytes/msg':>10}  "
          f"{'Neighbors':>10}  {'Sent':>10}  {'Recv':>10}  {'Total':>10}")
    print(f"  {'-'*75}")

    for d in devices:
        if d.id not in traffic:
            continue
        t   = traffic[d.id]
        byz = " [BYZ]" if d.is_byzantine else ""
        print(f"  Device {d.id:2d}{byz:7s}      "
              f"{t['bytes_per_message']:>10,}  "
              f"{t['num_neighbors']:>10}  "
              f"{t['bytes_sent']:>10,}  "
              f"{t['bytes_received']:>10,}  "
              f"{t['total_bytes']:>10,}")

    # Average traffic
    avg_total = sum(t["total_bytes"] for t in traffic.values()) / len(traffic)
    print(f"\n  Average total traffic per device: {avg_total:,.0f} bytes")
    print(f"  (Sparse transmission — only nonzero values sent)")

    # Confirm received is ready for Phase 5
    section("READY FOR PHASE 5")
    print(f"  received dict keys  : {list(received.keys())}")
    print(f"  Each value is a list of messages from neighbors.")
    print(f"  Example — Device 1 inbox:")
    for msg in received[1]:
        print(f"    from Device {msg['sender']:2d} | "
              f"layers: {list(msg['gradient'].keys())}")

    print(f"\n[OK] Step 4 (Phase 4) complete!")
    print(f"     'received' → Phase 5 (Byzantine Detection) এ pass হবে")

    return received, devices, noisy_gradients, X_test, y_test


if __name__ == "__main__":
    main()
