"""
test_step6.py
-------------
Phase 6: Aggregation & Model Update for all 20 devices (1 round).
Run with: python test_step6.py
"""

import torch
import copy
from topology import create_devices, build_topology
from data_loader import load_mnist, distribute_iid
from grad_compression import compress_gradient
from privacy import apply_differential_privacy
from gossip import gossip_exchange
from byzantine import run_phase5
from aggregation import run_phase6

MODEL_SIZE    = 101_770
LEARNING_RATE = 0.1


def section(title):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")


def main():
    section("PHASE 0 — Initialization")
    devices    = create_devices()
    G, manager = build_topology(devices)
    W0 = copy.deepcopy(devices[0].model.state_dict())
    for d in devices:
        d.model.load_state_dict(W0)
        d.init_reputation()
    X_train, y_train, X_test, y_test = load_mnist()
    distribute_iid(X_train, y_train, devices)
    print(f"  {len(devices)} devices ready.")

    # Accuracy before any training
    acc_initial = {d.id: d.evaluate(X_test, y_test) for d in devices}
    avg_initial = sum(acc_initial.values()) / len(acc_initial)
    print(f"  Initial accuracy (random weights): {avg_initial:.2f}%")

    section("PHASE 1 to 3 — All Devices")
    noisy_gradients = {}
    for d in devices:
        gradient              = d.local_train(batch_size=32)
        compressed            = compress_gradient(gradient, d, MODEL_SIZE)
        noisy_grad, _         = apply_differential_privacy(compressed)
        noisy_gradients[d.id] = noisy_grad
    print(f"  Done.")

    section("PHASE 4 — Gossip Exchange")
    received = gossip_exchange(devices, noisy_gradients, round_num=1)
    print(f"  Done.")

    section("PHASE 5 — Byzantine Detection")
    all_quality = run_phase5(devices, received, noisy_gradients)
    byz_flags = sum(
        1 for scores in all_quality.values()
        for info in scores.values() if info["is_byzantine"]
    )
    print(f"  Done. Total Byzantine flags: {byz_flags}")

    section("PHASE 6 — Aggregation & Model Update (PDF Section 3.6)")

    acc_before = {d.id: d.evaluate(X_test, y_test) for d in devices}
    agg_info   = run_phase6(devices, received, all_quality, LEARNING_RATE)
    acc_after  = {d.id: d.evaluate(X_test, y_test) for d in devices}

    # Device 1 aggregation detail
    info1 = agg_info.get(1, {})
    print(f"\n  Device 1 aggregation (PDF Section 3.6 format):")
    print(f"    weight_own = 1.0000  (always trust self)")
    for nid, w in sorted(info1.get("weights_used", {}).items()):
        if nid == "own":
            continue
        is_byz = all_quality[1].get(nid, {}).get("is_byzantine", False)
        tag    = "  <- EXCLUDED (Byzantine)" if is_byz else ""
        print(f"    weight_D{nid:02d} = {w:.4f}{tag}")
    print(f"    total_weight     = {info1.get('total_weight', 0):.4f}")
    print(f"    contributors     = {info1.get('num_contributors', 0)}")
    print(f"    excluded (byz)   = {info1.get('num_excluded', 0)}")

    # Accuracy comparison
    print(f"\n  Model accuracy after Round 1:")
    print(f"  {'Device':<22} {'Before':>8} {'After':>8} {'Change':>8}")
    print(f"  {'-'*48}")
    for d in devices:
        bef    = acc_before[d.id]
        aft    = acc_after[d.id]
        change = aft - bef
        tag    = " [BYZ]" if d.is_byzantine else ""
        sign   = "+" if change >= 0 else ""
        print(f"  Device {d.id:2d}{tag:7s}       "
              f"{bef:>7.2f}%  {aft:>7.2f}%  {sign}{change:>6.2f}%")

    honest_before = [acc_before[d.id] for d in devices if not d.is_byzantine]
    honest_after  = [acc_after[d.id]  for d in devices if not d.is_byzantine]
    print(f"\n  Average accuracy (honest devices):")
    print(f"    Before: {sum(honest_before)/len(honest_before):.2f}%")
    print(f"    After : {sum(honest_after)/len(honest_after):.2f}%")

    print(f"\n[OK] Step 6 (Phase 6) complete!")
    print(f"     One full round complete: Phase 0 -> Phase 6")


if __name__ == "__main__":
    main()
