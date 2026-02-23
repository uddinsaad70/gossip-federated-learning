"""
test_step5.py
-------------
Phase 5: Byzantine Detection for all 20 devices (1 round).
Run with: python test_step5.py
"""

import torch
import copy
from topology import create_devices, build_topology
from data_loader import load_mnist, distribute_iid
from grad_compression import compress_gradient
from privacy import apply_differential_privacy
from gossip import gossip_exchange
from byzantine import run_phase5

MODEL_SIZE = 101_770


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

    section("PHASE 5 — Byzantine Detection (PDF Section 3.5)")
    all_quality = run_phase5(devices, received, noisy_gradients)

    # Device 1 detail
    print(f"\n  Device 1 evaluating received gradients:")
    print(f"  {'Neighbor':<12} {'cos_sim':>9} {'mag_ratio':>10} "
          f"{'outlier%':>10} {'quality':>9} {'rep_old':>8} "
          f"{'rep_new':>8} {'flag':>12}")
    print(f"  {'-'*80}")
    for nid, info in sorted(all_quality[1].items()):
        flag = "BYZANTINE" if info["is_byzantine"] else "trusted"
        print(f"  Device {nid:2d}      "
              f"{info['cos_sim']:>9.4f} "
              f"{info['mag_ratio']:>10.4f} "
              f"{info['outlier_pct']*100:>9.1f}% "
              f"{info['quality']:>9.4f} "
              f"{info['rep_old']:>8.4f} "
              f"{info['rep_new']:>8.4f} "
              f"{flag:>12}")

    section("BYZANTINE DETECTION SUMMARY")
    print(f"  {'Device':<22} {'Evaluated':>10} {'Byzantine detected':>25}")
    print(f"  {'-'*60}")
    total_flagged = 0
    for d in devices:
        scores   = all_quality[d.id]
        byz_list = [nid for nid, info in scores.items() if info["is_byzantine"]]
        byz_tag  = " [BYZ]" if d.is_byzantine else ""
        total_flagged += len(byz_list)
        print(f"  Device {d.id:2d}{byz_tag:7s}        "
              f"{len(scores):>10}  "
              f"{str(byz_list):>25}")
    print(f"\n  Total Byzantine flags: {total_flagged}")

    print(f"\n  How honest devices rated Device 17 (Byzantine):")
    print(f"  {'Evaluator':<15} {'cos_sim':>9} {'mag_ratio':>10} "
          f"{'quality':>9} {'flag':>12}")
    print(f"  {'-'*58}")
    for d in devices:
        if d.is_byzantine:
            continue
        if 17 in all_quality[d.id]:
            info = all_quality[d.id][17]
            flag = "BYZANTINE" if info["is_byzantine"] else "trusted"
            print(f"  Device {d.id:2d}        "
                  f"{info['cos_sim']:>9.4f} "
                  f"{info['mag_ratio']:>10.4f} "
                  f"{info['quality']:>9.4f} "
                  f"{flag:>12}")

    section("REPUTATION AFTER ROUND 1")
    print(f"  {'Device':<22} {'Min rep':>10} {'Max rep':>10} {'Avg rep':>10}")
    print(f"  {'-'*54}")
    for d in devices:
        if not d.reputation:
            continue
        reps    = list(d.reputation.values())
        byz_tag = " [BYZ]" if d.is_byzantine else ""
        print(f"  Device {d.id:2d}{byz_tag:7s}      "
              f"{min(reps):>10.4f} "
              f"{max(reps):>10.4f} "
              f"{sum(reps)/len(reps):>10.4f}")

    print(f"\n[OK] Step 5 (Phase 5) complete!")
    print(f"     all_quality -> Phase 6 (Aggregation)")


if __name__ == "__main__":
    main()