# """
# main.py
# -------
# Full Gossip Federated Learning — 30 Training Rounds

# Architecture:
#     Phase 0 : Initialization (once, before all rounds)
#     Round 1 to T (30 rounds):
#         Phase 1 : Local Training
#         Phase 2 : Gradient Compression
#         Phase 3 : Differential Privacy
#         Phase 4 : Gossip Exchange
#         Phase 5 : Byzantine Detection
#         Phase 6 : Aggregation & Model Update

# Files and their roles:
#     device.py          — EdgeDevice class, SimpleCNN model, local_train()
#     topology.py        — build network graph, assign neighbors
#     data_loader.py     — load MNIST, distribute IID
#     grad_compression.py — Phase 2: DCT compression
#     privacy.py         — Phase 3: DP noise (AGC-DP)
#     gossip.py          — Phase 4: bidirectional gradient exchange
#     byzantine.py       — Phase 5: quality scores, reputation update
#     aggregation.py     — Phase 6: weighted average, model update
#     main.py            — this file: orchestrates all rounds

# Run with: python main.py
# """

# import torch
# import copy
# import time
# from topology import create_devices, build_topology
# from data_loader import load_mnist, distribute_iid
# from grad_compression import compress_gradient
# from privacy import apply_differential_privacy
# from gossip import gossip_exchange
# from byzantine import run_phase5
# from aggregation import run_phase6

# # ── Configuration ──────────────────────────────────
# NUM_ROUNDS    = 500
# BATCH_SIZE    = 32
# LEARNING_RATE = 0.1
# MODEL_SIZE    = 101_770


# def section(title):
#     print(f"\n{'='*65}")
#     print(f"  {title}")
#     print(f"{'='*65}")


# def evaluate_all(devices, X_test, y_test):
#     """Evaluate all devices and return accuracy dict."""
#     return {d.id: d.evaluate(X_test, y_test) for d in devices}


# def avg_honest(acc_dict, devices):
#     honest = [acc_dict[d.id] for d in devices if not d.is_byzantine]
#     return sum(honest) / len(honest)


# def main():
#     # ──────────────────────────────────────────────
#     # PHASE 0: Initialization (runs ONCE)
#     # ──────────────────────────────────────────────
#     section("PHASE 0 — Initialization (once)")

#     devices    = create_devices()
#     G, manager = build_topology(devices)

#     # W0: Xavier-initialized model, propagated to all devices
#     # (PDF Section 2.2: "W0 propagated via gossip ~2 rounds")
#     W0 = copy.deepcopy(devices[0].model.state_dict())
#     for d in devices:
#         d.model.load_state_dict(W0)
#         d.init_reputation()

#     X_train, y_train, X_test, y_test = load_mnist()
#     distribute_iid(X_train, y_train, devices)

#     acc_init = evaluate_all(devices, X_test, y_test)
#     print(f"  Devices         : {len(devices)}")
#     print(f"  Byzantine       : {[d.id for d in devices if d.is_byzantine]}")
#     print(f"  Initial accuracy: {avg_honest(acc_init, devices):.2f}% (random weights)")

#     # Track metrics per round
#     history = []

#     # ──────────────────────────────────────────────
#     # TRAINING LOOP: Round 1 to NUM_ROUNDS
#     # ──────────────────────────────────────────────
#     section(f"TRAINING — {NUM_ROUNDS} Rounds")
#     print(f"\n  {'Round':>6}  {'Avg Acc':>9}  {'Byz Flags':>10}  {'Time':>7}")
#     print(f"  {'-'*38}")

#     for round_num in range(1, NUM_ROUNDS + 1):
#         t_start = time.time()

#         # Phase 1 → 3: All devices
#         noisy_gradients = {}
#         for d in devices:
#             gradient              = d.local_train(batch_size=BATCH_SIZE)
#             compressed            = compress_gradient(gradient, d, MODEL_SIZE)
#             noisy_grad, _         = apply_differential_privacy(compressed)
#             noisy_gradients[d.id] = noisy_grad

#         # Phase 4: Gossip Exchange
#         received = gossip_exchange(devices, noisy_gradients, round_num)

#         # Phase 5: Byzantine Detection
#         all_quality = run_phase5(devices, received, noisy_gradients)
#         byz_flags   = sum(
#             1 for scores in all_quality.values()
#             for info in scores.values() if info["is_byzantine"]
#         )

#         # Phase 6: Aggregation & Model Update
#         run_phase6(devices, received, all_quality, LEARNING_RATE)

#         # Evaluate
#         acc      = evaluate_all(devices, X_test, y_test)
#         avg_acc  = avg_honest(acc, devices)
#         elapsed  = time.time() - t_start

#         history.append({
#             "round":     round_num,
#             "avg_acc":   avg_acc,
#             "byz_flags": byz_flags,
#         })

#         print(f"  {round_num:>6}  {avg_acc:>8.2f}%  {byz_flags:>10}  {elapsed:>6.1f}s")

#     # ──────────────────────────────────────────────
#     # Final Results
#     # ──────────────────────────────────────────────
#     section("FINAL RESULTS")

#     final_acc = evaluate_all(devices, X_test, y_test)
#     print(f"\n  Per-device final accuracy:")
#     print(f"  {'Device':<22} {'Accuracy':>10}")
#     print(f"  {'-'*34}")
#     for d in devices:
#         tag = " [BYZ]" if d.is_byzantine else ""
#         print(f"  Device {d.id:2d}{tag:7s}       {final_acc[d.id]:>9.2f}%")

#     honest_final = [final_acc[d.id] for d in devices if not d.is_byzantine]
#     print(f"\n  Honest device avg accuracy : {sum(honest_final)/len(honest_final):.2f}%")
#     print(f"  Byzantine device accuracy  : {final_acc[[d.id for d in devices if d.is_byzantine][0]]:.2f}%")

#     # Reputation of Byzantine device after 30 rounds
#     byz_id = [d.id for d in devices if d.is_byzantine][0]
#     honest_devs = [d for d in devices if not d.is_byzantine and byz_id in d.reputation]
#     if honest_devs:
#         avg_byz_rep = sum(d.reputation[byz_id] for d in honest_devs) / len(honest_devs)
#         print(f"  Byzantine (D{byz_id}) avg reputation: {avg_byz_rep:.4f}")

#     # Accuracy progression
#     print(f"\n  Accuracy progression:")
#     milestones = [1, 5, 10, 15, 20, 25, 30]
#     for h in history:
#         if h["round"] in milestones:
#             print(f"    Round {h['round']:2d}: {h['avg_acc']:.2f}%  "
#                   f"(Byzantine flags: {h['byz_flags']})")

#     print(f"\n[DONE] Training complete — {NUM_ROUNDS} rounds")


# if __name__ == "__main__":
#     main()


"""
main.py
-------
Full Gossip Federated Learning — Training Loop

Changes from previous version:
    1. MODEL_SIZE = 421,642  (4-layer CNN, up from 101,770)
    2. BATCH_SIZE = 64       (up from 32, more stable gradients)
    3. Learning rate decay   (0.01 base, ×0.95 every 50 rounds)
    4. Optimizer: SGD with momentum=0.9 and weight_decay=1e-4 (in device.py)

Run with: python main.py
"""

import torch
import copy
import time
from topology import create_devices, build_topology
from data_loader import load_mnist, distribute_iid
from grad_compression import compress_gradient
from privacy import apply_differential_privacy
from gossip import gossip_exchange
from byzantine import run_phase5
from aggregation import run_phase6

# ── Configuration ──────────────────────────────────
NUM_ROUNDS    = 200          # 200 rounds যথেষ্ট CNN এর জন্য
BATCH_SIZE    = 64           # 32 → 64 (more stable)
BASE_LR       = 0.01         # SGD+momentum এ 0.01 ভালো
LR_DECAY      = 0.95         # প্রতি 50 round এ multiply
LR_DECAY_STEP = 50
MODEL_SIZE    = 421_642      # নতুন 4-layer CNN


def section(title):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")


def get_lr(round_num):
    """Learning rate decay: 0.01 × 0.95^(round//50)"""
    steps = (round_num - 1) // LR_DECAY_STEP
    return BASE_LR * (LR_DECAY ** steps)


def evaluate_all(devices, X_test, y_test):
    return {d.id: d.evaluate(X_test, y_test) for d in devices}


def avg_honest(acc_dict, devices):
    honest = [acc_dict[d.id] for d in devices if not d.is_byzantine]
    return sum(honest) / len(honest)


def main():
    section("PHASE 0 — Initialization (once)")

    devices    = create_devices()
    G, manager = build_topology(devices)

    W0 = copy.deepcopy(devices[0].model.state_dict())
    for d in devices:
        d.model.load_state_dict(W0)
        d.init_reputation()

    X_train, y_train, X_test, y_test = load_mnist()
    distribute_iid(X_train, y_train, devices)

    acc_init = evaluate_all(devices, X_test, y_test)
    print(f"  Devices         : {len(devices)}")
    print(f"  Byzantine       : {[d.id for d in devices if d.is_byzantine]}")
    print(f"  Model params    : {MODEL_SIZE:,}  (4-layer CNN)")
    print(f"  Batch size      : {BATCH_SIZE}")
    print(f"  LR schedule     : {BASE_LR} × {LR_DECAY}^(round//{LR_DECAY_STEP})")
    print(f"  Initial accuracy: {avg_honest(acc_init, devices):.2f}%")

    history = []

    section(f"TRAINING — {NUM_ROUNDS} Rounds")
    print(f"\n  {'Round':>6}  {'LR':>7}  {'Avg Acc':>9}  {'Byz Flags':>10}  {'Time':>7}")
    print(f"  {'-'*46}")

    for round_num in range(1, NUM_ROUNDS + 1):
        t_start = time.time()
        lr      = get_lr(round_num)

        # Phase 1 → 3
        noisy_gradients = {}
        for d in devices:
            gradient              = d.local_train(batch_size=BATCH_SIZE)
            compressed            = compress_gradient(gradient, d, MODEL_SIZE)
            noisy_grad, _         = apply_differential_privacy(compressed)
            noisy_gradients[d.id] = noisy_grad

        # Phase 4
        received = gossip_exchange(devices, noisy_gradients, round_num)

        # Phase 5
        all_quality = run_phase5(devices, received, noisy_gradients)
        byz_flags   = sum(
            1 for scores in all_quality.values()
            for info in scores.values() if info["is_byzantine"]
        )

        # Phase 6
        run_phase6(devices, received, all_quality, lr)

        acc     = evaluate_all(devices, X_test, y_test)
        avg_acc = avg_honest(acc, devices)
        elapsed = time.time() - t_start

        history.append({
            "round": round_num, "avg_acc": avg_acc,
            "byz_flags": byz_flags, "lr": lr,
        })

        print(f"  {round_num:>6}  {lr:>7.5f}  {avg_acc:>8.2f}%  "
              f"{byz_flags:>10}  {elapsed:>6.1f}s")

    section("FINAL RESULTS")

    final_acc = evaluate_all(devices, X_test, y_test)
    print(f"\n  Per-device final accuracy:")
    print(f"  {'Device':<22} {'Accuracy':>10}")
    print(f"  {'-'*34}")
    for d in devices:
        tag = " [BYZ]" if d.is_byzantine else ""
        print(f"  Device {d.id:2d}{tag:7s}       {final_acc[d.id]:>9.2f}%")

    honest_final = [final_acc[d.id] for d in devices if not d.is_byzantine]
    byz_id       = [d.id for d in devices if d.is_byzantine][0]
    honest_devs  = [d for d in devices if not d.is_byzantine
                    and byz_id in d.reputation]

    print(f"\n  Honest device avg accuracy : {sum(honest_final)/len(honest_final):.2f}%")
    print(f"  Byzantine device accuracy  : {final_acc[byz_id]:.2f}%")

    if honest_devs:
        avg_byz_rep = sum(d.reputation[byz_id] for d in honest_devs) / len(honest_devs)
        print(f"  Byzantine (D{byz_id}) avg reputation: {avg_byz_rep:.4f}")

    print(f"\n  Accuracy progression:")
    milestones = [1, 10, 25, 50, 100, 150, 200]
    for h in history:
        if h["round"] in milestones:
            print(f"    Round {h['round']:3d}: {h['avg_acc']:.2f}%  "
                  f"lr={h['lr']:.5f}  (Byzantine flags: {h['byz_flags']})")

    print(f"\n[DONE] Training complete — {NUM_ROUNDS} rounds")


if __name__ == "__main__":
    main()