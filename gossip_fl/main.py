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

# Reputation tracking config
REP_TRACK_DEVICE = 17        # যে device এর reputation track করবো (Byzantine)
REP_PRINT_EVERY  = 10        # কত round পর পর reputation table দেখাবে


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
    byz_id = [d.id for d in devices if d.is_byzantine][0]
    print(f"\n  Tracking Device {byz_id} reputation across all neighbors.")
    print(f"  Reputation table shown every {REP_PRINT_EVERY} rounds.")
    print(f"\n  {'Round':>6}  {'LR':>7}  {'Avg Acc':>9}  {'Byz Flags':>10}  {'Time':>7}")
    print(f"  {'-'*46}")

    for round_num in range(1, NUM_ROUNDS + 1):
        t_start = time.time()
        lr      = get_lr(round_num)

        # Phase 1 → 3
        noisy_gradients      = {}
        compressed_gradients = {}
        for d in devices:
            gradient                   = d.local_train(batch_size=BATCH_SIZE)
            compressed                 = compress_gradient(gradient, d, MODEL_SIZE)
            noisy_grad, _              = apply_differential_privacy(compressed)
            noisy_gradients[d.id]      = noisy_grad
            compressed_gradients[d.id] = compressed  # Phase 2 output — pre-noise

        # Phase 4
        received = gossip_exchange(devices, noisy_gradients, round_num, compressed_gradients)

        # Phase 5
        all_quality = run_phase5(devices, received, noisy_gradients, compressed_gradients)
        byz_flags   = sum(
            1 for scores in all_quality.values()
            for info in scores.values() if info["is_byzantine"]
        )

        # Phase 6
        run_phase6(devices, received, all_quality, lr)

        acc     = evaluate_all(devices, X_test, y_test)
        avg_acc = avg_honest(acc, devices)
        elapsed = time.time() - t_start

        # Reputation snapshot: প্রতি device এ REP_TRACK_DEVICE এর reputation
        rep_snapshot = {}
        for d in devices:
            if not d.is_byzantine and REP_TRACK_DEVICE in d.reputation:
                rep_snapshot[d.id] = round(d.reputation[REP_TRACK_DEVICE], 4)

        history.append({
            "round":       round_num,
            "avg_acc":     avg_acc,
            "byz_flags":   byz_flags,
            "lr":          lr,
            "rep_snapshot": rep_snapshot,
        })

        print(f"  {round_num:>6}  {lr:>7.5f}  {avg_acc:>8.2f}%  "
              f"{byz_flags:>10}  {elapsed:>6.1f}s")

        # Reputation table প্রতি REP_PRINT_EVERY round এ
        if round_num % REP_PRINT_EVERY == 0:
            reps = rep_snapshot
            if reps:
                avg_r = sum(reps.values()) / len(reps)
                min_r = min(reps.values())
                max_r = max(reps.values())
                evaluators = sorted(reps.keys())
                rep_vals   = "  ".join(f"D{d:2d}:{reps[d]:.3f}" for d in evaluators)
                print(f"         ↳ D{REP_TRACK_DEVICE}[BYZ] rep → "
                      f"avg={avg_r:.4f}  min={min_r:.4f}  max={max_r:.4f}")
                # প্রতি evaluator এর value
                for i in range(0, len(evaluators), 5):
                    chunk = evaluators[i:i+5]
                    line  = "  ".join(f"D{d:2d}:{reps[d]:.4f}" for d in chunk)
                    print(f"           {line}")

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

    # ── Reputation Progression Table ──────────────────────
    print(f"\n  Device {byz_id} [BYZ] Reputation — How neighbors view it over time:")
    # কোন device গুলো neighbors ছিল বের করো
    rep_history = [(h["round"], h["rep_snapshot"]) for h in history
                   if h["rep_snapshot"] and h["round"] in milestones]
    if rep_history:
        # header
        sample_devs = sorted(rep_history[0][1].keys())
        header_devs = "  ".join(f"  D{d:2d} " for d in sample_devs)
        print(f"\n  {'Round':>6}  {'Avg rep':>8}  {header_devs}")
        print(f"  {'-'*(14 + 8*len(sample_devs))}")
        for rnd, reps in rep_history:
            avg_r    = sum(reps.values()) / len(reps) if reps else 0
            rep_vals = "  ".join(f"{reps.get(d, 0):.4f}" for d in sample_devs)
            print(f"  {rnd:>6}  {avg_r:>8.4f}  {rep_vals}")

    # ── Full final reputation matrix ───────────────────────
    print(f"\n  Final Reputation Matrix (row=evaluator, col=what they think of D{byz_id}):")
    honest_devs_with_rep = [d for d in devices
                            if not d.is_byzantine and byz_id in d.reputation]
    if honest_devs_with_rep:
        print(f"\n  {'Evaluator':<12} {'Rep of D'+str(byz_id):>14}  {'Status'}")
        print(f"  {'-'*40}")
        for d in sorted(honest_devs_with_rep, key=lambda x: x.id):
            rep = d.reputation[byz_id]
            status = "LOW (Byzantine soft-excluded)" if rep < 0.5 else "normal"
            print(f"  Device {d.id:2d}      {rep:>14.4f}  {status}")

    print(f"\n[DONE] Training complete — {NUM_ROUNDS} rounds")


if __name__ == "__main__":
    main()