"""
test_step3.py
-------------
Phase 0 → Phase 1 → Phase 2 → Phase 3, একটা device-এর পুরো pipeline।
noisy_gradient explicitly দেখানো হয়েছে।
Run with: python test_step3.py
"""

import torch
import copy
from topology import create_devices, build_topology
from data_loader import load_mnist, distribute_iid
from grad_compression import (compress_gradient, decompress_gradient,
                               compression_stats, compute_compression_ratio)
from privacy import apply_differential_privacy, PRIVACY_PARAMS
from device import count_parameters

MODEL_SIZE = 101_770


def section(title):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")


def main():
    # ─────────────────────────────────────────
    # PHASE 0: Initialization
    # ─────────────────────────────────────────
    section("PHASE 0 — Initialization (PDF Section 2)")

    devices    = create_devices()
    G, manager = build_topology(devices)

    # Model parameter count check
    param_count = count_parameters(devices[0].model)
    print(f"  Model parameters : {param_count:,}  "
          f"({'OK' if param_count == MODEL_SIZE else 'MISMATCH'})")

    # W0 sync — all devices get identical initial weights (Xavier init)
    # PDF Section 2.2: propagated via gossip ≈2 rounds
    # Simulation: direct copy (same result)
    W0 = copy.deepcopy(devices[0].model.state_dict())
    for d in devices:
        d.model.load_state_dict(W0)
    print(f"  W0 synchronized  : all {len(devices)} devices start with identical weights")

    # Reputation init — PDF Section 2.3: rep(i,j)(0) = 1.0
    for d in devices:
        d.init_reputation()
    print(f"  Reputation init  : rep(i,j)(0) = 1.0 for all neighbors")

    # MNIST
    X_train, y_train, X_test, y_test = load_mnist()
    distribute_iid(X_train, y_train, devices)
    print(f"  MNIST distributed: {len(X_train):,} train / {len(X_test):,} test")

    # Fixed privacy parameters — PDF Section 3.3
    # epsilon and delta are FIXED, they do NOT accumulate per round
    print(f"\n  Privacy parameters (fixed, PDF Section 3.3):")
    print(f"    epsilon          = {PRIVACY_PARAMS['epsilon']}   (fixed, NOT per-round)")
    print(f"    delta            = {PRIVACY_PARAMS['delta']}  (fixed)")
    print(f"    noise_multiplier = {PRIVACY_PARAMS['noise_multiplier']}")
    print(f"    C (clip bound)   = {PRIVACY_PARAMS['clipping_bound']}")
    print(f"    sigma            = noise_multiplier × C = "
          f"{PRIVACY_PARAMS['noise_multiplier']} × {PRIVACY_PARAMS['clipping_bound']} = "
          f"{PRIVACY_PARAMS['noise_multiplier'] * PRIVACY_PARAMS['clipping_bound']}")

    # ─────────────────────────────────────────
    # PHASE 1 → 2 → 3 Pipeline for Device 1
    # ─────────────────────────────────────────
    section("PHASE 1 → 2 → 3 Pipeline (Device 1)")
    d1 = devices[0]
    print(f"  Device: {d1}\n")

    # ── Phase 1: Local Training ──────────────
    print("  ── Phase 1: Local Training (PDF Section 3.1) ──")
    gradient = d1.local_train(batch_size=32)
    print(f"  gradient ∇W (after loss.backward()):")
    for name, t in gradient.items():
        print(f"    {name:30s} shape={str(tuple(t.shape)):20s} "
              f"L2={t.norm().item():.6f}  nonzero={( t != 0).sum().item()}")

    # ── Phase 2: Compression ─────────────────
    print(f"\n  ── Phase 2: Compression (PDF Section 3.2, Algorithm 2) ──")
    Cr = compute_compression_ratio(d1, MODEL_SIZE)
    print(f"  Cr = {Cr:.4f}")

    compressed = compress_gradient(gradient, d1, MODEL_SIZE)
    stats      = compression_stats(gradient, compressed)
    print(f"  compressed_gradient ∇W_compressed:")
    for name, pkg in compressed.items():
        nz = (pkg["data"] != 0).sum().item()
        print(f"    {name:30s} shape={str(tuple(pkg['data'].shape)):20s} "
              f"nonzero={nz}  sparsity={100*(1-nz/pkg['data'].numel()):.1f}%")
    print(f"  Overall sparsity : {stats['sparsity_%']}%")
    print(f"  Cr={stats['Cr']} → lt=Cr×max(weights) → Pi বেশি prune → বেশি sparsity")

    # ── Phase 3: Differential Privacy ────────
    print(f"\n  ── Phase 3: Differential Privacy (PDF Section 3.3, Algorithm 3) ──")
    print(f"  Step 1 — Adaptive Clipping (C = {PRIVACY_PARAMS['clipping_bound']}):")

    noisy_gradient, dp_log = apply_differential_privacy(compressed)

    for name, log in dp_log.items():
        clip_tag = "→ CLIPPED" if log["was_clipped"] else "→ no clipping needed"
        print(f"    {name:30s} L2={log['l2_norm']:.6f}  C={log['C']}  {clip_tag}")

    sigma = PRIVACY_PARAMS["noise_multiplier"] * PRIVACY_PARAMS["clipping_bound"]
    print(f"\n  Step 2 — Gaussian Noise (sigma={sigma}):")
    print(f"    noise ~ N(0, {sigma}²) added to each element")
    print(f"\n  noisy_gradient ∇W_noisy (= Phase 3 output, sent to neighbors):")
    for name, pkg in noisy_gradient.items():
        nz   = (pkg["data"] != 0).sum().item()
        norm = pkg["data"].norm().item()
        # Show first 5 values as sample
        sample = pkg["data"].detach().numpy().flatten()[:5]
        sample_str = "  ".join(f"{v:.4f}" for v in sample)
        print(f"    {name:30s} L2={norm:.6f}  "
              f"first 5 values: [{sample_str} ...]")

    # ── noisy_gradient goes to gossip ────────
    print(f"\n  ── What happens next? ──")
    print(f"  noisy_gradient is packaged into a message:")
    sample_layer = list(noisy_gradient.keys())[0]
    pkg = noisy_gradient[sample_layer]
    print(f"    message = {{")
    print(f"      'sender'           : {d1.id},")
    print(f"      'round'            : 1,")
    print(f"      'noisy_gradient'   : {{layer: tensor, ...}},  ← ∇W_noisy")
    print(f"      'compression_ratio': {pkg['Cr']:.4f},")
    print(f"      'mask'             : dct_mask array")
    print(f"    }}")
    print(f"  → This message is sent to Device 1's neighbors: {d1.neighbors}")
    print(f"  → Phase 4 (Gossip Exchange) handles the actual sending")

    # ─────────────────────────────────────────
    # Cr comparison across device types
    # ─────────────────────────────────────────
    section("COMPRESSION RATIO BY DEVICE TYPE")
    print(f"  {'Device':<48} {'Cr':>7}  {'Sparsity%':>10}")
    print(f"  {'-'*68}")
    for d in devices:
        grad = d.local_train(batch_size=32)
        comp = compress_gradient(grad, d, MODEL_SIZE)
        s    = compression_stats(grad, comp)
        byz  = " [BYZ]" if d.is_byzantine else ""
        print(f"  Device {d.id:2d} ({d.device_type:12s}){byz:7s} "
              f"R={d.resource_score:.3f}  "
              f"Cr={s['Cr']:7.4f}  "
              f"sparsity={s['sparsity_%']:5.1f}%")

    print("\n[OK] Step 3 complete!")
    print("     noisy_gradient → Phase 4 (gossip.py) এ পাঠানো হবে")


if __name__ == "__main__":
    main()