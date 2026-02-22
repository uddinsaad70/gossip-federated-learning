"""
test_step2.py
-------------
Validates MNIST loading and IID data distribution across 20 devices.
Device 21 joins dynamically with its own data, then appears in the final chart.
Run with: python test_step2.py
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from topology import create_devices, build_topology
from data_loader import load_mnist, distribute_iid, print_distribution_summary
from device import EdgeDevice


def plot_distribution(devices, title, save_path):
    """Bar chart showing class distribution per device."""
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    fig, ax = plt.subplots(figsize=(max(12, len(devices) * 0.8), 5))

    x      = np.arange(len(devices))
    bottom = np.zeros(len(devices))

    for c in range(10):
        counts = [(d.local_data[1] == c).sum().item()
                  if d.local_data is not None else 0
                  for d in devices]
        ax.bar(x, counts, bottom=bottom, label=f"Digit {c}",
               color=colors[c], alpha=0.85)
        bottom += np.array(counts)

    ax.set_xticks(x)
    ax.set_xticklabels([f"D{d.id}" for d in devices], fontsize=8)
    ax.set_xlabel("Device")
    ax.set_ylabel("Number of samples")
    ax.set_title(title)
    ax.legend(loc="upper right", ncol=5, fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Chart saved -> {save_path}")
    plt.show()


def main():
    # ── Setup ──────────────────────────────────────
    print("\n[1] Creating devices and topology...")
    devices    = create_devices()          # 20 devices
    G, manager = build_topology(devices)
    print(f"    {len(devices)} devices ready.")

    # ── Load MNIST ─────────────────────────────────
    print("\n[2] Loading MNIST dataset...")
    X_train, y_train, X_test, y_test = load_mnist()
    print(f"    Train : {X_train.shape}  |  Test : {X_test.shape}")

    # ── IID Distribution (20 devices) ─────────────
    print("\n[3] IID distribution across 20 devices...")
    distribute_iid(X_train, y_train, devices)
    print_distribution_summary(devices)

    # ── Verify every device has data ───────────────
    print("\n[4] Verifying all 20 devices received data...")
    all_ok = True
    for d in devices:
        if d.local_data is None:
            print(f"    [MISSING] Device {d.id} has no data!")
            all_ok = False
        else:
            n = len(d.local_data[1])
            print(f"    Device {d.id:2d} ({d.device_type:12s}) : {n} samples")
    if all_ok:
        print("    All 20 devices have data. OK")

    # ── Test set info ──────────────────────────────
    print("\n[5] Test set (shared for global evaluation):")
    print(f"    Shape  : {X_test.shape}")
    counts = {c: (y_test == c).sum().item() for c in range(10)}
    print(f"    Classes: {counts}")

    # ── Device 21 joins WITH its own data ──────────
    # Important: data must be assigned BEFORE add_device()
    # so that Device 21 appears correctly in charts
    print("\n[6] Adding Device 21 dynamically (with its own data)...")
    new_dev = EdgeDevice(device_id=21, device_type="laptop")

    # In real deployment, the device brings its own local dataset.
    # Here we use the last 1000 test samples as a stand-in.
    new_dev.local_data = (X_test[-1000:], y_test[-1000:])

    manager.add_device(new_dev)

    print(f"\n    Device 21 neighbors    : {new_dev.neighbors}")
    print(f"    Device 21 data samples : {len(new_dev.local_data[1])}")
    print(f"    Device 21 reputation   : {new_dev.reputation}")

    # ── Plot chart AFTER Device 21 has joined ──────
    # manager.devices now contains all 21 devices
    print("\n[7] Plotting distribution chart (21 devices, including Device 21)...")
    all_devices = manager.devices   # 21 devices
    plot_distribution(
        all_devices,
        "IID Distribution — 21 devices (Device 21 joined dynamically)",
        "distribution_iid_21.png"
    )

    print("\n[OK] Step 2 complete!")
    print("     distribution_iid_21.png -> IID chart for all 21 devices")

    return X_test, y_test


if __name__ == "__main__":
    main()
