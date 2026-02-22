"""
test_step1.py
-------------
Validates device creation, topology construction, and dynamic device addition.
Run with: python test_step1.py
"""

from device import EdgeDevice
from topology import (create_devices, build_topology,
                      print_topology_summary, plot_topology, TopologyManager)


def main():
    # ── Part 1: Create devices ─────────────────────
    print("\n[1] Creating 20 devices...")
    devices = create_devices()

    print("\n--- Resource Score and k for each device ---")
    for d in devices:
        print(f"  {d}")

    # ── Part 2: Build topology ─────────────────────
    print("\n[2] Building network topology...")
    G, manager = build_topology(devices)

    print()
    print_topology_summary(devices, G)

    # ── Part 3: Plot initial topology (20 devices) ─
    print("\n[3] Plotting initial topology (20 devices)...")
    plot_topology(devices, G,
                  save_path="topology_20devices.png",
                  title_suffix="Before: 20 devices")

    # ── Part 4: Add Device 21 dynamically ──────────
    print("\n[4] Testing dynamic device addition...")
    new_dev = EdgeDevice(device_id=21, device_type="laptop")
    manager.add_device(new_dev)

    print(f"\n    Device 21 neighbor list : {new_dev.neighbors}")
    print(f"    Reputation initialized  : {new_dev.reputation}")

    # ── Part 5: Plot updated topology (21 devices) ─
    print("\n[5] Plotting updated topology (21 devices)...")
    all_devices = manager.devices   # now includes device 21
    plot_topology(all_devices, manager.G,
                  save_path="topology_21devices.png",
                  title_suffix="After: Device 21 joined")

    print("\n[OK] Step 1 complete!")
    print("     topology_20devices.png  -> initial network")
    print("     topology_21devices.png  -> after device 21 joined")


if __name__ == "__main__":
    main()
