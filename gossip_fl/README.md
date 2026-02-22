# Gossip-Based Federated Learning with Byzantine Tolerance

A decentralized, privacy-preserving Federated Learning system that combines:
- **Resource-Adaptive Dynamic Topology** — each device gets a personalized neighbor count based on its hardware capabilities
- **Gossip Protocol** — decentralized model exchange without a central server
- **Differential Privacy** — gradient noise injection before sharing
- **DCT Compression** — reduces communication cost
- **Byzantine Detection** — cosine similarity + IQR-based outlier filtering

---

## Project Structure

```
gossip_fl/
├── device.py           # EdgeDevice class, resource profiling, dynamic k
├── topology.py         # Topology construction, saturation handling, add_device()
├── data_loader.py      # MNIST loading, IID and Non-IID distribution    [Step 2]
├── compression.py      # DCT-based gradient compression                 [Step 3]
├── privacy.py          # Differential privacy (Gaussian noise)          [Step 3]
├── gossip.py           # Gossip exchange, Byzantine detection            [Step 4]
├── main.py             # Full training loop, 30 rounds                  [Step 5]
│
├── test_step1.py       # Test: device creation + topology
├── test_step2.py       # Test: MNIST distribution                       [Step 2]
├── test_step3.py       # Test: compression + privacy                    [Step 3]
├── test_step4.py       # Test: gossip + Byzantine detection              [Step 4]
└── test_step5.py       # Test: full training run                        [Step 5]
```

---

## System Overview

### Network Configuration
| Parameter | Value |
|-----------|-------|
| Total devices (N) | 20 |
| Byzantine devices | 1 (Device 17) |
| Training rounds (T) | 30 |
| Dataset | MNIST (60,000 train / 10,000 test) |
| Model | Simple CNN (101,770 parameters) |

### Device Distribution
| Device Type | Count | IDs | R range | k range |
|-------------|-------|-----|---------|---------|
| Raspberry Pi (low) | 6 | 1–6 | 0.14–0.28 | 3–4 |
| Laptop (medium) | 10 | 7–16 | 0.46–1.27 | 6–10 |
| Desktop (high) | 4 | 17–20 | 1.61–3.07 | 10 |

---

## Installation

```bash
pip install torch torchvision numpy networkx matplotlib scipy
```

---

## Usage

Run each step independently to build and validate the system incrementally.

### Step 1 — Device Creation and Topology
```bash
python test_step1.py
```
**What it does:**
- Creates 20 devices with varied hardware specs
- Computes resource score R(i) and dynamic neighbor count k(i) for each device
- Builds the network graph using the greedy algorithm
- Demonstrates dynamic device addition (Device 21 joins)
- Saves `topology_20devices.png` and `topology_21devices.png`

**Expected output:**
```
Device  1 (raspberry_pi) | R=0.145 | k=4
Device  7 (laptop      ) | R=1.266 | k=10
Device 17 (desktop     ) | R=3.068 | k=10 [BYZANTINE]

Total Edges : 74  |  Average k : 7.40  |  Connected : True  |  Diameter : 3
```

---

## Core Algorithms

### Resource Score (Paper Eq. 1–4)
```
R(i) = 0.4 × (cores × freq / 10)
     + 0.4 × (RAM_GB / 32)
     + 0.2 × (bandwidth_Mbps / 100)
```

### Dynamic Neighbor Count (Paper Eq. 5)
```
k(i) = clip( k_min + floor(R(i) × (k_max − k_min)), k_min, k_max )
```
where k_min = 3 and k_max = 10.

### Topology Construction
1. Sort devices by R(i) descending
2. Greedily assign neighbors (highest unmet demand first)
3. Handle saturation via capacity relaxation
4. Guarantee connectivity with DFS bridge-edge insertion

---

## Key Design Decisions

**Why varied specs within the same device type?**  
Real-world devices of the same category (e.g., laptops) differ in CPU speed, RAM, and bandwidth. Using fixed specs for all laptops would make k identical for all of them — defeating the purpose of dynamic assignment.

**Why seed = device_id?**  
Using the device ID as the random seed ensures reproducible results across runs. The same device always gets the same hardware specs, making experiments comparable.

**Why is Device 17 Byzantine?**  
Device 17 is the highest-resource desktop, making it a worst-case attacker — it has the most neighbors and the most influence on the network. The Byzantine detection module must identify and down-weight it during aggregation.

---

## References

1. Hidayat, M. A., Nakamura, Y., & Arakawa, Y. (2024). *Privacy-Preserving Federated Learning With Resource-Adaptive Compression for Edge Devices.* IEEE Internet of Things Journal, 11(8).
2. Koloskova, A., et al. (2020). *Decentralized Stochastic Optimization and Gossip Algorithms with Compressed Communication.* ICML.
3. McMahan, B., et al. (2017). *Communication-Efficient Learning of Deep Networks from Decentralized Data.* AISTATS.

---

## Status

| Step | Module | Status |
|------|--------|--------|
| 1 | device.py, topology.py | ✅ Complete |
| 2 | data_loader.py | 🔄 In progress |
| 3 | compression.py, privacy.py | ⏳ Pending |
| 4 | gossip.py | ⏳ Pending |
| 5 | main.py | ⏳ Pending |
