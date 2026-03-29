# Gossip-Based Federated Learning with Byzantine Tolerance

A decentralized, privacy-preserving Federated Learning system combining resource-adaptive topology, gossip-based model exchange, differential privacy, DCT compression, and Byzantine detection — implemented without any central server.

---

## Key Results (MNIST, 200 rounds, 20 devices)

| Scenario | Final Accuracy | Notes |
|---|---|---|
| No Byzantine (baseline) | ~92% | All 20 devices honest |
| With Byzantine (Device 17) | ~92% | Sign-flip attack, detected by Phase 5 |
| Accuracy drop | ~0% | Byzantine resilience of the system |

Byzantine Device 17 reputation decays to near 0 by round 30, effectively removing its influence from aggregation.

---

## Project Structure

```
gossip_fl/
├── device.py           # EdgeDevice: resource profiling, dynamic k, local training
├── topology.py         # Topology construction, saturation handling, add_device()
├── data_loader.py      # MNIST loading, IID and Non-IID distribution
├── grad_compression.py # DCT-based adaptive gradient compression          [Phase 2]
├── privacy.py          # Differential privacy — AGC-DP noise injection    [Phase 3]
├── gossip.py           # Bidirectional gossip exchange                    [Phase 4]
├── byzantine.py        # Cosine + IQR quality assessment & detection      [Phase 5]
├── aggregation.py      # Reputation-weighted aggregation & model update   [Phase 6]
└── main.py             # Full training loop, 200 rounds, two experiments
```

---

## System Overview

### Network Configuration

| Parameter | Value |
|---|---|
| Total devices (N) | 20 |
| Byzantine devices | 1 (Device 17, sign-flip attack) |
| Training rounds (T) | 200 |
| Dataset | MNIST (60,000 train / 10,000 test) |
| Model | 4-layer CNN, 421,642 parameters |
| Batch size | 64 |
| Base learning rate | 0.01 (SGD, momentum=0.9, weight_decay=1e-4) |
| LR decay | ×0.95 every 50 rounds |
| DP noise multiplier | 0.1 (Hidayat 2024, Scenario 1 equivalent) |

### Device Distribution

| Device Type | Count | IDs | Resource score R | Neighbor count k |
|---|---|---|---|---|
| Raspberry Pi (low) | 6 | 1–6 | 0.14–0.28 | 3–4 |
| Laptop (medium) | 10 | 7–16 | 0.46–1.27 | 5–7 |
| Desktop (high) | 4 | 17–20 | 1.61–3.07 | 9–10 |

---

## Core Algorithms

### Resource score (topology paper, Eq. 1–4)

```
R(i) = 0.4 × (cores × freq_GHz / 10)
     + 0.4 × (RAM_GB / 32)
     + 0.2 × (bandwidth_Mbps / 100)
```

### Dynamic neighbor count (topology paper, Eq. 5)

```
k(i) = clip( k_min + floor(R(i) × (k_max − k_min)), 3, 10 )
```

### Compression ratio (Hidayat et al. 2024, Eq. 13–14)

```
Mu = (Ms + Mt) / Ma
Cr = Mu / (Cc / Cu)
```

Higher Cr → more aggressive DCT pruning (Raspberry Pi: Cr≈0.15, Desktop: Cr≈0.019).

### Byzantine detection (Phase 5)

Three tests on pre-noise DCT-domain gradients:

```
cos_sim   < -0.5   → Byzantine (sign-flip detected)
mag_ratio < 0.1    → Byzantine (scale attack detected)
outlier%  > 30%    → Byzantine (IQR anomaly detected)

quality = 0.6 × cos_sim + 0.4 × mag_ratio   (if not Byzantine)
        = 0.0                                 (if Byzantine)

reputation_new = 0.8 × reputation_old + 0.2 × quality
```

### Weighted aggregation (Phase 6)

```
weight_j      = reputation_j × quality_j
weight_own    = 1.0
aggregated    = Σ(weight_j × grad_j) / Σ(weight_j)
W_new         = W_old + lr × aggregated
```

---

## Installation

```bash
pip install torch torchvision numpy networkx matplotlib scipy
```

---

## Usage

```bash
python main.py
```

Runs two experiments sequentially: Run 1 (no Byzantine), Run 2 (Device 17 Byzantine). Results saved to `results_no_byzantine.txt` and `results_with_byzantine.txt`.

---

## Design Decisions

**Why sign-flip for Byzantine attack?**
Sign-flip (`gradient = -honest_gradient`) keeps the same L2 magnitude as an honest gradient, so DP clipping does not reduce it. The opposite direction produces `cos_sim ≈ -1.0`, which Phase 5's cosine test detects in the first round. The previous implementation (`randn × 50`) was clipped down by DP and produced a `mag_ratio` that fell in an ambiguous range, causing Byz Flags to stay at 0 throughout training.

**Why `param.data += lr × aggregated`?**
The gossip FL aggregation convention treats the aggregated vector as an update direction. Local training via `optimizer.step()` already applies the device's own gradient; the aggregated term adds the collaborative correction from neighbors. Using `-=` inverts this correction for all 200 rounds.

**Why `optimizer.step()` inside `local_train()`?**
Without the local step, each device's model is only updated through neighbor aggregation. This removes the self-learning signal and cuts effective batch size from 64 to near 0 for local knowledge. Including it means each device improves on its own data before sharing.

**Why seed = device_id?**
Reproducibility. The same device always gets the same hardware specs, making experiments directly comparable across runs without global random seed management.

**Why IID data distribution?**
The Hidayat et al. (2024) base paper distributes data proportionately without specifying Non-IID. IID is the closest faithful match. Non-IID support is included in `data_loader.py` via `distribute_non_iid()` for ablation.

---

## References

1. Hidayat, M. A., Nakamura, Y., & Arakawa, Y. (2024). *Privacy-Preserving Federated Learning With Resource-Adaptive Compression for Edge Devices.* IEEE Internet of Things Journal, 11(8), 13180–13198.
2. Resource-Adaptive Dynamic Network Topology for Gossip-Based Federated Learning. (Thesis proposal, extended from Hidayat et al.)
3. Koloskova, A., et al. (2020). *Decentralized Stochastic Optimization and Gossip Algorithms with Compressed Communication.* ICML.
4. McMahan, B., et al. (2017). *Communication-Efficient Learning of Deep Networks from Decentralized Data.* AISTATS.

---

## Module Status

| Phase | Module | Status |
|---|---|---|
| 0 | topology.py, device.py | Complete |
| 1 | device.py local_train | Complete — sign-flip Byzantine, optimizer.step() for honest |
| 2 | grad_compression.py | Complete — resource-adaptive DCT |
| 3 | privacy.py | Complete — AGC-DP, noise_multiplier=0.1 |
| 4 | gossip.py | Complete — bidirectional exchange |
| 5 | byzantine.py | Complete — cosine + mag_ratio + IQR detection |
| 6 | aggregation.py | Complete — reputation-weighted, param.data += |
