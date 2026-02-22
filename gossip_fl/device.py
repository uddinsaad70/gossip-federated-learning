"""
device.py
---------
Represents each Edge Device in the federated learning network.
Calculates resource score and dynamic neighbor count (k) based on device capabilities.

Each device gets slightly varied specs to simulate real-world heterogeneity.
This ensures truly dynamic k assignment — even within the same device type.
"""

import numpy as np
import random
import torch
import torch.nn as nn


# ──────────────────────────────────────────
# CNN Model (Simple CNN as described in paper)
# Architecture: 784 -> 128 (ReLU) -> 10
# Total parameters: 101,770
# ──────────────────────────────────────────
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.network(x)


# ──────────────────────────────────────────
# Resource Profile Ranges per Device Type
# Each device gets randomly sampled specs
# within these realistic ranges.
#
# Target R ranges (paper Table 1):
#   raspberry_pi -> R ≈ 0.15 - 0.30  -> k = 3-4
#   laptop       -> R ≈ 0.40 - 0.80  -> k = 5-8
#   desktop      -> R ≈ 1.20 - 2.00  -> k = 9-10
# ──────────────────────────────────────────
DEVICE_RANGES = {
    "raspberry_pi": {
        "cpu_cores":      (2, 4),        # 2 or 4 cores
        "cpu_freq_ghz":   (1.0, 1.5),    # 1.0 to 1.5 GHz
        "ram_gb":         (2, 4),         # 2 or 4 GB
        "bandwidth_mbps": (5, 20),        # 5 to 20 Mbps
    },
    "laptop": {
        "cpu_cores":      (4, 8),         # 4 or 8 cores
        "cpu_freq_ghz":   (1.8, 3.0),    # 1.8 to 3.0 GHz
        "ram_gb":         (8, 16),        # 8 or 16 GB
        "bandwidth_mbps": (30, 80),       # 30 to 80 Mbps
    },
    "desktop": {
        "cpu_cores":      (8, 16),        # 8 or 16 cores
        "cpu_freq_ghz":   (3.0, 4.0),    # 3.0 to 4.0 GHz
        "ram_gb":         (16, 32),       # 16 or 32 GB
        "bandwidth_mbps": (80, 200),      # 80 to 200 Mbps
    },
}


def sample_resources(device_type: str, seed: int = None) -> dict:
    """
    Randomly samples hardware specs for a given device type.
    Each call returns slightly different values — simulating
    real-world device heterogeneity within the same category.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    ranges = DEVICE_RANGES[device_type]
    return {
        "cpu_cores":      random.choice(range(ranges["cpu_cores"][0],
                                              ranges["cpu_cores"][1] + 1,
                                              ranges["cpu_cores"][1] - ranges["cpu_cores"][0]
                                              or 1)),
        "cpu_freq_ghz":   round(random.uniform(*ranges["cpu_freq_ghz"]), 2),
        "ram_gb":         random.choice([ranges["ram_gb"][0], ranges["ram_gb"][1]]),
        "bandwidth_mbps": round(random.uniform(*ranges["bandwidth_mbps"]), 1),
    }


# ──────────────────────────────────────────
# EdgeDevice Class
# ──────────────────────────────────────────
class EdgeDevice:
    def __init__(self, device_id: int, device_type: str,
                 is_byzantine: bool = False, seed: int = None):
        """
        Parameters
        ----------
        device_id    : Integer ID (e.g. 1 to 20)
        device_type  : "raspberry_pi" / "laptop" / "desktop"
        is_byzantine : If True, device sends poisoned gradients
        seed         : Random seed for reproducible resource sampling
                       (use device_id as seed for consistent results)
        """
        self.id           = device_id
        self.device_type  = device_type
        self.is_byzantine = is_byzantine

        # Sample unique hardware specs for this device
        # Using device_id as seed ensures same device always gets same specs
        self.resources = sample_resources(device_type, seed=seed if seed is not None
                                          else device_id)

        # Compute resource score and dynamic neighbor count
        self.resource_score = self._compute_resource_score()
        self.k              = self._compute_k()

        # Network topology
        self.neighbors  = []      # List of neighbor device IDs
        self.reputation = {}      # {neighbor_id: reputation_score}

        # Machine learning components
        self.model          = SimpleCNN()
        self.optimizer      = torch.optim.SGD(self.model.parameters(), lr=0.1)
        self.criterion      = nn.CrossEntropyLoss()
        self.local_data     = None   # (X_tensor, y_tensor) set later
        self.local_gradient = None   # Most recently computed gradient

    # ── Resource Score (Paper Eq. 1-4) ───────────
    def _compute_resource_score(self) -> float:
        """
        R(i) = 0.4 * (cores * freq / 10)
             + 0.4 * (ram / 32)
             + 0.2 * (bandwidth / 100)
        """
        r         = self.resources
        cpu_score = (r["cpu_cores"] * r["cpu_freq_ghz"]) / 10
        mem_score = r["ram_gb"] / 32
        bw_score  = r["bandwidth_mbps"] / 100
        return round(0.4 * cpu_score + 0.4 * mem_score + 0.2 * bw_score, 4)

    # ── Dynamic k (Paper Eq. 5) ───────────────────
    def _compute_k(self, k_min: int = 3, k_max: int = 10) -> int:
        """
        k(i) = clip( k_min + floor(R(i) * (k_max - k_min)), k_min, k_max )
        """
        k = k_min + int(self.resource_score * (k_max - k_min))
        return int(np.clip(k, k_min, k_max))

    # ── Initialize Reputation ─────────────────────
    def init_reputation(self):
        """Sets all neighbor reputations to 1.0. Call after neighbors are assigned."""
        for nid in self.neighbors:
            self.reputation[nid] = 1.0

    # ── Local Training ────────────────────────────
    def local_train(self, batch_size: int = 32) -> dict:
        """
        Trains on local data and returns the gradient.
        Byzantine devices return large random noise instead.
        """
        if self.local_data is None:
            raise ValueError(f"Device {self.id}: local_data has not been set!")

        X, y = self.local_data

        if self.is_byzantine:
            gradient = {name: torch.randn_like(param) * 5.0
                        for name, param in self.model.named_parameters()}
            self.local_gradient = gradient
            return gradient

        self.model.train()
        self.optimizer.zero_grad()

        indices = torch.randperm(len(X))[:batch_size]
        loss    = self.criterion(self.model(X[indices]), y[indices])
        loss.backward()

        gradient = {name: param.grad.clone()
                    for name, param in self.model.named_parameters()
                    if param.grad is not None}

        self.optimizer.step()
        self.local_gradient = gradient
        return gradient

    # ── Evaluation ────────────────────────────────
    def evaluate(self, X_test: torch.Tensor, y_test: torch.Tensor) -> float:
        """Returns classification accuracy (%) on the provided test set."""
        self.model.eval()
        with torch.no_grad():
            predicted = self.model(X_test).argmax(dim=1)
            correct   = (predicted == y_test).sum().item()
        return round((correct / len(y_test)) * 100, 2)

    # ── String Representation ─────────────────────
    def __repr__(self):
        byz_tag = " [BYZANTINE]" if self.is_byzantine else ""
        r = self.resources
        specs = (f"cores={r['cpu_cores']} "
                 f"freq={r['cpu_freq_ghz']}GHz "
                 f"ram={r['ram_gb']}GB "
                 f"bw={r['bandwidth_mbps']}Mbps")
        return (f"Device {self.id:2d} ({self.device_type:12s}) | "
                f"R={self.resource_score:.3f} | k={self.k} | "
                f"{specs}{byz_tag}")
