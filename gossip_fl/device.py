"""
device.py
---------
Represents each Edge Device in the federated learning network.

Model architecture (PDF Section 2.2):
  Layer 1: 784 → 128 + ReLU
  Layer 2: 128 → 10  + Softmax
  Total  : 101,770 parameters

Resource profiling and dynamic k assignment (Paper Eq. 1-5).
"""

import numpy as np
import random
import torch
import torch.nn as nn


# ──────────────────────────────────────────
# CNN Model — exactly as PDF Section 2.2
# Xavier initialization applied in __init__
# ──────────────────────────────────────────
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1     = nn.Linear(784, 128)
        self.relu    = nn.ReLU()
        self.fc2     = nn.Linear(128, 10)
        # Note: Softmax is implicit in CrossEntropyLoss during training.
        # For inference/evaluation, we apply it explicitly.

        # Xavier initialization (PDF Section 2.2 reference)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x   # raw logits (CrossEntropyLoss applies softmax internally)

    def predict(self, x):
        """Returns softmax probabilities (for evaluation)."""
        return torch.softmax(self.forward(x), dim=1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ──────────────────────────────────────────
# Device Resource Profiles
# ──────────────────────────────────────────
DEVICE_RANGES = {
    "raspberry_pi": {
        "cpu_cores":      (2, 4),
        "cpu_freq_ghz":   (1.0, 1.5),
        "ram_gb":         (2, 4),
        "bandwidth_mbps": (5, 20),
    },
    "laptop": {
        "cpu_cores":      (4, 8),
        "cpu_freq_ghz":   (1.8, 3.0),
        "ram_gb":         (8, 16),
        "bandwidth_mbps": (30, 80),
    },
    "desktop": {
        "cpu_cores":      (8, 16),
        "cpu_freq_ghz":   (3.0, 4.0),
        "ram_gb":         (16, 32),
        "bandwidth_mbps": (80, 200),
    },
}


def sample_resources(device_type: str, seed: int = None) -> dict:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    ranges = DEVICE_RANGES[device_type]
    step_cores = ranges["cpu_cores"][1] - ranges["cpu_cores"][0] or 1
    return {
        "cpu_cores":      random.choice(range(
                              ranges["cpu_cores"][0],
                              ranges["cpu_cores"][1] + 1,
                              step_cores)),
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
        self.id           = device_id
        self.device_type  = device_type
        self.is_byzantine = is_byzantine
        self.resources    = sample_resources(
            device_type, seed=seed if seed is not None else device_id
        )

        self.resource_score = self._compute_resource_score()
        self.k              = self._compute_k()

        self.neighbors  = []
        self.reputation = {}

        # Model — Xavier initialized (PDF Section 2.2)
        self.model          = SimpleCNN()
        self.optimizer      = torch.optim.SGD(self.model.parameters(), lr=0.1)
        self.criterion      = nn.CrossEntropyLoss()
        self.local_data     = None
        self.local_gradient = None

    def _compute_resource_score(self) -> float:
        r         = self.resources
        cpu_score = (r["cpu_cores"] * r["cpu_freq_ghz"]) / 10
        mem_score = r["ram_gb"] / 32
        bw_score  = r["bandwidth_mbps"] / 100
        return round(0.4 * cpu_score + 0.4 * mem_score + 0.2 * bw_score, 4)

    def _compute_k(self, k_min: int = 3, k_max: int = 10) -> int:
        k = k_min + int(self.resource_score * (k_max - k_min))
        return int(np.clip(k, k_min, k_max))

    def init_reputation(self):
        for nid in self.neighbors:
            self.reputation[nid] = 1.0

    def local_train(self, batch_size: int = 32) -> dict:
        """
        PDF Phase 1:
          1. predictions = model.forward(samples)
          2. loss = cross_entropy(predictions, labels)
          3. gradients = loss.backward()
        Byzantine device returns large random noise instead.
        """
        if self.local_data is None:
            raise ValueError(f"Device {self.id}: local_data not set!")

        X, y = self.local_data

        # Byzantine attack: poisoned random gradient
        if self.is_byzantine:
            gradient = {name: torch.randn_like(p) * 5.0
                        for name, p in self.model.named_parameters()}
            self.local_gradient = gradient
            return gradient

        # Phase 1 — Normal local training
        self.model.train()
        self.optimizer.zero_grad()

        indices = torch.randperm(len(X))[:batch_size]
        output  = self.model(X[indices])          # forward pass
        loss    = self.criterion(output, y[indices])  # cross entropy
        loss.backward()                           # compute gradients

        gradient = {name: p.grad.clone()
                    for name, p in self.model.named_parameters()
                    if p.grad is not None}
        self.optimizer.step()
        self.local_gradient = gradient
        return gradient

    def evaluate(self, X_test: torch.Tensor, y_test: torch.Tensor) -> float:
        self.model.eval()
        with torch.no_grad():
            predicted = self.model(X_test).argmax(dim=1)
            correct   = (predicted == y_test).sum().item()
        return round((correct / len(y_test)) * 100, 2)

    def __repr__(self):
        byz_tag = " [BYZANTINE]" if self.is_byzantine else ""
        r = self.resources
        return (f"Device {self.id:2d} ({self.device_type:12s}) | "
                f"R={self.resource_score:.3f} | k={self.k} | "
                f"cores={r['cpu_cores']} freq={r['cpu_freq_ghz']}GHz "
                f"ram={r['ram_gb']}GB bw={r['bandwidth_mbps']}Mbps"
                f"{byz_tag}")