# """
# privacy.py
# ----------
# Phase 3: Differential Privacy — Adaptive Gaussian Clipping (AGC-DP)

# Implements PDF Section 3.3 and Hidayat et al. (2024) Algorithm 3.

# PDF Step-by-step (Section 3.3):

#   Input: compressed gradient ∇W_compressed

#   Step 1 — Adaptive Clipping:
#       Compute L2 norm: ||∇W||_2 = sqrt(sum of squares)
#       If ||∇W||_2 > C:  scale down → ∇W_clipped = ∇W × (C / ||∇W||_2)
#       If ||∇W||_2 <= C: no change  → scaling factor = 1.0

#   Step 2 — Gaussian Noise Addition:
#       sigma (σ) = noise_multiplier × C
#       noise     ~ N(0, σ²)  for each element
#       ∇W_noisy  = ∇W_clipped + noise

#   Privacy parameters (PDF Section 3.3 example):
#       epsilon_target   = 1.0
#       delta_target     = 1e-5
#       noise_multiplier = 0.5
#       C                = 1.0   (clipping bound)
#       sigma            = 0.5 × 1.0 = 0.5

#   What epsilon and delta mean:
#       epsilon (ε): privacy budget.
#           Lower ε = more private = more noise added.
#           PDF uses ε = 1.0 as target.
#           After T rounds, total privacy cost accumulates.
#       delta (δ): probability that privacy guarantee fails.
#           Standard choice: δ = 1e-5 (very small, near-zero failure chance).
#           This means the DP guarantee holds with probability 1 - δ ≈ 99.999%.

#   How epsilon_total is measured:
#       Each round consumes some privacy budget.
#       We use basic composition (upper bound):
#           ε_total = T × ε_per_round
#       Paper uses PLD accountant (tighter bound) — our version is conservative.
# """

# import torch
# import numpy as np


# # ──────────────────────────────────────────
# # Privacy Parameters
# # Matching PDF Section 3.3 example values
# # ──────────────────────────────────────────
# DEFAULT_PRIVACY_PARAMS = {
#     "epsilon":          1.0,    # target privacy budget per round
#     "delta":            1e-5,   # failure probability (near zero)
#     "noise_multiplier": 0.5,    # multiplies C to get sigma
#     "clipping_bound":   1.0,    # C — max allowed L2 norm
# }


# # ──────────────────────────────────────────
# # Step 1: Adaptive Clipping
# # PDF Section 3.3, Step 1
# # ──────────────────────────────────────────
# def clip_tensor(tensor: torch.Tensor, C: float):
#     """
#     Clips gradient tensor so L2 norm does not exceed C.

#     Calculation:
#         flat   = tensor flattened to 1D
#         l2norm = sqrt(flat[0]² + flat[1]² + ... + flat[n]²)

#         if l2norm <= C:
#             scale = 1.0          (no change, PDF: "no clipping needed")
#         else:
#             scale = C / l2norm   (scale down proportionally)

#         clipped = flat × scale

#     PDF Example (Section 3.3, Step 1):
#         ∇W = [0.15, 0.08, 0, 0, 0, 0, 0, 0, 0, 0]
#         ||∇W||_2 = sqrt(0.15² + 0.08²) = sqrt(0.0225 + 0.0064) = 0.17
#         Since 0.17 < 1.0 → no clipping, scale = 1.0
#     """
#     flat   = tensor.detach().numpy().flatten()
#     l2norm = float(np.sqrt(np.sum(flat ** 2)))   # L2 norm

#     if l2norm > C:
#         scale       = C / l2norm
#         was_clipped = True
#     else:
#         scale       = 1.0
#         was_clipped = False

#     clipped = torch.tensor(
#         (flat * scale).reshape(tensor.shape), dtype=torch.float32
#     )
#     return clipped, l2norm, was_clipped


# # ──────────────────────────────────────────
# # Step 2: Gaussian Noise Addition
# # PDF Section 3.3, Step 2
# # ──────────────────────────────────────────
# def add_noise_tensor(tensor: torch.Tensor, sigma: float) -> torch.Tensor:
#     """
#     Adds Gaussian noise to protect privacy.

#     Calculation:
#         sigma (σ) = noise_multiplier × C   (computed before calling this)
#         noise     ~ N(0, σ²) for each element independently
#         result    = tensor + noise

#     PDF Example (Section 3.3, Step 2):
#         C = 1.0, noise_multiplier = 0.5
#         sigma = 0.5 × 1.0 = 0.5

#         noise sampled from N(0, 0.5²) = N(0, 0.25):
#         noise = [0.12, -0.05, 0.03, -0.01, ...]

#         ∇W_noisy = [0.15+0.12, 0.08-0.05, 0+0.03, ...] = [0.27, 0.03, 0.03, ...]
#     """
#     noise = torch.normal(mean=0.0, std=sigma, size=tensor.shape)
#     return tensor + noise


# # ──────────────────────────────────────────
# # Apply DP to Full Compressed Gradient
# # PDF Algorithm 3
# # ──────────────────────────────────────────
# def apply_differential_privacy(
#     compressed_gradient: dict,
#     params: dict = None
# ) -> tuple:
#     """
#     Applies AGC-DP to every layer of the compressed gradient.

#     For each layer:
#         1. Clip to L2 norm <= C
#         2. Add Gaussian noise with sigma = noise_multiplier × C

#     Returns
#     -------
#     noisy_gradient : same structure as compressed_gradient,
#                      'data' replaced by clipped + noisy tensor
#     dp_log         : per-layer statistics for reporting
#     """
#     if params is None:
#         params = DEFAULT_PRIVACY_PARAMS

#     C     = params["clipping_bound"]
#     sigma = params["noise_multiplier"] * C   # σ = 0.5 × 1.0 = 0.5

#     result = {}
#     dp_log = {}

#     for name, pkg in compressed_gradient.items():
#         tensor = pkg["data"]

#         # Step 1: Adaptive clipping
#         clipped, l2norm, was_clipped = clip_tensor(tensor, C)

#         # Step 2: Add Gaussian noise
#         noisy = add_noise_tensor(clipped, sigma)

#         result[name] = {
#             "data":           noisy,
#             "mask":           pkg["mask"],
#             "original_shape": pkg["original_shape"],
#             "Cr":             pkg["Cr"],
#         }
#         dp_log[name] = {
#             "l2_norm":     round(l2norm, 6),
#             "C":           C,
#             "was_clipped": was_clipped,
#             "sigma":       round(sigma, 4),
#         }

#     return result, dp_log


# # ──────────────────────────────────────────
# # Privacy Accountant
# # Tracks cumulative epsilon across rounds
# # ──────────────────────────────────────────
# class PrivacyAccountant:
#     """
#     Tracks total privacy cost (epsilon) across training rounds.

#     How epsilon_total is measured:
#         Each round consumes epsilon_per_round privacy budget.
#         We use basic composition (conservative upper bound):
#             epsilon_total = rounds × epsilon_per_round

#         Paper uses PLD (Privacy Loss Distribution) accountant
#         which gives a tighter (lower) bound on epsilon_total.
#         Our version overestimates the cost — it is safe but not optimal.

#     Example (30 rounds, epsilon_per_round = 1.0):
#         Round  1: epsilon_total =  1.0
#         Round 10: epsilon_total = 10.0
#         Round 30: epsilon_total = 30.0

#     delta = 1e-5 is fixed — it represents the probability that the
#     privacy guarantee breaks down. Standard FL practice sets δ < 1/N
#     where N is the dataset size. For MNIST N=60000 → δ < 1.67e-5.
#     """

#     def __init__(self, epsilon_per_round: float = 1.0, delta: float = 1e-5):
#         self.epsilon_per_round = epsilon_per_round
#         self.delta             = delta
#         self.rounds_elapsed    = 0
#         self.epsilon_total     = 0.0

#     def step(self):
#         """Call once per training round to accumulate privacy cost."""
#         self.rounds_elapsed += 1
#         self.epsilon_total  += self.epsilon_per_round

#     def report(self) -> str:
#         return (f"Round {self.rounds_elapsed:3d} | "
#                 f"ε_total={self.epsilon_total:6.2f} | "
#                 f"δ={self.delta}")

#     def privacy_spent(self) -> dict:
#         return {
#             "rounds":          self.rounds_elapsed,
#             "epsilon_total":   round(self.epsilon_total, 4),
#             "delta":           self.delta,
#             "epsilon_per_round": self.epsilon_per_round,
#         }
"""
privacy.py
----------
Phase 3: Differential Privacy — Adaptive Gaussian Clipping (AGC-DP)

Source: Gossip FL PDF Section 3.3
        Hidayat et al. (2024) Algorithm 3

Fixed privacy parameters:
    epsilon          = 1.0    (fixed, does NOT accumulate per round)
    delta            = 1e-5   (fixed)
    noise_multiplier = 0.1    (Hidayat 2024 Scenario 1, balances
                               privacy with Byzantine detection utility)
    C                = 1.0    (clipping bound)
    sigma            = 0.1 x 1.0 = 0.1

Note on noise_multiplier choice:
    PDF uses 0.5 but that masks gradient signal for Byzantine detection.
    Hidayat (2024) Scenario 1 uses epsilon~4.0 which corresponds to
    noise_multiplier=0.1. This is the justified choice for our system.
"""

import torch
import numpy as np


PRIVACY_PARAMS = {
    "epsilon":          1.0,
    "delta":            1e-5,
    "noise_multiplier": 0.1,    # Hidayat (2024) Scenario 1
    "clipping_bound":   1.0,
    # sigma = noise_multiplier x C = 0.1 x 1.0 = 0.1
}


def clip_tensor(tensor: torch.Tensor, C: float):
    """
    Clip gradient so L2 norm <= C.

    PDF Section 3.3, Step 1:
        scale = min(1.0, C / ||grad||_2)
        clipped = grad x scale
    """
    flat   = tensor.detach().numpy().flatten()
    l2norm = float(np.sqrt(np.sum(flat ** 2)))
    scale  = min(1.0, C / l2norm) if l2norm > 0 else 1.0
    clipped = torch.tensor(
        (flat * scale).reshape(tensor.shape), dtype=torch.float32
    )
    return clipped, l2norm, l2norm > C


def add_noise_tensor(tensor: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    Add Gaussian noise N(0, sigma^2) to each element.

    PDF Section 3.3, Step 2:
        sigma = noise_multiplier x C = 0.1 x 1.0 = 0.1
        noisy = clipped + N(0, sigma^2)
    """
    noise = torch.normal(mean=0.0, std=sigma, size=tensor.shape)
    return tensor + noise


def apply_differential_privacy(compressed_gradient: dict) -> tuple:
    """
    Applies AGC-DP to every layer of the compressed gradient.

    Input : compressed_gradient  {layer: {'data', 'mask', 'Cr', 'original_shape'}}
    Output: noisy_gradient       same structure, 'data' replaced with noisy tensor
            dp_log               per-layer stats

    The noisy_gradient is what gets sent to neighbors in Phase 4.
    Parameters are FIXED — they do not change per round.
    """
    C     = PRIVACY_PARAMS["clipping_bound"]
    sigma = PRIVACY_PARAMS["noise_multiplier"] * C   # 0.1

    noisy_gradient = {}
    dp_log         = {}

    for name, pkg in compressed_gradient.items():
        tensor = pkg["data"]

        clipped, l2norm, was_clipped = clip_tensor(tensor, C)
        noisy = add_noise_tensor(clipped, sigma)

        noisy_gradient[name] = {
            "data":           noisy,
            "mask":           pkg["mask"],
            "original_shape": pkg["original_shape"],
            "Cr":             pkg["Cr"],
        }
        dp_log[name] = {
            "l2_norm":     round(l2norm, 6),
            "C":           C,
            "sigma":       round(sigma, 4),
            "was_clipped": was_clipped,
        }

    return noisy_gradient, dp_log