"""
byzantine.py
------------
Phase 5: Quality Assessment & Byzantine Detection

Source: Gossip FL PDF Section 3.5

Comparison method (DCT domain, no IDCT needed):
    own  = noisy_gradient['data']  — DCT domain + noise, L2 ~ 3.15
    recv = msg['gradient'][layer]  — DCT domain + noise, L2 ~ 3.15 (honest)
                                                           L2 ~ 159  (Byzantine)

    mag_ratio honest   = 3.15/3.15 = 0.997 -> trusted    (>0.1)
    mag_ratio Byzantine = 3.15/159  = 0.020 -> BYZANTINE  (<0.1)

    Both are in the same DCT domain, so comparison is valid.
    IDCT is NOT used here — it would make both L2 equal
    (IDCT is orthogonal, preserves L2) and lose the magnitude difference.

PDF Section 3.5.1 — Three Tests:
    Test 1: Cosine Similarity  cos_sim = A.B / (||A|| x ||B||)
    Test 2: Magnitude Ratio    mag_ratio = min(||A||,||B||) / max(||A||,||B||)
    Test 3: IQR Outlier        outlier_pct > 30% flags anomaly

Quality Score:
    if outlier_pct > 0.30 or cos_sim < -0.5 or mag_ratio < 0.1:
        quality = 0.0   <- Byzantine
    else:
        quality = 0.6 x cos_sim + 0.4 x mag_ratio

Reputation Update (PDF Section 3.5.4):
    rep_new = 0.8 x rep_old + 0.2 x quality
"""

import numpy as np
import torch


def flatten_noisy(noisy_gradient: dict) -> np.ndarray:
    """Flatten noisy_gradient (Phase 3 output) to 1D — stays in DCT domain."""
    return np.concatenate([
        pkg["data"].detach().numpy().flatten()
        for pkg in noisy_gradient.values()
    ])


def flatten_message(msg: dict) -> np.ndarray:
    """Flatten received message gradient to 1D — stays in DCT domain."""
    return np.concatenate([
        tensor.detach().numpy().flatten()
        for tensor in msg["gradient"].values()
    ])


# ── Test 1: Cosine Similarity ──────────────────────
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# ── Test 2: Magnitude Ratio ────────────────────────
def magnitude_ratio(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    denom  = max(norm_a, norm_b)
    if denom == 0:
        return 1.0
    return float(min(norm_a, norm_b) / denom)


# ── Test 3: IQR Outlier ────────────────────────────
def outlier_pct_iqr(a: np.ndarray, b: np.ndarray) -> float:
    diff = np.abs(b - a)
    if len(diff) < 4:
        return 0.0
    Q1        = np.percentile(diff, 25)
    Q3        = np.percentile(diff, 75)
    threshold = Q3 + 1.5 * (Q3 - Q1)
    return float(np.sum(diff > threshold) / len(diff))


# ── Quality Score (PDF Section 3.5.1) ──────────────
def compute_quality(own: np.ndarray, recv: np.ndarray) -> tuple:
    cos_sim = cosine_similarity(own, recv)
    mag_rat = magnitude_ratio(own, recv)
    out_pct = outlier_pct_iqr(own, recv)

    if out_pct > 0.30 or cos_sim < -0.5 or mag_rat < 0.1:
        quality = 0.0
    else:
        quality = 0.6 * cos_sim + 0.4 * mag_rat

    return quality, cos_sim, mag_rat, out_pct


# ── Main Phase 5 Function ──────────────────────────
def assess_received_gradients(device, received_messages: list,
                               own_noisy: dict) -> dict:
    """
    For each received message:
        1. Flatten own noisy gradient (DCT domain)
        2. Flatten received gradient  (DCT domain)
        3. Compute quality score
        4. Update reputation

    Parameters
    ----------
    device            : EdgeDevice
    received_messages : list of messages from Phase 4
    own_noisy         : device's noisy_gradient (Phase 3 output)
    """
    if own_noisy is None or not received_messages:
        return {}

    own_flat       = flatten_noisy(own_noisy)
    quality_scores = {}

    for msg in received_messages:
        sender_id = msg["sender"]
        recv_flat = flatten_message(msg)

        n        = min(len(own_flat), len(recv_flat))
        own_cmp  = own_flat[:n]
        recv_cmp = recv_flat[:n]

        quality, cos_sim, mag_rat, out_pct = compute_quality(own_cmp, recv_cmp)

        rep_old = device.reputation.get(sender_id, 1.0)
        rep_new = round(0.8 * rep_old + 0.2 * quality, 4)
        device.reputation[sender_id] = rep_new

        quality_scores[sender_id] = {
            "quality":      round(quality, 4),
            "cos_sim":      round(cos_sim, 4),
            "mag_ratio":    round(mag_rat, 4),
            "outlier_pct":  round(out_pct, 4),
            "rep_old":      round(rep_old, 4),
            "rep_new":      rep_new,
            "is_byzantine": quality == 0.0,
        }

    return quality_scores


def run_phase5(devices: list, received: dict,
               noisy_gradients: dict) -> dict:
    """
    Run Phase 5 for all devices.

    Parameters
    ----------
    noisy_gradients : {device_id: noisy_gradient}  — Phase 3 output
    """
    all_quality = {}
    for d in devices:
        msgs      = received.get(d.id, [])
        own_noisy = noisy_gradients.get(d.id)
        all_quality[d.id] = assess_received_gradients(d, msgs, own_noisy)
    return all_quality