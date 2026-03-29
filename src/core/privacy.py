"""
Adaptive Gaussian Clipping Differential Privacy (AGC-DP)
"""
import torch
import numpy as np
from typing import Dict, Tuple
from collections import deque

class AdaptiveGaussianClippingDP:
    """
    Implements AGC-DP for gradient privacy.
    Based on: Hidayat et al. paper
    """
    
    def __init__(
        self,
        epsilon_target: float = 1.0,
        delta_target: float = 1e-5,
        initial_quantile: float = 0.5,
        quantile_decay: float = 0.95,
        decay_interval: int = 20
    ):
        """
        Args:
            epsilon_target: Target privacy budget
            delta_target: Failure probability
            initial_quantile: Initial unclipped quantile
            quantile_decay: Decay factor for quantile
            decay_interval: Rounds between decay
        """
        self.epsilon_target = epsilon_target
        self.delta_target = delta_target
        self.unclipped_quantile = initial_quantile
        self.quantile_decay = quantile_decay
        self.decay_interval = decay_interval
        
        # Adaptive clipping state
        self.clipping_threshold = None
        self.unclipped_fractions = deque(maxlen=20)
        
        # Privacy accounting
        self.rounds_completed = 0
        self.epsilon_spent = 0.0
        
    def add_noise(
        self,
        gradient: Dict[str, torch.Tensor],
        round_num: int
    ) -> Tuple[Dict[str, torch.Tensor], Dict]:
        """
        Add differential privacy noise to gradient.
        
        Returns:
            noisy_gradient: Gradient with DP noise
            privacy_info: Privacy budget information
        """
        # Decay quantile periodically
        if round_num % self.decay_interval == 0 and round_num > 0:
            self.unclipped_quantile *= self.quantile_decay
            self.unclipped_quantile = max(self.unclipped_quantile, 0.1)
        
        # Initialize or update clipping threshold
        if self.clipping_threshold is None:
            self.clipping_threshold = self._initialize_threshold(gradient)
        else:
            self.clipping_threshold = self._update_threshold()
        
        # Clip gradients
        clipped_gradient, unclipped_fraction = self._clip_gradient(
            gradient, self.clipping_threshold
        )
        self.unclipped_fractions.append(unclipped_fraction)
        
        # Calculate noise scale
        noise_multiplier = self._calculate_noise_multiplier(round_num)
        
        # Add Gaussian noise
        noisy_gradient = {}
        for layer_name, grad_tensor in clipped_gradient.items():
            noise = torch.randn_like(grad_tensor)
            noise_scale = noise_multiplier * self.clipping_threshold
            noisy_gradient[layer_name] = grad_tensor + noise * noise_scale
        
        # Update privacy accounting
        self.rounds_completed = round_num
        self.epsilon_spent = self._compute_epsilon_spent(noise_multiplier, round_num)
        
        privacy_info = {
            'epsilon': self.epsilon_spent,
            'delta': self.delta_target,
            'rounds': round_num,
            'clipping_threshold': self.clipping_threshold,
            'noise_multiplier': noise_multiplier,
            'unclipped_fraction': unclipped_fraction
        }
        
        return noisy_gradient, privacy_info
    
    def _initialize_threshold(
        self,
        gradient: Dict[str, torch.Tensor]
    ) -> float:
        """Initialize clipping threshold based on gradient norms."""
        norms = []
        for grad_tensor in gradient.values():
            norm = torch.norm(grad_tensor.flatten(), p=2).item()
            norms.append(norm)
        
        if len(norms) == 0:
            return 1.0
        
        norms_array = np.array(norms)
        threshold = np.percentile(norms_array, self.unclipped_quantile * 100)
        
        return float(threshold)
    
    def _update_threshold(self) -> float:
        """Adaptively update clipping threshold."""
        if len(self.unclipped_fractions) == 0:
            return self.clipping_threshold
        
        # Average recent unclipped fraction
        recent_unclipped = np.mean(list(self.unclipped_fractions))
        
        # Adjust threshold based on target quantile
        tolerance = 0.1
        
        if recent_unclipped < self.unclipped_quantile - tolerance:
            # Too many clipped -> increase threshold
            new_threshold = self.clipping_threshold * 1.1
        elif recent_unclipped > self.unclipped_quantile + tolerance:
            # Too few clipped -> decrease threshold
            new_threshold = self.clipping_threshold * 0.9
        else:
            # Within tolerance
            new_threshold = self.clipping_threshold
        
        return new_threshold
    
    def _clip_gradient(
        self,
        gradient: Dict[str, torch.Tensor],
        threshold: float
    ) -> Tuple[Dict[str, torch.Tensor], float]:
        """
        Clip gradient layers to threshold.
        
        Returns:
            clipped_gradient: Clipped gradient dict
            unclipped_fraction: Fraction of layers not clipped
        """
        clipped = {}
        unclipped_count = 0
        total_count = 0
        
        for layer_name, grad_tensor in gradient.items():
            layer_norm = torch.norm(grad_tensor.flatten(), p=2).item()
            
            if layer_norm > threshold:
                # Clip by scaling
                scale_factor = threshold / layer_norm
                clipped[layer_name] = grad_tensor * scale_factor
            else:
                # No clipping needed
                clipped[layer_name] = grad_tensor
                unclipped_count += 1
            
            total_count += 1
        
        unclipped_fraction = (
            unclipped_count / total_count if total_count > 0 else 0.0
        )
        
        return clipped, unclipped_fraction
    
    def _calculate_noise_multiplier(self, round_num: int) -> float:
        """
        Calculate Gaussian noise multiplier.
        Simplified version - in production use proper privacy accountant.
        """
        # Base noise calculation
        # noise_scale = C × sqrt(2 × ln(1.25/δ)) / ε
        
        base_noise = 0.5
        privacy_factor = np.sqrt(2 * np.log(1.25 / self.delta_target))
        
        noise_multiplier = (base_noise * privacy_factor / self.epsilon_target)
        
        # Adjust for composition over rounds (simplified)
        composition_factor = np.sqrt(round_num)
        noise_multiplier = noise_multiplier / composition_factor
        
        return max(noise_multiplier, 0.01)  # Minimum noise
    
    def _compute_epsilon_spent(
        self,
        noise_multiplier: float,
        rounds: int
    ) -> float:
        """
        Compute cumulative epsilon spent.
        Simplified composition - use proper accountant in production.
        """
        # Basic composition theorem
        # ε_total ≈ ε_single × sqrt(k) for k rounds
        
        if noise_multiplier == 0:
            return float('inf')
        
        epsilon_per_round = 1.0 / noise_multiplier
        epsilon_total = epsilon_per_round * np.sqrt(rounds)
        
        # Cap at 1.5x target
        return min(epsilon_total, self.epsilon_target * 1.5)
    
    def reset(self):
        """Reset privacy accounting for new experiment."""
        self.clipping_threshold = None
        self.unclipped_fractions.clear()
        self.rounds_completed = 0
        self.epsilon_spent = 0.0