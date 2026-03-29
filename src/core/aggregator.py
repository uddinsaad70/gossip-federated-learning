"""
Reputation-Weighted Gradient Aggregation
"""
import torch
from typing import Dict, List

class ReputationWeightedAggregator:
    """
    Aggregates gradients using reputation and quality weighting.
    """
    
    def __init__(self, own_weight: float = 1.0):
        """
        Args:
            own_weight: Weight for device's own gradient
        """
        self.own_weight = own_weight
    
    def aggregate(
        self,
        own_gradient: Dict[str, torch.Tensor],
        received_gradients: List[Dict],
        quality_scores: Dict[int, float],
        reputations: Dict[int, float]
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate gradients with reputation-quality weighting.
        
        Args:
            own_gradient: Device's own gradient
            received_gradients: List of dicts with 'sender_id' and 'gradient'
            quality_scores: Quality score for each sender
            reputations: Reputation score for each sender
        
        Returns:
            aggregated_gradient: Weighted average gradient
        """
        if len(received_gradients) == 0:
            return own_gradient
        
        # Initialize aggregated gradient
        aggregated = {}
        for layer_name, tensor in own_gradient.items():
            aggregated[layer_name] = torch.zeros_like(tensor)
        
        total_weight = self.own_weight
        
        # Add own gradient
        for layer_name, tensor in own_gradient.items():
            aggregated[layer_name] += self.own_weight * tensor
        
        # Add received gradients
        for received in received_gradients:
            sender_id = received['sender_id']
            gradient = received['gradient']
            
            # Combined weight: reputation × quality
            reputation = reputations.get(sender_id, 0.5)
            quality = quality_scores.get(sender_id, 0.5)
            weight = reputation * quality
            
            # Skip if weight too low
            if weight < 0.01:
                continue
            
            total_weight += weight
            
            # Add weighted gradient
            for layer_name, tensor in gradient.items():
                if layer_name in aggregated:
                    aggregated[layer_name] += weight * tensor
        
        # Normalize by total weight
        for layer_name in aggregated:
            aggregated[layer_name] /= total_weight
        
        return aggregated
    
    def fedavg_aggregate(
        self,
        gradients: List[Dict[str, torch.Tensor]],
        weights: List[float] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Standard FedAvg aggregation (for comparison).
        
        Args:
            gradients: List of gradient dicts
            weights: Optional weights (default: uniform)
        
        Returns:
            averaged_gradient: Simple weighted average
        """
        if len(gradients) == 0:
            raise ValueError("No gradients to aggregate")
        
        # Default to uniform weights
        if weights is None:
            weights = [1.0] * len(gradients)
        
        # Normalize weights
        weights = torch.tensor(weights, dtype=torch.float32)
        weights = weights / weights.sum()
        
        # Initialize
        aggregated = {}
        for layer_name, tensor in gradients[0].items():
            aggregated[layer_name] = torch.zeros_like(tensor)
        
        # Weighted sum
        for i, gradient in enumerate(gradients):
            weight = weights[i].item()
            for layer_name, tensor in gradient.items():
                if layer_name in aggregated:
                    aggregated[layer_name] += weight * tensor
        
        return aggregated