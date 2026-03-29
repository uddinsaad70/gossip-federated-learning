"""
Gradient Quality Assessment and Reputation System
"""
import torch
import numpy as np
from typing import Dict
from scipy.spatial.distance import cosine

class GradientQualityAssessor:
    """
    Assesses quality of received gradients for reputation system.
    """
    
    def __init__(
        self,
        outlier_threshold: float = 0.3,
        cosine_weight: float = 0.6,
        magnitude_weight: float = 0.4
    ):
        """
        Args:
            outlier_threshold: Fraction of outliers to flag as suspicious
            cosine_weight: Weight for cosine similarity
            magnitude_weight: Weight for magnitude consistency
        """
        self.outlier_threshold = outlier_threshold
        self.cosine_weight = cosine_weight
        self.magnitude_weight = magnitude_weight
    
    def assess_quality(
        self,
        local_gradient: Dict[str, torch.Tensor],
        received_gradient: Dict[str, torch.Tensor]
    ) -> float:
        """
        Assess quality of received gradient compared to local.
        
        Returns:
            quality_score: Score between 0 and 1
        """
        # Flatten both gradients
        local_flat = self._flatten_gradient(local_gradient)
        received_flat = self._flatten_gradient(received_gradient)
        
        # Check compatibility
        if local_flat.shape != received_flat.shape:
            return 0.0
        
        # Compute metrics
        cos_sim = self._cosine_similarity(local_flat, received_flat)
        mag_ratio = self._magnitude_consistency(local_flat, received_flat)
        is_outlier = self._detect_outlier(local_flat, received_flat)
        
        # If outlier, penalize heavily
        if is_outlier:
            return 0.0
        
        # Combined quality score
        quality = (
            self.cosine_weight * cos_sim +
            self.magnitude_weight * mag_ratio
        )
        
        return float(np.clip(quality, 0.0, 1.0))
    
    def _flatten_gradient(
        self,
        gradient: Dict[str, torch.Tensor]
    ) -> np.ndarray:
        """Flatten all layers into single vector."""
        flat_list = []
        for grad_tensor in gradient.values():
            flat_list.append(grad_tensor.cpu().detach().numpy().flatten())
        
        if len(flat_list) == 0:
            return np.array([])
        
        return np.concatenate(flat_list)
    
    def _cosine_similarity(
        self,
        vec1: np.ndarray,
        vec2: np.ndarray
    ) -> float:
        """Compute cosine similarity."""
        # Handle zero vectors
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Cosine distance -> similarity
        cos_dist = cosine(vec1, vec2)
        
        # Handle numerical issues
        if np.isnan(cos_dist):
            return 0.0
        
        cos_sim = 1.0 - cos_dist
        
        return max(cos_sim, 0.0)
    
    def _magnitude_consistency(
        self,
        vec1: np.ndarray,
        vec2: np.ndarray
    ) -> float:
        """Compute magnitude consistency ratio."""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        # Both zero
        if norm1 == 0 and norm2 == 0:
            return 1.0
        
        # One zero
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Ratio of smaller to larger
        ratio = min(norm1, norm2) / max(norm1, norm2)
        
        return ratio
    
    def _detect_outlier(
        self,
        local: np.ndarray,
        received: np.ndarray
    ) -> bool:
        """
        Detect outliers using IQR method.
        
        Returns:
            True if outlier detected
        """
        if len(local) == 0 or len(received) == 0:
            return True
        
        # Element-wise differences
        differences = np.abs(received - local)
        
        # Compute IQR
        Q1 = np.percentile(differences, 25)
        Q3 = np.percentile(differences, 75)
        IQR = Q3 - Q1
        
        # Handle case where IQR is zero
        if IQR == 0:
            return False
        
        # Outlier threshold
        upper_bound = Q3 + 1.5 * IQR
        
        # Count outliers
        outlier_count = np.sum(differences > upper_bound)
        outlier_fraction = outlier_count / len(differences)
        
        return outlier_fraction > self.outlier_threshold
    
    def assess_batch(
        self,
        local_gradient: Dict[str, torch.Tensor],
        received_gradients: list
    ) -> Dict[int, float]:
        """
        Assess quality for multiple received gradients.
        
        Args:
            local_gradient: Local device gradient
            received_gradients: List of dicts with 'sender_id' and 'gradient'
        
        Returns:
            quality_scores: Dict mapping sender_id -> quality_score
        """
        scores = {}
        
        for received in received_gradients:
            sender_id = received['sender_id']
            gradient = received['gradient']
            
            quality = self.assess_quality(local_gradient, gradient)
            scores[sender_id] = quality
        
        return scores