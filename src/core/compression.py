"""
DCT-based Resource-Adaptive Compression for Gradients
"""
import torch
import numpy as np
from scipy.fft import dct, idct
from typing import Dict, Tuple
import pickle

class DCTCompressor:
    """
    Implements DCT-based gradient compression with weight pruning.
    """
    
    def __init__(self, dimension_threshold: int = 100):
        """
        Args:
            dimension_threshold: Minimum size to apply compression
        """
        self.dimension_threshold = dimension_threshold
    
    def compress(
        self,
        gradient: Dict[str, torch.Tensor],
        compression_ratio: float
    ) -> Tuple[Dict, Dict, int]:
        """
        Compress gradient dictionary.
        
        Args:
            gradient: Dict of layer_name -> gradient tensor
            compression_ratio: Target compression (0.1 = high, 0.9 = low)
        
        Returns:
            compressed_gradient: Compressed gradient dict
            metadata: Decompression metadata
            compressed_size_bytes: Estimated size in bytes
        """
        compressed = {}
        metadata = {}
        total_size = 0
        
        for layer_name, grad_tensor in gradient.items():
            # Skip very small layers
            if grad_tensor.numel() < self.dimension_threshold:
                compressed[layer_name] = grad_tensor
                metadata[layer_name] = {'compressed': False}
                total_size += grad_tensor.numel() * 4  # float32
            else:
                comp_tensor, meta = self._compress_layer(
                    grad_tensor, compression_ratio
                )
                compressed[layer_name] = comp_tensor
                metadata[layer_name] = meta
                
                # Estimate compressed size (non-zero elements only)
                non_zero = torch.count_nonzero(comp_tensor).item()
                total_size += non_zero * 4  # Only non-zero values
        
        return compressed, metadata, total_size
    
    def _compress_layer(
        self,
        tensor: torch.Tensor,
        compression_ratio: float
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compress a single layer using DCT + pruning.
        """
        original_shape = tensor.shape
        original_device = tensor.device
        
        # Flatten
        flat = tensor.cpu().numpy().flatten()
        
        # Step 1: Weight Pruning by magnitude
        sorted_abs = np.sort(np.abs(flat))
        if len(sorted_abs) > 0:
            threshold = compression_ratio * sorted_abs[-1]
        else:
            threshold = 0.0
        
        # Zero out small values
        pruned = np.where(np.abs(flat) < threshold, 0.0, flat)
        
        # Step 2: DCT Transform
        dct_coeffs = dct(pruned, type=2, norm='ortho')
        
        # Step 3: DCT Coefficient Pruning (keep top 10%)
        max_coeff = np.max(np.abs(dct_coeffs))
        dct_threshold = 0.1 * max_coeff
        
        # Create sparse representation
        mask = (np.abs(dct_coeffs) >= dct_threshold).astype(np.float32)
        compressed_dct = dct_coeffs * mask
        
        # Convert back to tensor
        compressed_tensor = torch.from_numpy(compressed_dct).float().to(original_device)
        
        metadata = {
            'compressed': True,
            'original_shape': original_shape,
            'weight_threshold': float(threshold),
            'dct_threshold': float(dct_threshold),
            'compression_ratio': compression_ratio
        }
        
        return compressed_tensor, metadata
    
    def decompress(
        self,
        compressed: Dict[str, torch.Tensor],
        metadata: Dict[str, Dict]
    ) -> Dict[str, torch.Tensor]:
        """
        Decompress gradient using IDCT.
        """
        decompressed = {}
        
        for layer_name, comp_tensor in compressed.items():
            meta = metadata[layer_name]
            
            if not meta['compressed']:
                # No compression was applied
                decompressed[layer_name] = comp_tensor
            else:
                # Apply IDCT and reshape
                decomp = self._decompress_layer(comp_tensor, meta)
                decompressed[layer_name] = decomp
        
        return decompressed
    
    def _decompress_layer(
        self,
        compressed_tensor: torch.Tensor,
        metadata: Dict
    ) -> torch.Tensor:
        """
        Decompress a single layer.
        """
        original_shape = metadata['original_shape']
        original_device = compressed_tensor.device
        
        # Move to CPU for IDCT
        compressed_np = compressed_tensor.cpu().numpy()
        
        # Inverse DCT
        reconstructed = idct(compressed_np, type=2, norm='ortho')
        
        # Reshape and move back
        reconstructed_tensor = torch.from_numpy(reconstructed).float()
        reconstructed_tensor = reconstructed_tensor.reshape(original_shape)
        reconstructed_tensor = reconstructed_tensor.to(original_device)
        
        return reconstructed_tensor
    
    def calculate_compression_rate(
        self,
        original_gradient: Dict[str, torch.Tensor],
        compressed_size_bytes: int
    ) -> float:
        """
        Calculate actual compression rate achieved.
        """
        original_size = sum(
            tensor.numel() * 4  # float32 = 4 bytes
            for tensor in original_gradient.values()
        )
        
        if original_size == 0:
            return 0.0
        
        compression_rate = (1 - compressed_size_bytes / original_size) * 100
        return compression_rate


def estimate_gradient_size(gradient: Dict[str, torch.Tensor]) -> int:
    """
    Estimate gradient size in bytes (for simulation).
    """
    # Simple pickle serialization
    serialized = pickle.dumps(gradient)
    return len(serialized)