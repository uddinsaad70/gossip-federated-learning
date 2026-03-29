"""
Data loading and partitioning for federated learning simulation
"""
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
import numpy as np
from typing import List, Tuple

class DataPartitioner:
    """
    Partition dataset for multiple devices.
    """
    
    def __init__(
        self,
        dataset: Dataset,
        num_devices: int,
        partition_type: str = "iid",
        alpha: float = 0.5
    ):
        """
        Args:
            dataset: PyTorch dataset
            num_devices: Number of devices
            partition_type: 'iid' or 'non_iid'
            alpha: Dirichlet distribution parameter for non-IID
        """
        self.dataset = dataset
        self.num_devices = num_devices
        self.partition_type = partition_type
        self.alpha = alpha
        
        self.device_indices = self._partition()
    
    def _partition(self) -> List[List[int]]:
        """Partition dataset indices."""
        n = len(self.dataset)
        
        if self.partition_type == "iid":
            # Shuffle and split equally
            indices = np.random.permutation(n)
            split_size = n // self.num_devices
            
            device_indices = []
            for i in range(self.num_devices):
                start = i * split_size
                end = start + split_size if i < self.num_devices - 1 else n
                device_indices.append(indices[start:end].tolist())
            
            return device_indices
        
        elif self.partition_type == "non_iid":
            # Dirichlet distribution for non-IID partitioning
            return self._dirichlet_partition()
        
        else:
            raise ValueError(f"Unknown partition type: {self.partition_type}")
    
    def _dirichlet_partition(self) -> List[List[int]]:
        """
        Non-IID partitioning using Dirichlet distribution.
        """
        # Get labels
        labels = np.array([self.dataset[i][1] for i in range(len(self.dataset))])
        num_classes = len(np.unique(labels))
        
        # Group indices by class
        class_indices = [np.where(labels == c)[0] for c in range(num_classes)]
        
        # Initialize device indices
        device_indices = [[] for _ in range(self.num_devices)]
        
        # For each class, distribute to devices using Dirichlet
        for c_indices in class_indices:
            # Sample proportions from Dirichlet
            proportions = np.random.dirichlet([self.alpha] * self.num_devices)
            
            # Shuffle class indices
            np.random.shuffle(c_indices)
            
            # Split according to proportions
            splits = (proportions * len(c_indices)).astype(int)
            splits[-1] = len(c_indices) - splits[:-1].sum()  # Adjust last split
            
            current_idx = 0
            for device_id, split_size in enumerate(splits):
                device_indices[device_id].extend(
                    c_indices[current_idx:current_idx + split_size].tolist()
                )
                current_idx += split_size
        
        # Shuffle each device's indices
        for indices in device_indices:
            np.random.shuffle(indices)
        
        return device_indices
    
    def get_device_dataset(self, device_id: int) -> Subset:
        """Get dataset subset for a device."""
        indices = self.device_indices[device_id]
        return Subset(self.dataset, indices)


def load_mnist(
    num_devices: int,
    batch_size: int = 32,
    partition_type: str = "iid",
    alpha: float = 0.5
) -> Tuple[List[DataLoader], DataLoader]:
    """
    Load and partition MNIST dataset.
    
    Returns:
        train_loaders: List of train DataLoaders for each device
        test_loader: Global test DataLoader
    """
    # Transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Download datasets
    train_dataset = datasets.MNIST(
        './data', train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        './data', train=False, transform=transform
    )
    
    # Partition training data
    partitioner = DataPartitioner(
        train_dataset,
        num_devices,
        partition_type,
        alpha
    )
    
    # Create data loaders
    train_loaders = []
    for device_id in range(num_devices):
        device_dataset = partitioner.get_device_dataset(device_id)
        loader = DataLoader(
            device_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False
        )
        train_loaders.append(loader)
    
    # Test loader (same for all devices)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_loaders, test_loader


def print_partition_stats(train_loaders: List[DataLoader]):
    """Print statistics about data partition."""
    print("\n=== Data Partition Statistics ===")
    print(f"Number of devices: {len(train_loaders)}")
    
    for device_id, loader in enumerate(train_loaders):
        num_samples = len(loader.dataset)
        print(f"Device {device_id}: {num_samples} samples")
    
    print("=" * 35)