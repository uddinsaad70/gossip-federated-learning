"""
Complete Gossip-Based Federated Learning Simulation
এটাই main simulation script যেখানে সব কিছু একসাথে চলবে
"""
import sys
sys.path.append('..')

import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import List, Dict
import networkx as nx

# Import our modules
from src.core.device import VirtualGossipDevice
from src.core.compression import DCTCompressor, estimate_gradient_size
from src.core.privacy import AdaptiveGaussianClippingDP
from src.communication.quality_assessor import GradientQualityAssessor
from src.core.aggregator import ReputationWeightedAggregator
from src.models.simple_cnn import get_model
from src.utils.data_loader import load_mnist, print_partition_stats


class GossipFLSimulator:    
    """
    Main simulator for Gossip-based Federated Learning.
    """
    
    def __init__(
        self,
        num_devices: int = 20,
        neighbors_per_device: int = 5,
        num_rounds: int = 50,
        local_epochs: int = 1,
        batch_size: int = 32,
        learning_rate: float = 0.01,
        compression_enabled: bool = True,
        privacy_enabled: bool = True,
        epsilon_target: float = 1.0,
        device_types: List[str] = None
    ):
        """
        Initialize simulator.
        
        Args:
            num_devices: Number of devices in network
            neighbors_per_device: Gossip k parameter
            num_rounds: Total training rounds
            local_epochs: Local training epochs per round
            batch_size: Batch size for training
            learning_rate: Learning rate
            compression_enabled: Enable DCT compression
            privacy_enabled: Enable differential privacy
            epsilon_target: Privacy budget
            device_types: List of device types (or None for auto)
        """
        self.num_devices = num_devices
        self.k_gossip = neighbors_per_device
        self.num_rounds = num_rounds
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.compression_enabled = compression_enabled
        self.privacy_enabled = privacy_enabled
        self.epsilon_target = epsilon_target
        
        # Auto-assign device types if not provided
        if device_types is None:
            device_types = self._auto_assign_device_types()
        self.device_types = device_types
        
        # Components
        self.devices = []
        self.network_graph = None
        self.compressor = DCTCompressor()
        self.quality_assessor = GradientQualityAssessor()
        self.aggregator = ReputationWeightedAggregator()
        
        # Results tracking
        self.round_accuracies = []
        self.round_losses = []
        self.communication_costs = []
        self.privacy_budgets = []
        
    def _auto_assign_device_types(self) -> List[str]:
        """Auto-assign device types (30% low, 50% medium, 20% high)."""
        types = []
        
        num_low = int(0.3 * self.num_devices)
        num_medium = int(0.5 * self.num_devices)
        num_high = self.num_devices - num_low - num_medium
        
        types.extend(['low'] * num_low)
        types.extend(['medium'] * num_medium)
        types.extend(['high'] * num_high)
        
        np.random.shuffle(types)
        return types
    
    def setup_network(self):
        """Create gossip network topology."""
        print("Setting up gossip network...")
        
        # Create random graph
        G = nx.random_regular_graph(self.k_gossip, self.num_devices)
        self.network_graph = G
        
        # Get neighbors for each device
        neighbors_dict = {i: list(G.neighbors(i)) for i in range(self.num_devices)}
        
        # Load and partition data
        print("Loading MNIST dataset...")
        train_loaders, test_loader = load_mnist(
            self.num_devices,
            self.batch_size,
            partition_type="iid"  # Use "non_iid" for heterogeneous data
        )
        print_partition_stats(train_loaders)
        
        # Create devices
        print("Creating virtual devices...")
        for device_id in range(self.num_devices):
            model = get_model("simple", num_classes=10)
            
            device = VirtualGossipDevice(
                device_id=device_id,
                model=model,
                neighbors=neighbors_dict[device_id],
                train_data=train_loaders[device_id],
                test_data=test_loader,
                device_type=self.device_types[device_id],
                learning_rate=self.learning_rate
            )
            
            self.devices.append(device)
        
        print(f"✓ Created {len(self.devices)} devices")
        print(f"✓ Network topology: {self.k_gossip}-regular graph")
    
    def run_round(self, round_num: int):
        """Execute one round of gossip FL."""
        
        # Phase 1: Local Training (parallel simulation)
        print(f"  Phase 1: Local training...")
        local_gradients = []
        
        for device in self.devices:
            gradient = device.local_training(epochs=self.local_epochs)
            local_gradients.append(gradient)
        
        # Phase 2: Compression (if enabled)
        print(f"  Phase 2: Compression...")
        compressed_gradients = []
        compression_sizes = []
        
        for device_id, device in enumerate(self.devices):
            gradient = local_gradients[device_id]
            
            if self.compression_enabled:
                C_r = device.calculate_compression_ratio()
                compressed, metadata, size = self.compressor.compress(
                    gradient, C_r
                )
                compressed_gradients.append((compressed, metadata))
                compression_sizes.append(size)
            else:
                # No compression
                size = estimate_gradient_size(gradient)
                compressed_gradients.append((gradient, {}))
                compression_sizes.append(size)
        
        # Phase 3: Add Differential Privacy (if enabled)
        print(f"  Phase 3: Privacy protection...")
        private_gradients = []
        
        for device_id, device in enumerate(self.devices):
            compressed, metadata = compressed_gradients[device_id]
            
            if self.privacy_enabled:
                if not hasattr(device, 'dp_mechanism'):
                    device.dp_mechanism = AdaptiveGaussianClippingDP(
                        epsilon_target=self.epsilon_target
                    )
                
                noisy_gradient, privacy_info = device.dp_mechanism.add_noise(
                    compressed, round_num
                )
                device.epsilon_spent = privacy_info['epsilon']
                private_gradients.append((noisy_gradient, metadata))
            else:
                private_gradients.append((compressed, metadata))
        
        # Phase 4: Gossip Exchange (simulated)
        print(f"  Phase 4: Gossip exchange...")
        
        for device in self.devices:
            device.increment_contact_timers()
        
        # Each device selects gossip targets and exchanges
        exchanges = {}  # device_id -> list of (sender_id, gradient, metadata)
        
        for device in self.devices:
            targets = device.select_gossip_targets(self.k_gossip)
            exchanges[device.id] = []
            
            # Simulate sending to targets and receiving from them
            for target_id in targets:
                # Send own gradient to target
                own_grad, own_meta = private_gradients[device.id]
                
                # Receive target's gradient
                target_grad, target_meta = private_gradients[target_id]
                
                # Log communication
                bytes_sent = compression_sizes[device.id]
                bytes_received = compression_sizes[target_id]
                device.log_communication(bytes_sent, bytes_received)
                
                # Store received gradient
                exchanges[device.id].append({
                    'sender_id': target_id,
                    'gradient': target_grad,
                    'metadata': target_meta
                })
        
        # Phase 5: Quality Assessment
        print(f"  Phase 5: Quality assessment...")
        
        for device in self.devices:
            own_gradient = local_gradients[device.id]
            received = exchanges[device.id]
            
            # Decompress received gradients
            decompressed_received = []
            for recv in received:
                decomp_grad = self.compressor.decompress(
                    recv['gradient'],
                    recv['metadata']
                )
                decompressed_received.append({
                    'sender_id': recv['sender_id'],
                    'gradient': decomp_grad
                })
            
            # Assess quality
            quality_scores = self.quality_assessor.assess_batch(
                own_gradient,
                decompressed_received
            )
            
            # Update reputations
            for sender_id, quality in quality_scores.items():
                device.update_reputation(sender_id, quality)
        
        # Phase 6: Aggregation
        print(f"  Phase 6: Aggregation...")
        
        for device in self.devices:
            own_gradient = local_gradients[device.id]
            received = exchanges[device.id]
            
            # Decompress received
            decompressed_received = []
            for recv in received:
                decomp_grad = self.compressor.decompress(
                    recv['gradient'],
                    recv['metadata']
                )
                decompressed_received.append({
                    'sender_id': recv['sender_id'],
                    'gradient': decomp_grad
                })
            
            # Quality scores (already computed)
            quality_scores = self.quality_assessor.assess_batch(
                own_gradient,
                decompressed_received
            )
            
            # Aggregate
            aggregated = self.aggregator.aggregate(
                own_gradient,
                decompressed_received,
                quality_scores,
                device.reputation
            )
            
            # Update model
            device.update_model(aggregated)
    
    def evaluate_round(self, round_num: int):
        """Evaluate all devices at end of round."""
        print(f"  Evaluating...")
        
        accuracies = []
        losses = []
        total_comm = 0
        total_epsilon = 0
        
        for device in self.devices:
            acc, loss = device.evaluate()
            accuracies.append(acc)
            losses.append(loss)
            total_comm += device.total_bytes_sent + device.total_bytes_received
            total_epsilon += device.epsilon_spent
        
        avg_acc = np.mean(accuracies)
        avg_loss = np.mean(losses)
        avg_epsilon = total_epsilon / self.num_devices
        
        self.round_accuracies.append(avg_acc)
        self.round_losses.append(avg_loss)
        self.communication_costs.append(total_comm)
        self.privacy_budgets.append(avg_epsilon)
        
        print(f"  ✓ Avg Accuracy: {avg_acc:.2f}%")
        print(f"  ✓ Avg Loss: {avg_loss:.4f}")
        print(f"  ✓ Total Communication: {total_comm / 1e6:.2f} MB")
        print(f"  ✓ Avg ε spent: {avg_epsilon:.3f}")
    
    def train(self):
        """Run complete training simulation."""
        print("\n" + "="*60)
        print("Starting Gossip-Based Federated Learning Simulation")
        print("="*60)
        print(f"Devices: {self.num_devices}")
        print(f"Rounds: {self.num_rounds}")
        print(f"Compression: {'Enabled' if self.compression_enabled else 'Disabled'}")
        print(f"Privacy: {'Enabled (ε=' + str(self.epsilon_target) + ')' if self.privacy_enabled else 'Disabled'}")
        print("="*60 + "\n")
        
        # Setup
        self.setup_network()
        
        # Training loop
        for round_num in range(1, self.num_rounds + 1):
            print(f"\n{'='*60}")
            print(f"Round {round_num}/{self.num_rounds}")
            print(f"{'='*60}")
            
            self.run_round(round_num)
            self.evaluate_round(round_num)
        
        print("\n" + "="*60)
        print("Training Complete!")
        print("="*60)
        
        self.plot_results()
    
    def plot_results(self):
        """Plot training results."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Accuracy
        axes[0, 0].plot(self.round_accuracies, 'b-', linewidth=2)
        axes[0, 0].set_xlabel('Round')
        axes[0, 0].set_ylabel('Accuracy (%)')
        axes[0, 0].set_title('Average Accuracy over Rounds')
        axes[0, 0].grid(True)
        
        # Loss
        axes[0, 1].plot(self.round_losses, 'r-', linewidth=2)
        axes[0, 1].set_xlabel('Round')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Average Loss over Rounds')
        axes[0, 1].grid(True)
        
        # Communication Cost
        comm_mb = [c / 1e6 for c in self.communication_costs]
        axes[1, 0].plot(comm_mb, 'g-', linewidth=2)
        axes[1, 0].set_xlabel('Round')
        axes[1, 0].set_ylabel('Communication (MB)')
        axes[1, 0].set_title('Total Communication Cost')
        axes[1, 0].grid(True)
        
        # Privacy Budget
        axes[1, 1].plot(self.privacy_budgets, 'm-', linewidth=2)
        axes[1, 1].axhline(y=self.epsilon_target, color='r', linestyle='--', label='Target ε')
        axes[1, 1].set_xlabel('Round')
        axes[1, 1].set_ylabel('ε spent')
        axes[1, 1].set_title('Privacy Budget Spent')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('results/gossip_fl_results.png', dpi=300)
        print(f"\n✓ Results saved to 'results/gossip_fl_results.png'")
        plt.show()


if __name__ == "__main__":
    # Create results directory
    import os
    os.makedirs('results', exist_ok=True)
    
    # Run simulation
    simulator = GossipFLSimulator(
        num_devices=20,
        neighbors_per_device=5,
        num_rounds=30,
        local_epochs=1,
        batch_size=32,
        learning_rate=0.01,
        compression_enabled=True,
        privacy_enabled=True,
        epsilon_target=1.0
    )
    
    simulator.train()