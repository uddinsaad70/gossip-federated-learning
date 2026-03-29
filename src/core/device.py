"""
Virtual Gossip Device for Federated Learning Simulation
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple
import copy

class VirtualGossipDevice:
    """
    Simulates a single device in gossip federated learning network.
    এটি একটা virtual device যা real device এর behavior simulate করে।
    """
    
    def __init__(
        self,
        device_id: int,
        model: nn.Module,
        neighbors: List[int],
        train_data: torch.utils.data.DataLoader,
        test_data: torch.utils.data.DataLoader = None,
        device_type: str = "medium",  # low, medium, high
        learning_rate: float = 0.01
    ):
        self.id = device_id
        self.model = model
        self.neighbors = neighbors
        self.train_loader = train_data
        self.test_loader = test_data
        self.device_type = device_type
        self.lr = learning_rate
        
        # Reputation tracking (প্রতিটি neighbor এর জন্য)
        self.reputation = {neighbor: 1.0 for neighbor in neighbors}
        self.rounds_since_contact = {neighbor: 0 for neighbor in neighbors}
        
        # Resource profile (device capability)
        self.resource_profile = self._get_resource_profile()
        
        # Communication logs
        self.communication_log = []
        self.total_bytes_sent = 0
        self.total_bytes_received = 0
        
        # Training history
        self.loss_history = []
        self.accuracy_history = []
        
        # Privacy budget
        self.epsilon_spent = 0.0
        self.delta = 1e-5
        
        # Gradient storage
        self.current_gradient = None
        
        # Device compute backend
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
    def _get_resource_profile(self) -> Dict:
        """
        Simulate different device capabilities.
        এখানে আমরা device type অনুযায়ী capability define করছি।
        """
        profiles = {
            "low": {  # Raspberry Pi জাতীয়
                "cpu_cores": 2,
                "memory_gb": 1,
                "capability_factor": 0.3,
                "compression_base": 0.7  # বেশি compression
            },
            "medium": {  # Smartphone জাতীয়
                "cpu_cores": 4,
                "memory_gb": 4,
                "capability_factor": 0.6,
                "compression_base": 0.5
            },
            "high": {  # Laptop/Desktop
                "cpu_cores": 8,
                "memory_gb": 16,
                "capability_factor": 1.0,
                "compression_base": 0.2  # কম compression
            }
        }
        return profiles.get(self.device_type, profiles["medium"])
    
    def calculate_compression_ratio(self) -> float:
        """
        Calculate dynamic compression ratio based on simulated resources.
        Formula from paper: C_r = M_u / (C_c / C_u)
        """
        # Simulate current resource usage (randomly vary)
        base_compression = self.resource_profile["compression_base"]
        
        # Add some randomness to simulate dynamic conditions
        noise = np.random.uniform(-0.1, 0.1)
        C_r = base_compression + noise
        
        # Clip to valid range
        C_r = np.clip(C_r, 0.1, 0.9)
        
        return C_r
    
    def local_training(self, epochs: int = 1) -> Dict[str, torch.Tensor]:
        """
        Perform local training and return gradient.
        """
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        
        # Store initial model
        initial_state = copy.deepcopy(self.model.state_dict())
        
        epoch_losses = []
        
        for epoch in range(epochs):
            total_loss = 0.0
            num_batches = 0
            
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            epoch_losses.append(avg_loss)
        
        self.loss_history.append(np.mean(epoch_losses))
        
        # Calculate gradient (difference between new and old model)
        gradient = {}
        current_state = self.model.state_dict()
        
        for name in current_state:
            gradient[name] = current_state[name] - initial_state[name]
        
        self.current_gradient = gradient
        return gradient
    
    def evaluate(self) -> Tuple[float, float]:
        """
        Evaluate model on test data.
        Returns: (accuracy, loss)
        """
        if self.test_loader is None:
            return 0.0, 0.0
        
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        
        correct = 0
        total = 0
        total_loss = 0.0
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = 100 * correct / total
        avg_loss = total_loss / len(self.test_loader)
        
        self.accuracy_history.append(accuracy)
        
        return accuracy, avg_loss
    
    def select_gossip_targets(self, k: int) -> List[int]:
        """
        Select k neighbors for gossip based on reputation.
        High reputation = higher selection probability
        """
        if len(self.neighbors) <= k:
            return self.neighbors.copy()
        
        # Calculate scores
        scores = []
        for neighbor in self.neighbors:
            reputation_score = self.reputation[neighbor]
            
            # Freshness: encourage talking to less recent contacts
            rounds_since = self.rounds_since_contact[neighbor]
            freshness_score = 1.0 / (1.0 + rounds_since)
            
            # Combined score (70% reputation, 30% freshness)
            combined = 0.7 * reputation_score + 0.3 * freshness_score
            scores.append(combined)
        
        # Normalize to probabilities
        scores = np.array(scores)
        probabilities = scores / scores.sum()
        
        # Weighted random selection
        selected_indices = np.random.choice(
            len(self.neighbors),
            size=k,
            replace=False,
            p=probabilities
        )
        
        return [self.neighbors[i] for i in selected_indices]
    
    def update_reputation(self, neighbor_id: int, quality_score: float):
        """
        Update reputation using exponential moving average.
        """
        alpha = 0.2  # Learning rate
        old_rep = self.reputation.get(neighbor_id, 0.5)
        new_rep = (1 - alpha) * old_rep + alpha * quality_score
        
        # Clip to [0.1, 1.0]
        self.reputation[neighbor_id] = np.clip(new_rep, 0.1, 1.0)
        
        # Reset contact timer
        self.rounds_since_contact[neighbor_id] = 0
    
    def increment_contact_timers(self):
        """Increment timers for all neighbors."""
        for neighbor in self.neighbors:
            self.rounds_since_contact[neighbor] += 1
    
    def update_model(self, aggregated_gradient: Dict[str, torch.Tensor]):
        """
        Update model with aggregated gradient.
        """
        current_state = self.model.state_dict()
        
        for name in current_state:
            if name in aggregated_gradient:
                current_state[name] = current_state[name] + aggregated_gradient[name]
        
        self.model.load_state_dict(current_state)
    
    def log_communication(self, bytes_sent: int, bytes_received: int):
        """Log communication costs."""
        self.total_bytes_sent += bytes_sent
        self.total_bytes_received += bytes_received
        
        self.communication_log.append({
            'sent': bytes_sent,
            'received': bytes_received,
            'total': bytes_sent + bytes_received
        })
    
    def get_stats(self) -> Dict:
        """Get device statistics."""
        return {
            'id': self.id,
            'type': self.device_type,
            'accuracy': self.accuracy_history[-1] if self.accuracy_history else 0.0,
            'loss': self.loss_history[-1] if self.loss_history else 0.0,
            'total_comm': self.total_bytes_sent + self.total_bytes_received,
            'epsilon_spent': self.epsilon_spent,
            'avg_reputation': np.mean(list(self.reputation.values()))
        }
    
    def __repr__(self):
        return f"Device({self.id}, type={self.device_type}, neighbors={len(self.neighbors)})"