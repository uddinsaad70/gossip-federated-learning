"""
topology.py
-----------
Builds a resource-adaptive network topology for the gossip FL system.

Implements Paper Section 1.3:
  - Greedy neighbor assignment (high-resource devices first)
  - Connectivity guarantee via DFS bridge-edge insertion
  - Neighbor saturation handling via capacity relaxation
  - New device join support via add_device()
"""

import networkx as nx
import matplotlib.pyplot as plt
from device import EdgeDevice, DEVICE_RANGES


# ──────────────────────────────────────────
# Create the 20-Device Network
# ──────────────────────────────────────────
def create_devices() -> list:
    """Creates and returns the default list of 20 EdgeDevice objects."""
    config = []
    for i in range(1, 7):
        config.append((i, "raspberry_pi", False))
    for i in range(7, 17):
        config.append((i, "laptop", False))
    config.append((17, "desktop", True))
    for i in range(18, 21):
        config.append((i, "desktop", False))
    return [EdgeDevice(did, dtype, byz) for did, dtype, byz in config]


# ──────────────────────────────────────────
# Topology Manager
# ──────────────────────────────────────────
class TopologyManager:
    K_MIN    = 3
    K_MAX    = 10
    HARD_CAP = 12

    def __init__(self, devices: list):
        self.devices = list(devices)
        self.did     = {d.id: d for d in self.devices}
        self.G       = nx.Graph()
        for d in self.devices:
            self.G.add_node(d.id, resource_score=d.resource_score, k=d.k)

    def _greedy_assign(self):
        sorted_devices = sorted(self.devices,
                                key=lambda d: d.resource_score, reverse=True)
        for device in sorted_devices:
            if self.G.degree(device.id) >= device.k:
                continue
            candidates = [
                d for d in self.devices
                if d.id != device.id
                and not self.G.has_edge(device.id, d.id)
                and self.G.degree(d.id) < d.k
            ]
            candidates.sort(key=lambda d: d.k - self.G.degree(d.id), reverse=True)
            for c in candidates:
                if self.G.degree(device.id) >= device.k:
                    break
                self.G.add_edge(device.id, c.id)

    def _handle_saturation(self):
        for device in self.devices:
            while self.G.degree(device.id) < device.k:
                unconnected = [
                    d for d in self.devices
                    if d.id != device.id
                    and not self.G.has_edge(device.id, d.id)
                ]
                if not unconnected:
                    break
                available = [d for d in unconnected if self.G.degree(d.id) < d.k]
                saturated = [d for d in unconnected if self.G.degree(d.id) >= d.k]
                if available:
                    best = max(available, key=lambda d: d.k - self.G.degree(d.id))
                    self.G.add_edge(device.id, best.id)
                elif saturated:
                    def relax(d):
                        return d.resource_score * (1 - self.G.degree(d.id) / self.HARD_CAP)
                    best = max(saturated, key=relax)
                    if self.G.degree(best.id) < self.HARD_CAP:
                        best.k = min(best.k + 1, self.HARD_CAP)
                        self.G.add_edge(device.id, best.id)
                    else:
                        break
                else:
                    break

    def _ensure_connectivity(self):
        components = list(nx.connected_components(self.G))
        while len(components) > 1:
            best_a = max(components[0], key=lambda nid: self.did[nid].resource_score)
            best_b = max(components[1], key=lambda nid: self.did[nid].resource_score)
            self.G.add_edge(best_a, best_b)
            components = list(nx.connected_components(self.G))

    def _sync_neighbors(self):
        for d in self.devices:
            d.neighbors = list(self.G.neighbors(d.id))
            d.reputation = {nid: d.reputation.get(nid, 1.0) for nid in d.neighbors}

    def build(self) -> nx.Graph:
        self._greedy_assign()
        self._handle_saturation()
        self._ensure_connectivity()
        self._sync_neighbors()
        return self.G

    def add_device(self, new_device: EdgeDevice):
        """
        Adds a new device to the existing network without full rebuild.

        - New device connects to k(new) neighbors
        - Those neighbors get new_device added to their list (bidirectional)
        - All other devices are unaffected
        - Connectivity is re-checked after joining
        """
        self.devices.append(new_device)
        self.did[new_device.id] = new_device
        self.G.add_node(new_device.id,
                        resource_score=new_device.resource_score,
                        k=new_device.k)

        print(f"\n[+] New device joining: {new_device}")
        print(f"    Needs {new_device.k} neighbors.")

        candidates = sorted(
            [d for d in self.devices if d.id != new_device.id],
            key=lambda d: d.k - self.G.degree(d.id),
            reverse=True
        )

        connected_to = []
        for candidate in candidates:
            if self.G.degree(new_device.id) >= new_device.k:
                break
            if self.G.degree(candidate.id) < candidate.k:
                self.G.add_edge(new_device.id, candidate.id)
                connected_to.append(candidate.id)
            else:
                relax_score = candidate.resource_score * (
                    1 - self.G.degree(candidate.id) / self.HARD_CAP
                )
                if relax_score > 0 and self.G.degree(candidate.id) < self.HARD_CAP:
                    candidate.k = min(candidate.k + 1, self.HARD_CAP)
                    self.G.add_edge(new_device.id, candidate.id)
                    connected_to.append(candidate.id)

        self._ensure_connectivity()
        self._sync_neighbors()

        print(f"    Connected to {len(connected_to)} neighbors: {connected_to}")
        print(f"    Note: Only these {len(connected_to)} devices updated their neighbor list.")
        print(f"          All other devices in the network are NOT affected.")


# ──────────────────────────────────────────
# Convenience wrapper
# ──────────────────────────────────────────
def build_topology(devices: list):
    """Builds topology and returns (graph, manager) tuple."""
    manager = TopologyManager(devices)
    G       = manager.build()
    return G, manager


# ──────────────────────────────────────────
# Print Summary
# ──────────────────────────────────────────
def print_topology_summary(devices: list, G: nx.Graph):
    print("=" * 75)
    print(f"{'NETWORK TOPOLOGY SUMMARY':^75}")
    print("=" * 75)
    for d in devices:
        byz = "  << BYZANTINE" if d.is_byzantine else ""
        print(f"  {d}")
        print(f"      actual neighbors={G.degree(d.id)} | "
              f"neighbor list: {list(G.neighbors(d.id))}{byz}")
    print("-" * 75)
    avg_k = sum(G.degree(d.id) for d in devices) / len(devices)
    print(f"  Total Edges : {G.number_of_edges()}")
    print(f"  Average k   : {avg_k:.2f}")
    print(f"  Connected   : {nx.is_connected(G)}")
    print(f"  Diameter    : {nx.diameter(G)}")
    print("=" * 75)


# ──────────────────────────────────────────
# Plot Graph
# Fix: pos calculated ONCE and reused for all draw calls
# ──────────────────────────────────────────
def plot_topology(devices: list, G: nx.Graph,
                  save_path: str = None, title_suffix: str = ""):
    """
    Draws the network graph.
    pos is calculated once and shared across all draw calls
    to ensure nodes, edges, and labels are all aligned.
    """
    did       = {d.id: d for d in devices}
    color_map = {"raspberry_pi": "#FF6B6B", "laptop": "#4ECDC4", "desktop": "#45B7D1"}

    node_colors = [
        "#FF0000" if did[n].is_byzantine else color_map[did[n].device_type]
        for n in G.nodes()
    ]
    node_sizes = [200 + did[n].resource_score * 200 for n in G.nodes()]
    labels     = {d.id: f"{d.id}\nk={G.degree(d.id)}" for d in devices}

    # Calculate pos ONCE — reuse for nodes, edges, and labels
    pos = nx.spring_layout(G, seed=42)

    plt.figure(figsize=(14, 10))
    nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                           node_size=node_sizes, alpha=0.9)
    nx.draw_networkx_edges(G, pos, alpha=0.4, width=1.5)
    nx.draw_networkx_labels(G, pos, labels, font_size=7)

    from matplotlib.patches import Patch
    plt.legend(handles=[
        Patch(color="#FF6B6B", label="Raspberry Pi  (low)"),
        Patch(color="#4ECDC4", label="Laptop        (medium)"),
        Patch(color="#45B7D1", label="Desktop       (high)"),
        Patch(color="#FF0000", label="Byzantine attacker"),
    ], loc="upper left", fontsize=10)

    title = "Resource-Adaptive Dynamic Topology\n(Node size proportional to resource score)"
    if title_suffix:
        title += f"\n{title_suffix}"
    plt.title(title, fontsize=13)
    plt.axis("off")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Graph saved -> {save_path}")
    plt.show()
