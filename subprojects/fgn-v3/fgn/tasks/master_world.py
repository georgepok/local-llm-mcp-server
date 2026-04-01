"""MasterWorld — persistent topology with gradual mutations.

Maintains a single ContinuousWorld whose graph structure slowly shifts.
Each mutation severs, adds, or reweights an edge while preserving connectivity.
Episodes are generated from the current topology with fresh agent/object placement.
"""

import copy
import math
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from .continuous_gridworld import ContinuousWorld


class MasterWorld:
    """Persistent world topology that mutates every K steps.

    The model trains on episodes from this world. Every K steps, the
    topology shifts slightly — forcing the model to continuously adapt
    its internal geometry rather than memorizing a fixed lookup table.
    """

    def __init__(self, n_rooms: int = 15, n_objects: int = 4,
                 space_size: float = 150.0, connect_radius: float = 40.0,
                 locked_door_prob: float = 0.3,
                 seed: int = 42):
        self.n_rooms = n_rooms
        self.n_objects = n_objects
        self.space_size = space_size
        self.connect_radius = connect_radius
        self.locked_door_prob = locked_door_prob
        self.rng = random.Random(seed)

        # Create the base world
        self._base = ContinuousWorld(
            n_rooms=n_rooms, n_objects=n_objects,
            space_size=space_size, connect_radius=connect_radius,
            locked_door_prob=locked_door_prob,
            rng=random.Random(seed),
        )

        # Track topology state for mutations
        self.mutation_count = 0
        self.mutation_log = []

    @property
    def coords(self):
        return self._base.coords

    @property
    def graph(self):
        return self._base.graph

    @property
    def distances(self):
        return self._base.distances

    def create_episode_world(self) -> ContinuousWorld:
        """Create a fresh ContinuousWorld sharing this topology.

        Copies the graph structure but randomizes agent position,
        object placement, and room decorations for variety.
        """
        # Create a new world object that shares our topology
        w = ContinuousWorld.__new__(ContinuousWorld)
        w.n_rooms = self._base.n_rooms
        w.space_size = self._base.space_size
        w.connect_radius = self._base.connect_radius
        w.locked_door_prob = self._base.locked_door_prob
        w.rng = random.Random(self.rng.randint(0, 2**32))

        # Share topology (deep copy so episode mutations don't affect master)
        w.coords = list(self._base.coords)
        w.graph = {k: list(v) for k, v in self._base.graph.items()}
        w.distances = dict(self._base.distances)
        w.locked_edges = set(self._base.locked_edges)

        # Fresh room decoration
        w.room_types = w._assign_room_types()
        from .gridworld import ROOM_FIXTURES
        w.room_surfaces = [list(ROOM_FIXTURES[t][0]) for t in w.room_types]
        w.room_appliances = [list(ROOM_FIXTURES[t][1]) for t in w.room_types]

        # Fresh object placement and agent position
        w.objects = w._place_objects(self.n_objects)
        w.agent_room = w.rng.randint(0, w.n_rooms - 1)
        w.holding = None
        w.holding_clean = False
        w.holding_temp = "neutral"

        return w

    def mutate(self, mutation_type: str = "auto") -> str:
        """Apply a single topology mutation while preserving connectivity.

        Args:
            mutation_type: "sever", "add", "reweight", or "auto" (random choice)

        Returns:
            Description string of what changed.
        """
        if mutation_type == "auto":
            mutation_type = self.rng.choice(["sever", "add", "reweight"])

        if mutation_type == "sever":
            result = self._mutate_sever()
        elif mutation_type == "add":
            result = self._mutate_add()
        elif mutation_type == "reweight":
            result = self._mutate_reweight()
        else:
            raise ValueError(f"Unknown mutation type: {mutation_type}")

        self.mutation_count += 1
        self.mutation_log.append((self.mutation_count, mutation_type, result))
        return result

    def _mutate_sever(self) -> str:
        """Remove a random edge while keeping graph connected."""
        edges = []
        for i in self._base.graph:
            for j in self._base.graph[i]:
                if i < j:
                    edges.append((i, j))

        if not edges:
            return "no edges to sever"

        self.rng.shuffle(edges)
        for i, j in edges:
            # Temporarily remove
            self._base.graph[i].remove(j)
            self._base.graph[j].remove(i)

            if self._base._is_connected():
                # Keep it severed
                self._base.distances.pop((i, j), None)
                self._base.distances.pop((j, i), None)
                self._base.locked_edges.add((min(i, j), max(i, j)))
                return f"severed edge ({i},{j})"
            else:
                # Restore
                self._base.graph[i].append(j)
                self._base.graph[j].append(i)
                self._base.graph[i].sort()
                self._base.graph[j].sort()

        return "no safe edge to sever"

    def _mutate_add(self) -> str:
        """Add a new edge between disconnected nearby rooms, or restore a locked edge."""
        # First try restoring a locked edge
        if self._base.locked_edges:
            locked = list(self._base.locked_edges)
            self.rng.shuffle(locked)
            i, j = locked[0]

            # Compute distance
            dx = self._base.coords[i][0] - self._base.coords[j][0]
            dy = self._base.coords[i][1] - self._base.coords[j][1]
            dist = math.hypot(dx, dy)

            self._base.graph.setdefault(i, []).append(j)
            self._base.graph.setdefault(j, []).append(i)
            self._base.graph[i].sort()
            self._base.graph[j].sort()
            self._base.distances[(i, j)] = dist
            self._base.distances[(j, i)] = dist
            self._base.locked_edges.remove((i, j))
            return f"restored edge ({i},{j}), dist={dist:.1f}"

        # Otherwise add a new edge between unconnected rooms within 2x radius
        candidates = []
        for i in range(self._base.n_rooms):
            neighbors = set(self._base.graph.get(i, []))
            for j in range(i + 1, self._base.n_rooms):
                if j not in neighbors:
                    dx = self._base.coords[i][0] - self._base.coords[j][0]
                    dy = self._base.coords[i][1] - self._base.coords[j][1]
                    dist = math.hypot(dx, dy)
                    if dist <= self._base.connect_radius * 2.0:
                        candidates.append((i, j, dist))

        if not candidates:
            return "no candidate edges to add"

        i, j, dist = self.rng.choice(candidates)
        self._base.graph.setdefault(i, []).append(j)
        self._base.graph.setdefault(j, []).append(i)
        self._base.graph[i].sort()
        self._base.graph[j].sort()
        self._base.distances[(i, j)] = dist
        self._base.distances[(j, i)] = dist
        return f"added edge ({i},{j}), dist={dist:.1f}"

    def _mutate_reweight(self) -> str:
        """Perturb the weight of a random edge by ±30%."""
        edges = []
        for i in self._base.graph:
            for j in self._base.graph[i]:
                if i < j:
                    edges.append((i, j))

        if not edges:
            return "no edges to reweight"

        i, j = self.rng.choice(edges)
        old_dist = self._base.distances[(i, j)]
        factor = self.rng.uniform(0.7, 1.3)
        new_dist = old_dist * factor
        self._base.distances[(i, j)] = new_dist
        self._base.distances[(j, i)] = new_dist
        return f"reweighted ({i},{j}): {old_dist:.1f} -> {new_dist:.1f}"

    def catastrophic_mutate(self, fraction: float = 0.2) -> str:
        """Mutate a fraction of all edges simultaneously.

        More violent than single-edge mutations — forces the model to
        continuously adapt rather than absorb small perturbations.

        Args:
            fraction: proportion of edges to mutate (0.2 = 20%)

        Returns:
            Description string of what changed.
        """
        edges = []
        for i in self._base.graph:
            for j in self._base.graph[i]:
                if i < j:
                    edges.append((i, j))

        if not edges:
            return "no edges to mutate"

        n_mutate = max(1, int(len(edges) * fraction))
        targets = self.rng.sample(edges, min(n_mutate, len(edges)))

        results = []
        for i, j in targets:
            action = self.rng.choice(["reweight_hard", "reweight_hard", "sever"])

            if action == "sever":
                # Try to sever, preserving connectivity
                self._base.graph[i].remove(j)
                self._base.graph[j].remove(i)
                if self._base._is_connected():
                    self._base.distances.pop((i, j), None)
                    self._base.distances.pop((j, i), None)
                    self._base.locked_edges.add((min(i, j), max(i, j)))
                    results.append(f"sever({i},{j})")
                else:
                    # Restore — can't sever without disconnecting
                    self._base.graph[i].append(j)
                    self._base.graph[j].append(i)
                    self._base.graph[i].sort()
                    self._base.graph[j].sort()
                    # Fall back to hard reweight
                    old_dist = self._base.distances[(i, j)]
                    factor = self.rng.uniform(0.3, 3.0)
                    new_dist = old_dist * factor
                    self._base.distances[(i, j)] = new_dist
                    self._base.distances[(j, i)] = new_dist
                    results.append(f"reweight({i},{j}):{old_dist:.0f}->{new_dist:.0f}")

            elif action == "reweight_hard":
                # Aggressive reweight: ±70% (was ±30%)
                old_dist = self._base.distances[(i, j)]
                factor = self.rng.uniform(0.3, 3.0)
                new_dist = old_dist * factor
                self._base.distances[(i, j)] = new_dist
                self._base.distances[(j, i)] = new_dist
                results.append(f"reweight({i},{j}):{old_dist:.0f}->{new_dist:.0f}")

        # Also try restoring some locked edges to keep topology alive
        if self._base.locked_edges and self.rng.random() < 0.3:
            locked = list(self._base.locked_edges)
            restore = self.rng.choice(locked)
            ri, rj = restore
            dx = self._base.coords[ri][0] - self._base.coords[rj][0]
            dy = self._base.coords[ri][1] - self._base.coords[rj][1]
            dist = math.hypot(dx, dy)
            self._base.graph.setdefault(ri, []).append(rj)
            self._base.graph.setdefault(rj, []).append(ri)
            self._base.graph[ri].sort()
            self._base.graph[rj].sort()
            self._base.distances[(ri, rj)] = dist
            self._base.distances[(rj, ri)] = dist
            self._base.locked_edges.remove(restore)
            results.append(f"restore({ri},{rj})")

        self.mutation_count += 1
        desc = f"catastrophic: {len(results)} ops [{', '.join(results[:3])}{'...' if len(results) > 3 else ''}]"
        self.mutation_log.append((self.mutation_count, "catastrophic", desc))
        return desc

    def get_topology_stats(self) -> Dict:
        """Return summary stats of current topology."""
        n_edges = sum(len(v) for v in self._base.graph.values()) // 2
        n_locked = len(self._base.locked_edges)
        edge_dists = [d for (i, j), d in self._base.distances.items() if i < j]
        avg_dist = sum(edge_dists) / len(edge_dists) if edge_dists else 0.0
        return {
            "n_rooms": self._base.n_rooms,
            "n_edges": n_edges,
            "n_locked": n_locked,
            "avg_edge_dist": avg_dist,
            "mutations": self.mutation_count,
        }


if __name__ == "__main__":
    print("Testing MasterWorld...")

    mw = MasterWorld(n_rooms=10, space_size=100.0, connect_radius=40.0,
                     locked_door_prob=0.2, seed=42)
    stats = mw.get_topology_stats()
    print(f"  Initial: {stats}")

    # Generate episode worlds
    for i in range(3):
        w = mw.create_episode_world()
        assert w.n_rooms == 10
        assert w._is_connected()
        print(f"  Episode {i}: agent at room {w.agent_room}, "
              f"{len(w.objects)} objects")

    # Apply mutations
    for i in range(10):
        result = mw.mutate()
        print(f"  Mutation {i+1}: {result}")

    stats = mw.get_topology_stats()
    print(f"  After 10 mutations: {stats}")

    # Verify connectivity after mutations
    w = mw.create_episode_world()
    assert w._is_connected(), "Graph disconnected after mutations!"

    print("MasterWorld OK")
