"""Continuous Grid World Task — distance-weighted navigation and object manipulation.

Like GridWorld but with continuous 2D coordinates for rooms. Connections are
based on distance threshold R, and navigation cost is proportional to Euclidean
distance. Uses Dijkstra instead of BFS.

Key difference from discrete GridWorld:
- Two rooms can both be connected, but one is 5 units away and another 40.
- The model must learn to choose distance-efficient paths.
- A learned metric that encodes spatial distances should have an advantage.
"""

import heapq
import math
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch

# Reuse constants from discrete gridworld
from .gridworld import ROOM_TYPES, ROOM_FIXTURES, OBJECT_TYPES, TEMPS


# --- Continuous World ---

class ContinuousWorld:
    """Grid world with continuous 2D room coordinates."""

    def __init__(self, n_rooms: int = 10, n_objects: int = 4,
                 space_size: float = 100.0, connect_radius: float = 30.0,
                 locked_door_prob: float = 0.0,
                 rng: Optional[random.Random] = None):
        self.n_rooms = n_rooms
        self.space_size = space_size
        self.connect_radius = connect_radius
        self.locked_door_prob = locked_door_prob
        self.rng = rng or random.Random()

        # Generate rooms with coordinates
        self.coords = self._generate_coords()
        self.room_types = self._assign_room_types()
        self.room_surfaces = [list(ROOM_FIXTURES[t][0]) for t in self.room_types]
        self.room_appliances = [list(ROOM_FIXTURES[t][1]) for t in self.room_types]

        # Build distance-based graph
        self.graph, self.distances = self._build_graph()

        # Locked doors: randomly mask edges while keeping graph connected
        self.locked_edges = set()  # set of (i,j) tuples, i < j
        if locked_door_prob > 0:
            self._apply_locked_doors()

        # Place objects
        self.objects = self._place_objects(n_objects)

        # Agent state
        self.agent_room = self.rng.randint(0, n_rooms - 1)
        self.holding = None
        self.holding_clean = False
        self.holding_temp = "neutral"

    def _generate_coords(self) -> List[Tuple[float, float]]:
        """Generate random 2D coordinates for rooms, ensuring minimum separation."""
        coords = []
        min_dist = self.space_size * 0.05  # minimum 5% of space between rooms
        for _ in range(self.n_rooms):
            for _ in range(200):
                x = self.rng.uniform(0, self.space_size)
                y = self.rng.uniform(0, self.space_size)
                if all(math.hypot(x - cx, y - cy) >= min_dist for cx, cy in coords):
                    coords.append((x, y))
                    break
            else:
                # Fallback: accept any position
                coords.append((self.rng.uniform(0, self.space_size),
                                self.rng.uniform(0, self.space_size)))
        return coords

    def _assign_room_types(self) -> List[str]:
        types = ["kitchen"]  # guarantee a kitchen
        remaining = [t for t in ROOM_TYPES if t != "kitchen"]
        for _ in range(self.n_rooms - 1):
            types.append(self.rng.choice(remaining))
        self.rng.shuffle(types)
        return types

    def _build_graph(self) -> Tuple[Dict[int, List[int]], Dict[Tuple[int, int], float]]:
        """Build graph: rooms within connect_radius are connected."""
        graph = defaultdict(list)
        distances = {}

        for i in range(self.n_rooms):
            for j in range(i + 1, self.n_rooms):
                dx = self.coords[i][0] - self.coords[j][0]
                dy = self.coords[i][1] - self.coords[j][1]
                dist = math.hypot(dx, dy)
                if dist <= self.connect_radius:
                    graph[i].append(j)
                    graph[j].append(i)
                    distances[(i, j)] = dist
                    distances[(j, i)] = dist

        # Ensure connectivity: if any room is isolated, connect to nearest
        visited = set()
        if graph:
            start = next(iter(graph))
        else:
            start = 0
        queue = [start]
        visited.add(start)
        while queue:
            node = queue.pop(0)
            for nb in graph[node]:
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)

        # Connect unreachable rooms to their nearest reachable room
        for r in range(self.n_rooms):
            if r not in visited:
                best_dist = float('inf')
                best_target = None
                for v in visited:
                    dx = self.coords[r][0] - self.coords[v][0]
                    dy = self.coords[r][1] - self.coords[v][1]
                    d = math.hypot(dx, dy)
                    if d < best_dist:
                        best_dist = d
                        best_target = v
                if best_target is not None:
                    graph[r].append(best_target)
                    graph[best_target].append(r)
                    distances[(r, best_target)] = best_dist
                    distances[(best_target, r)] = best_dist
                    visited.add(r)

        # Sort adjacency lists
        for k in graph:
            graph[k].sort()

        return dict(graph), distances

    def _apply_locked_doors(self):
        """Randomly lock edges while keeping the graph connected.

        For each edge, lock it with probability locked_door_prob.
        Before locking, verify the graph remains connected without it.
        Locked edges are removed from self.graph and self.distances.
        """
        # Collect all edges as (i, j) with i < j
        all_edges = []
        for i in self.graph:
            for j in self.graph[i]:
                if i < j:
                    all_edges.append((i, j))

        # Shuffle to randomize which edges get considered first
        self.rng.shuffle(all_edges)

        for i, j in all_edges:
            if self.rng.random() >= self.locked_door_prob:
                continue

            # Temporarily remove edge and check connectivity
            self.graph[i].remove(j)
            self.graph[j].remove(i)

            if self._is_connected():
                # Lock it — keep it removed
                self.locked_edges.add((min(i, j), max(i, j)))
                self.distances.pop((i, j), None)
                self.distances.pop((j, i), None)
            else:
                # Restore — can't lock this one (would disconnect graph)
                self.graph[i].append(j)
                self.graph[j].append(i)
                self.graph[i].sort()
                self.graph[j].sort()

    def _is_connected(self) -> bool:
        """Check if graph is connected via BFS."""
        if self.n_rooms == 0:
            return True
        visited = set()
        queue = [0]
        visited.add(0)
        while queue:
            node = queue.pop(0)
            for nb in self.graph.get(node, []):
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        return len(visited) == self.n_rooms

    def _place_objects(self, n_objects: int) -> Dict[str, dict]:
        available = list(OBJECT_TYPES)
        self.rng.shuffle(available)
        selected = available[:n_objects]
        objects = {}
        for obj_name in selected:
            rooms_with_surfaces = [r for r in range(self.n_rooms) if self.room_surfaces[r]]
            if not rooms_with_surfaces:
                continue
            room = self.rng.choice(rooms_with_surfaces)
            surface = self.rng.choice(self.room_surfaces[room])
            objects[obj_name] = {
                "room": room,
                "surface": surface,
                "clean": self.rng.choice([True, False]),
                "temp": self.rng.choice(TEMPS),
            }
        return objects

    def all_pairs_shortest_paths(self) -> Dict[Tuple[int, int], float]:
        """Compute shortest-path distance between all room pairs via Dijkstra."""
        result = {}
        for source in range(self.n_rooms):
            dist = {source: 0.0}
            heap = [(0.0, source)]
            while heap:
                d, u = heapq.heappop(heap)
                if d > dist.get(u, float('inf')):
                    continue
                for v in self.graph.get(u, []):
                    edge_d = self.distances.get((u, v), float('inf'))
                    new_d = d + edge_d
                    if new_d < dist.get(v, float('inf')):
                        dist[v] = new_d
                        heapq.heappush(heap, (new_d, v))
            for target in range(self.n_rooms):
                result[(source, target)] = dist.get(target, float('inf'))
        return result

    def edge_distance(self, a: int, b: int) -> float:
        """Get distance between connected rooms a and b."""
        return self.distances.get((a, b), float('inf'))

    def get_observation(self) -> str:
        r = self.agent_room
        rtype = self.room_types[r]
        cx, cy = self.coords[r]

        surface_contents = {}
        for obj_name, obj in self.objects.items():
            if obj["room"] == r and obj_name != self.holding:
                surf = obj["surface"]
                desc = self._describe_object(obj_name, obj)
                surface_contents.setdefault(surf, []).append(desc)

        see_parts = []
        for surf in self.room_surfaces[r]:
            if surf in surface_contents:
                items = ", ".join(surface_contents[surf])
                see_parts.append(f"{surf} ({items})")
            else:
                see_parts.append(surf)
        for appl in self.room_appliances[r]:
            see_parts.append(appl)
        see_str = ", ".join(see_parts) if see_parts else "nothing"

        neighbors = self.graph.get(r, [])
        doors_parts = []
        for n in neighbors:
            d = self.distances.get((r, n), 0.0)
            doors_parts.append(f"room {n} (dist={d:.1f})")
        doors_str = ", ".join(doors_parts)

        if self.holding:
            hold_desc = self._describe_object(
                self.holding, {"clean": self.holding_clean, "temp": self.holding_temp})
        else:
            hold_desc = "nothing"

        return (f"Room {r} ({rtype}) pos({cx:.1f},{cy:.1f}). You see: {see_str}. "
                f"Doors: {doors_str}. Holding: {hold_desc}.")

    def _describe_object(self, name: str, obj: dict) -> str:
        parts = []
        if not obj["clean"]:
            parts.append("dirty")
        if obj["temp"] == "hot":
            parts.append("hot")
        elif obj["temp"] == "cold":
            parts.append("cold")
        if parts:
            return f"{' '.join(parts)} {name}"
        return name

    def apply_action(self, action: str) -> Tuple[bool, float]:
        """Apply action, return (valid, cost). Cost is edge distance for navigation."""
        parts = action.split()
        if not parts:
            return False, 0.0

        cmd = parts[0]

        if cmd == "go" and len(parts) == 2:
            try:
                target = int(parts[1])
            except ValueError:
                return False, 0.0
            if target in self.graph.get(self.agent_room, []):
                cost = self.distances.get((self.agent_room, target), 0.0)
                self.agent_room = target
                return True, cost
            return False, 0.0

        elif cmd == "take" and len(parts) == 2:
            obj_name = parts[1]
            if self.holding is not None:
                return False, 0.0
            if obj_name in self.objects and self.objects[obj_name]["room"] == self.agent_room:
                obj = self.objects[obj_name]
                self.holding = obj_name
                self.holding_clean = obj["clean"]
                self.holding_temp = obj["temp"]
                obj["room"] = -1
                obj["surface"] = ""
                return True, 0.0
            return False, 0.0

        elif cmd == "put" and len(parts) == 3:
            obj_name, surface = parts[1], parts[2]
            if self.holding != obj_name:
                return False, 0.0
            if surface not in self.room_surfaces[self.agent_room]:
                return False, 0.0
            self.objects[obj_name]["room"] = self.agent_room
            self.objects[obj_name]["surface"] = surface
            self.objects[obj_name]["clean"] = self.holding_clean
            self.objects[obj_name]["temp"] = self.holding_temp
            self.holding = None
            return True, 0.0

        elif cmd == "clean" and len(parts) == 2:
            if self.holding != parts[1]:
                return False, 0.0
            if "sink" not in self.room_appliances[self.agent_room]:
                return False, 0.0
            self.holding_clean = True
            return True, 0.0

        elif cmd == "heat" and len(parts) == 2:
            if self.holding != parts[1]:
                return False, 0.0
            if "stove" not in self.room_appliances[self.agent_room]:
                return False, 0.0
            self.holding_temp = "hot"
            return True, 0.0

        elif cmd == "cool" and len(parts) == 2:
            if self.holding != parts[1]:
                return False, 0.0
            if "fridge" not in self.room_appliances[self.agent_room]:
                return False, 0.0
            self.holding_temp = "cold"
            return True, 0.0

        return False, 0.0

    def get_state(self) -> tuple:
        obj_states = tuple(
            (name, obj["room"], obj["surface"], obj["clean"], obj["temp"])
            for name, obj in sorted(self.objects.items())
        )
        return (self.agent_room, self.holding, self.holding_clean,
                self.holding_temp, obj_states)

    def set_state(self, state: tuple):
        (self.agent_room, self.holding, self.holding_clean,
         self.holding_temp, obj_states) = state
        for name, room, surface, clean, temp in obj_states:
            self.objects[name]["room"] = room
            self.objects[name]["surface"] = surface
            self.objects[name]["clean"] = clean
            self.objects[name]["temp"] = temp


# --- Dijkstra Solver ---

class DijkstraSolver:
    """Dijkstra-based planner for distance-weighted navigation."""

    def __init__(self, world: ContinuousWorld, goal: dict):
        self.world = world
        self.goal = goal
        self.goal_obj = goal["obj"]

    def _compact_state(self) -> tuple:
        w = self.world
        obj = w.objects[self.goal_obj]
        if w.holding == self.goal_obj:
            return (w.agent_room, True, w.holding_clean, w.holding_temp)
        else:
            return (w.agent_room, False, obj["room"], obj["surface"],
                    obj["clean"], obj["temp"])

    def solve(self) -> Optional[Tuple[List[str], float]]:
        """Find minimum-cost action sequence. Returns (actions, total_cost) or None."""
        initial_full = self.world.get_state()
        initial = self._compact_state()

        # Priority queue: (cost, counter, full_state, compact_state, actions)
        counter = 0
        heap = [(0.0, counter, initial_full, initial, [])]
        best_cost = {initial: 0.0}

        while heap:
            cost, _, full_state, compact, actions = heapq.heappop(heap)

            if self._is_goal(compact):
                self.world.set_state(initial_full)
                return actions, cost

            if cost > best_cost.get(compact, float('inf')):
                continue

            for action in self._get_valid_actions(full_state):
                self.world.set_state(full_state)
                valid, step_cost = self.world.apply_action(action)
                if not valid:
                    continue

                next_full = self.world.get_state()
                next_compact = self._compact_state()
                new_cost = cost + step_cost

                if new_cost < best_cost.get(next_compact, float('inf')):
                    best_cost[next_compact] = new_cost
                    counter += 1
                    heapq.heappush(heap,
                        (new_cost, counter, next_full, next_compact, actions + [action]))

        self.world.set_state(initial_full)
        return None

    def _is_goal(self, compact: tuple) -> bool:
        g = self.goal
        if compact[1]:  # holding goal obj
            return False
        _, _, obj_room, obj_surface, obj_clean, obj_temp = compact
        return (obj_room == g["room"] and obj_surface == g["surface"] and
                obj_clean == g["clean"] and obj_temp == g["temp"])

    def _get_valid_actions(self, full_state: tuple) -> List[str]:
        self.world.set_state(full_state)
        w = self.world
        actions = []

        for neighbor in w.graph.get(w.agent_room, []):
            actions.append(f"go {neighbor}")

        if w.holding is None:
            obj = w.objects[self.goal_obj]
            if obj["room"] == w.agent_room:
                actions.append(f"take {self.goal_obj}")
        elif w.holding == self.goal_obj:
            for surface in w.room_surfaces[w.agent_room]:
                actions.append(f"put {self.goal_obj} {surface}")
            if "sink" in w.room_appliances[w.agent_room]:
                actions.append(f"clean {self.goal_obj}")
            if "stove" in w.room_appliances[w.agent_room]:
                actions.append(f"heat {self.goal_obj}")
            if "fridge" in w.room_appliances[w.agent_room]:
                actions.append(f"cool {self.goal_obj}")

        return actions


# --- Episode Generation ---

def generate_goal(world: ContinuousWorld, rng: random.Random,
                  min_state_changes: int = 1) -> Optional[dict]:
    """Generate a random goal requiring state transformations."""
    obj_names = list(world.objects.keys())
    if not obj_names:
        return None

    rng.shuffle(obj_names)

    for obj_name in obj_names:
        obj = world.objects[obj_name]
        target_clean = rng.choice([True, False])
        target_temp = rng.choice(TEMPS)

        changes = 0
        if target_clean != obj["clean"]:
            changes += 1
        if target_temp != obj["temp"]:
            changes += 1

        if changes < min_state_changes:
            if not obj["clean"]:
                target_clean = True
                changes += 1
            elif obj["temp"] == "neutral":
                target_temp = rng.choice(["hot", "cold"])
                changes += 1

        if changes < min_state_changes:
            continue

        if target_clean and not obj["clean"]:
            if not any("sink" in world.room_appliances[r] for r in range(world.n_rooms)):
                continue
        if target_temp == "hot" and obj["temp"] != "hot":
            if not any("stove" in world.room_appliances[r] for r in range(world.n_rooms)):
                continue
        if target_temp == "cold" and obj["temp"] != "cold":
            if not any("fridge" in world.room_appliances[r] for r in range(world.n_rooms)):
                continue

        possible_rooms = [r for r in range(world.n_rooms)
                          if world.room_surfaces[r] and r != obj["room"]]
        if not possible_rooms:
            continue

        target_room = rng.choice(possible_rooms)
        target_surface = rng.choice(world.room_surfaces[target_room])

        return {
            "obj": obj_name,
            "clean": target_clean,
            "temp": target_temp,
            "room": target_room,
            "surface": target_surface,
        }

    return None


def render_world_description(world: ContinuousWorld) -> str:
    """Render world with coordinates and distances."""
    lines = []
    for r in range(world.n_rooms):
        rtype = world.room_types[r]
        cx, cy = world.coords[r]
        parts = list(world.room_surfaces[r]) + list(world.room_appliances[r])
        fixtures_str = ", ".join(parts) if parts else "empty"

        neighbors = world.graph.get(r, [])
        conn_parts = []
        for n in neighbors:
            d = world.distances.get((r, n), 0.0)
            conn_parts.append(f"room {n} (dist={d:.1f})")

        # Show locked doors (nearby but inaccessible)
        locked_parts = []
        for i, j in world.locked_edges:
            if i == r:
                locked_parts.append(f"room {j} (LOCKED)")
            elif j == r:
                locked_parts.append(f"room {i} (LOCKED)")

        conn_str = ", ".join(conn_parts) if conn_parts else "none"
        if locked_parts:
            conn_str += ". Locked: " + ", ".join(locked_parts)

        lines.append(f"[WORLD] Room {r} ({rtype}) pos({cx:.1f},{cy:.1f}): "
                      f"{fixtures_str}. Connects to: {conn_str}.\n")

    obj_parts = []
    for obj_name, obj in sorted(world.objects.items()):
        desc = world._describe_object(obj_name, obj)
        obj_parts.append(f"{desc} on {obj['surface']} in room {obj['room']}")
    if obj_parts:
        lines.append(f"[OBJECTS] {'. '.join(obj_parts)}.\n")

    lines.append(f"[START] You are in room {world.agent_room}. Holding: nothing.\n")
    return "".join(lines)


def render_goal_text(goal: dict) -> str:
    parts = []
    if goal["clean"]:
        parts.append("clean")
    if goal["temp"] == "hot":
        parts.append("hot")
    elif goal["temp"] == "cold":
        parts.append("cold")

    state_str = " ".join(parts)
    if state_str:
        return (f"[GOAL] Put a {state_str} {goal['obj']} on the "
                f"{goal['surface']} in room {goal['room']}.\n")
    else:
        return (f"[GOAL] Put the {goal['obj']} on the "
                f"{goal['surface']} in room {goal['room']}.\n")


def render_episode(world: ContinuousWorld, goal: dict,
                   plan: List[str]) -> Tuple[str, List[str], List[float]]:
    """Render episode. Returns (text, actions, per_step_costs)."""
    lines = []
    lines.append(render_world_description(world))
    lines.append(render_goal_text(goal))
    lines.append(f"[OBS] {world.get_observation()}\n")

    action_strings = []
    step_costs = []
    for action in plan:
        lines.append(f"[ACT] {action}\n")
        action_strings.append(action)
        valid, cost = world.apply_action(action)
        assert valid, f"Solver produced invalid action: {action}"
        step_costs.append(cost)
        lines.append(f"[OBS] {world.get_observation()}\n")

    return "".join(lines), action_strings, step_costs


# --- Task Class ---

class ContinuousGridWorldTask:
    """Continuous grid world task with distance-weighted navigation.

    Args:
        tokenizer: HuggingFace tokenizer
        seq_len: maximum sequence length
        n_rooms_min, n_rooms_max: room count range
        space_size: coordinate space extent [0, space_size]
        connect_radius: rooms within this distance are connected
        n_objects: number of objects
        min_steps, max_steps: plan length constraints
        min_state_changes: minimum object state changes
        max_retries: generation retry limit
    """

    def __init__(self, tokenizer, seq_len: int = 1024,
                 n_rooms_min: int = 10, n_rooms_max: int = 15,
                 space_size: float = 100.0, connect_radius: float = 30.0,
                 locked_door_prob: float = 0.0,
                 n_objects: int = 4,
                 min_steps: int = 4, max_steps: int = 10,
                 min_state_changes: int = 1,
                 max_retries: int = 200,
                 **kwargs):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.n_rooms_min = n_rooms_min
        self.n_rooms_max = n_rooms_max
        self.space_size = space_size
        self.connect_radius = connect_radius
        self.locked_door_prob = locked_door_prob
        self.n_objects = n_objects
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.min_state_changes = min_state_changes
        self.max_retries = max_retries

    def generate_batch(self, batch_size: int,
                       device: Optional[torch.device] = None
                       ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        pad_id = self.tokenizer.eos_token_id or 0
        all_input_ids = []
        all_labels = []
        all_context_masks = []
        all_action_spans = []
        all_optimal_costs = []
        all_step_costs = []
        step_counts = []
        episode_worlds = []
        episode_room_positions = []

        for _ in range(batch_size):
            result = self._generate_valid_episode()
            episode_text, _actions, n_steps, optimal_cost, step_costs, world = result

            input_ids, labels, context_end_pos, action_spans, room_token_pos = \
                self._tokenize_episode(episode_text)
            step_counts.append(n_steps)
            all_optimal_costs.append(optimal_cost)
            all_step_costs.append(step_costs)
            episode_worlds.append(world)
            episode_room_positions.append(room_token_pos)

            if len(input_ids) > self.seq_len:
                input_ids = input_ids[:self.seq_len]
                labels = labels[:self.seq_len]
            else:
                pad_len = self.seq_len - len(input_ids)
                input_ids += [pad_id] * pad_len
                labels += [-100] * pad_len

            context_mask_row = [False] * self.seq_len
            for i in range(min(context_end_pos, self.seq_len)):
                context_mask_row[i] = True
            all_context_masks.append(context_mask_row)
            all_action_spans.append(action_spans)

            all_input_ids.append(input_ids)
            all_labels.append(labels)

        # Build graph-distance tensors
        R_max = max(w.n_rooms for w in episode_worlds)
        room_distances = torch.ones(batch_size, R_max, R_max)
        room_positions = torch.full((batch_size, R_max), -1, dtype=torch.long)
        n_rooms_tensor = torch.zeros(batch_size, dtype=torch.long)

        for b, (world, rtp) in enumerate(zip(episode_worlds, episode_room_positions)):
            R = world.n_rooms
            n_rooms_tensor[b] = R

            sp = world.all_pairs_shortest_paths()

            finite_dists = [d for d in sp.values() if d < float('inf') and d > 0]
            max_dist = max(finite_dists) if finite_dists else 1.0

            for i in range(R):
                for j in range(R):
                    d = sp.get((i, j), float('inf'))
                    if d < float('inf'):
                        room_distances[b, i, j] = d / max_dist

                if i in rtp and rtp[i] < self.seq_len:
                    room_positions[b, i] = rtp[i]

        input_ids_t = torch.tensor(all_input_ids, dtype=torch.long)
        labels_t = torch.tensor(all_labels, dtype=torch.long)

        if device is not None:
            input_ids_t = input_ids_t.to(device)
            labels_t = labels_t.to(device)

        metadata = {
            "task": "continuous_gridworld",
            "n_rooms_min": self.n_rooms_min,
            "n_rooms_max": self.n_rooms_max,
            "space_size": self.space_size,
            "connect_radius": self.connect_radius,
            "avg_steps": sum(step_counts) / max(len(step_counts), 1),
            "context_mask": torch.tensor(all_context_masks, dtype=torch.bool),
            "action_spans": all_action_spans,
            "optimal_costs": all_optimal_costs,
            "step_costs": all_step_costs,
            "room_distances": room_distances,
            "room_token_positions": room_positions,
            "n_rooms": n_rooms_tensor,
        }

        if device is not None:
            metadata["context_mask"] = metadata["context_mask"].to(device)
            metadata["room_distances"] = metadata["room_distances"].to(device)
            metadata["room_token_positions"] = metadata["room_token_positions"].to(device)
            metadata["n_rooms"] = metadata["n_rooms"].to(device)

        return input_ids_t, labels_t, metadata

    def _generate_valid_episode(self):
        for _ in range(self.max_retries):
            result = self._try_generate_episode()
            if result is not None:
                return result

        result = self._try_generate_episode(relax=True)
        if result is not None:
            return result
        return self._minimal_episode()

    def _try_generate_episode(self, relax: bool = False, override_world=None):
        rng = random.Random()

        if override_world is not None:
            world = override_world
        else:
            n_rooms = rng.randint(self.n_rooms_min, self.n_rooms_max)
            world = ContinuousWorld(
                n_rooms=n_rooms,
                n_objects=self.n_objects,
                space_size=self.space_size,
                connect_radius=self.connect_radius,
                locked_door_prob=self.locked_door_prob,
                rng=rng,
            )

        min_changes = 1 if relax else self.min_state_changes
        goal = generate_goal(world, rng, min_state_changes=min_changes)
        if goal is None:
            return None

        solver = DijkstraSolver(world, goal)
        solution = solver.solve()
        if solution is None:
            return None

        plan, total_cost = solution
        n_steps = len(plan)
        min_s = 1 if relax else self.min_steps
        max_s = self.max_steps * 2 if relax else self.max_steps

        if not (min_s <= n_steps <= max_s):
            return None

        episode_text, action_strings, step_costs = render_episode(world, goal, plan)
        return episode_text, action_strings, n_steps, total_cost, step_costs, world

    def _minimal_episode(self):
        rng = random.Random()
        world = ContinuousWorld(
            n_rooms=max(2, self.n_rooms_min),
            n_objects=1,
            space_size=self.space_size,
            connect_radius=self.space_size,  # connect everything
            rng=rng,
        )

        for obj_name, obj in world.objects.items():
            for neighbor in world.graph.get(world.agent_room, []):
                if world.room_surfaces[neighbor]:
                    goal = {
                        "obj": obj_name,
                        "clean": obj["clean"],
                        "temp": obj["temp"],
                        "room": neighbor,
                        "surface": world.room_surfaces[neighbor][0],
                    }
                    world.agent_room = obj["room"]
                    solver = DijkstraSolver(world, goal)
                    solution = solver.solve()
                    if solution:
                        plan, cost = solution
                        text, acts, step_costs = render_episode(world, goal, plan)
                        return text, acts, len(plan), cost, step_costs, world

        # Absolute fallback — create a minimal world for graph distances
        fallback_world = ContinuousWorld(
            n_rooms=2, n_objects=1,
            space_size=self.space_size,
            connect_radius=self.space_size,
            rng=rng,
        )
        text = ("[WORLD] Room 0 (kitchen) pos(10.0,10.0): counter, sink. "
                "Connects to: room 1 (dist=20.0).\n"
                "[WORLD] Room 1 (bedroom) pos(30.0,10.0): desk. "
                "Connects to: room 0 (dist=20.0).\n"
                "[OBJECTS] mug on counter in room 0.\n"
                "[START] You are in room 0. Holding: nothing.\n"
                "[GOAL] Put the mug on the desk in room 1.\n"
                "[OBS] Room 0 (kitchen) pos(10.0,10.0). You see: counter (mug), sink. "
                "Doors: room 1 (dist=20.0). Holding: nothing.\n"
                "[ACT] take mug\n"
                "[OBS] Room 0 (kitchen) pos(10.0,10.0). You see: counter, sink. "
                "Doors: room 1 (dist=20.0). Holding: mug.\n"
                "[ACT] go 1\n"
                "[OBS] Room 1 (bedroom) pos(30.0,10.0). You see: desk. "
                "Doors: room 0 (dist=20.0). Holding: mug.\n"
                "[ACT] put mug desk\n"
                "[OBS] Room 1 (bedroom) pos(30.0,10.0). You see: desk (mug). "
                "Doors: room 0 (dist=20.0). Holding: nothing.\n")
        return text, ["take mug", "go 1", "put mug desk"], 3, 20.0, [0.0, 20.0, 0.0], fallback_world

    def _tokenize_episode(self, text: str) -> Tuple[List[int], List[int], int, List[Tuple[int, int, str]], Dict[int, int]]:
        """Tokenize episode. Supervise only action tokens.

        Returns:
            (input_ids, labels, context_end_pos, action_spans, room_token_positions)
        """
        lines = text.split("\n")
        all_ids = []
        all_labels = []
        context_end_pos = 0
        action_spans = []
        room_token_positions: Dict[int, int] = {}

        for line in lines:
            if not line:
                continue

            line_ids = self.tokenizer.encode(line + "\n", add_special_tokens=False)
            offset = len(all_ids)

            # Track room token positions from [WORLD] lines
            if line.startswith("[WORLD] Room "):
                try:
                    room_num = int(line.split()[2])
                    room_token_positions[room_num] = offset
                except (IndexError, ValueError):
                    pass

            if line.startswith("[ACT]"):
                prefix_ids = self.tokenizer.encode("[ACT] ", add_special_tokens=False)
                prefix_len = len(prefix_ids)

                sup_start = offset + prefix_len
                sup_end = offset + len(line_ids)
                action_text = line.split("] ", 1)[1] if "] " in line else ""
                action_type = "nav" if action_text.startswith("go") else "manip"
                action_spans.append((sup_start, sup_end, action_type))

                all_ids.extend(line_ids)
                labels = [-100] * prefix_len + line_ids[prefix_len:]
                all_labels.extend(labels)
            else:
                all_ids.extend(line_ids)
                all_labels.extend([-100] * len(line_ids))

                if line.startswith(("[WORLD]", "[OBJECTS]", "[START]")):
                    context_end_pos = len(all_ids)

        return all_ids, all_labels, context_end_pos, action_spans, room_token_positions


# --- Self-test ---

if __name__ == "__main__":
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    print("=== Continuous Grid World Self-Test ===\n")

    # Test world generation
    rng = random.Random(42)
    world = ContinuousWorld(n_rooms=10, n_objects=4, space_size=100.0,
                            connect_radius=30.0, rng=rng)
    print(f"Rooms: {world.n_rooms}")
    print(f"Coords: {[(f'{x:.1f}',f'{y:.1f}') for x,y in world.coords]}")
    print(f"Graph edges: {sum(len(v) for v in world.graph.values()) // 2}")
    print(f"Objects: {list(world.objects.keys())}")
    print(f"Agent in room {world.agent_room}")
    print(f"Observation: {world.get_observation()}\n")

    # Test Dijkstra solver
    goal = generate_goal(world, random.Random(123), min_state_changes=1)
    print(f"Goal: {goal}")
    if goal:
        solver = DijkstraSolver(world, goal)
        solution = solver.solve()
        if solution:
            plan, cost = solution
            print(f"Plan ({len(plan)} steps, cost={cost:.1f}): {plan}")

            text, actions, step_costs = render_episode(world, goal, plan)
            print(f"\n--- Episode (first 500 chars) ---")
            print(text[:500])
            print("---\n")

            nav_cost = sum(c for c in step_costs if c > 0)
            print(f"Navigation cost: {nav_cost:.1f}")
            print(f"Step costs: {[f'{c:.1f}' for c in step_costs]}")

    # Test batch generation
    print("\n=== Batch Generation ===")
    task = ContinuousGridWorldTask(
        tokenizer, seq_len=1024,
        n_rooms_min=10, n_rooms_max=15,
        space_size=100.0, connect_radius=30.0,
        n_objects=4, min_steps=4, max_steps=10,
    )

    input_ids, labels, meta = task.generate_batch(4)
    print(f"input_ids: {input_ids.shape}")
    print(f"labels: {labels.shape}")
    print(f"avg_steps: {meta['avg_steps']:.1f}")
    print(f"optimal_costs: {[f'{c:.1f}' for c in meta['optimal_costs']]}")

    for i in range(4):
        n_sup = (labels[i] != -100).sum().item()
        print(f"  Example {i}: {n_sup} supervised tokens, "
              f"cost={meta['optimal_costs'][i]:.1f}")

    print("\nContinuousGridWorldTask OK")
