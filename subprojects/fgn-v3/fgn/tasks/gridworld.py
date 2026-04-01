"""Interactive Grid World Task — text-based navigation and object manipulation.

The agent navigates rooms, picks up objects, uses appliances (sink/stove/fridge)
to change object state (clean/dirty, hot/cold), and places objects to achieve
composite goals like "Put a clean hot mug on the desk in room 3."

Partial observability: the agent only sees the current room.
State-dependent actions: clean/heat/cool require specific appliances.
Novel configurations: room layouts and object placements vary per episode.

This is the first FGN task with intrinsic geometry — state proximity in the
environment's state graph differs from textual similarity, giving a learned
metric tensor something meaningful to capture.
"""

import random
from collections import deque
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import torch


# --- Constants ---

ROOM_TYPES = ["kitchen", "bathroom", "bedroom", "study", "living_room", "hallway"]

# Room type -> (surfaces, appliances)
ROOM_FIXTURES = {
    "kitchen":     (["counter"], ["sink", "stove", "fridge"]),
    "bathroom":    (["counter"], ["sink"]),
    "bedroom":     (["desk", "shelf"], []),
    "study":       (["desk", "shelf"], []),
    "living_room": (["table"], []),
    "hallway":     ([], []),
}

OBJECT_TYPES = ["mug", "plate", "bowl", "apple", "tomato", "potato", "bread", "pen", "book"]

TEMPS = ["neutral", "hot", "cold"]


# --- Environment ---

class GridWorld:
    """Text-based grid world with rooms, objects, and appliances."""

    def __init__(self, n_rooms: int = 5, n_objects: int = 4,
                 rng: Optional[random.Random] = None):
        self.n_rooms = n_rooms
        self.rng = rng or random.Random()

        # Generate world
        self.graph = self._generate_room_graph()
        self.room_types = self._assign_room_types()
        self.room_surfaces = self._compute_surfaces()
        self.room_appliances = self._compute_appliances()

        # Place objects
        self.objects = self._place_objects(n_objects)

        # Agent state
        self.agent_room = self.rng.randint(0, n_rooms - 1)
        self.holding = None  # object name or None
        self.holding_clean = False
        self.holding_temp = "neutral"

    def _generate_room_graph(self) -> Dict[int, List[int]]:
        """Generate a random connected graph via spanning tree + extra edges."""
        n = self.n_rooms
        graph = {i: [] for i in range(n)}

        # Random spanning tree (ensures connectivity)
        nodes = list(range(n))
        self.rng.shuffle(nodes)
        for i in range(1, n):
            parent = nodes[self.rng.randint(0, i - 1)]
            child = nodes[i]
            graph[parent].append(child)
            graph[child].append(parent)

        # Add 1-2 extra edges for shortcuts
        n_extra = self.rng.randint(1, min(2, n * (n - 1) // 2 - (n - 1)))
        for _ in range(n_extra):
            a, b = self.rng.sample(range(n), 2)
            if b not in graph[a]:
                graph[a].append(b)
                graph[b].append(a)

        # Sort adjacency lists for deterministic observations
        for k in graph:
            graph[k].sort()

        return graph

    def _assign_room_types(self) -> List[str]:
        """Assign types to rooms. Ensure at least one kitchen."""
        types = ["kitchen"]  # guarantee a kitchen
        remaining = [t for t in ROOM_TYPES if t != "kitchen"]
        for _ in range(self.n_rooms - 1):
            types.append(self.rng.choice(remaining))
        self.rng.shuffle(types)
        return types

    def _compute_surfaces(self) -> List[List[str]]:
        """Get surfaces for each room based on its type."""
        return [list(ROOM_FIXTURES[t][0]) for t in self.room_types]

    def _compute_appliances(self) -> List[List[str]]:
        """Get appliances for each room based on its type."""
        return [list(ROOM_FIXTURES[t][1]) for t in self.room_types]

    def _place_objects(self, n_objects: int) -> Dict[str, dict]:
        """Place objects randomly on surfaces."""
        available_types = list(OBJECT_TYPES)
        self.rng.shuffle(available_types)
        selected = available_types[:n_objects]

        objects = {}
        for obj_name in selected:
            # Find rooms with surfaces
            rooms_with_surfaces = [r for r in range(self.n_rooms)
                                   if self.room_surfaces[r]]
            if not rooms_with_surfaces:
                continue

            room = self.rng.choice(rooms_with_surfaces)
            surface = self.rng.choice(self.room_surfaces[room])
            is_clean = self.rng.choice([True, False])
            temp = self.rng.choice(TEMPS)

            objects[obj_name] = {
                "room": room,
                "surface": surface,
                "clean": is_clean,
                "temp": temp,
            }

        return objects

    def get_observation(self) -> str:
        """Text observation of current room (partial observability)."""
        r = self.agent_room
        rtype = self.room_types[r]

        # Objects on surfaces in this room
        surface_contents = {}
        for obj_name, obj in self.objects.items():
            if obj["room"] == r and obj_name != self.holding:
                surf = obj["surface"]
                desc = self._describe_object(obj_name, obj)
                surface_contents.setdefault(surf, []).append(desc)

        # Build "You see:" section
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

        # Doors
        neighbors = self.graph[r]
        doors_str = ", ".join(f"room {n}" for n in neighbors)

        # Holding
        if self.holding:
            hold_desc = self._describe_object(
                self.holding, {"clean": self.holding_clean, "temp": self.holding_temp})
            hold_str = hold_desc
        else:
            hold_str = "nothing"

        return (f"Room {r} ({rtype}). You see: {see_str}. "
                f"Doors: {doors_str}. Holding: {hold_str}.")

    def _describe_object(self, name: str, obj: dict) -> str:
        """Short description of an object with states."""
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

    def apply_action(self, action: str) -> bool:
        """Apply action, return True if valid."""
        parts = action.split()
        if not parts:
            return False

        cmd = parts[0]

        if cmd == "go" and len(parts) == 2:
            try:
                target = int(parts[1])
            except ValueError:
                return False
            if target in self.graph[self.agent_room]:
                self.agent_room = target
                return True
            return False

        elif cmd == "take" and len(parts) == 2:
            obj_name = parts[1]
            if self.holding is not None:
                return False
            if obj_name in self.objects and self.objects[obj_name]["room"] == self.agent_room:
                obj = self.objects[obj_name]
                self.holding = obj_name
                self.holding_clean = obj["clean"]
                self.holding_temp = obj["temp"]
                # Mark object as held (remove from room)
                obj["room"] = -1
                obj["surface"] = ""
                return True
            return False

        elif cmd == "put" and len(parts) == 3:
            obj_name, surface = parts[1], parts[2]
            if self.holding != obj_name:
                return False
            if surface not in self.room_surfaces[self.agent_room]:
                return False
            self.objects[obj_name]["room"] = self.agent_room
            self.objects[obj_name]["surface"] = surface
            self.objects[obj_name]["clean"] = self.holding_clean
            self.objects[obj_name]["temp"] = self.holding_temp
            self.holding = None
            return True

        elif cmd == "clean" and len(parts) == 2:
            obj_name = parts[1]
            if self.holding != obj_name:
                return False
            if "sink" not in self.room_appliances[self.agent_room]:
                return False
            self.holding_clean = True
            return True

        elif cmd == "heat" and len(parts) == 2:
            obj_name = parts[1]
            if self.holding != obj_name:
                return False
            if "stove" not in self.room_appliances[self.agent_room]:
                return False
            self.holding_temp = "hot"
            return True

        elif cmd == "cool" and len(parts) == 2:
            obj_name = parts[1]
            if self.holding != obj_name:
                return False
            if "fridge" not in self.room_appliances[self.agent_room]:
                return False
            self.holding_temp = "cold"
            return True

        return False

    def get_state(self) -> tuple:
        """Hashable state for BFS (only goal-relevant object tracked)."""
        obj_states = tuple(
            (name, obj["room"], obj["surface"], obj["clean"], obj["temp"])
            for name, obj in sorted(self.objects.items())
        )
        return (self.agent_room, self.holding, self.holding_clean,
                self.holding_temp, obj_states)

    def set_state(self, state: tuple):
        """Restore from hashable state."""
        (self.agent_room, self.holding, self.holding_clean,
         self.holding_temp, obj_states) = state
        for name, room, surface, clean, temp in obj_states:
            self.objects[name]["room"] = room
            self.objects[name]["surface"] = surface
            self.objects[name]["clean"] = clean
            self.objects[name]["temp"] = temp


# --- BFS Solver ---

class BFSSolver:
    """BFS optimal planner for single-object goals.

    Uses a compact state representation that only tracks the goal object,
    not all objects. This keeps the state space small (~1000 states)
    regardless of how many distractor objects exist.
    """

    def __init__(self, world: GridWorld, goal: dict):
        """
        goal: {"obj": "mug", "clean": True, "temp": "hot",
               "room": 3, "surface": "desk"}
        """
        self.world = world
        self.goal = goal
        self.goal_obj = goal["obj"]

    def _compact_state(self) -> tuple:
        """Compact state: only agent + goal object."""
        w = self.world
        obj = w.objects[self.goal_obj]
        if w.holding == self.goal_obj:
            return (w.agent_room, True, w.holding_clean, w.holding_temp)
        else:
            return (w.agent_room, False, obj["room"], obj["surface"],
                    obj["clean"], obj["temp"])

    def solve(self) -> Optional[List[str]]:
        """Find shortest action sequence to achieve goal."""
        initial_full = self.world.get_state()
        initial = self._compact_state()
        queue = deque([(initial_full, initial, [])])
        visited = {initial}

        while queue:
            full_state, compact, actions = queue.popleft()

            if self._is_goal(compact):
                self.world.set_state(initial_full)
                return actions

            for action in self._get_valid_actions(full_state):
                self.world.set_state(full_state)
                self.world.apply_action(action)
                next_full = self.world.get_state()
                next_compact = self._compact_state()

                if next_compact not in visited:
                    visited.add(next_compact)
                    queue.append((next_full, next_compact, actions + [action]))

        self.world.set_state(initial_full)
        return None

    def _is_goal(self, compact: tuple) -> bool:
        """Check if goal is satisfied."""
        g = self.goal
        if compact[1]:  # holding goal obj
            return False
        # compact = (agent_room, False, obj_room, obj_surface, obj_clean, obj_temp)
        _, _, obj_room, obj_surface, obj_clean, obj_temp = compact
        return (obj_room == g["room"] and
                obj_surface == g["surface"] and
                obj_clean == g["clean"] and
                obj_temp == g["temp"])

    def _get_valid_actions(self, full_state: tuple) -> List[str]:
        """Generate valid actions (only goal-relevant take/put/use)."""
        self.world.set_state(full_state)
        w = self.world
        actions = []

        # Movement
        for neighbor in w.graph[w.agent_room]:
            actions.append(f"go {neighbor}")

        if w.holding is None:
            # Only take the goal object (other objects are distractors)
            obj = w.objects[self.goal_obj]
            if obj["room"] == w.agent_room:
                actions.append(f"take {self.goal_obj}")
        elif w.holding == self.goal_obj:
            # Put on surfaces
            for surface in w.room_surfaces[w.agent_room]:
                actions.append(f"put {self.goal_obj} {surface}")

            # Appliance actions
            if "sink" in w.room_appliances[w.agent_room]:
                actions.append(f"clean {self.goal_obj}")
            if "stove" in w.room_appliances[w.agent_room]:
                actions.append(f"heat {self.goal_obj}")
            if "fridge" in w.room_appliances[w.agent_room]:
                actions.append(f"cool {self.goal_obj}")

        return actions


# --- Episode Generation ---

def generate_goal(world: GridWorld, rng: random.Random,
                  min_state_changes: int = 1) -> Optional[dict]:
    """Generate a random goal requiring state transformations.

    Returns goal dict or None if no valid goal found.
    """
    obj_names = list(world.objects.keys())
    if not obj_names:
        return None

    rng.shuffle(obj_names)

    for obj_name in obj_names:
        obj = world.objects[obj_name]

        # Determine target states
        target_clean = rng.choice([True, False])
        target_temp = rng.choice(TEMPS)

        # Count required state changes
        changes = 0
        if target_clean != obj["clean"]:
            changes += 1
        if target_temp != obj["temp"]:
            changes += 1

        if changes < min_state_changes:
            # Force at least one change
            if not obj["clean"]:
                target_clean = True
                changes += 1
            elif obj["temp"] == "neutral":
                target_temp = rng.choice(["hot", "cold"])
                changes += 1

        if changes < min_state_changes:
            continue

        # Check required appliances exist in the world
        if target_clean and not obj["clean"]:
            if not any("sink" in world.room_appliances[r]
                       for r in range(world.n_rooms)):
                continue
        if target_temp == "hot" and obj["temp"] != "hot":
            if not any("stove" in world.room_appliances[r]
                       for r in range(world.n_rooms)):
                continue
        if target_temp == "cold" and obj["temp"] != "cold":
            if not any("fridge" in world.room_appliances[r]
                       for r in range(world.n_rooms)):
                continue

        # Target location (different from current)
        possible_rooms = []
        for r in range(world.n_rooms):
            if world.room_surfaces[r] and r != obj["room"]:
                possible_rooms.append(r)

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


def render_goal_text(goal: dict) -> str:
    """Render goal as natural language."""
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


def render_world_description(world: GridWorld) -> str:
    """Render full world topology as text prefix.

    Used in v5 harder mode where the model must read the topology
    to plan, preventing memorization of fixed layouts.
    """
    lines = []

    # Room descriptions with connections
    for r in range(world.n_rooms):
        rtype = world.room_types[r]
        parts = []

        # Surfaces
        for surf in world.room_surfaces[r]:
            parts.append(surf)

        # Appliances
        for appl in world.room_appliances[r]:
            parts.append(appl)

        fixtures_str = ", ".join(parts) if parts else "empty"
        neighbors = ", ".join(f"room {n}" for n in world.graph[r])
        lines.append(f"[WORLD] Room {r} ({rtype}): {fixtures_str}. "
                      f"Connects to: {neighbors}.\n")

    # Object descriptions
    obj_parts = []
    for obj_name, obj in sorted(world.objects.items()):
        desc = world._describe_object(obj_name, obj)
        obj_parts.append(f"{desc} on {obj['surface']} in room {obj['room']}")
    if obj_parts:
        lines.append(f"[OBJECTS] {'. '.join(obj_parts)}.\n")

    # Starting position
    lines.append(f"[START] You are in room {world.agent_room}. "
                  f"Holding: nothing.\n")

    return "".join(lines)


def render_episode(world: GridWorld, goal: dict,
                   plan: List[str],
                   include_world_desc: bool = False) -> Tuple[str, List[str]]:
    """Render a full episode as text.

    Args:
        world: the grid world
        goal: goal dict
        plan: list of action strings
        include_world_desc: if True, prepend [WORLD]/[OBJECTS]/[START] prefix

    Returns (episode_text, action_strings).
    """
    lines = []

    # Optional world description prefix
    if include_world_desc:
        lines.append(render_world_description(world))

    lines.append(render_goal_text(goal))

    # Initial observation
    lines.append(f"[OBS] {world.get_observation()}\n")

    action_strings = []
    for action in plan:
        lines.append(f"[ACT] {action}\n")
        action_strings.append(action)

        valid = world.apply_action(action)
        assert valid, f"BFS produced invalid action: {action}"

        lines.append(f"[OBS] {world.get_observation()}\n")

    return "".join(lines), action_strings


# --- Task Class ---

class GridWorldTask:
    """Interactive grid world task for FGN training.

    Args:
        tokenizer: HuggingFace tokenizer
        seq_len: maximum sequence length (default: 1024)
        n_rooms: fixed number of rooms (default: 5, used if n_rooms_min not set)
        n_rooms_min: minimum rooms per episode (randomized topology)
        n_rooms_max: maximum rooms per episode (randomized topology)
        n_objects: number of objects (default: 4)
        min_steps: minimum plan length (default: 4)
        max_steps: maximum plan length (default: 7)
        min_state_changes: min object state changes required (default: 1)
        max_retries: max attempts to generate valid episode (default: 200)
        include_world_desc: include [WORLD]/[OBJECTS]/[START] prefix (default: False)
    """

    def __init__(self, tokenizer, seq_len: int = 1024,
                 n_rooms: int = 5, n_objects: int = 4,
                 min_steps: int = 4, max_steps: int = 7,
                 min_state_changes: int = 1,
                 max_retries: int = 200,
                 n_rooms_min: Optional[int] = None,
                 n_rooms_max: Optional[int] = None,
                 include_world_desc: bool = False,
                 randomize_topology: bool = False,
                 **kwargs):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.n_objects = n_objects
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.min_state_changes = min_state_changes
        self.max_retries = max_retries

        # Room count: either fixed or randomized range
        if n_rooms_min is not None and n_rooms_max is not None:
            self.n_rooms_min = n_rooms_min
            self.n_rooms_max = n_rooms_max
        else:
            self.n_rooms_min = n_rooms
            self.n_rooms_max = n_rooms
        # Keep n_rooms for backward compat metadata
        self.n_rooms = n_rooms

        # Include world description prefix (v5 harder mode)
        self.include_world_desc = include_world_desc or randomize_topology

        # Pre-tokenize markers for fast label creation
        self._act_prefix_ids = tokenizer.encode("[ACT] ", add_special_tokens=False)
        self._act_prefix_len = len(self._act_prefix_ids)

    def generate_batch(self, batch_size: int,
                       device: Optional[torch.device] = None
                       ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Generate batch of grid world episodes."""
        pad_id = self.tokenizer.eos_token_id or 0
        all_input_ids = []
        all_labels = []
        all_context_masks = []
        all_action_spans = []
        step_counts = []

        for _ in range(batch_size):
            episode_text, actions, n_steps = self._generate_valid_episode()

            input_ids, labels, context_end_pos, action_spans = self._tokenize_episode(episode_text)
            step_counts.append(n_steps)

            # Truncate or pad
            if len(input_ids) > self.seq_len:
                input_ids = input_ids[:self.seq_len]
                labels = labels[:self.seq_len]
            else:
                pad_len = self.seq_len - len(input_ids)
                input_ids += [pad_id] * pad_len
                labels += [-100] * pad_len

            # Create context mask
            context_mask_row = [False] * self.seq_len
            for i in range(min(context_end_pos, self.seq_len)):
                context_mask_row[i] = True
            all_context_masks.append(context_mask_row)
            all_action_spans.append(action_spans)

            all_input_ids.append(input_ids)
            all_labels.append(labels)

        input_ids_t = torch.tensor(all_input_ids, dtype=torch.long)
        labels_t = torch.tensor(all_labels, dtype=torch.long)

        if device is not None:
            input_ids_t = input_ids_t.to(device)
            labels_t = labels_t.to(device)

        metadata = {
            "task": "gridworld",
            "n_rooms_min": self.n_rooms_min,
            "n_rooms_max": self.n_rooms_max,
            "n_objects": self.n_objects,
            "avg_steps": sum(step_counts) / max(len(step_counts), 1),
            "context_mask": torch.tensor(all_context_masks, dtype=torch.bool),
            "action_spans": all_action_spans,
        }

        # Move context_mask to device if specified
        if device is not None:
            metadata["context_mask"] = metadata["context_mask"].to(device)

        return input_ids_t, labels_t, metadata

    def _generate_valid_episode(self) -> Tuple[str, List[str], int]:
        """Generate a valid episode, retrying up to max_retries times."""
        for _ in range(self.max_retries):
            result = self._try_generate_episode()
            if result is not None:
                return result

        # Fallback: generate with relaxed constraints
        return self._try_generate_episode(relax=True) or self._minimal_episode()

    def _try_generate_episode(self, relax: bool = False
                              ) -> Optional[Tuple[str, List[str], int]]:
        """Attempt to generate one episode."""
        rng = random.Random()  # fresh RNG per episode

        # Randomize room count within range
        n_rooms = rng.randint(self.n_rooms_min, self.n_rooms_max)

        world = GridWorld(
            n_rooms=n_rooms,
            n_objects=self.n_objects,
            rng=rng,
        )

        min_changes = 1 if relax else self.min_state_changes
        goal = generate_goal(world, rng, min_state_changes=min_changes)
        if goal is None:
            return None

        solver = BFSSolver(world, goal)
        plan = solver.solve()
        if plan is None:
            return None

        n_steps = len(plan)
        min_s = 1 if relax else self.min_steps
        max_s = self.max_steps * 2 if relax else self.max_steps

        if not (min_s <= n_steps <= max_s):
            return None

        episode_text, action_strings = render_episode(
            world, goal, plan, include_world_desc=self.include_world_desc)
        return episode_text, action_strings, n_steps

    def _minimal_episode(self) -> Tuple[str, List[str], int]:
        """Fallback: generate a trivial 1-step episode (just navigation)."""
        rng = random.Random()
        world = GridWorld(n_rooms=max(2, self.n_rooms), n_objects=1, rng=rng)

        # Simple goal: put any object on a surface in adjacent room
        for obj_name, obj in world.objects.items():
            for neighbor in world.graph[world.agent_room]:
                if world.room_surfaces[neighbor]:
                    goal = {
                        "obj": obj_name,
                        "clean": obj["clean"],
                        "temp": obj["temp"],
                        "room": neighbor,
                        "surface": world.room_surfaces[neighbor][0],
                    }
                    world.agent_room = obj["room"]
                    solver = BFSSolver(world, goal)
                    plan = solver.solve()
                    if plan:
                        text, acts = render_episode(world, goal, plan)
                        return text, acts, len(plan)

        # Absolute fallback
        text = "[GOAL] Put the mug on the desk in room 0.\n[OBS] Room 0 (kitchen). You see: counter (mug). Doors: room 1. Holding: nothing.\n[ACT] take mug\n[OBS] Room 0 (kitchen). You see: counter. Doors: room 1. Holding: mug.\n"
        return text, ["take mug"], 1

    def _tokenize_episode(self, text: str) -> Tuple[List[int], List[int], int, List[Tuple[int, int, str]]]:
        """Tokenize episode text and create labels.

        Supervise only action tokens (after '[ACT] ' prefix).

        Returns:
            all_ids: list of token IDs
            all_labels: list of labels (-100 for non-supervised tokens)
            context_end_pos: token position where world-description prefix ends
            action_spans: list of (start_pos, end_pos, action_type) tuples
        """
        lines = text.split("\n")
        all_ids = []
        all_labels = []
        context_end_pos = 0
        action_spans = []
        offset = 0  # Track current position in all_ids

        for line in lines:
            if not line:
                continue

            # Tokenize line with newline
            line_ids = self.tokenizer.encode(line + "\n", add_special_tokens=False)

            if line.startswith("[ACT]"):
                # Supervise action tokens (after "[ACT] " prefix)
                prefix_ids = self.tokenizer.encode("[ACT] ", add_special_tokens=False)
                prefix_len = len(prefix_ids)

                # Compute action span
                sup_start = offset + prefix_len
                sup_end = offset + len(line_ids)
                action_text = line.split("] ", 1)[1] if "] " in line else ""
                action_type = "nav" if action_text.startswith("go") else "manip"
                action_spans.append((sup_start, sup_end, action_type))

                # The rest are action tokens to supervise
                all_ids.extend(line_ids)
                labels = [-100] * prefix_len + line_ids[prefix_len:]
                all_labels.extend(labels)
            else:
                # Don't supervise goal/observation lines
                all_ids.extend(line_ids)
                all_labels.extend([-100] * len(line_ids))

                # Track context end position
                if line.startswith("[WORLD]") or line.startswith("[OBJECTS]") or line.startswith("[START]"):
                    context_end_pos = len(all_ids)

            # Update offset for next iteration
            offset = len(all_ids)

        return all_ids, all_labels, context_end_pos, action_spans


# --- Self-test ---

if __name__ == "__main__":
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    print("=== Grid World Task Self-Test ===\n")

    # Test environment
    rng = random.Random(42)
    world = GridWorld(n_rooms=5, n_objects=4, rng=rng)
    print(f"Rooms: {world.n_rooms}")
    print(f"Room types: {world.room_types}")
    print(f"Graph: {world.graph}")
    print(f"Objects: {world.objects}")
    print(f"Agent in room {world.agent_room}")
    print(f"Observation: {world.get_observation()}\n")

    # Test goal generation
    goal = generate_goal(world, random.Random(123), min_state_changes=1)
    print(f"Goal: {goal}")
    print(f"Goal text: {render_goal_text(goal)}")

    # Test BFS solver
    solver = BFSSolver(world, goal)
    plan = solver.solve()
    print(f"Plan ({len(plan)} steps): {plan}\n")

    if plan:
        # Render episode
        text, actions = render_episode(world, goal, plan)
        print("--- Episode ---")
        print(text)
        print("--- End ---\n")

        # Tokenize
        task = GridWorldTask(tokenizer, seq_len=1024,
                             n_rooms=5, n_objects=4,
                             min_steps=1, max_steps=20)
        ids, labels, context_end_pos, action_spans = task._tokenize_episode(text)
        n_supervised = sum(1 for l in labels if l != -100)
        print(f"Tokens: {len(ids)}, Supervised: {n_supervised}")
        print(f"Context end pos: {context_end_pos}, Action spans: {len(action_spans)}")

    # Test batch generation
    print("\n=== Batch Generation ===")
    task = GridWorldTask(tokenizer, seq_len=1024,
                         n_rooms=5, n_objects=4,
                         min_steps=4, max_steps=7,
                         min_state_changes=1)

    input_ids, labels, meta = task.generate_batch(4)
    print(f"input_ids: {input_ids.shape}")
    print(f"labels: {labels.shape}")
    print(f"metadata: {meta}")

    # Check supervised tokens per example
    for i in range(4):
        n_sup = (labels[i] != -100).sum().item()
        print(f"  Example {i}: {n_sup} supervised tokens")

    # Decode first example
    text = tokenizer.decode(input_ids[0].tolist())
    # Find first [ACT] and show context
    act_pos = text.find("[ACT]")
    if act_pos >= 0:
        print(f"\n  First action context: ...{text[max(0,act_pos-20):act_pos+40]}...")

    print("\n=== Difficulty Scaling ===")
    for n_rooms, min_s, max_s, min_ch in [(5, 4, 7, 1), (6, 8, 12, 2), (8, 13, 20, 2)]:
        successes = 0
        total_steps = 0
        for _ in range(20):
            t = GridWorldTask(tokenizer, seq_len=1024,
                              n_rooms=n_rooms, n_objects=4,
                              min_steps=min_s, max_steps=max_s,
                              min_state_changes=min_ch, max_retries=50)
            _, _, n = t._generate_valid_episode()
            successes += 1
            total_steps += n
        avg = total_steps / max(successes, 1)
        print(f"  n_rooms={n_rooms}, steps={min_s}-{max_s}, "
              f"changes≥{min_ch}: avg_steps={avg:.1f}")

    print("\nGridWorldTask OK")
