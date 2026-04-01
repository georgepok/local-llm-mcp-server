"""Wrapper connecting LiquidARC to Isaac Lab's DirectRLEnv.

Tokenizes flat observations into entity tokens, runs LiquidARC,
returns actions as flat tensors. Each environment subclass implements
its own tokenizer.
"""

import torch
from typing import Dict


class ObservationTokenizer:
    """Base class for converting flat observations to entity tokens."""

    def __init__(self, n_entities: int, state_dim: int, n_actuated: int):
        self.n_entities = n_entities
        self.state_dim = state_dim
        self.n_actuated = n_actuated

    def tokenize(self, obs: torch.Tensor) -> Dict[str, torch.Tensor]:
        raise NotImplementedError


class CartpoleTokenizer(ObservationTokenizer):
    """Tokenize Cartpole observations into 2 entity tokens.

    Cartpole obs: [cart_pos, cart_vel, pole_angle, pole_angular_vel]

    Token 0 (cart): state=[cart_pos, cart_vel, 0, 0], spatial=(cart_pos, 0, 0)
    Token 1 (pole): state=[pole_angle, pole_angular_vel, 0, 0], spatial=(cart_pos, pole_tip_y, 0)
    """

    def __init__(self):
        super().__init__(n_entities=2, state_dim=4, n_actuated=1)
        self.pole_length = 0.5

    def tokenize(self, obs: torch.Tensor) -> Dict[str, torch.Tensor]:
        B = obs.shape[0]
        device = obs.device

        # Isaac Lab Cartpole obs order: [pole_angle, pole_vel, cart_pos, cart_vel]
        pole_angle = obs[:, 0]
        pole_vel = obs[:, 1]
        cart_pos = obs[:, 2]
        cart_vel = obs[:, 3]

        state_features = torch.zeros(B, 2, 4, device=device)
        state_features[:, 0, 0] = cart_pos
        state_features[:, 0, 1] = cart_vel
        state_features[:, 1, 0] = pole_angle
        state_features[:, 1, 1] = pole_vel

        spatial_x = torch.zeros(B, 2, device=device)
        spatial_y = torch.zeros(B, 2, device=device)
        spatial_z = torch.zeros(B, 2, device=device)

        spatial_x[:, 0] = cart_pos
        spatial_x[:, 1] = cart_pos + torch.sin(pole_angle) * self.pole_length
        spatial_y[:, 0] = 0.0
        spatial_y[:, 1] = torch.cos(pole_angle) * self.pole_length

        entity_types = torch.tensor([0, 1], device=device).unsqueeze(0).expand(B, -1)
        entity_ids = torch.tensor([0, 1], device=device).unsqueeze(0).expand(B, -1)
        actuated_indices = torch.tensor([0], device=device)

        return {
            'state_features': state_features,
            'spatial_x': spatial_x,
            'spatial_y': spatial_y,
            'spatial_z': spatial_z,
            'entity_types': entity_types,
            'entity_ids': entity_ids,
            'actuated_indices': actuated_indices,
        }


class AnymalTokenizer(ObservationTokenizer):
    """Tokenize Anymal-C flat observations into 13 entity tokens.

    Anymal-C flat obs (48 floats):
      [0:3]   root_lin_vel_b
      [3:6]   root_ang_vel_b
      [6:9]   projected_gravity_b
      [9:12]  velocity commands
      [12:24] joint_pos - default (12 joints)
      [24:36] joint_vel (12 joints)
      [36:48] previous_actions (12 joints)

    Token 0: base body — lin_vel, ang_vel, gravity, commands (12 features)
    Tokens 1-12: joints — joint_pos, joint_vel, prev_action (3 features each)
    """

    def __init__(self, obs_dim: int = 48):
        super().__init__(n_entities=13, state_dim=16, n_actuated=12)
        self.obs_dim = obs_dim

    def tokenize(self, obs: torch.Tensor) -> Dict[str, torch.Tensor]:
        B = obs.shape[0]
        device = obs.device
        N = self.n_entities

        state_features = torch.zeros(B, N, self.state_dim, device=device)
        spatial_x = torch.zeros(B, N, device=device)
        spatial_y = torch.zeros(B, N, device=device)
        spatial_z = torch.zeros(B, N, device=device)

        # Base body: lin_vel(3) + ang_vel(3) + gravity(3) + commands(3) = 12 features
        state_features[:, 0, :12] = obs[:, :12]

        # Joint tokens: joint_pos, joint_vel, prev_action per joint
        for j in range(12):
            token_idx = j + 1
            state_features[:, token_idx, 0] = obs[:, 12 + j]   # joint_pos
            state_features[:, token_idx, 1] = obs[:, 24 + j]   # joint_vel
            state_features[:, token_idx, 2] = obs[:, 36 + j]   # prev_action

            # Approximate spatial positions based on kinematic structure
            # 4 legs × 3 joints: FL(0-2), FR(3-5), RL(6-8), RR(9-11)
            leg = j // 3
            leg_offsets_x = [0.3, 0.3, -0.3, -0.3]   # front/rear
            leg_offsets_y = [0.15, -0.15, 0.15, -0.15]  # left/right
            spatial_x[:, token_idx] = leg_offsets_x[leg]
            spatial_y[:, token_idx] = leg_offsets_y[leg]
            spatial_z[:, token_idx] = -0.1 * ((j % 3) + 1)  # lower for distal joints

        entity_types = torch.zeros(B, N, dtype=torch.long, device=device)
        entity_types[:, 1:] = 1  # joints
        entity_ids = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)
        actuated_indices = torch.arange(1, 13, device=device)

        return {
            'state_features': state_features,
            'spatial_x': spatial_x,
            'spatial_y': spatial_y,
            'spatial_z': spatial_z,
            'entity_types': entity_types,
            'entity_ids': entity_ids,
            'actuated_indices': actuated_indices,
        }


class AnymalRoughTokenizer(ObservationTokenizer):
    """Tokenize Anymal-C rough terrain observations into 14 entity tokens.

    Anymal-C rough obs (235 floats):
      [0:48]   same as flat (base + 12 joints)
      [48:235] height scanner: 187 terrain height samples around robot

    Tokens 0-12: same as flat tokenizer (base + 12 joints)
    Token 13: terrain summary — compresses height scanner into state features

    The height map is a 16x10 grid (resolution=0.1m, size=[1.6, 1.0]).
    We compress it into statistics that fit the state_dim: mean height,
    std, front/rear/left/right slope, min, max, roughness.
    """

    def __init__(self, obs_dim: int = 235):
        super().__init__(n_entities=14, state_dim=16, n_actuated=12)
        self.obs_dim = obs_dim
        self.height_start = 48
        self.height_dim = 187
        # Height grid is ~16x10 but may include padding
        self.grid_cols = 16  # along x (forward/backward)
        self.grid_rows = 10  # along y (left/right)
        # Flat tokenizer for base + joints
        self._flat = AnymalTokenizer(obs_dim=48)

    def tokenize(self, obs: torch.Tensor) -> Dict[str, torch.Tensor]:
        B = obs.shape[0]
        device = obs.device

        # Tokenize base + joints from first 48 dims
        flat_obs = obs[:, :48]
        result = self._flat.tokenize(flat_obs)

        # Expand all tensors to 14 entities
        N = 14
        old_N = 13

        state_features = torch.zeros(B, N, self.state_dim, device=device)
        state_features[:, :old_N, :] = result['state_features']

        spatial_x = torch.zeros(B, N, device=device)
        spatial_y = torch.zeros(B, N, device=device)
        spatial_z = torch.zeros(B, N, device=device)
        spatial_x[:, :old_N] = result['spatial_x']
        spatial_y[:, :old_N] = result['spatial_y']
        spatial_z[:, :old_N] = result['spatial_z']

        # Terrain token: compress height scanner into meaningful features
        height_data = obs[:, self.height_start:self.height_start + self.height_dim]  # [B, 187]

        # Statistics over the full height map
        h_mean = height_data.mean(dim=-1)
        h_std = height_data.std(dim=-1)
        h_min = height_data.min(dim=-1).values
        h_max = height_data.max(dim=-1).values

        # Directional slopes: split into quadrants for front/rear/left/right
        n_pts = height_data.shape[1]
        half = n_pts // 2
        quarter = n_pts // 4

        front_mean = height_data[:, :half].mean(dim=-1)
        rear_mean = height_data[:, half:].mean(dim=-1)
        left_mean = height_data[:, :quarter].mean(dim=-1)
        right_mean = height_data[:, quarter:half].mean(dim=-1)

        # Slope estimates
        front_rear_slope = front_mean - rear_mean
        left_right_slope = left_mean - right_mean

        # Roughness: mean absolute deviation
        roughness = (height_data - h_mean.unsqueeze(-1)).abs().mean(dim=-1)

        # Nearby terrain (close to robot) vs far terrain
        nearby = height_data[:, :quarter].mean(dim=-1)

        # Pack into state features for terrain token
        state_features[:, 13, 0] = h_mean
        state_features[:, 13, 1] = h_std
        state_features[:, 13, 2] = h_min
        state_features[:, 13, 3] = h_max
        state_features[:, 13, 4] = front_rear_slope
        state_features[:, 13, 5] = left_right_slope
        state_features[:, 13, 6] = roughness
        state_features[:, 13, 7] = nearby

        # Terrain token sits at robot center, ground level
        spatial_x[:, 13] = 0.0
        spatial_y[:, 13] = 0.0
        spatial_z[:, 13] = h_mean

        entity_types = torch.zeros(B, N, dtype=torch.long, device=device)
        entity_types[:, 1:13] = 1  # joints
        entity_types[:, 13] = 2    # terrain
        entity_ids = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)
        actuated_indices = torch.arange(1, 13, device=device)

        return {
            'state_features': state_features,
            'spatial_x': spatial_x,
            'spatial_y': spatial_y,
            'spatial_z': spatial_z,
            'entity_types': entity_types,
            'entity_ids': entity_ids,
            'actuated_indices': actuated_indices,
        }


class HumanoidTokenizer(ObservationTokenizer):
    """Tokenize Isaac-Humanoid-Direct-v0 observations into 22 entity tokens.

    Humanoid obs (75 floats):
      [0:13]  root state (pos 3, quat 4, lin_vel 3, ang_vel 3)
      [13:34] joint_pos (21 joints)
      [34:55] joint_vel (21 joints)
      [55:75] remaining (prev actions or sensor data)

    Token 0: body (root state, 13 features)
    Tokens 1-21: joints (joint_pos, joint_vel, extra = 3 features each)
    """

    def __init__(self, obs_dim: int = 75):
        super().__init__(n_entities=22, state_dim=16, n_actuated=21)
        self.obs_dim = obs_dim

    def tokenize(self, obs: torch.Tensor) -> Dict[str, torch.Tensor]:
        B = obs.shape[0]
        device = obs.device
        N = self.n_entities

        state_features = torch.zeros(B, N, self.state_dim, device=device)
        spatial_x = torch.zeros(B, N, device=device)
        spatial_y = torch.zeros(B, N, device=device)
        spatial_z = torch.zeros(B, N, device=device)

        # Body: root state (13 features)
        state_features[:, 0, :min(13, self.state_dim)] = obs[:, :13]

        # Joint tokens
        for j in range(21):
            idx = j + 1
            if 13 + j < obs.shape[1]:
                state_features[:, idx, 0] = obs[:, 13 + j]
            if 34 + j < obs.shape[1]:
                state_features[:, idx, 1] = obs[:, 34 + j]
            if 55 + j < obs.shape[1]:
                state_features[:, idx, 2] = obs[:, 55 + j]

            # Approximate spatial layout
            if j < 6:  # legs
                side = 0.1 if j % 2 == 0 else -0.1
                spatial_y[:, idx] = side
                spatial_z[:, idx] = -0.3 * ((j // 2) + 1)
            elif j < 9:  # torso
                spatial_z[:, idx] = 0.2 * (j - 5)
            else:  # arms
                side = 0.3 if j % 2 == 0 else -0.3
                spatial_y[:, idx] = side
                spatial_z[:, idx] = 0.4 - 0.15 * ((j - 9) // 2)

        entity_types = torch.zeros(B, N, dtype=torch.long, device=device)
        entity_types[:, 1:] = 1
        entity_ids = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)
        actuated_indices = torch.arange(1, 22, device=device)

        return {
            'state_features': state_features,
            'spatial_x': spatial_x,
            'spatial_y': spatial_y,
            'spatial_z': spatial_z,
            'entity_types': entity_types,
            'entity_ids': entity_ids,
            'actuated_indices': actuated_indices,
        }
