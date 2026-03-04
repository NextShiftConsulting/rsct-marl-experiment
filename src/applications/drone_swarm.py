"""
Drone Swarm Coordination Simulation

Simulates UAV swarm operations with:
- 3D airspace (discretized to grid layers)
- Wind disturbances
- Communication range limits
- Battery constraints
- Target tracking missions

Cost model: $50,000 per drone collision
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum

from ..environment.gridworld import Action


class DroneState(Enum):
    ACTIVE = "active"
    LOW_BATTERY = "low_battery"
    CRASHED = "crashed"
    RTB = "returning_to_base"  # Return to base


@dataclass
class DroneSwarmConfig:
    """Configuration for drone swarm simulation."""
    airspace_size: int = 50  # 50x50 grid
    num_drones: int = 8
    num_targets: int = 4
    max_steps: int = 200
    comm_range: int = 15  # Communication range in cells
    battery_capacity: int = 150  # Steps before RTB
    wind_strength: float = 0.1  # Probability of drift
    drone_cost: int = 50000  # USD per drone
    mission_value: int = 100000  # USD per target tracked


@dataclass
class SwarmStepResult:
    """Result of a simulation step."""
    positions: List[Tuple[int, int]]
    states: List[DroneState]
    rewards: List[float]
    done: bool
    info: Dict[str, Any]


class DroneSwarmEnv:
    """
    Drone swarm environment for target tracking missions.

    Drones must coordinate to:
    1. Track multiple moving targets
    2. Maintain communication network
    3. Avoid mid-air collisions
    4. Manage battery/RTB
    """

    def __init__(self, config: Optional[DroneSwarmConfig] = None):
        self.config = config or DroneSwarmConfig()
        self.grid_size = self.config.airspace_size
        self.num_drones = self.config.num_drones

        # State
        self.positions: List[Tuple[int, int]] = []
        self.goals: List[Tuple[int, int]] = []  # Current assigned targets
        self.target_positions: List[Tuple[int, int]] = []
        self.drone_states: List[DroneState] = []
        self.battery_levels: List[int] = []
        self.base_position = (0, 0)

        # Metrics
        self.total_collisions = 0
        self.total_targets_tracked = 0
        self.drones_lost = 0
        self.step_count = 0

        # Obstacles (no-fly zones)
        self.obstacles = set()

    def reset(self) -> List[Tuple[int, int]]:
        """Reset the environment."""
        # Spread drones from base in formation
        self.positions = []
        for i in range(self.num_drones):
            row = i // 4
            col = i % 4
            self.positions.append((row * 2, col * 2))

        # Random target positions
        self.target_positions = []
        for _ in range(self.config.num_targets):
            while True:
                pos = (
                    np.random.randint(20, self.grid_size - 5),
                    np.random.randint(20, self.grid_size - 5)
                )
                if pos not in self.target_positions:
                    self.target_positions.append(pos)
                    break

        # Assign targets to drones
        self.goals = []
        for i in range(self.num_drones):
            target_idx = i % len(self.target_positions)
            self.goals.append(self.target_positions[target_idx])

        # Reset states
        self.drone_states = [DroneState.ACTIVE] * self.num_drones
        self.battery_levels = [self.config.battery_capacity] * self.num_drones

        # Reset metrics
        self.total_collisions = 0
        self.total_targets_tracked = 0
        self.drones_lost = 0
        self.step_count = 0

        return self.positions.copy()

    def get_state(self, drone_id: int) -> Tuple:
        """Get state for a specific drone."""
        own_pos = self.positions[drone_id]
        goal = self.goals[drone_id]
        battery = self.battery_levels[drone_id]

        # Nearby drones (within comm range)
        nearby = []
        for i, pos in enumerate(self.positions):
            if i != drone_id:
                dist = abs(pos[0] - own_pos[0]) + abs(pos[1] - own_pos[1])
                if dist <= self.config.comm_range:
                    nearby.append(pos)

        return (own_pos, goal, battery, tuple(nearby[:4]))  # Max 4 nearby

    def step(self, actions: List[Action]) -> SwarmStepResult:
        """Execute actions for all drones."""
        self.step_count += 1

        # Store previous positions for swap detection
        prev_positions = self.positions.copy()

        # Apply actions with wind disturbance
        new_positions = []
        for i, (pos, action) in enumerate(zip(self.positions, actions)):
            if self.drone_states[i] == DroneState.CRASHED:
                new_positions.append(pos)
                continue

            # Compute intended position
            dr, dc = 0, 0
            if action == Action.UP:
                dr = -1
            elif action == Action.DOWN:
                dr = 1
            elif action == Action.LEFT:
                dc = -1
            elif action == Action.RIGHT:
                dc = 1

            # Wind disturbance
            if np.random.random() < self.config.wind_strength:
                wind_dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
                wind_dir = wind_dirs[np.random.randint(4)]
                dr += wind_dir[0]
                dc += wind_dir[1]

            new_r = max(0, min(self.grid_size - 1, pos[0] + dr))
            new_c = max(0, min(self.grid_size - 1, pos[1] + dc))
            new_positions.append((new_r, new_c))

            # Drain battery
            if action != Action.STAY:
                self.battery_levels[i] -= 1

        # Check collisions
        collision_occurred = False
        colliding_drones = set()

        # Same-cell collisions
        for i in range(len(new_positions)):
            for j in range(i + 1, len(new_positions)):
                if self.drone_states[i] == DroneState.CRASHED:
                    continue
                if self.drone_states[j] == DroneState.CRASHED:
                    continue
                if new_positions[i] == new_positions[j]:
                    collision_occurred = True
                    colliding_drones.add(i)
                    colliding_drones.add(j)

        # Swap collisions
        for i in range(len(new_positions)):
            for j in range(i + 1, len(new_positions)):
                if self.drone_states[i] == DroneState.CRASHED:
                    continue
                if self.drone_states[j] == DroneState.CRASHED:
                    continue
                if (new_positions[i] == prev_positions[j] and
                    new_positions[j] == prev_positions[i]):
                    collision_occurred = True
                    colliding_drones.add(i)
                    colliding_drones.add(j)

        # Update positions
        self.positions = new_positions

        # Handle collisions - drones crash
        if collision_occurred:
            self.total_collisions += 1
            for drone_id in colliding_drones:
                if self.drone_states[drone_id] != DroneState.CRASHED:
                    self.drone_states[drone_id] = DroneState.CRASHED
                    self.drones_lost += 1

        # Check battery levels
        for i, battery in enumerate(self.battery_levels):
            if battery <= 0 and self.drone_states[i] == DroneState.ACTIVE:
                self.drone_states[i] = DroneState.CRASHED  # Out of battery = crash
                self.drones_lost += 1
            elif battery < 30 and self.drone_states[i] == DroneState.ACTIVE:
                self.drone_states[i] = DroneState.LOW_BATTERY

        # Move targets (random walk)
        new_targets = []
        for pos in self.target_positions:
            if np.random.random() < 0.3:  # 30% chance to move
                dr = np.random.randint(-1, 2)
                dc = np.random.randint(-1, 2)
                new_r = max(5, min(self.grid_size - 5, pos[0] + dr))
                new_c = max(5, min(self.grid_size - 5, pos[1] + dc))
                new_targets.append((new_r, new_c))
            else:
                new_targets.append(pos)
        self.target_positions = new_targets

        # Check target tracking (drone within 3 cells of target)
        targets_tracked = set()
        for i, pos in enumerate(self.positions):
            if self.drone_states[i] == DroneState.CRASHED:
                continue
            for j, target in enumerate(self.target_positions):
                dist = abs(pos[0] - target[0]) + abs(pos[1] - target[1])
                if dist <= 3:
                    targets_tracked.add(j)

        self.total_targets_tracked += len(targets_tracked)

        # Compute rewards
        rewards = []
        for i in range(self.num_drones):
            if self.drone_states[i] == DroneState.CRASHED:
                rewards.append(-self.config.drone_cost)
            elif i in colliding_drones:
                rewards.append(-self.config.drone_cost)
            else:
                # Reward for tracking targets
                target = self.goals[i]
                dist = abs(self.positions[i][0] - target[0]) + abs(self.positions[i][1] - target[1])
                if dist <= 3:
                    rewards.append(100)  # Tracking reward
                else:
                    rewards.append(-dist)  # Distance penalty

        # Check if done
        active_drones = sum(1 for s in self.drone_states if s != DroneState.CRASHED)
        done = (active_drones == 0 or self.step_count >= self.config.max_steps)

        return SwarmStepResult(
            positions=self.positions.copy(),
            states=self.drone_states.copy(),
            rewards=rewards,
            done=done,
            info={
                "collision": collision_occurred,
                "colliding_drones": list(colliding_drones),
                "targets_tracked": len(targets_tracked),
                "active_drones": active_drones,
                "drones_lost": self.drones_lost,
                "total_collisions": self.total_collisions,
                "step": self.step_count,
                "financial_loss": self.drones_lost * self.config.drone_cost,
            }
        )

    def get_metrics(self) -> Dict[str, Any]:
        """Get final metrics."""
        active = sum(1 for s in self.drone_states if s != DroneState.CRASHED)
        return {
            "total_collisions": self.total_collisions,
            "drones_lost": self.drones_lost,
            "active_drones": active,
            "survival_rate": active / self.num_drones,
            "targets_tracked": self.total_targets_tracked,
            "financial_loss": self.drones_lost * self.config.drone_cost,
            "steps": self.step_count,
        }

    def render(self) -> str:
        """Render the environment as ASCII."""
        grid = [['.' for _ in range(self.grid_size)] for _ in range(self.grid_size)]

        # Mark targets
        for i, pos in enumerate(self.target_positions):
            grid[pos[0]][pos[1]] = 'T'

        # Mark drones
        drone_chars = '12345678'
        for i, pos in enumerate(self.positions):
            if self.drone_states[i] == DroneState.CRASHED:
                grid[pos[0]][pos[1]] = 'X'
            else:
                grid[pos[0]][pos[1]] = drone_chars[i % len(drone_chars)]

        return '\n'.join([''.join(row) for row in grid])
