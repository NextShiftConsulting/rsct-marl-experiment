"""
Warehouse Robot Coordination Simulation

Simulates Amazon Kiva-style warehouse operations with:
- Multiple robots picking/delivering items
- Shelf aisles and intersections
- Order queue management
- Charging stations
- 24/7 continuous operation

Based on Amazon fulfillment center operations
Cost model: Robot damage + order delays + throughput loss
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any, Set
from enum import Enum

from ..environment.gridworld import Action


class RobotState(Enum):
    IDLE = "idle"
    PICKING = "picking"
    DELIVERING = "delivering"
    CHARGING = "charging"
    DAMAGED = "damaged"


class TaskType(Enum):
    PICK = "pick"
    DELIVER = "deliver"
    RETURN = "return"


@dataclass
class WarehouseConfig:
    """Configuration for warehouse simulation."""
    warehouse_size: int = 40  # 40x40 grid
    num_robots: int = 12
    num_shelves: int = 100
    num_stations: int = 4  # Picking/packing stations
    max_steps: int = 500
    robot_cost: int = 30000  # USD per robot
    order_value: int = 50  # USD per order
    downtime_cost: int = 1000  # USD per hour of robot downtime


@dataclass
class WarehouseStepResult:
    """Result of warehouse step."""
    positions: List[Tuple[int, int]]
    robot_states: List[RobotState]
    rewards: List[float]
    done: bool
    info: Dict[str, Any]


class WarehouseEnv:
    """
    Warehouse robot coordination environment.

    Robots must:
    1. Pick items from shelves
    2. Deliver to packing stations
    3. Navigate without collisions
    4. Maximize throughput
    5. Manage battery/charging
    """

    def __init__(self, config: Optional[WarehouseConfig] = None):
        self.config = config or WarehouseConfig()
        self.grid_size = self.config.warehouse_size

        # Robots
        self.positions: List[Tuple[int, int]] = []
        self.goals: List[Tuple[int, int]] = []
        self.robot_states: List[RobotState] = []
        self.carrying_item: List[bool] = []
        self.battery_levels: List[int] = []

        # Warehouse layout
        self.shelves: Set[Tuple[int, int]] = set()
        self.stations: List[Tuple[int, int]] = []
        self.charging_stations: List[Tuple[int, int]] = []
        self.aisles: Set[Tuple[int, int]] = set()
        self.obstacles: Set[Tuple[int, int]] = set()

        # Metrics
        self.total_collisions = 0
        self.robots_damaged = 0
        self.orders_completed = 0
        self.step_count = 0

    def _generate_warehouse_layout(self):
        """Generate realistic warehouse layout with aisles and shelves."""
        self.shelves = set()
        self.aisles = set()
        self.obstacles = set()

        # Create shelf blocks with aisles between
        shelf_block_width = 4
        aisle_width = 2
        margin = 5

        for block_x in range(margin, self.grid_size - margin, shelf_block_width + aisle_width):
            for block_y in range(margin, self.grid_size - margin, shelf_block_width + aisle_width):
                # Create shelf block
                for dx in range(shelf_block_width):
                    for dy in range(shelf_block_width):
                        x = block_x + dx
                        y = block_y + dy
                        if x < self.grid_size - margin and y < self.grid_size - margin:
                            if dx in [0, shelf_block_width - 1] or dy in [0, shelf_block_width - 1]:
                                self.shelves.add((x, y))
                            else:
                                self.obstacles.add((x, y))  # Center of block is inaccessible

        # Packing stations at edges
        self.stations = [
            (self.grid_size // 2, 1),
            (self.grid_size // 2, self.grid_size - 2),
            (1, self.grid_size // 2),
            (self.grid_size - 2, self.grid_size // 2),
        ]

        # Charging stations in corners
        self.charging_stations = [
            (2, 2),
            (2, self.grid_size - 3),
            (self.grid_size - 3, 2),
            (self.grid_size - 3, self.grid_size - 3),
        ]

    def reset(self) -> List[Tuple[int, int]]:
        """Reset warehouse."""
        self._generate_warehouse_layout()

        # Position robots near stations
        self.positions = []
        for i in range(self.config.num_robots):
            station_idx = i % len(self.stations)
            base = self.stations[station_idx]
            offset_x = (i // len(self.stations)) % 3
            offset_y = (i // len(self.stations)) // 3
            pos = (base[0] + offset_x + 1, base[1] + offset_y + 1)
            # Make sure not on shelf
            while pos in self.shelves or pos in self.obstacles:
                pos = (pos[0] + 1, pos[1])
            self.positions.append(pos)

        # Assign initial goals (pick items)
        self.goals = []
        shelf_list = list(self.shelves)
        for i in range(self.config.num_robots):
            target = shelf_list[np.random.randint(len(shelf_list))]
            self.goals.append(target)

        # Reset states
        self.robot_states = [RobotState.PICKING] * self.config.num_robots
        self.carrying_item = [False] * self.config.num_robots
        self.battery_levels = [100] * self.config.num_robots

        # Reset metrics
        self.total_collisions = 0
        self.robots_damaged = 0
        self.orders_completed = 0
        self.step_count = 0

        return self.positions.copy()

    def get_state(self, robot_id: int) -> Tuple:
        """Get state for a specific robot."""
        own_pos = self.positions[robot_id]
        goal = self.goals[robot_id]
        state = self.robot_states[robot_id]
        carrying = self.carrying_item[robot_id]
        battery = self.battery_levels[robot_id]

        # Nearby robots
        nearby = []
        for i, pos in enumerate(self.positions):
            if i != robot_id:
                dist = abs(pos[0] - own_pos[0]) + abs(pos[1] - own_pos[1])
                if dist <= 5:
                    nearby.append(pos)

        return (own_pos, goal, state, carrying, battery, tuple(nearby[:4]))

    def step(self, actions: List[Action]) -> WarehouseStepResult:
        """Execute robot movements."""
        self.step_count += 1

        prev_positions = self.positions.copy()
        new_positions = []

        # Apply actions
        for i, (pos, action) in enumerate(zip(self.positions, actions)):
            if self.robot_states[i] == RobotState.DAMAGED:
                new_positions.append(pos)
                continue

            dr, dc = 0, 0
            if action == Action.UP:
                dr = -1
            elif action == Action.DOWN:
                dr = 1
            elif action == Action.LEFT:
                dc = -1
            elif action == Action.RIGHT:
                dc = 1

            new_r = max(0, min(self.grid_size - 1, pos[0] + dr))
            new_c = max(0, min(self.grid_size - 1, pos[1] + dc))

            # Check for obstacles (center of shelf blocks)
            if (new_r, new_c) in self.obstacles:
                new_positions.append(pos)
            else:
                new_positions.append((new_r, new_c))

            # Drain battery
            if action != Action.STAY:
                self.battery_levels[i] = max(0, self.battery_levels[i] - 1)

        # Check collisions
        collision_occurred = False
        colliding_robots = set()

        # Same-cell
        for i in range(len(new_positions)):
            if self.robot_states[i] == RobotState.DAMAGED:
                continue
            for j in range(i + 1, len(new_positions)):
                if self.robot_states[j] == RobotState.DAMAGED:
                    continue
                if new_positions[i] == new_positions[j]:
                    collision_occurred = True
                    colliding_robots.add(i)
                    colliding_robots.add(j)

        # Swap
        for i in range(len(new_positions)):
            if self.robot_states[i] == RobotState.DAMAGED:
                continue
            for j in range(i + 1, len(new_positions)):
                if self.robot_states[j] == RobotState.DAMAGED:
                    continue
                if (new_positions[i] == prev_positions[j] and
                    new_positions[j] == prev_positions[i]):
                    collision_occurred = True
                    colliding_robots.add(i)
                    colliding_robots.add(j)

        # Update positions
        self.positions = new_positions

        # Handle collisions
        if collision_occurred:
            self.total_collisions += 1
            for r_id in colliding_robots:
                if self.robot_states[r_id] != RobotState.DAMAGED:
                    self.robot_states[r_id] = RobotState.DAMAGED
                    self.robots_damaged += 1
                    if self.carrying_item[r_id]:
                        self.carrying_item[r_id] = False  # Item dropped

        # Task completion checks
        for i, pos in enumerate(self.positions):
            if self.robot_states[i] == RobotState.DAMAGED:
                continue

            # At shelf - pick item
            if pos in self.shelves and not self.carrying_item[i]:
                if self.robot_states[i] == RobotState.PICKING:
                    self.carrying_item[i] = True
                    self.robot_states[i] = RobotState.DELIVERING
                    # New goal: nearest station
                    nearest_station = min(self.stations,
                                         key=lambda s: abs(s[0]-pos[0]) + abs(s[1]-pos[1]))
                    self.goals[i] = nearest_station

            # At station - deliver item
            if pos in self.stations and self.carrying_item[i]:
                self.carrying_item[i] = False
                self.orders_completed += 1
                self.robot_states[i] = RobotState.PICKING
                # New goal: random shelf
                shelf_list = list(self.shelves)
                self.goals[i] = shelf_list[np.random.randint(len(shelf_list))]

            # Low battery - go charge
            if self.battery_levels[i] < 20 and self.robot_states[i] != RobotState.CHARGING:
                self.robot_states[i] = RobotState.CHARGING
                nearest_charger = min(self.charging_stations,
                                     key=lambda c: abs(c[0]-pos[0]) + abs(c[1]-pos[1]))
                self.goals[i] = nearest_charger

            # At charger - charge
            if pos in self.charging_stations:
                self.battery_levels[i] = min(100, self.battery_levels[i] + 10)
                if self.battery_levels[i] >= 80:
                    self.robot_states[i] = RobotState.PICKING
                    shelf_list = list(self.shelves)
                    self.goals[i] = shelf_list[np.random.randint(len(shelf_list))]

        # Compute rewards
        rewards = []
        for i in range(self.config.num_robots):
            if self.robot_states[i] == RobotState.DAMAGED:
                rewards.append(-self.config.robot_cost / 100)  # Spread damage cost
            elif i in colliding_robots:
                rewards.append(-1000)
            else:
                # Distance to goal progress
                goal = self.goals[i]
                dist = abs(self.positions[i][0] - goal[0]) + abs(self.positions[i][1] - goal[1])
                rewards.append(max(0, 10 - dist))

        # Done check
        active_robots = sum(1 for s in self.robot_states if s != RobotState.DAMAGED)
        done = active_robots == 0 or self.step_count >= self.config.max_steps

        return WarehouseStepResult(
            positions=self.positions.copy(),
            robot_states=self.robot_states.copy(),
            rewards=rewards,
            done=done,
            info={
                "collision": collision_occurred,
                "colliding_robots": list(colliding_robots),
                "total_collisions": self.total_collisions,
                "robots_damaged": self.robots_damaged,
                "orders_completed": self.orders_completed,
                "active_robots": active_robots,
                "step": self.step_count,
                "throughput": self.orders_completed / max(1, self.step_count) * 100,
                "financial_loss": self.robots_damaged * self.config.robot_cost,
            }
        )

    def get_metrics(self) -> Dict[str, Any]:
        """Get final metrics."""
        active = sum(1 for s in self.robot_states if s != RobotState.DAMAGED)
        return {
            "total_collisions": self.total_collisions,
            "robots_damaged": self.robots_damaged,
            "active_robots": active,
            "orders_completed": self.orders_completed,
            "throughput_per_100_steps": self.orders_completed / max(1, self.step_count) * 100,
            "financial_loss": self.robots_damaged * self.config.robot_cost,
            "steps": self.step_count,
        }
