"""
Military Autonomous Convoy Simulation

Simulates leader-follower convoy operations with:
- Multiple vehicle types (lead, cargo, security)
- Hostile threat zones
- Terrain obstacles
- Communication blackouts
- IED detection/avoidance

Based on US Army AMAS (Autonomous Mobility Applique System)
Cost model: Mission failure + vehicle loss + personnel risk
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any, Set
from enum import Enum

from ..environment.gridworld import Action


class VehicleType(Enum):
    LEAD = "lead"
    CARGO = "cargo"
    SECURITY = "security"
    MEDEVAC = "medevac"


class VehicleState(Enum):
    OPERATIONAL = "operational"
    DAMAGED = "damaged"
    DISABLED = "disabled"
    DESTROYED = "destroyed"


class ThreatType(Enum):
    IED = "ied"
    AMBUSH = "ambush"
    BLOCKED = "blocked"


@dataclass
class ConvoyConfig:
    """Configuration for convoy simulation."""
    route_length: int = 100  # Grid cells
    route_width: int = 20
    num_vehicles: int = 6
    num_threats: int = 8
    max_steps: int = 300
    vehicle_spacing: int = 3  # Minimum spacing
    vehicle_cost: int = 500000  # USD per vehicle
    cargo_value: int = 2000000  # USD cargo value
    mission_success_bonus: int = 5000000


@dataclass
class ConvoyStepResult:
    """Result of convoy step."""
    positions: List[Tuple[int, int]]
    vehicle_states: List[VehicleState]
    rewards: List[float]
    done: bool
    info: Dict[str, Any]


class MilitaryConvoyEnv:
    """
    Military convoy environment simulating AMAS-style operations.

    Convoy must:
    1. Navigate from start to destination
    2. Maintain formation and spacing
    3. Avoid collisions between vehicles
    4. Detect and avoid threats (IEDs, ambushes)
    5. Complete mission with minimal losses
    """

    def __init__(self, config: Optional[ConvoyConfig] = None):
        self.config = config or ConvoyConfig()
        self.grid_height = self.config.route_length
        self.grid_width = self.config.route_width
        self.grid_size = max(self.grid_height, self.grid_width)

        # Vehicles
        self.positions: List[Tuple[int, int]] = []
        self.goals: List[Tuple[int, int]] = []
        self.vehicle_types: List[VehicleType] = []
        self.vehicle_states: List[VehicleState] = []

        # Threats and obstacles
        self.threats: Dict[Tuple[int, int], ThreatType] = {}
        self.obstacles: Set[Tuple[int, int]] = set()
        self.detected_threats: Set[Tuple[int, int]] = set()

        # Metrics
        self.total_collisions = 0
        self.vehicles_lost = 0
        self.threats_detected = 0
        self.step_count = 0
        self.mission_complete = False

    def reset(self) -> List[Tuple[int, int]]:
        """Reset convoy to starting position."""
        # Create convoy formation at start
        self.positions = []
        self.vehicle_types = []
        center_col = self.grid_width // 2

        vehicle_order = [
            VehicleType.SECURITY,  # Point vehicle
            VehicleType.LEAD,
            VehicleType.CARGO,
            VehicleType.CARGO,
            VehicleType.CARGO,
            VehicleType.SECURITY,  # Rear security
        ]

        for i in range(self.config.num_vehicles):
            row = i * self.config.vehicle_spacing
            col = center_col + (i % 2)  # Slight offset
            self.positions.append((row, col))
            self.vehicle_types.append(
                vehicle_order[i] if i < len(vehicle_order) else VehicleType.CARGO
            )

        # Set goals (destination at end of route)
        self.goals = [(self.grid_height - 5, center_col)] * self.config.num_vehicles

        # Reset states
        self.vehicle_states = [VehicleState.OPERATIONAL] * self.config.num_vehicles

        # Generate threats along route
        self.threats = {}
        self.obstacles = set()
        for _ in range(self.config.num_threats):
            while True:
                row = np.random.randint(20, self.grid_height - 20)
                col = np.random.randint(3, self.grid_width - 3)
                pos = (row, col)
                if pos not in self.threats and pos not in self.obstacles:
                    threat_type = np.random.choice([
                        ThreatType.IED,
                        ThreatType.AMBUSH,
                        ThreatType.BLOCKED
                    ])
                    self.threats[pos] = threat_type
                    break

        # Add terrain obstacles
        for _ in range(self.config.num_threats * 2):
            row = np.random.randint(10, self.grid_height - 10)
            col = np.random.randint(0, 3) if np.random.random() < 0.5 else \
                  np.random.randint(self.grid_width - 3, self.grid_width)
            self.obstacles.add((row, col))

        self.detected_threats = set()

        # Reset metrics
        self.total_collisions = 0
        self.vehicles_lost = 0
        self.threats_detected = 0
        self.step_count = 0
        self.mission_complete = False

        return self.positions.copy()

    def get_state(self, vehicle_id: int) -> Tuple:
        """Get state for a specific vehicle."""
        own_pos = self.positions[vehicle_id]
        goal = self.goals[vehicle_id]
        v_type = self.vehicle_types[vehicle_id]

        # Get positions of other convoy vehicles
        others = [p for i, p in enumerate(self.positions) if i != vehicle_id]

        # Scan for threats in detection range (5 cells ahead)
        visible_threats = []
        for threat_pos, threat_type in self.threats.items():
            if threat_pos in self.detected_threats:
                continue
            dr = threat_pos[0] - own_pos[0]
            dc = abs(threat_pos[1] - own_pos[1])
            if 0 < dr < 10 and dc < 5:  # Ahead and nearby
                visible_threats.append((threat_pos, threat_type))

        return (own_pos, goal, v_type, tuple(others), tuple(visible_threats))

    def step(self, actions: List[Action]) -> ConvoyStepResult:
        """Execute convoy movement."""
        self.step_count += 1

        prev_positions = self.positions.copy()
        new_positions = []

        # Apply actions
        for i, (pos, action) in enumerate(zip(self.positions, actions)):
            if self.vehicle_states[i] in [VehicleState.DISABLED, VehicleState.DESTROYED]:
                new_positions.append(pos)
                continue

            dr, dc = 0, 0
            if action == Action.UP:
                dr = -1  # Backward
            elif action == Action.DOWN:
                dr = 1   # Forward (toward goal)
            elif action == Action.LEFT:
                dc = -1
            elif action == Action.RIGHT:
                dc = 1

            new_r = max(0, min(self.grid_height - 1, pos[0] + dr))
            new_c = max(0, min(self.grid_width - 1, pos[1] + dc))

            # Check for obstacles
            if (new_r, new_c) in self.obstacles:
                new_positions.append(pos)  # Can't move into obstacle
            else:
                new_positions.append((new_r, new_c))

        # Check vehicle-to-vehicle collisions
        collision_occurred = False
        colliding_vehicles = set()

        # Same-cell collisions
        for i in range(len(new_positions)):
            if self.vehicle_states[i] == VehicleState.DESTROYED:
                continue
            for j in range(i + 1, len(new_positions)):
                if self.vehicle_states[j] == VehicleState.DESTROYED:
                    continue
                if new_positions[i] == new_positions[j]:
                    collision_occurred = True
                    colliding_vehicles.add(i)
                    colliding_vehicles.add(j)

        # Swap collisions
        for i in range(len(new_positions)):
            if self.vehicle_states[i] == VehicleState.DESTROYED:
                continue
            for j in range(i + 1, len(new_positions)):
                if self.vehicle_states[j] == VehicleState.DESTROYED:
                    continue
                if (new_positions[i] == prev_positions[j] and
                    new_positions[j] == prev_positions[i]):
                    collision_occurred = True
                    colliding_vehicles.add(i)
                    colliding_vehicles.add(j)

        # Update positions
        self.positions = new_positions

        # Handle collisions
        if collision_occurred:
            self.total_collisions += 1
            for v_id in colliding_vehicles:
                if self.vehicle_states[v_id] == VehicleState.OPERATIONAL:
                    self.vehicle_states[v_id] = VehicleState.DAMAGED
                elif self.vehicle_states[v_id] == VehicleState.DAMAGED:
                    self.vehicle_states[v_id] = VehicleState.DISABLED
                    self.vehicles_lost += 1

        # Check for threats
        for i, pos in enumerate(self.positions):
            if self.vehicle_states[i] == VehicleState.DESTROYED:
                continue
            if pos in self.threats and pos not in self.detected_threats:
                threat = self.threats[pos]
                self.detected_threats.add(pos)
                self.threats_detected += 1

                if threat == ThreatType.IED:
                    # IED hit - vehicle destroyed
                    self.vehicle_states[i] = VehicleState.DESTROYED
                    self.vehicles_lost += 1
                elif threat == ThreatType.AMBUSH:
                    # Ambush - vehicle damaged
                    if self.vehicle_states[i] == VehicleState.OPERATIONAL:
                        self.vehicle_states[i] = VehicleState.DAMAGED

        # Lead vehicle detection (scan ahead)
        for i, pos in enumerate(self.positions):
            if self.vehicle_types[i] in [VehicleType.LEAD, VehicleType.SECURITY]:
                for threat_pos in self.threats:
                    if threat_pos in self.detected_threats:
                        continue
                    dr = threat_pos[0] - pos[0]
                    dc = abs(threat_pos[1] - pos[1])
                    if 0 < dr < 8 and dc < 3:
                        self.detected_threats.add(threat_pos)
                        self.threats_detected += 1

        # Check mission completion
        operational = [i for i, s in enumerate(self.vehicle_states)
                      if s in [VehicleState.OPERATIONAL, VehicleState.DAMAGED]]

        # At least one cargo vehicle must reach destination
        cargo_delivered = False
        for i in operational:
            if self.vehicle_types[i] == VehicleType.CARGO:
                if self.positions[i][0] >= self.grid_height - 10:
                    cargo_delivered = True

        self.mission_complete = cargo_delivered

        # Compute rewards
        rewards = []
        for i in range(self.config.num_vehicles):
            if self.vehicle_states[i] == VehicleState.DESTROYED:
                rewards.append(-self.config.vehicle_cost)
            elif i in colliding_vehicles:
                rewards.append(-50000)  # Collision penalty
            else:
                # Progress reward
                progress = self.positions[i][0] / self.grid_height
                rewards.append(progress * 100)

        # Done conditions
        all_destroyed = all(s == VehicleState.DESTROYED for s in self.vehicle_states)
        done = all_destroyed or self.mission_complete or self.step_count >= self.config.max_steps

        return ConvoyStepResult(
            positions=self.positions.copy(),
            vehicle_states=self.vehicle_states.copy(),
            rewards=rewards,
            done=done,
            info={
                "collision": collision_occurred,
                "colliding_vehicles": list(colliding_vehicles),
                "total_collisions": self.total_collisions,
                "vehicles_lost": self.vehicles_lost,
                "threats_detected": self.threats_detected,
                "mission_complete": self.mission_complete,
                "step": self.step_count,
                "financial_loss": self.vehicles_lost * self.config.vehicle_cost,
            }
        )

    def get_metrics(self) -> Dict[str, Any]:
        """Get final metrics."""
        operational = sum(1 for s in self.vehicle_states
                         if s in [VehicleState.OPERATIONAL, VehicleState.DAMAGED])
        return {
            "total_collisions": self.total_collisions,
            "vehicles_lost": self.vehicles_lost,
            "operational_vehicles": operational,
            "survival_rate": operational / self.config.num_vehicles,
            "threats_detected": self.threats_detected,
            "mission_complete": self.mission_complete,
            "financial_loss": self.vehicles_lost * self.config.vehicle_cost,
            "steps": self.step_count,
        }
