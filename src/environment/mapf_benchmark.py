"""
MovingAI MAPF Benchmark Environment Adapter

Loads real MAPF benchmarks from MovingAI and provides an environment
compatible with our RSCT gatekeeper experiments.

Supports:
- Loading .map files (grid maps with obstacles)
- Loading .scen files (agent start/goal positions)
- Running multi-agent pathfinding episodes
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional, Set
import numpy as np
import os

from .gridworld import Action, ACTION_EFFECTS, StepResult


@dataclass
class MAPFScenario:
    """A single MAPF scenario with multiple agents."""
    map_name: str
    width: int
    height: int
    obstacles: Set[Tuple[int, int]]
    agents: List[Tuple[Tuple[int, int], Tuple[int, int]]]  # (start, goal) pairs
    optimal_lengths: List[float]


class MAPFBenchmarkEnv:
    """
    Environment wrapper for MovingAI MAPF benchmarks.

    Provides the same interface as MultiAgentGridworld but loads
    real benchmark maps and scenarios.
    """

    def __init__(
        self,
        map_path: str,
        scenario_path: str,
        num_agents: int = 4,
        max_steps: int = 100,
        goal_reward: float = 10.0,
        step_penalty: float = -0.1,
        collision_penalty: float = -20.0,
    ):
        self.map_path = map_path
        self.scenario_path = scenario_path
        self.num_agents = num_agents
        self.max_steps = max_steps
        self.goal_reward = goal_reward
        self.step_penalty = step_penalty
        self.collision_penalty = collision_penalty

        # Load map
        self.grid, self.width, self.height = self._load_map(map_path)
        self.obstacles = self._extract_obstacles()
        self.grid_size = max(self.width, self.height)

        # Load scenarios
        self.scenarios = self._load_scenarios(scenario_path)

        # Current episode state
        self.positions: List[Tuple[int, int]] = []
        self.goals: List[Tuple[int, int]] = []
        self.reached_goal: List[bool] = []
        self.step_count = 0
        self.total_collisions = 0
        self.collision_history: List[Tuple[int, List[int]]] = []
        self.current_scenario_idx = 0

    def _load_map(self, path: str) -> Tuple[List[List[str]], int, int]:
        """Load a MovingAI .map file."""
        with open(path, 'r') as f:
            lines = f.readlines()

        # Parse header
        height = width = 0
        map_start = 0
        for i, line in enumerate(lines):
            line = line.strip()
            if line.startswith('height'):
                height = int(line.split()[1])
            elif line.startswith('width'):
                width = int(line.split()[1])
            elif line == 'map':
                map_start = i + 1
                break

        # Parse grid
        grid = []
        for i in range(map_start, map_start + height):
            if i < len(lines):
                row = list(lines[i].strip())
                # Pad if necessary
                while len(row) < width:
                    row.append('.')
                grid.append(row[:width])

        return grid, width, height

    def _extract_obstacles(self) -> Set[Tuple[int, int]]:
        """Extract obstacle positions from grid."""
        obstacles = set()
        for r in range(len(self.grid)):
            for c in range(len(self.grid[r])):
                # '@', 'T', and other non-passable chars
                if self.grid[r][c] not in '.G':
                    obstacles.add((r, c))
        return obstacles

    def _load_scenarios(self, path: str) -> List[List[Tuple[Tuple[int, int], Tuple[int, int], float]]]:
        """Load a MovingAI .scen file."""
        scenarios = []
        current_agents = []

        with open(path, 'r') as f:
            lines = f.readlines()

        for line in lines[1:]:  # Skip version line
            parts = line.strip().split('\t')
            if len(parts) >= 9:
                # bucket, map, width, height, start_x, start_y, goal_x, goal_y, optimal
                start_x, start_y = int(parts[4]), int(parts[5])
                goal_x, goal_y = int(parts[6]), int(parts[7])
                optimal = float(parts[8])

                # Convert to (row, col) format
                start = (start_y, start_x)
                goal = (goal_y, goal_x)

                current_agents.append((start, goal, optimal))

                # Group into scenarios of num_agents
                if len(current_agents) >= self.num_agents:
                    scenarios.append(current_agents[:self.num_agents])
                    current_agents = current_agents[self.num_agents:]

        return scenarios

    def _is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """Check if position is within grid and not an obstacle."""
        r, c = pos
        if r < 0 or r >= self.height or c < 0 or c >= self.width:
            return False
        if pos in self.obstacles:
            return False
        return True

    def _compute_next_position(
        self, current: Tuple[int, int], action: Action
    ) -> Tuple[int, int]:
        """Compute next position given current position and action."""
        dr, dc = ACTION_EFFECTS[action]
        next_pos = (current[0] + dr, current[1] + dc)
        if self._is_valid_position(next_pos):
            return next_pos
        return current

    def _detect_collisions(
        self,
        current_positions: List[Tuple[int, int]],
        next_positions: List[Tuple[int, int]],
    ) -> List[int]:
        """Detect collision (same cell or swap)."""
        colliding = set()

        # Same cell
        for i in range(len(next_positions)):
            for j in range(i + 1, len(next_positions)):
                if next_positions[i] == next_positions[j]:
                    colliding.add(i)
                    colliding.add(j)

        # Swap collision
        for i in range(len(current_positions)):
            for j in range(i + 1, len(current_positions)):
                if (next_positions[i] == current_positions[j] and
                    next_positions[j] == current_positions[i]):
                    colliding.add(i)
                    colliding.add(j)

        return list(colliding)

    def reset(self, scenario_idx: Optional[int] = None) -> List[Tuple[int, int]]:
        """Reset to a specific scenario or next in sequence."""
        if scenario_idx is not None:
            self.current_scenario_idx = scenario_idx
        else:
            self.current_scenario_idx = (self.current_scenario_idx + 1) % len(self.scenarios)

        scenario = self.scenarios[self.current_scenario_idx]

        self.positions = [agent[0] for agent in scenario]
        self.goals = [agent[1] for agent in scenario]
        self.reached_goal = [False] * len(self.positions)
        self.step_count = 0
        self.total_collisions = 0
        self.collision_history = []

        return self.positions.copy()

    def step(self, actions: List[Action]) -> StepResult:
        """Execute joint action."""
        assert len(actions) == len(self.positions)

        self.step_count += 1

        # Compute next positions
        next_positions = [
            self._compute_next_position(self.positions[i], actions[i])
            for i in range(len(self.positions))
        ]

        # Detect collisions
        colliding = self._detect_collisions(self.positions, next_positions)
        collision = len(colliding) > 0

        if collision:
            self.total_collisions += 1
            self.collision_history.append((self.step_count, colliding))

        # Compute rewards
        rewards = []
        for i in range(len(self.positions)):
            reward = self.step_penalty

            if i in colliding:
                reward += self.collision_penalty

            if next_positions[i] == self.goals[i] and not self.reached_goal[i]:
                reward += self.goal_reward
                self.reached_goal[i] = True

            rewards.append(reward)

        # Update positions
        self.positions = next_positions

        # Check termination
        all_reached = all(self.reached_goal)
        max_steps = self.step_count >= self.max_steps
        done = all_reached or max_steps

        info = {
            "collision": collision,
            "colliding_agents": colliding,
            "reached_goals": self.reached_goal.copy(),
            "all_goals_reached": all_reached,
            "step": self.step_count,
            "total_collisions": self.total_collisions,
        }

        return StepResult(
            next_positions=self.positions.copy(),
            rewards=rewards,
            done=done,
            info=info,
        )

    def get_state(self, agent_id: int) -> Tuple:
        """Get state for agent."""
        own_pos = self.positions[agent_id]
        other_positions = tuple(
            self.positions[i] for i in range(len(self.positions)) if i != agent_id
        )
        own_goal = self.goals[agent_id]
        return (own_pos, other_positions, own_goal)

    def get_metrics(self) -> Dict[str, Any]:
        """Get episode metrics."""
        return {
            "total_steps": self.step_count,
            "total_collisions": self.total_collisions,
            "collision_history": self.collision_history.copy(),
            "all_goals_reached": all(self.reached_goal),
            "goals_reached_count": sum(self.reached_goal),
            "final_positions": self.positions.copy(),
            "scenario_idx": self.current_scenario_idx,
        }

    def render(self, mode: str = "ascii") -> str:
        """Render current state."""
        display = [row.copy() for row in self.grid]

        # Mark goals
        for i, goal in enumerate(self.goals):
            if not self.reached_goal[i]:
                r, c = goal
                if 0 <= r < len(display) and 0 <= c < len(display[0]):
                    display[r][c] = '*'

        # Mark agents
        pos_count: Dict[Tuple[int, int], List[int]] = {}
        for i, pos in enumerate(self.positions):
            if pos not in pos_count:
                pos_count[pos] = []
            pos_count[pos].append(i)

        for pos, agents in pos_count.items():
            r, c = pos
            if 0 <= r < len(display) and 0 <= c < len(display[0]):
                if len(agents) > 1:
                    display[r][c] = 'X'  # Collision
                else:
                    display[r][c] = chr(ord('A') + agents[0])

        lines = [''.join(row) for row in display[:30]]  # Truncate for display
        lines.append(f"Step: {self.step_count}, Collisions: {self.total_collisions}")
        return '\n'.join(lines)


def list_available_benchmarks(data_dir: str) -> List[Dict[str, str]]:
    """List available map/scenario pairs."""
    benchmarks = []

    maps_dir = data_dir
    scen_dir = os.path.join(data_dir, "scen-random")

    if not os.path.exists(maps_dir):
        return benchmarks

    for map_file in os.listdir(maps_dir):
        if map_file.endswith('.map'):
            map_name = map_file[:-4]
            map_path = os.path.join(maps_dir, map_file)

            # Find corresponding scenarios
            if os.path.exists(scen_dir):
                for scen_file in os.listdir(scen_dir):
                    if scen_file.startswith(map_name) and scen_file.endswith('.scen'):
                        scen_path = os.path.join(scen_dir, scen_file)
                        benchmarks.append({
                            "map": map_path,
                            "scenario": scen_path,
                            "name": f"{map_name}_{scen_file[:-5]}",
                        })

    return benchmarks
