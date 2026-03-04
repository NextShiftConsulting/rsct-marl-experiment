"""
Multi-Agent Gridworld Environment

A clean, extensible gridworld for MARL vs RSCT experiments.
Supports N agents, configurable grids, and detailed collision tracking.
"""

from enum import IntEnum
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any
import numpy as np

from .config import GridworldConfig


class Action(IntEnum):
    """Available actions for agents."""
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    STAY = 4


# Action effects: (delta_row, delta_col)
ACTION_EFFECTS = {
    Action.UP: (-1, 0),
    Action.DOWN: (1, 0),
    Action.LEFT: (0, -1),
    Action.RIGHT: (0, 1),
    Action.STAY: (0, 0),
}


@dataclass
class StepResult:
    """Result of a single environment step."""
    next_positions: List[Tuple[int, int]]
    rewards: List[float]
    done: bool
    info: Dict[str, Any]


class MultiAgentGridworld:
    """
    Multi-agent gridworld environment.

    Supports arbitrary number of agents, obstacles, and goal positions.
    Tracks collisions and provides detailed step information for analysis.
    """

    def __init__(self, config: Optional[GridworldConfig] = None):
        self.config = config or GridworldConfig()
        self.grid_size = self.config.grid_size
        self.num_agents = self.config.num_agents
        self.obstacles = set(self.config.obstacles)

        # Agent state
        self.positions: List[Tuple[int, int]] = []
        self.goals: List[Tuple[int, int]] = []
        self.reached_goal: List[bool] = []

        # Episode tracking
        self.step_count = 0
        self.total_collisions = 0
        self.collision_history: List[Tuple[int, List[int]]] = []  # (step, [agent_ids])

        self.reset()

    def reset(self) -> List[Tuple[int, int]]:
        """Reset environment to initial state."""
        self.positions = [cfg[0] for cfg in self.config.agent_configs]
        self.goals = [cfg[1] for cfg in self.config.agent_configs]
        self.reached_goal = [False] * self.num_agents
        self.step_count = 0
        self.total_collisions = 0
        self.collision_history = []
        return self.positions.copy()

    def _is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """Check if position is within grid and not an obstacle."""
        r, c = pos
        if r < 0 or r >= self.grid_size or c < 0 or c >= self.grid_size:
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
        return current  # Stay in place if move is invalid

    def _detect_collisions(
        self,
        current_positions: List[Tuple[int, int]],
        next_positions: List[Tuple[int, int]],
    ) -> List[int]:
        """
        Detect all collision types:
        1. Same cell collision: two agents move to the same cell
        2. Swap collision: two agents swap positions (cross through each other)

        Returns list of agent indices involved in collisions.
        """
        colliding_agents = set()

        # Check same-cell collisions
        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                if next_positions[i] == next_positions[j]:
                    colliding_agents.add(i)
                    colliding_agents.add(j)

        # Check swap collisions
        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                if (
                    next_positions[i] == current_positions[j]
                    and next_positions[j] == current_positions[i]
                ):
                    colliding_agents.add(i)
                    colliding_agents.add(j)

        return list(colliding_agents)

    def step(self, actions: List[Action]) -> StepResult:
        """
        Execute joint action for all agents.

        Args:
            actions: List of actions, one per agent

        Returns:
            StepResult with next positions, rewards, done flag, and info dict
        """
        assert len(actions) == self.num_agents, "Must provide action for each agent"

        self.step_count += 1

        # Compute proposed next positions
        next_positions = [
            self._compute_next_position(self.positions[i], actions[i])
            for i in range(self.num_agents)
        ]

        # Detect collisions
        colliding_agents = self._detect_collisions(self.positions, next_positions)
        collision_occurred = len(colliding_agents) > 0

        if collision_occurred:
            self.total_collisions += 1
            self.collision_history.append((self.step_count, colliding_agents))

        # Compute rewards
        rewards = []
        for i in range(self.num_agents):
            reward = self.config.step_penalty

            # Collision penalty
            if i in colliding_agents:
                reward += self.config.collision_penalty

            # Goal reward (only once per agent)
            if next_positions[i] == self.goals[i] and not self.reached_goal[i]:
                reward += self.config.goal_reward
                self.reached_goal[i] = True

            rewards.append(reward)

        # Update positions
        self.positions = next_positions

        # Check termination
        all_reached_goals = all(self.reached_goal)
        max_steps_reached = self.step_count >= self.config.max_steps
        done = all_reached_goals or max_steps_reached

        info = {
            "collision": collision_occurred,
            "colliding_agents": colliding_agents,
            "reached_goals": self.reached_goal.copy(),
            "all_goals_reached": all_reached_goals,
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
        """
        Get state representation for a specific agent.

        State: (own_pos, other_positions, own_goal)
        Flattened for Q-table indexing.
        """
        own_pos = self.positions[agent_id]
        other_positions = tuple(
            self.positions[i] for i in range(self.num_agents) if i != agent_id
        )
        own_goal = self.goals[agent_id]
        return (own_pos, other_positions, own_goal)

    def get_joint_state(self) -> Tuple:
        """Get full joint state (all positions and goals)."""
        return (tuple(self.positions), tuple(self.goals))

    def manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """Compute Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def render(self, mode: str = "ascii") -> str:
        """
        Render current grid state.

        Legend:
            A, B, C, ... = Agents
            *, **, ... = Goals (for agents that haven't reached)
            # = Obstacle
            . = Empty cell
            X = Collision (if multiple agents in same cell)
        """
        grid = [["." for _ in range(self.grid_size)] for _ in range(self.grid_size)]

        # Place obstacles
        for r, c in self.obstacles:
            grid[r][c] = "#"

        # Place goals (only if agent hasn't reached)
        for i, goal in enumerate(self.goals):
            if not self.reached_goal[i]:
                r, c = goal
                if grid[r][c] == ".":
                    grid[r][c] = "*"

        # Place agents (detect same-cell collisions for rendering)
        position_counts: Dict[Tuple[int, int], List[int]] = {}
        for i, pos in enumerate(self.positions):
            if pos not in position_counts:
                position_counts[pos] = []
            position_counts[pos].append(i)

        for pos, agent_ids in position_counts.items():
            r, c = pos
            if len(agent_ids) > 1:
                grid[r][c] = "X"  # Collision
            else:
                grid[r][c] = chr(ord("A") + agent_ids[0])

        # Build string
        lines = []
        lines.append("+" + "-" * self.grid_size + "+")
        for row in grid:
            lines.append("|" + "".join(row) + "|")
        lines.append("+" + "-" * self.grid_size + "+")
        lines.append(f"Step: {self.step_count}, Collisions: {self.total_collisions}")

        return "\n".join(lines)

    def get_metrics(self) -> Dict[str, Any]:
        """Get episode metrics for analysis."""
        return {
            "total_steps": self.step_count,
            "total_collisions": self.total_collisions,
            "collision_history": self.collision_history.copy(),
            "all_goals_reached": all(self.reached_goal),
            "goals_reached_count": sum(self.reached_goal),
            "final_positions": self.positions.copy(),
        }
