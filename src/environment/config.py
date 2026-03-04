"""
Environment configuration for gridworld experiments.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class GridworldConfig:
    """Configuration for multi-agent gridworld environment."""

    # Grid dimensions
    grid_size: int = 5

    # Agent configurations: list of (start_pos, goal_pos) tuples
    agent_configs: List[Tuple[Tuple[int, int], Tuple[int, int]]] = field(
        default_factory=lambda: [
            ((0, 0), (4, 4)),  # Agent A: bottom-left to top-right
            ((4, 0), (0, 4)),  # Agent B: bottom-right to top-left
        ]
    )

    # Obstacle positions (walls)
    obstacles: List[Tuple[int, int]] = field(
        default_factory=lambda: [
            (2, 1),  # Force paths to cross in center
            (2, 3),
        ]
    )

    # Episode parameters
    max_steps: int = 30

    # Reward structure
    goal_reward: float = 10.0
    step_penalty: float = -1.0
    collision_penalty: float = -20.0

    # Scaling configurations for experiments
    @classmethod
    def small(cls) -> "GridworldConfig":
        """5x5 grid, 2 agents - base experiment."""
        return cls()

    @classmethod
    def medium(cls) -> "GridworldConfig":
        """10x10 grid, 4 agents."""
        return cls(
            grid_size=10,
            agent_configs=[
                ((0, 0), (9, 9)),
                ((9, 0), (0, 9)),
                ((0, 9), (9, 0)),
                ((9, 9), (0, 0)),
            ],
            obstacles=[
                (4, 3), (4, 6), (5, 3), (5, 6),
            ],
            max_steps=50,
        )

    @classmethod
    def large(cls) -> "GridworldConfig":
        """20x20 grid, 8 agents."""
        agents = []
        # Place agents in corners and edge midpoints
        positions = [
            ((0, 0), (19, 19)),
            ((19, 0), (0, 19)),
            ((0, 19), (19, 0)),
            ((19, 19), (0, 0)),
            ((0, 10), (19, 10)),
            ((19, 10), (0, 10)),
            ((10, 0), (10, 19)),
            ((10, 19), (10, 0)),
        ]
        return cls(
            grid_size=20,
            agent_configs=positions,
            obstacles=[
                (9, 8), (9, 11), (10, 8), (10, 11),
            ],
            max_steps=80,
        )

    @property
    def num_agents(self) -> int:
        return len(self.agent_configs)
