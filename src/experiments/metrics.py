"""
Metrics collection and analysis for RSCT vs MARL experiments.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from collections import defaultdict


@dataclass
class EpisodeMetrics:
    """Metrics for a single episode."""

    episode_id: int
    regime: str  # "marl" or "rsct_gated"

    # Safety metrics
    total_collisions: int
    collision_steps: List[int]  # Steps where collisions occurred

    # Performance metrics
    total_steps: int
    all_goals_reached: bool
    goals_reached_count: int
    total_return: float  # Sum of rewards across all agents

    # Per-agent metrics
    agent_returns: List[float]
    agent_goal_reached: List[bool]

    # RSCT-specific (only for rsct_gated regime)
    gate_blocks: int = 0
    gate_block_rate: float = 0.0

    # Trajectory data (for visualization)
    position_history: List[List[Tuple[int, int]]] = field(default_factory=list)


class MetricsCollector:
    """
    Collects and aggregates metrics across episodes and experiments.
    """

    def __init__(self):
        self.episodes: List[EpisodeMetrics] = []
        self.regime_episodes: Dict[str, List[EpisodeMetrics]] = defaultdict(list)

        # Cell visitation tracking (for heatmaps)
        self.cell_visits: Dict[str, Dict[int, np.ndarray]] = {}  # regime -> agent_id -> grid

        # Collision location tracking
        self.collision_locations: Dict[str, List[Tuple[int, int]]] = defaultdict(list)

    def record_episode(self, metrics: EpisodeMetrics, grid_size: int = 5):
        """Record metrics for a completed episode."""
        self.episodes.append(metrics)
        self.regime_episodes[metrics.regime].append(metrics)

        # Initialize cell visit tracking if needed
        if metrics.regime not in self.cell_visits:
            self.cell_visits[metrics.regime] = {}

        # Update cell visitation counts
        for agent_id, positions in enumerate(metrics.position_history):
            if agent_id not in self.cell_visits[metrics.regime]:
                self.cell_visits[metrics.regime][agent_id] = np.zeros(
                    (grid_size, grid_size), dtype=np.int32
                )

            for pos in positions:
                r, c = pos
                self.cell_visits[metrics.regime][agent_id][r, c] += 1

        # Track collision locations
        if metrics.position_history and metrics.collision_steps:
            for step in metrics.collision_steps:
                if step < len(metrics.position_history[0]):
                    # Record position of first agent at collision (approximation)
                    pos = metrics.position_history[0][step]
                    self.collision_locations[metrics.regime].append(pos)

    def get_summary(self, regime: Optional[str] = None) -> Dict[str, Any]:
        """
        Get summary statistics for a regime or all regimes.
        """
        episodes = (
            self.regime_episodes[regime] if regime else self.episodes
        )

        if not episodes:
            return {}

        # Safety metrics
        collision_counts = [e.total_collisions for e in episodes]
        collision_rate = sum(1 for c in collision_counts if c > 0) / len(episodes)

        # Find time-to-zero-collisions (first episode with 50 consecutive collision-free)
        time_to_zero = None
        consecutive_safe = 0
        for i, e in enumerate(episodes):
            if e.total_collisions == 0:
                consecutive_safe += 1
                if consecutive_safe >= 50 and time_to_zero is None:
                    time_to_zero = i - 49
            else:
                consecutive_safe = 0

        # Performance metrics
        steps_to_goal = [
            e.total_steps for e in episodes if e.all_goals_reached
        ]
        success_rate = sum(1 for e in episodes if e.all_goals_reached) / len(episodes)

        returns = [e.total_return for e in episodes]

        # RSCT-specific
        gate_blocks = [e.gate_blocks for e in episodes if e.regime == "rsct_gated"]

        return {
            "num_episodes": len(episodes),
            "regime": regime or "all",

            # Safety
            "total_collisions": sum(collision_counts),
            "mean_collisions_per_episode": np.mean(collision_counts),
            "collision_rate": collision_rate,  # Fraction of episodes with at least one collision
            "time_to_zero_collisions": time_to_zero,

            # Performance
            "success_rate": success_rate,
            "mean_steps_to_goal": np.mean(steps_to_goal) if steps_to_goal else None,
            "std_steps_to_goal": np.std(steps_to_goal) if steps_to_goal else None,
            "mean_return": np.mean(returns),
            "std_return": np.std(returns),

            # RSCT-specific
            "mean_gate_blocks": np.mean(gate_blocks) if gate_blocks else None,

            # Time series data
            "collision_history": collision_counts,
            "return_history": returns,
            "steps_history": [e.total_steps for e in episodes],
        }

    def get_comparison(self) -> Dict[str, Any]:
        """Get side-by-side comparison of MARL vs RSCT-gated."""
        marl_summary = self.get_summary("marl")
        rsct_summary = self.get_summary("rsct_gated")

        return {
            "marl": marl_summary,
            "rsct_gated": rsct_summary,
            "comparison": {
                "collision_reduction": (
                    (marl_summary.get("mean_collisions_per_episode", 0) -
                     rsct_summary.get("mean_collisions_per_episode", 0))
                    if marl_summary and rsct_summary else None
                ),
                "success_rate_diff": (
                    (rsct_summary.get("success_rate", 0) -
                     marl_summary.get("success_rate", 0))
                    if marl_summary and rsct_summary else None
                ),
            },
        }

    def get_heatmap_data(self, regime: str, agent_id: int = 0) -> Optional[np.ndarray]:
        """Get cell visitation heatmap for a specific regime and agent."""
        if regime in self.cell_visits and agent_id in self.cell_visits[regime]:
            return self.cell_visits[regime][agent_id].copy()
        return None

    def get_collision_heatmap(self, regime: str, grid_size: int = 5) -> np.ndarray:
        """Get heatmap of collision locations."""
        heatmap = np.zeros((grid_size, grid_size), dtype=np.int32)
        for pos in self.collision_locations.get(regime, []):
            r, c = pos
            if 0 <= r < grid_size and 0 <= c < grid_size:
                heatmap[r, c] += 1
        return heatmap

    def export_for_plotting(self) -> Dict[str, Any]:
        """Export data in format ready for visualization module."""
        return {
            "marl": {
                "collision_history": [
                    e.total_collisions for e in self.regime_episodes.get("marl", [])
                ],
                "steps_history": [
                    e.total_steps for e in self.regime_episodes.get("marl", [])
                ],
                "return_history": [
                    e.total_return for e in self.regime_episodes.get("marl", [])
                ],
                "success_history": [
                    e.all_goals_reached for e in self.regime_episodes.get("marl", [])
                ],
            },
            "rsct_gated": {
                "collision_history": [
                    e.total_collisions for e in self.regime_episodes.get("rsct_gated", [])
                ],
                "steps_history": [
                    e.total_steps for e in self.regime_episodes.get("rsct_gated", [])
                ],
                "return_history": [
                    e.total_return for e in self.regime_episodes.get("rsct_gated", [])
                ],
                "success_history": [
                    e.all_goals_reached for e in self.regime_episodes.get("rsct_gated", [])
                ],
                "gate_blocks_history": [
                    e.gate_blocks for e in self.regime_episodes.get("rsct_gated", [])
                ],
            },
            "cell_visits": self.cell_visits,
            "collision_locations": dict(self.collision_locations),
        }
