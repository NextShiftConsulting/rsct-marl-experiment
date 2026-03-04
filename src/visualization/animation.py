"""
Gridworld animation for visualizing agent trajectories.

Creates frame-by-frame animations showing:
- Agent movements
- Collision events (red flash)
- Gate interventions (orange outline)
"""

from typing import List, Tuple, Optional, Dict, Any
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.animation import FuncAnimation
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class GridworldAnimator:
    """
    Animator for gridworld episodes.

    Creates visual representations of agent trajectories,
    highlighting collisions and gate interventions.
    """

    def __init__(
        self,
        grid_size: int = 5,
        obstacles: List[Tuple[int, int]] = None,
        goals: List[Tuple[int, int]] = None,
    ):
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib required for animation")

        self.grid_size = grid_size
        self.obstacles = obstacles or []
        self.goals = goals or []

        # Colors for agents
        self.agent_colors = ["blue", "green", "orange", "purple", "cyan", "magenta", "yellow", "brown"]

    def render_frame(
        self,
        positions: List[Tuple[int, int]],
        step: int = 0,
        collision: bool = False,
        blocked_agents: List[int] = None,
        ax: Optional[Any] = None,
    ) -> Any:
        """
        Render a single frame of the gridworld.

        Args:
            positions: Current position of each agent
            step: Step number
            collision: Whether a collision occurred this step
            blocked_agents: List of agents whose actions were blocked by gate
            ax: Matplotlib axes to draw on (creates new if None)

        Returns:
            matplotlib axes
        """
        blocked_agents = blocked_agents or []

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))

        ax.clear()

        # Draw grid
        for i in range(self.grid_size + 1):
            ax.axhline(i - 0.5, color="gray", linewidth=0.5)
            ax.axvline(i - 0.5, color="gray", linewidth=0.5)

        # Draw obstacles
        for r, c in self.obstacles:
            rect = patches.Rectangle(
                (c - 0.5, r - 0.5), 1, 1,
                facecolor="black", edgecolor="black"
            )
            ax.add_patch(rect)

        # Draw goals
        for i, (r, c) in enumerate(self.goals):
            color = self.agent_colors[i % len(self.agent_colors)]
            ax.plot(c, r, marker="*", markersize=30, color=color, alpha=0.5,
                   markeredgecolor="black", markeredgewidth=1)

        # Check for collision at current step
        position_counts: Dict[Tuple[int, int], List[int]] = {}
        for i, pos in enumerate(positions):
            if pos not in position_counts:
                position_counts[pos] = []
            position_counts[pos].append(i)

        # Draw agents
        for i, pos in enumerate(positions):
            r, c = pos
            color = self.agent_colors[i % len(self.agent_colors)]

            # Check if collision at this position
            is_collision = len(position_counts[pos]) > 1

            # Draw agent
            if is_collision or collision:
                # Collision: red background
                circle = patches.Circle(
                    (c, r), 0.4,
                    facecolor="red", edgecolor="darkred", linewidth=3
                )
                ax.add_patch(circle)
            elif i in blocked_agents:
                # Gate blocked: orange outline
                circle = patches.Circle(
                    (c, r), 0.4,
                    facecolor=color, edgecolor="orange", linewidth=4
                )
                ax.add_patch(circle)
            else:
                # Normal
                circle = patches.Circle(
                    (c, r), 0.4,
                    facecolor=color, edgecolor="black", linewidth=2
                )
                ax.add_patch(circle)

            # Agent label
            ax.text(c, r, chr(ord("A") + i), ha="center", va="center",
                   fontsize=14, fontweight="bold", color="white")

        # Configure axes
        ax.set_xlim(-0.5, self.grid_size - 0.5)
        ax.set_ylim(self.grid_size - 0.5, -0.5)  # Flip y-axis for intuitive display
        ax.set_aspect("equal")
        ax.set_title(f"Step {step}" + (" - COLLISION!" if collision else ""), fontsize=14)

        return ax

    def create_episode_animation(
        self,
        position_history: List[List[Tuple[int, int]]],
        collision_steps: List[int] = None,
        blocked_steps: Dict[int, List[int]] = None,
        interval: int = 500,
        save_path: Optional[str] = None,
    ) -> Any:
        """
        Create animation for a full episode.

        Args:
            position_history: List of position lists for each agent over time
                            Shape: [num_agents][num_steps]
            collision_steps: List of steps where collisions occurred
            blocked_steps: Dict mapping step -> list of blocked agent IDs
            interval: Milliseconds between frames
            save_path: Optional path to save animation (requires ffmpeg or pillow)

        Returns:
            matplotlib FuncAnimation object
        """
        collision_steps = collision_steps or []
        blocked_steps = blocked_steps or {}

        # Transpose position history: [step][agent] instead of [agent][step]
        num_agents = len(position_history)
        num_steps = len(position_history[0]) if position_history else 0

        if num_steps == 0:
            print("No trajectory data for animation")
            return None

        fig, ax = plt.subplots(figsize=(8, 8))

        def animate(step):
            positions = [position_history[agent][step] for agent in range(num_agents)]
            collision = step in collision_steps
            blocked = blocked_steps.get(step, [])
            self.render_frame(positions, step, collision, blocked, ax)
            return []

        anim = FuncAnimation(
            fig, animate,
            frames=num_steps,
            interval=interval,
            blit=False,
            repeat=True,
        )

        if save_path:
            try:
                if save_path.endswith(".gif"):
                    anim.save(save_path, writer="pillow", fps=1000 // interval)
                else:
                    anim.save(save_path, writer="ffmpeg", fps=1000 // interval)
                print(f"Saved animation to: {save_path}")
            except Exception as e:
                print(f"Could not save animation: {e}")
                print("Install pillow for GIF or ffmpeg for MP4 support")

        return anim

    def render_trajectory_overlay(
        self,
        position_history: List[List[Tuple[int, int]]],
        collision_steps: List[int] = None,
        save_path: Optional[str] = None,
    ) -> Any:
        """
        Render static image with full trajectories overlaid.

        Shows the complete path of each agent with collision points highlighted.

        Args:
            position_history: Position history per agent
            collision_steps: Steps where collisions occurred
            save_path: Optional path to save figure

        Returns:
            matplotlib figure
        """
        collision_steps = collision_steps or []

        fig, ax = plt.subplots(figsize=(10, 10))

        # Draw grid
        for i in range(self.grid_size + 1):
            ax.axhline(i - 0.5, color="gray", linewidth=0.5, alpha=0.5)
            ax.axvline(i - 0.5, color="gray", linewidth=0.5, alpha=0.5)

        # Draw obstacles
        for r, c in self.obstacles:
            rect = patches.Rectangle(
                (c - 0.5, r - 0.5), 1, 1,
                facecolor="black", edgecolor="black"
            )
            ax.add_patch(rect)

        # Draw goals
        for i, (r, c) in enumerate(self.goals):
            color = self.agent_colors[i % len(self.agent_colors)]
            ax.plot(c, r, marker="*", markersize=25, color=color, alpha=0.7,
                   markeredgecolor="black", markeredgewidth=1)

        # Draw trajectories
        for agent_id, positions in enumerate(position_history):
            color = self.agent_colors[agent_id % len(self.agent_colors)]

            # Extract coordinates
            rows = [p[0] for p in positions]
            cols = [p[1] for p in positions]

            # Plot path
            ax.plot(cols, rows, color=color, linewidth=2, alpha=0.7,
                   label=f"Agent {chr(ord('A') + agent_id)}")

            # Mark start and end
            ax.plot(cols[0], rows[0], marker="o", markersize=12, color=color,
                   markeredgecolor="black", markeredgewidth=2)
            ax.plot(cols[-1], rows[-1], marker="s", markersize=10, color=color,
                   markeredgecolor="black", markeredgewidth=2)

        # Mark collision points
        for step in collision_steps:
            for agent_id, positions in enumerate(position_history):
                if step < len(positions):
                    r, c = positions[step]
                    ax.plot(c, r, marker="X", markersize=15, color="red",
                           markeredgecolor="darkred", markeredgewidth=2)

        ax.set_xlim(-0.5, self.grid_size - 0.5)
        ax.set_ylim(self.grid_size - 0.5, -0.5)
        ax.set_aspect("equal")
        ax.legend(loc="upper right")
        ax.set_title("Agent Trajectories (X = collision)", fontsize=14)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved trajectory overlay to: {save_path}")

        return fig
