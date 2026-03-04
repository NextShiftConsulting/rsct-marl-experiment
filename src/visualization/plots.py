"""
Visualization module for RSCT vs MARL experiments.

Creates publication-quality figures showing:
1. Collision comparison (the money shot)
2. Learning curves
3. Trajectory heatmaps
4. Collision location heatmaps
"""

from typing import Dict, Any, Optional, List, Tuple
import numpy as np

# Try to import matplotlib, provide fallback if not available
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.colors import LinearSegmentedColormap
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Visualization functions will not work.")


def _check_matplotlib():
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for visualization. Install with: pip install matplotlib")


def plot_collision_comparison(
    plotting_data: Dict[str, Any],
    save_path: Optional[str] = None,
    window_size: int = 20,
) -> Optional[Any]:
    """
    Plot collision count over episodes for MARL vs RSCT-gated.

    This is the key visualization showing learned vs certified safety.

    Args:
        plotting_data: Data from MetricsCollector.export_for_plotting()
        save_path: Optional path to save figure
        window_size: Window size for rolling average

    Returns:
        matplotlib figure if matplotlib is available
    """
    _check_matplotlib()

    marl_collisions = plotting_data.get("marl", {}).get("collision_history", [])
    rsct_collisions = plotting_data.get("rsct_gated", {}).get("collision_history", [])

    if not marl_collisions and not rsct_collisions:
        print("No collision data available")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left plot: Raw collision count per episode
    ax1 = axes[0]
    if marl_collisions:
        ax1.plot(marl_collisions, alpha=0.3, color="red", label="MARL (raw)")
        # Rolling average
        if len(marl_collisions) >= window_size:
            marl_smooth = np.convolve(
                marl_collisions,
                np.ones(window_size) / window_size,
                mode="valid"
            )
            ax1.plot(
                range(window_size - 1, len(marl_collisions)),
                marl_smooth,
                color="red",
                linewidth=2,
                label=f"MARL ({window_size}-ep avg)"
            )

    if rsct_collisions:
        ax1.plot(rsct_collisions, color="green", linewidth=2, label="RSCT-Gated")

    ax1.set_xlabel("Episode", fontsize=12)
    ax1.set_ylabel("Collisions per Episode", fontsize=12)
    ax1.set_title("Collision Count: Learned vs Certified", fontsize=14)
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=-0.1)

    # Right plot: Cumulative collisions
    ax2 = axes[1]
    if marl_collisions:
        marl_cumsum = np.cumsum(marl_collisions)
        ax2.plot(marl_cumsum, color="red", linewidth=2, label="MARL")

    if rsct_collisions:
        rsct_cumsum = np.cumsum(rsct_collisions)
        ax2.plot(rsct_cumsum, color="green", linewidth=2, label="RSCT-Gated")

    ax2.set_xlabel("Episode", fontsize=12)
    ax2.set_ylabel("Cumulative Collisions", fontsize=12)
    ax2.set_title("Total Collisions Over Time", fontsize=14)
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)

    # Add annotation showing final totals
    if marl_collisions:
        ax2.annotate(
            f"Total: {sum(marl_collisions)}",
            xy=(len(marl_collisions) - 1, sum(marl_collisions)),
            xytext=(10, 0),
            textcoords="offset points",
            fontsize=10,
            color="red",
        )
    if rsct_collisions:
        ax2.annotate(
            f"Total: {sum(rsct_collisions)}",
            xy=(len(rsct_collisions) - 1, sum(rsct_collisions)),
            xytext=(10, -15),
            textcoords="offset points",
            fontsize=10,
            color="green",
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved collision comparison to: {save_path}")

    return fig


def plot_learning_curves(
    plotting_data: Dict[str, Any],
    save_path: Optional[str] = None,
    window_size: int = 20,
) -> Optional[Any]:
    """
    Plot learning curves: return and success rate over episodes.

    Args:
        plotting_data: Data from MetricsCollector.export_for_plotting()
        save_path: Optional path to save figure
        window_size: Window size for rolling average

    Returns:
        matplotlib figure
    """
    _check_matplotlib()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    regimes = [("marl", "red", "MARL"), ("rsct_gated", "green", "RSCT-Gated")]

    for regime, color, label in regimes:
        data = plotting_data.get(regime, {})

        # Return history
        returns = data.get("return_history", [])
        if returns:
            ax = axes[0, 0]
            ax.plot(returns, alpha=0.2, color=color)
            if len(returns) >= window_size:
                smooth = np.convolve(returns, np.ones(window_size) / window_size, mode="valid")
                ax.plot(range(window_size - 1, len(returns)), smooth, color=color, linewidth=2, label=label)

        # Steps history
        steps = data.get("steps_history", [])
        if steps:
            ax = axes[0, 1]
            ax.plot(steps, alpha=0.2, color=color)
            if len(steps) >= window_size:
                smooth = np.convolve(steps, np.ones(window_size) / window_size, mode="valid")
                ax.plot(range(window_size - 1, len(steps)), smooth, color=color, linewidth=2, label=label)

        # Success rate (cumulative)
        success = data.get("success_history", [])
        if success:
            ax = axes[1, 0]
            success_rate = np.cumsum(success) / (np.arange(len(success)) + 1)
            ax.plot(success_rate, color=color, linewidth=2, label=label)

        # Gate blocks (RSCT only)
        if regime == "rsct_gated":
            blocks = data.get("gate_blocks_history", [])
            if blocks:
                ax = axes[1, 1]
                ax.plot(blocks, alpha=0.3, color=color)
                if len(blocks) >= window_size:
                    smooth = np.convolve(blocks, np.ones(window_size) / window_size, mode="valid")
                    ax.plot(range(window_size - 1, len(blocks)), smooth, color=color, linewidth=2, label=label)

    # Configure axes
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Total Return")
    axes[0, 0].set_title("Return per Episode")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Steps to Completion")
    axes[0, 1].set_title("Episode Length")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("Cumulative Success Rate")
    axes[1, 0].set_title("Success Rate Over Training")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(0, 1)

    axes[1, 1].set_xlabel("Episode")
    axes[1, 1].set_ylabel("Gate Blocks")
    axes[1, 1].set_title("RSCT Gate Interventions per Episode")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved learning curves to: {save_path}")

    return fig


def plot_trajectory_heatmap(
    cell_visits: Dict[str, Dict[int, np.ndarray]],
    grid_size: int = 5,
    obstacles: List[Tuple[int, int]] = None,
    goals: List[Tuple[int, int]] = None,
    save_path: Optional[str] = None,
) -> Optional[Any]:
    """
    Plot trajectory heatmaps showing cell visitation frequency.

    Args:
        cell_visits: Dict mapping regime -> agent_id -> visit counts
        grid_size: Size of the grid
        obstacles: List of obstacle positions
        goals: List of goal positions
        save_path: Optional path to save figure

    Returns:
        matplotlib figure
    """
    _check_matplotlib()

    obstacles = obstacles or []
    goals = goals or []

    num_regimes = len(cell_visits)
    num_agents = max(len(agents) for agents in cell_visits.values()) if cell_visits else 0

    if num_agents == 0:
        print("No trajectory data available")
        return None

    fig, axes = plt.subplots(num_regimes, num_agents, figsize=(5 * num_agents, 5 * num_regimes))

    if num_regimes == 1:
        axes = [axes]
    if num_agents == 1:
        axes = [[ax] for ax in axes]

    for i, (regime, agent_data) in enumerate(cell_visits.items()):
        for j, (agent_id, visits) in enumerate(sorted(agent_data.items())):
            ax = axes[i][j]

            # Plot heatmap
            im = ax.imshow(visits, cmap="Blues", origin="upper")
            plt.colorbar(im, ax=ax, label="Visit Count")

            # Mark obstacles
            for r, c in obstacles:
                rect = patches.Rectangle(
                    (c - 0.5, r - 0.5), 1, 1,
                    linewidth=2, edgecolor="black", facecolor="gray"
                )
                ax.add_patch(rect)

            # Mark goals
            if j < len(goals):
                gr, gc = goals[j]
                ax.plot(gc, gr, marker="*", markersize=15, color="gold", markeredgecolor="black")

            ax.set_title(f"{regime.upper()} - Agent {agent_id}")
            ax.set_xlabel("Column")
            ax.set_ylabel("Row")

            # Grid lines
            ax.set_xticks(np.arange(-0.5, grid_size, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, grid_size, 1), minor=True)
            ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved trajectory heatmap to: {save_path}")

    return fig


def plot_collision_heatmap(
    collision_locations: Dict[str, List[Tuple[int, int]]],
    grid_size: int = 5,
    obstacles: List[Tuple[int, int]] = None,
    save_path: Optional[str] = None,
) -> Optional[Any]:
    """
    Plot heatmap of collision locations.

    This visualization shows WHERE collisions occur - typically in the center
    where paths cross for MARL, and nowhere for RSCT-gated.

    Args:
        collision_locations: Dict mapping regime -> list of collision positions
        grid_size: Size of the grid
        obstacles: List of obstacle positions
        save_path: Optional path to save figure

    Returns:
        matplotlib figure
    """
    _check_matplotlib()

    obstacles = obstacles or []

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    regimes = [("marl", "MARL"), ("rsct_gated", "RSCT-Gated")]

    # Custom colormap: white to red
    colors = ["white", "yellow", "orange", "red", "darkred"]
    cmap = LinearSegmentedColormap.from_list("collision", colors)

    max_collisions = 1  # Avoid division by zero

    for regime, _ in regimes:
        locs = collision_locations.get(regime, [])
        if locs:
            heatmap = np.zeros((grid_size, grid_size))
            for r, c in locs:
                if 0 <= r < grid_size and 0 <= c < grid_size:
                    heatmap[r, c] += 1
            max_collisions = max(max_collisions, heatmap.max())

    for idx, (regime, title) in enumerate(regimes):
        ax = axes[idx]

        locs = collision_locations.get(regime, [])
        heatmap = np.zeros((grid_size, grid_size))
        for r, c in locs:
            if 0 <= r < grid_size and 0 <= c < grid_size:
                heatmap[r, c] += 1

        im = ax.imshow(heatmap, cmap=cmap, origin="upper", vmin=0, vmax=max_collisions)
        plt.colorbar(im, ax=ax, label="Collision Count")

        # Mark obstacles
        for r, c in obstacles:
            rect = patches.Rectangle(
                (c - 0.5, r - 0.5), 1, 1,
                linewidth=2, edgecolor="black", facecolor="gray"
            )
            ax.add_patch(rect)

        ax.set_title(f"{title}\nTotal: {int(heatmap.sum())} collisions", fontsize=14)
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")

        # Grid lines
        ax.set_xticks(np.arange(-0.5, grid_size, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, grid_size, 1), minor=True)
        ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5)

    plt.suptitle("Collision Locations: Learned vs Certified Safety", fontsize=16, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved collision heatmap to: {save_path}")

    return fig


def create_experiment_report(
    results: Dict[str, Any],
    output_dir: str = "results/figures",
) -> List[str]:
    """
    Generate all standard figures for experiment report.

    Args:
        results: Full results from ExperimentRunner.run_all()
        output_dir: Directory to save figures

    Returns:
        List of saved figure paths
    """
    _check_matplotlib()

    import os
    os.makedirs(output_dir, exist_ok=True)

    saved_paths = []
    plotting_data = results.get("plotting_data", {})
    config = results.get("config", {})
    grid_size = config.get("grid_size", 5)

    # 1. Collision comparison (the money shot)
    path = os.path.join(output_dir, "collision_comparison.png")
    plot_collision_comparison(plotting_data, save_path=path)
    saved_paths.append(path)

    # 2. Learning curves
    path = os.path.join(output_dir, "learning_curves.png")
    plot_learning_curves(plotting_data, save_path=path)
    saved_paths.append(path)

    # 3. Trajectory heatmaps
    cell_visits = plotting_data.get("cell_visits", {})
    if cell_visits:
        path = os.path.join(output_dir, "trajectory_heatmaps.png")
        plot_trajectory_heatmap(cell_visits, grid_size=grid_size, save_path=path)
        saved_paths.append(path)

    # 4. Collision heatmaps
    collision_locs = plotting_data.get("collision_locations", {})
    if collision_locs:
        path = os.path.join(output_dir, "collision_heatmap.png")
        plot_collision_heatmap(collision_locs, grid_size=grid_size, save_path=path)
        saved_paths.append(path)

    print(f"\nGenerated {len(saved_paths)} figures in {output_dir}/")
    return saved_paths
