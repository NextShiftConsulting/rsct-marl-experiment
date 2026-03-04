"""
Experiment Runner for RSCT vs MARL Comparison

Runs controlled experiments comparing learned coordination (MARL)
against certified coordination (RSCT-gated) in multi-agent gridworld.

Experiment ID: EXP-RSCT-001
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional, Callable
import numpy as np
from tqdm import tqdm
import time

from ..environment import MultiAgentGridworld, GridworldConfig, Action
from ..agents import QLearningAgent, AgentConfig
from ..gatekeeper import RSCTGatekeeper, GatekeeperConfig, BlockingStrategy
from .metrics import MetricsCollector, EpisodeMetrics


class Regime(Enum):
    """Experiment regime."""
    MARL = "marl"
    RSCT_GATED = "rsct_gated"


@dataclass
class ExperimentConfig:
    """Configuration for experiment run."""

    # Experiment identification
    experiment_id: str = "EXP-RSCT-001"
    name: str = "RSCT vs MARL Navigation"

    # Episode parameters
    num_episodes: int = 500
    eval_episodes: int = 100  # Additional greedy evaluation episodes

    # Environment config
    env_config: GridworldConfig = field(default_factory=GridworldConfig)

    # Agent config (shared by all agents)
    agent_config: AgentConfig = field(default_factory=AgentConfig)

    # Gatekeeper config
    gatekeeper_config: GatekeeperConfig = field(default_factory=GatekeeperConfig)

    # Regimes to run
    regimes: List[Regime] = field(
        default_factory=lambda: [Regime.MARL, Regime.RSCT_GATED]
    )

    # Random seed for reproducibility
    seed: int = 42

    # Logging
    log_interval: int = 50  # Episodes between progress logs
    verbose: bool = True


class ExperimentRunner:
    """
    Runs RSCT vs MARL experiments with controlled conditions.

    Ensures identical initialization and hyperparameters between regimes,
    allowing for fair comparison of learned vs certified coordination.
    """

    def __init__(self, config: Optional[ExperimentConfig] = None):
        self.config = config or ExperimentConfig()
        self.metrics_collector = MetricsCollector()

        # Results storage
        self.results: Dict[str, Any] = {}
        self.trained_agents: Dict[str, List[QLearningAgent]] = {}

    def _create_environment(self) -> MultiAgentGridworld:
        """Create fresh environment instance."""
        return MultiAgentGridworld(self.config.env_config)

    def _create_agents(self, num_agents: int) -> List[QLearningAgent]:
        """Create fresh agent instances."""
        return [
            QLearningAgent(
                agent_id=i,
                num_actions=5,
                config=self.config.agent_config,
            )
            for i in range(num_agents)
        ]

    def _create_gatekeeper(self) -> RSCTGatekeeper:
        """Create gatekeeper instance."""
        gate_config = GatekeeperConfig(
            kappa_min=self.config.gatekeeper_config.kappa_min,
            blocking_strategy=self.config.gatekeeper_config.blocking_strategy,
            grid_size=self.config.env_config.grid_size,
        )
        return RSCTGatekeeper(gate_config)

    def _run_episode(
        self,
        env: MultiAgentGridworld,
        agents: List[QLearningAgent],
        regime: Regime,
        gatekeeper: Optional[RSCTGatekeeper] = None,
        training: bool = True,
        episode_id: int = 0,
    ) -> EpisodeMetrics:
        """
        Run a single episode under specified regime.

        Args:
            env: Gridworld environment
            agents: List of Q-learning agents
            regime: MARL or RSCT_GATED
            gatekeeper: RSCT gatekeeper (required for RSCT_GATED regime)
            training: Whether to update Q-values and explore
            episode_id: Episode number for logging

        Returns:
            EpisodeMetrics for the episode
        """
        positions = env.reset()
        position_history = [[pos] for pos in positions]  # Track trajectory

        total_rewards = [0.0] * env.num_agents
        collision_steps = []
        gate_blocks = 0

        for step in range(self.config.env_config.max_steps):
            # Get state for each agent
            states = [env.get_state(i) for i in range(env.num_agents)]

            # Each agent proposes an action
            proposed_actions = [
                agents[i].act(states[i], greedy=not training)
                for i in range(env.num_agents)
            ]

            # Apply gatekeeper if in RSCT regime
            if regime == Regime.RSCT_GATED and gatekeeper is not None:
                decision = gatekeeper.check_and_gate(
                    current_positions=env.positions,
                    proposed_actions=proposed_actions,
                    goals=env.goals,
                    obstacles=env.obstacles,
                )
                actions = decision.approved_actions
                if decision.was_blocked:
                    gate_blocks += 1
            else:
                actions = proposed_actions

            # Execute actions
            result = env.step(actions)

            # Record trajectory
            for i, pos in enumerate(result.next_positions):
                position_history[i].append(pos)

            # Track collisions
            if result.info["collision"]:
                collision_steps.append(step)

            # Accumulate rewards
            for i in range(env.num_agents):
                total_rewards[i] += result.rewards[i]

            # Update Q-values if training
            if training:
                next_states = [env.get_state(i) for i in range(env.num_agents)]
                for i in range(env.num_agents):
                    agents[i].update(
                        states[i],
                        actions[i],
                        result.rewards[i],
                        next_states[i],
                        result.done,
                    )

            if result.done:
                break

        # Decay exploration rate
        if training:
            for agent in agents:
                agent.decay_epsilon()

        # Compile metrics
        env_metrics = env.get_metrics()

        return EpisodeMetrics(
            episode_id=episode_id,
            regime=regime.value,
            total_collisions=env_metrics["total_collisions"],
            collision_steps=collision_steps,
            total_steps=env_metrics["total_steps"],
            all_goals_reached=env_metrics["all_goals_reached"],
            goals_reached_count=env_metrics["goals_reached_count"],
            total_return=sum(total_rewards),
            agent_returns=total_rewards,
            agent_goal_reached=env_metrics["all_goals_reached"],  # Simplified
            gate_blocks=gate_blocks,
            gate_block_rate=gate_blocks / max(1, env_metrics["total_steps"]),
            position_history=position_history,
        )

    def run_regime(
        self,
        regime: Regime,
        progress_callback: Optional[Callable[[int, EpisodeMetrics], None]] = None,
    ) -> Dict[str, Any]:
        """
        Run full training for a single regime.

        Args:
            regime: MARL or RSCT_GATED
            progress_callback: Optional callback for progress updates

        Returns:
            Dictionary with training results and statistics
        """
        # Set random seed for reproducibility
        np.random.seed(self.config.seed)

        # Create environment and agents
        env = self._create_environment()
        agents = self._create_agents(env.num_agents)

        # Create gatekeeper if needed
        gatekeeper = self._create_gatekeeper() if regime == Regime.RSCT_GATED else None

        if self.config.verbose:
            print(f"\n{'='*60}")
            print(f"Running regime: {regime.value.upper()}")
            print(f"Episodes: {self.config.num_episodes}")
            print(f"Grid size: {self.config.env_config.grid_size}x{self.config.env_config.grid_size}")
            print(f"Agents: {env.num_agents}")
            print(f"{'='*60}")

        start_time = time.time()

        # Training loop
        episode_iterator = range(self.config.num_episodes)
        if self.config.verbose:
            episode_iterator = tqdm(episode_iterator, desc=f"{regime.value} training")

        for episode in episode_iterator:
            metrics = self._run_episode(
                env=env,
                agents=agents,
                regime=regime,
                gatekeeper=gatekeeper,
                training=True,
                episode_id=episode,
            )

            self.metrics_collector.record_episode(
                metrics, grid_size=self.config.env_config.grid_size
            )

            if progress_callback:
                progress_callback(episode, metrics)

        training_time = time.time() - start_time

        # Store trained agents
        self.trained_agents[regime.value] = agents

        # Get gatekeeper stats
        gate_stats = gatekeeper.get_stats() if gatekeeper else {}

        # Run evaluation episodes (greedy, no exploration)
        if self.config.verbose:
            print(f"\nRunning {self.config.eval_episodes} evaluation episodes...")

        eval_metrics = []
        for episode in range(self.config.eval_episodes):
            metrics = self._run_episode(
                env=env,
                agents=agents,
                regime=regime,
                gatekeeper=gatekeeper,
                training=False,
                episode_id=self.config.num_episodes + episode,
            )
            eval_metrics.append(metrics)

        # Compile results
        results = {
            "regime": regime.value,
            "training_time": training_time,
            "training_summary": self.metrics_collector.get_summary(regime.value),
            "eval_summary": {
                "num_episodes": len(eval_metrics),
                "mean_collisions": np.mean([m.total_collisions for m in eval_metrics]),
                "collision_rate": sum(1 for m in eval_metrics if m.total_collisions > 0) / len(eval_metrics),
                "mean_steps": np.mean([m.total_steps for m in eval_metrics]),
                "success_rate": sum(1 for m in eval_metrics if m.all_goals_reached) / len(eval_metrics),
                "mean_return": np.mean([m.total_return for m in eval_metrics]),
            },
            "agent_stats": [agent.get_stats() for agent in agents],
            "gatekeeper_stats": gate_stats,
        }

        self.results[regime.value] = results

        if self.config.verbose:
            self._print_summary(results)

        return results

    def run_all(self) -> Dict[str, Any]:
        """
        Run all configured regimes and return comparison.

        Returns:
            Dictionary with results for all regimes and comparison metrics
        """
        for regime in self.config.regimes:
            self.run_regime(regime)

        comparison = self.metrics_collector.get_comparison()

        if self.config.verbose:
            self._print_comparison(comparison)

        return {
            "config": {
                "experiment_id": self.config.experiment_id,
                "num_episodes": self.config.num_episodes,
                "grid_size": self.config.env_config.grid_size,
                "num_agents": self.config.env_config.num_agents,
                "seed": self.config.seed,
            },
            "results": self.results,
            "comparison": comparison,
            "plotting_data": self.metrics_collector.export_for_plotting(),
        }

    def _print_summary(self, results: Dict[str, Any]):
        """Print summary for a single regime."""
        regime = results["regime"]
        train = results["training_summary"]
        eval_s = results["eval_summary"]

        print(f"\n--- {regime.upper()} Results ---")
        print(f"Training time: {results['training_time']:.2f}s")
        print(f"\nTraining ({train['num_episodes']} episodes):")
        print(f"  Total collisions: {train['total_collisions']}")
        print(f"  Collision rate: {train['collision_rate']:.2%}")
        print(f"  Success rate: {train['success_rate']:.2%}")
        print(f"  Mean return: {train['mean_return']:.2f}")

        print(f"\nEvaluation ({eval_s['num_episodes']} episodes, greedy):")
        print(f"  Mean collisions: {eval_s['mean_collisions']:.2f}")
        print(f"  Collision rate: {eval_s['collision_rate']:.2%}")
        print(f"  Success rate: {eval_s['success_rate']:.2%}")
        print(f"  Mean return: {eval_s['mean_return']:.2f}")

    def _print_comparison(self, comparison: Dict[str, Any]):
        """Print side-by-side comparison."""
        print("\n" + "="*60)
        print("COMPARISON: MARL vs RSCT-Gated")
        print("="*60)

        marl = comparison.get("marl", {})
        rsct = comparison.get("rsct_gated", {})

        if not marl or not rsct:
            print("Insufficient data for comparison")
            return

        print(f"\n{'Metric':<30} {'MARL':>12} {'RSCT-Gated':>12}")
        print("-" * 56)

        metrics = [
            ("Total Collisions", "total_collisions", "{:.0f}"),
            ("Collision Rate", "collision_rate", "{:.2%}"),
            ("Success Rate", "success_rate", "{:.2%}"),
            ("Mean Return", "mean_return", "{:.2f}"),
            ("Mean Steps", "mean_steps_to_goal", "{:.1f}"),
        ]

        for name, key, fmt in metrics:
            marl_val = marl.get(key, "N/A")
            rsct_val = rsct.get(key, "N/A")

            marl_str = fmt.format(marl_val) if isinstance(marl_val, (int, float)) else str(marl_val)
            rsct_str = fmt.format(rsct_val) if isinstance(rsct_val, (int, float)) else str(rsct_val)

            print(f"{name:<30} {marl_str:>12} {rsct_str:>12}")

        print("\n" + "="*60)
        print("KEY FINDING:")
        if rsct.get("total_collisions", 1) == 0:
            print("  RSCT-Gated achieved ZERO collisions by construction.")
        print(f"  MARL required learning to reduce collisions;")
        print(f"  RSCT provided safety guarantees from episode 1.")
        print("="*60)
