#!/usr/bin/env python3
"""
RSCT vs MARL Gridworld Experiment

EXP-RSCT-001: Gated vs MARL Navigation

Run this script to execute the full experiment comparing:
1. Pure MARL: Q-learning agents with collision penalties (learned coordination)
2. RSCT-Gated: Same Q-learners with static safety gate (certified coordination)

Usage:
    python run_experiment.py                    # Full experiment
    python run_experiment.py --quick            # Quick test (100 episodes)
    python run_experiment.py --verify           # Run formal verification only
    python run_experiment.py --scale medium     # Run on larger grid
"""

import argparse
import json
import os
import sys
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.environment import GridworldConfig
from src.agents import AgentConfig
from src.gatekeeper import GatekeeperConfig, BlockingStrategy
from src.experiments import ExperimentRunner, ExperimentConfig, Regime
from src.theory import verify_all_theorems


def run_quick_test():
    """Run a quick test with 100 episodes."""
    print("\n" + "="*60)
    print("QUICK TEST: 100 episodes, 5x5 grid, 2 agents")
    print("="*60)

    config = ExperimentConfig(
        experiment_id="EXP-RSCT-001-QUICK",
        num_episodes=100,
        eval_episodes=20,
        verbose=True,
    )

    runner = ExperimentRunner(config)
    results = runner.run_all()

    return results


def run_full_experiment(scale: str = "small"):
    """Run full experiment with specified scale."""
    print("\n" + "="*60)
    print(f"FULL EXPERIMENT: Scale={scale}")
    print("="*60)

    # Configure environment based on scale
    if scale == "small":
        env_config = GridworldConfig.small()
        num_episodes = 500
    elif scale == "medium":
        env_config = GridworldConfig.medium()
        num_episodes = 1000
    elif scale == "large":
        env_config = GridworldConfig.large()
        num_episodes = 2000
    else:
        raise ValueError(f"Unknown scale: {scale}")

    config = ExperimentConfig(
        experiment_id=f"EXP-RSCT-001-{scale.upper()}",
        num_episodes=num_episodes,
        eval_episodes=100,
        env_config=env_config,
        agent_config=AgentConfig(
            learning_rate=0.1,
            discount_factor=0.95,
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_decay=0.995,
        ),
        gatekeeper_config=GatekeeperConfig(
            kappa_min=0.2,
            blocking_strategy=BlockingStrategy.PRIORITY_BY_DISTANCE,
            grid_size=env_config.grid_size,
        ),
        verbose=True,
    )

    runner = ExperimentRunner(config)
    results = runner.run_all()

    # Generate figures
    try:
        from src.visualization import create_experiment_report
        output_dir = f"results/figures/{scale}"
        create_experiment_report(results, output_dir)
    except ImportError:
        print("\nMatplotlib not available - skipping figure generation")
        print("Install with: pip install matplotlib")

    return results


def run_formal_verification():
    """Run formal theorem verification."""
    print("\n" + "="*60)
    print("FORMAL VERIFICATION")
    print("="*60)

    # Verify for different grid sizes
    for grid_size in [3, 5]:
        for num_agents in [2, 3] if grid_size <= 4 else [2]:
            print(f"\n--- Grid {grid_size}x{grid_size}, {num_agents} agents ---")
            verify_all_theorems(grid_size=grid_size, num_agents=num_agents)


def save_results(results: dict, name: str):
    """Save results to JSON file."""
    os.makedirs("results/data", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = f"results/data/{name}_{timestamp}.json"

    # Convert non-serializable objects
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        elif hasattr(obj, "__dict__"):
            return make_serializable(obj.__dict__)
        else:
            return str(obj)

    serializable = make_serializable(results)

    with open(filepath, "w") as f:
        json.dump(serializable, f, indent=2)

    print(f"\nResults saved to: {filepath}")
    return filepath


def main():
    parser = argparse.ArgumentParser(
        description="RSCT vs MARL Gridworld Experiment"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Run quick test (100 episodes)"
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Run formal verification only"
    )
    parser.add_argument(
        "--scale", choices=["small", "medium", "large"], default="small",
        help="Experiment scale (default: small)"
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Don't save results to file"
    )

    args = parser.parse_args()

    if args.verify:
        run_formal_verification()
        return

    if args.quick:
        results = run_quick_test()
        name = "quick_test"
    else:
        results = run_full_experiment(args.scale)
        name = f"full_{args.scale}"

    if not args.no_save:
        save_results(results, name)

    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE")
    print("="*60)

    # Print key findings
    comparison = results.get("comparison", {})
    marl = comparison.get("marl", {})
    rsct = comparison.get("rsct_gated", {})

    if marl and rsct:
        print("\nKEY FINDINGS:")
        print(f"  MARL collisions: {marl.get('total_collisions', 'N/A')}")
        print(f"  RSCT collisions: {rsct.get('total_collisions', 'N/A')}")
        print(f"\n  MARL success rate: {marl.get('success_rate', 0):.1%}")
        print(f"  RSCT success rate: {rsct.get('success_rate', 0):.1%}")


if __name__ == "__main__":
    main()
