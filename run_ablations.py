#!/usr/bin/env python3
"""
Ablation Studies for RSCT Gatekeeper

Tests:
1. kappa_min sweep: How does safety margin affect performance?
2. Blocking strategy comparison: BOTH_STAY vs PRIORITY_BY_DISTANCE
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
from datetime import datetime
import numpy as np

from src.environment import GridworldConfig
from src.agents import AgentConfig
from src.gatekeeper import GatekeeperConfig, BlockingStrategy
from src.experiments import ExperimentRunner, ExperimentConfig, Regime


def run_kappa_ablation():
    """Sweep kappa_min from 0.0 to 0.5"""
    print("\n" + "="*60)
    print("ABLATION: kappa_min sweep")
    print("="*60)

    kappa_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    results = []

    for kappa in kappa_values:
        print(f"\n--- kappa_min = {kappa} ---")

        config = ExperimentConfig(
            experiment_id=f"ABLATION-KAPPA-{kappa}",
            num_episodes=200,
            eval_episodes=50,
            env_config=GridworldConfig.small(),
            gatekeeper_config=GatekeeperConfig(
                kappa_min=kappa,
                blocking_strategy=BlockingStrategy.PRIORITY_BY_DISTANCE,
                grid_size=5,
            ),
            regimes=[Regime.RSCT_GATED],
            verbose=False,
        )

        runner = ExperimentRunner(config)
        run_results = runner.run_all()

        rsct = run_results["comparison"].get("rsct_gated", {})

        result = {
            "kappa_min": kappa,
            "collisions": rsct.get("total_collisions", 0),
            "collision_rate": rsct.get("collision_rate", 0),
            "success_rate": rsct.get("success_rate", 0),
            "mean_return": rsct.get("mean_return", 0),
            "gate_blocks": runner.results.get("rsct_gated", {}).get("gatekeeper_stats", {}).get("total_blocks", 0),
        }
        results.append(result)

        print(f"  Collisions: {result['collisions']}")
        print(f"  Gate blocks: {result['gate_blocks']}")
        print(f"  Mean return: {result['mean_return']:.2f}")

    return results


def run_strategy_ablation():
    """Compare blocking strategies"""
    print("\n" + "="*60)
    print("ABLATION: Blocking Strategy Comparison")
    print("="*60)

    strategies = [
        (BlockingStrategy.BOTH_STAY, "BOTH_STAY"),
        (BlockingStrategy.PRIORITY_BY_DISTANCE, "PRIORITY_BY_DISTANCE"),
        (BlockingStrategy.PRIORITY_BY_ID, "PRIORITY_BY_ID"),
    ]

    results = []

    for strategy, name in strategies:
        print(f"\n--- Strategy: {name} ---")

        config = ExperimentConfig(
            experiment_id=f"ABLATION-STRATEGY-{name}",
            num_episodes=200,
            eval_episodes=50,
            env_config=GridworldConfig.small(),
            gatekeeper_config=GatekeeperConfig(
                kappa_min=0.0,  # No kappa threshold, just collision avoidance
                blocking_strategy=strategy,
                grid_size=5,
            ),
            regimes=[Regime.RSCT_GATED],
            verbose=False,
        )

        runner = ExperimentRunner(config)
        run_results = runner.run_all()

        rsct = run_results["comparison"].get("rsct_gated", {})

        result = {
            "strategy": name,
            "collisions": rsct.get("total_collisions", 0),
            "collision_rate": rsct.get("collision_rate", 0),
            "success_rate": rsct.get("success_rate", 0),
            "mean_return": rsct.get("mean_return", 0),
            "mean_steps": rsct.get("mean_steps_to_goal", None),
        }
        results.append(result)

        print(f"  Collisions: {result['collisions']}")
        print(f"  Success rate: {result['success_rate']*100:.1f}%")
        print(f"  Mean return: {result['mean_return']:.2f}")

    return results


def run_scaling_summary():
    """Load and summarize scaling experiment results"""
    print("\n" + "="*60)
    print("SCALING EXPERIMENT SUMMARY")
    print("="*60)

    scales = ["small", "medium", "large"]
    results = []

    for scale in scales:
        # Find the most recent result file for this scale
        data_dir = "results/data"
        files = [f for f in os.listdir(data_dir) if f.startswith(f"full_{scale}")]
        if not files:
            continue

        latest = sorted(files)[-1]
        with open(os.path.join(data_dir, latest)) as f:
            data = json.load(f)

        config = data.get("config", {})
        comp = data.get("comparison", {})
        marl = comp.get("marl", {})
        rsct = comp.get("rsct_gated", {})

        result = {
            "scale": scale,
            "grid_size": config.get("grid_size"),
            "num_agents": config.get("num_agents"),
            "episodes": config.get("num_episodes"),
            "marl_collisions": marl.get("total_collisions", 0),
            "rsct_collisions": rsct.get("total_collisions", 0),
            "marl_collision_rate": marl.get("collision_rate", 0),
            "rsct_collision_rate": rsct.get("collision_rate", 0),
            "marl_return": marl.get("mean_return", 0),
            "rsct_return": rsct.get("mean_return", 0),
        }
        results.append(result)

    # Print summary table
    print(f"\n{'Scale':<10} {'Grid':<8} {'Agents':<8} {'MARL Coll':<12} {'RSCT Coll':<12} {'MARL Rate':<12} {'RSCT Rate':<12}")
    print("-" * 80)
    for r in results:
        print(f"{r['scale']:<10} {r['grid_size']}x{r['grid_size']:<4} {r['num_agents']:<8} "
              f"{r['marl_collisions']:<12} {r['rsct_collisions']:<12} "
              f"{r['marl_collision_rate']*100:.1f}%{'':6} {r['rsct_collision_rate']*100:.1f}%")

    return results


def main():
    print("\n" + "="*60)
    print("RSCT GATEKEEPER ABLATION STUDIES")
    print("="*60)

    all_results = {}

    # Run ablations
    all_results["kappa_sweep"] = run_kappa_ablation()
    all_results["strategy_comparison"] = run_strategy_ablation()
    all_results["scaling_summary"] = run_scaling_summary()

    # Save results
    os.makedirs("results/data", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = f"results/data/ablations_{timestamp}.json"

    with open(filepath, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n\nAblation results saved to: {filepath}")

    # Print final summary
    print("\n" + "="*60)
    print("ABLATION STUDY COMPLETE")
    print("="*60)

    print("\nKappa Sweep Results:")
    print(f"{'kappa':<10} {'Collisions':<12} {'Gate Blocks':<12} {'Return':<12}")
    print("-" * 46)
    for r in all_results["kappa_sweep"]:
        print(f"{r['kappa_min']:<10} {r['collisions']:<12} {r['gate_blocks']:<12} {r['mean_return']:.2f}")

    print("\nStrategy Comparison:")
    print(f"{'Strategy':<25} {'Collisions':<12} {'Success':<12} {'Return':<12}")
    print("-" * 51)
    for r in all_results["strategy_comparison"]:
        print(f"{r['strategy']:<25} {r['collisions']:<12} {r['success_rate']*100:.1f}%{'':5} {r['mean_return']:.2f}")


if __name__ == "__main__":
    main()
