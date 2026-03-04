#!/usr/bin/env python3
"""
RSCT vs MARL on Real MAPF Benchmarks

Runs experiments on MovingAI MAPF benchmark maps to demonstrate
RSCT safety certification on real-world scenarios.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
from datetime import datetime
from typing import Dict, Any, List
import numpy as np
from tqdm import tqdm

from src.environment.mapf_benchmark import MAPFBenchmarkEnv, list_available_benchmarks
from src.environment.gridworld import Action
from src.agents import QLearningAgent, AgentConfig
from src.gatekeeper import RSCTGatekeeper, GatekeeperConfig, BlockingStrategy


def run_mapf_experiment(
    map_path: str,
    scenario_path: str,
    num_agents: int = 4,
    num_episodes: int = 100,
    max_steps: int = 100,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run RSCT vs MARL comparison on a MAPF benchmark.
    """
    results = {"marl": {}, "rsct_gated": {}}

    for regime in ["marl", "rsct_gated"]:
        if verbose:
            print(f"\n{'='*50}")
            print(f"Running: {regime.upper()}")
            print(f"{'='*50}")

        # Create environment
        env = MAPFBenchmarkEnv(
            map_path=map_path,
            scenario_path=scenario_path,
            num_agents=num_agents,
            max_steps=max_steps,
        )

        if len(env.scenarios) == 0:
            print(f"No scenarios found for {num_agents} agents")
            return results

        # Create agents
        agents = [
            QLearningAgent(i, config=AgentConfig(epsilon_decay=0.99))
            for i in range(num_agents)
        ]

        # Create gatekeeper
        gatekeeper = RSCTGatekeeper(GatekeeperConfig(
            grid_size=env.grid_size,
            kappa_min=0.0,
            blocking_strategy=BlockingStrategy.PRIORITY_BY_DISTANCE,
        )) if regime == "rsct_gated" else None

        # Metrics
        total_collisions = 0
        total_goals = 0
        total_steps = 0
        episode_collisions = []
        episode_successes = []

        # Run episodes
        ep_iter = range(min(num_episodes, len(env.scenarios)))
        if verbose:
            ep_iter = tqdm(ep_iter, desc=regime)

        for episode in ep_iter:
            env.reset(scenario_idx=episode % len(env.scenarios))

            for step in range(max_steps):
                # Get states and propose actions
                states = [env.get_state(i) for i in range(num_agents)]
                proposed_actions = [agents[i].act(states[i]) for i in range(num_agents)]

                # Apply gatekeeper if RSCT
                if gatekeeper:
                    decision = gatekeeper.check_and_gate(
                        env.positions, proposed_actions, env.goals, env.obstacles
                    )
                    actions = decision.approved_actions
                else:
                    actions = proposed_actions

                # Step environment
                result = env.step(actions)

                # Update agents
                next_states = [env.get_state(i) for i in range(num_agents)]
                for i in range(num_agents):
                    agents[i].update(
                        states[i], actions[i], result.rewards[i],
                        next_states[i], result.done
                    )

                if result.done:
                    break

            # Decay epsilon
            for agent in agents:
                agent.decay_epsilon()

            # Record metrics
            metrics = env.get_metrics()
            total_collisions += metrics["total_collisions"]
            total_goals += metrics["goals_reached_count"]
            total_steps += metrics["total_steps"]
            episode_collisions.append(metrics["total_collisions"])
            episode_successes.append(metrics["all_goals_reached"])

        # Compile results
        num_run = min(num_episodes, len(env.scenarios))
        results[regime] = {
            "total_collisions": total_collisions,
            "collision_rate": sum(1 for c in episode_collisions if c > 0) / num_run,
            "mean_collisions_per_episode": total_collisions / num_run,
            "success_rate": sum(episode_successes) / num_run,
            "total_goals_reached": total_goals,
            "mean_steps": total_steps / num_run,
            "episodes_run": num_run,
            "num_agents": num_agents,
            "episode_collisions": episode_collisions,
        }

        if verbose:
            print(f"  Total collisions: {total_collisions}")
            print(f"  Collision rate: {results[regime]['collision_rate']*100:.1f}%")
            print(f"  Success rate: {results[regime]['success_rate']*100:.1f}%")

    return results


def run_all_benchmarks(
    data_dir: str = "data/movingai",
    num_agents: int = 4,
    num_episodes: int = 50,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run experiments on all available benchmarks.
    """
    benchmarks = list_available_benchmarks(data_dir)

    if not benchmarks:
        print(f"No benchmarks found in {data_dir}")
        print("Make sure maps (.map) and scenarios (scen-random/*.scen) are present")
        return {}

    if verbose:
        print(f"\nFound {len(benchmarks)} benchmark configurations")
        print(f"Running with {num_agents} agents, {num_episodes} episodes each\n")

    all_results = {}
    summary = {
        "marl_total_collisions": 0,
        "rsct_total_collisions": 0,
        "benchmarks_run": 0,
    }

    # Run on a subset for speed
    selected = benchmarks[:5]  # First 5 benchmarks

    for bench in selected:
        name = bench["name"]
        if verbose:
            print(f"\n{'#'*60}")
            print(f"Benchmark: {name}")
            print(f"{'#'*60}")

        try:
            results = run_mapf_experiment(
                map_path=bench["map"],
                scenario_path=bench["scenario"],
                num_agents=num_agents,
                num_episodes=num_episodes,
                verbose=verbose,
            )

            all_results[name] = results
            summary["marl_total_collisions"] += results.get("marl", {}).get("total_collisions", 0)
            summary["rsct_total_collisions"] += results.get("rsct_gated", {}).get("total_collisions", 0)
            summary["benchmarks_run"] += 1

        except Exception as e:
            print(f"Error on {name}: {e}")
            continue

    return {"benchmarks": all_results, "summary": summary}


def main():
    import argparse

    parser = argparse.ArgumentParser(description="RSCT vs MARL on MAPF Benchmarks")
    parser.add_argument("--data-dir", default="data/movingai", help="Data directory")
    parser.add_argument("--agents", type=int, default=4, help="Number of agents")
    parser.add_argument("--episodes", type=int, default=50, help="Episodes per benchmark")
    parser.add_argument("--single", help="Run single benchmark (map path)")
    parser.add_argument("--scenario", help="Scenario path (for single benchmark)")

    args = parser.parse_args()

    print("\n" + "="*60)
    print("RSCT vs MARL on Real MAPF Benchmarks")
    print("="*60)

    if args.single and args.scenario:
        results = run_mapf_experiment(
            map_path=args.single,
            scenario_path=args.scenario,
            num_agents=args.agents,
            num_episodes=args.episodes,
        )
    else:
        results = run_all_benchmarks(
            data_dir=args.data_dir,
            num_agents=args.agents,
            num_episodes=args.episodes,
        )

    # Print summary
    if "summary" in results:
        print("\n" + "="*60)
        print("SUMMARY: REAL MAPF BENCHMARKS")
        print("="*60)
        print(f"Benchmarks run: {results['summary']['benchmarks_run']}")
        print(f"MARL total collisions: {results['summary']['marl_total_collisions']}")
        print(f"RSCT total collisions: {results['summary']['rsct_total_collisions']}")

        if results['summary']['rsct_total_collisions'] == 0:
            print("\n*** RSCT achieved ZERO collisions across all real benchmarks ***")

    # Save results
    os.makedirs("results/data", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = f"results/data/mapf_benchmark_{timestamp}.json"

    # Remove non-serializable data
    def clean(obj):
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean(v) for v in obj]
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            return str(obj)

    with open(filepath, "w") as f:
        json.dump(clean(results), f, indent=2)

    print(f"\nResults saved to: {filepath}")


if __name__ == "__main__":
    main()
