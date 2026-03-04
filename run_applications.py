#!/usr/bin/env python3
"""
Real-World Application Simulations

Compares RSCT vs MARL on:
1. Drone Swarm Coordination ($50K/drone)
2. Military Convoy Operations ($500K/vehicle)
3. Warehouse Robot Logistics ($30K/robot)

Shows financial impact of safety certification.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from tqdm import tqdm
from typing import Dict, Any, List

from src.applications.drone_swarm import DroneSwarmEnv, DroneSwarmConfig
from src.applications.military_convoy import MilitaryConvoyEnv, ConvoyConfig
from src.applications.warehouse_robots import WarehouseEnv, WarehouseConfig
from src.agents import QLearningAgent, AgentConfig
from src.gatekeeper import RSCTGatekeeper, GatekeeperConfig, BlockingStrategy
from src.environment.gridworld import Action


def run_drone_swarm_experiment(num_episodes: int = 50, verbose: bool = True) -> Dict[str, Any]:
    """Run drone swarm comparison."""
    if verbose:
        print("\n" + "="*60)
        print("🚁 DRONE SWARM SIMULATION")
        print("   8 drones, target tracking mission")
        print("   Cost per drone: $50,000")
        print("="*60)

    config = DroneSwarmConfig(
        airspace_size=30,
        num_drones=8,
        num_targets=4,
        max_steps=150,
        drone_cost=50000,
    )

    results = {"marl": {}, "rsct": {}}

    for regime in ["marl", "rsct"]:
        env = DroneSwarmEnv(config)
        agents = [QLearningAgent(i, config=AgentConfig(epsilon_decay=0.99))
                  for i in range(config.num_drones)]

        gatekeeper = None
        if regime == "rsct":
            gatekeeper = RSCTGatekeeper(GatekeeperConfig(
                grid_size=config.airspace_size,
                kappa_min=0.0,
                blocking_strategy=BlockingStrategy.PRIORITY_BY_DISTANCE,
            ))

        total_collisions = 0
        total_drones_lost = 0
        total_targets_tracked = 0

        ep_iter = range(num_episodes)
        if verbose:
            ep_iter = tqdm(ep_iter, desc=f"  {regime.upper()}")

        for episode in ep_iter:
            env.reset()

            for step in range(config.max_steps):
                states = [env.get_state(i) for i in range(config.num_drones)]
                actions = [agents[i].act(states[i]) for i in range(config.num_drones)]

                if gatekeeper:
                    # Adapt positions/goals format
                    decision = gatekeeper.check_and_gate(
                        env.positions, actions, env.goals, env.obstacles
                    )
                    actions = decision.approved_actions

                result = env.step(actions)

                # Update agents
                next_states = [env.get_state(i) for i in range(config.num_drones)]
                for i in range(config.num_drones):
                    agents[i].update(states[i], actions[i], result.rewards[i],
                                    next_states[i], result.done)

                if result.done:
                    break

            for agent in agents:
                agent.decay_epsilon()

            metrics = env.get_metrics()
            total_collisions += metrics["total_collisions"]
            total_drones_lost += metrics["drones_lost"]
            total_targets_tracked += metrics["targets_tracked"]

        results[regime] = {
            "total_collisions": total_collisions,
            "drones_lost": total_drones_lost,
            "financial_loss": total_drones_lost * config.drone_cost,
            "targets_tracked": total_targets_tracked,
            "episodes": num_episodes,
        }

        if verbose:
            print(f"    Collisions: {total_collisions}")
            print(f"    Drones Lost: {total_drones_lost}")
            print(f"    💰 Financial Loss: ${total_drones_lost * config.drone_cost:,}")

    return results


def run_convoy_experiment(num_episodes: int = 30, verbose: bool = True) -> Dict[str, Any]:
    """Run military convoy comparison."""
    if verbose:
        print("\n" + "="*60)
        print("🚛 MILITARY CONVOY SIMULATION")
        print("   6 vehicles, supply route mission")
        print("   Cost per vehicle: $500,000")
        print("="*60)

    config = ConvoyConfig(
        route_length=60,
        route_width=15,
        num_vehicles=6,
        num_threats=5,
        max_steps=200,
        vehicle_cost=500000,
    )

    results = {"marl": {}, "rsct": {}}

    for regime in ["marl", "rsct"]:
        env = MilitaryConvoyEnv(config)
        agents = [QLearningAgent(i, config=AgentConfig(epsilon_decay=0.98))
                  for i in range(config.num_vehicles)]

        gatekeeper = None
        if regime == "rsct":
            gatekeeper = RSCTGatekeeper(GatekeeperConfig(
                grid_size=max(config.route_length, config.route_width),
                kappa_min=0.0,
                blocking_strategy=BlockingStrategy.PRIORITY_BY_DISTANCE,
            ))

        total_collisions = 0
        total_vehicles_lost = 0
        missions_complete = 0

        ep_iter = range(num_episodes)
        if verbose:
            ep_iter = tqdm(ep_iter, desc=f"  {regime.upper()}")

        for episode in ep_iter:
            env.reset()

            for step in range(config.max_steps):
                states = [env.get_state(i) for i in range(config.num_vehicles)]
                actions = [agents[i].act(states[i]) for i in range(config.num_vehicles)]

                if gatekeeper:
                    decision = gatekeeper.check_and_gate(
                        env.positions, actions, env.goals, env.obstacles
                    )
                    actions = decision.approved_actions

                result = env.step(actions)

                next_states = [env.get_state(i) for i in range(config.num_vehicles)]
                for i in range(config.num_vehicles):
                    agents[i].update(states[i], actions[i], result.rewards[i],
                                    next_states[i], result.done)

                if result.done:
                    break

            for agent in agents:
                agent.decay_epsilon()

            metrics = env.get_metrics()
            total_collisions += metrics["total_collisions"]
            total_vehicles_lost += metrics["vehicles_lost"]
            if metrics["mission_complete"]:
                missions_complete += 1

        results[regime] = {
            "total_collisions": total_collisions,
            "vehicles_lost": total_vehicles_lost,
            "financial_loss": total_vehicles_lost * config.vehicle_cost,
            "missions_complete": missions_complete,
            "mission_success_rate": missions_complete / num_episodes,
            "episodes": num_episodes,
        }

        if verbose:
            print(f"    Collisions: {total_collisions}")
            print(f"    Vehicles Lost: {total_vehicles_lost}")
            print(f"    💰 Financial Loss: ${total_vehicles_lost * config.vehicle_cost:,}")
            print(f"    Mission Success: {missions_complete}/{num_episodes}")

    return results


def run_warehouse_experiment(num_episodes: int = 30, verbose: bool = True) -> Dict[str, Any]:
    """Run warehouse robot comparison."""
    if verbose:
        print("\n" + "="*60)
        print("📦 WAREHOUSE ROBOT SIMULATION")
        print("   12 robots, order fulfillment")
        print("   Cost per robot: $30,000")
        print("="*60)

    config = WarehouseConfig(
        warehouse_size=30,
        num_robots=12,
        max_steps=300,
        robot_cost=30000,
    )

    results = {"marl": {}, "rsct": {}}

    for regime in ["marl", "rsct"]:
        env = WarehouseEnv(config)
        agents = [QLearningAgent(i, config=AgentConfig(epsilon_decay=0.99))
                  for i in range(config.num_robots)]

        gatekeeper = None
        if regime == "rsct":
            gatekeeper = RSCTGatekeeper(GatekeeperConfig(
                grid_size=config.warehouse_size,
                kappa_min=0.0,
                blocking_strategy=BlockingStrategy.PRIORITY_BY_DISTANCE,
            ))

        total_collisions = 0
        total_robots_damaged = 0
        total_orders = 0

        ep_iter = range(num_episodes)
        if verbose:
            ep_iter = tqdm(ep_iter, desc=f"  {regime.upper()}")

        for episode in ep_iter:
            env.reset()

            for step in range(config.max_steps):
                states = [env.get_state(i) for i in range(config.num_robots)]
                actions = [agents[i].act(states[i]) for i in range(config.num_robots)]

                if gatekeeper:
                    decision = gatekeeper.check_and_gate(
                        env.positions, actions, env.goals, env.obstacles
                    )
                    actions = decision.approved_actions

                result = env.step(actions)

                next_states = [env.get_state(i) for i in range(config.num_robots)]
                for i in range(config.num_robots):
                    agents[i].update(states[i], actions[i], result.rewards[i],
                                    next_states[i], result.done)

                if result.done:
                    break

            for agent in agents:
                agent.decay_epsilon()

            metrics = env.get_metrics()
            total_collisions += metrics["total_collisions"]
            total_robots_damaged += metrics["robots_damaged"]
            total_orders += metrics["orders_completed"]

        results[regime] = {
            "total_collisions": total_collisions,
            "robots_damaged": total_robots_damaged,
            "financial_loss": total_robots_damaged * config.robot_cost,
            "orders_completed": total_orders,
            "throughput": total_orders / num_episodes,
            "episodes": num_episodes,
        }

        if verbose:
            print(f"    Collisions: {total_collisions}")
            print(f"    Robots Damaged: {total_robots_damaged}")
            print(f"    💰 Financial Loss: ${total_robots_damaged * config.robot_cost:,}")
            print(f"    Orders Completed: {total_orders}")

    return results


def print_summary(drone_results, convoy_results, warehouse_results):
    """Print financial impact summary."""
    print("\n" + "="*70)
    print("💰 FINANCIAL IMPACT SUMMARY: RSCT vs MARL")
    print("="*70)

    total_marl_loss = (
        drone_results["marl"]["financial_loss"] +
        convoy_results["marl"]["financial_loss"] +
        warehouse_results["marl"]["financial_loss"]
    )

    total_rsct_loss = (
        drone_results["rsct"]["financial_loss"] +
        convoy_results["rsct"]["financial_loss"] +
        warehouse_results["rsct"]["financial_loss"]
    )

    savings = total_marl_loss - total_rsct_loss

    print(f"\n{'Application':<25} {'MARL Loss':>15} {'RSCT Loss':>15} {'Savings':>15}")
    print("-" * 70)

    print(f"{'Drone Swarm':<25} "
          f"${drone_results['marl']['financial_loss']:>14,} "
          f"${drone_results['rsct']['financial_loss']:>14,} "
          f"${drone_results['marl']['financial_loss'] - drone_results['rsct']['financial_loss']:>14,}")

    print(f"{'Military Convoy':<25} "
          f"${convoy_results['marl']['financial_loss']:>14,} "
          f"${convoy_results['rsct']['financial_loss']:>14,} "
          f"${convoy_results['marl']['financial_loss'] - convoy_results['rsct']['financial_loss']:>14,}")

    print(f"{'Warehouse Robots':<25} "
          f"${warehouse_results['marl']['financial_loss']:>14,} "
          f"${warehouse_results['rsct']['financial_loss']:>14,} "
          f"${warehouse_results['marl']['financial_loss'] - warehouse_results['rsct']['financial_loss']:>14,}")

    print("-" * 70)
    print(f"{'TOTAL':<25} ${total_marl_loss:>14,} ${total_rsct_loss:>14,} ${savings:>14,}")
    print("="*70)

    print(f"\n🎯 RSCT SAVED: ${savings:,} in this simulation")
    print(f"   ({(savings/total_marl_loss*100) if total_marl_loss > 0 else 0:.1f}% reduction in losses)")

    # Collision comparison
    print("\n" + "="*70)
    print("🚨 COLLISION COMPARISON")
    print("="*70)
    print(f"\n{'Application':<25} {'MARL Collisions':>18} {'RSCT Collisions':>18}")
    print("-" * 70)
    print(f"{'Drone Swarm':<25} {drone_results['marl']['total_collisions']:>18} {drone_results['rsct']['total_collisions']:>18}")
    print(f"{'Military Convoy':<25} {convoy_results['marl']['total_collisions']:>18} {convoy_results['rsct']['total_collisions']:>18}")
    print(f"{'Warehouse Robots':<25} {warehouse_results['marl']['total_collisions']:>18} {warehouse_results['rsct']['total_collisions']:>18}")

    total_marl_collisions = (
        drone_results["marl"]["total_collisions"] +
        convoy_results["marl"]["total_collisions"] +
        warehouse_results["marl"]["total_collisions"]
    )
    total_rsct_collisions = (
        drone_results["rsct"]["total_collisions"] +
        convoy_results["rsct"]["total_collisions"] +
        warehouse_results["rsct"]["total_collisions"]
    )

    print("-" * 70)
    print(f"{'TOTAL':<25} {total_marl_collisions:>18} {total_rsct_collisions:>18}")

    if total_rsct_collisions == 0:
        print("\n✅ RSCT achieved ZERO COLLISIONS across all applications!")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Real-World Application Simulations")
    parser.add_argument("--episodes", type=int, default=30, help="Episodes per scenario")
    parser.add_argument("--quick", action="store_true", help="Quick test (10 episodes)")
    args = parser.parse_args()

    episodes = 10 if args.quick else args.episodes

    print("\n" + "#"*70)
    print("#" + " "*20 + "REAL-WORLD APPLICATIONS" + " "*23 + "#")
    print("#" + " "*15 + "RSCT Safety Certification Value" + " "*18 + "#")
    print("#"*70)

    drone_results = run_drone_swarm_experiment(episodes)
    convoy_results = run_convoy_experiment(episodes)
    warehouse_results = run_warehouse_experiment(episodes)

    print_summary(drone_results, convoy_results, warehouse_results)


if __name__ == "__main__":
    main()
