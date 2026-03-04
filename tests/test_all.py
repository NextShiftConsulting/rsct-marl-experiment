"""
Unit tests for RSCT vs MARL experiment components.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import numpy as np

from src.environment import MultiAgentGridworld, GridworldConfig, Action
from src.agents import QLearningAgent, AgentConfig
from src.gatekeeper import RSCTGatekeeper, GatekeeperConfig, BlockingStrategy
from src.experiments import MetricsCollector, EpisodeMetrics
from src.theory import verify_all_theorems


class TestGridworld(unittest.TestCase):
    """Test gridworld environment."""

    def test_initialization(self):
        env = MultiAgentGridworld()
        self.assertEqual(env.grid_size, 5)
        self.assertEqual(env.num_agents, 2)

    def test_reset(self):
        env = MultiAgentGridworld()
        positions = env.reset()
        self.assertEqual(len(positions), 2)
        self.assertEqual(positions[0], (0, 0))
        self.assertEqual(positions[1], (4, 0))

    def test_step(self):
        env = MultiAgentGridworld()
        env.reset()
        result = env.step([Action.RIGHT, Action.LEFT])
        self.assertEqual(len(result.next_positions), 2)
        self.assertEqual(len(result.rewards), 2)

    def test_collision_detection(self):
        config = GridworldConfig(
            agent_configs=[((2, 2), (4, 4)), ((2, 3), (0, 0))]
        )
        env = MultiAgentGridworld(config)
        env.reset()

        # Both agents move to same cell
        result = env.step([Action.RIGHT, Action.LEFT])
        self.assertTrue(result.info["collision"])

    def test_swap_collision(self):
        config = GridworldConfig(
            agent_configs=[((2, 2), (4, 4)), ((2, 3), (0, 0))]
        )
        env = MultiAgentGridworld(config)
        env.reset()

        # Agents swap positions
        result = env.step([Action.RIGHT, Action.LEFT])
        # This is a collision (they meet in the middle or swap)
        self.assertTrue(result.info["collision"])


class TestQLearningAgent(unittest.TestCase):
    """Test Q-learning agent."""

    def test_initialization(self):
        agent = QLearningAgent(agent_id=0)
        self.assertEqual(agent.agent_id, 0)
        self.assertEqual(agent.epsilon, 1.0)

    def test_act(self):
        agent = QLearningAgent(agent_id=0)
        state = ((0, 0), ((1, 1),), (4, 4))
        action = agent.act(state)
        self.assertIsInstance(action, Action)

    def test_greedy_act(self):
        agent = QLearningAgent(agent_id=0)
        agent.epsilon = 0.0  # Force greedy
        state = ((0, 0), ((1, 1),), (4, 4))
        action = agent.act(state, greedy=True)
        self.assertIsInstance(action, Action)

    def test_update(self):
        agent = QLearningAgent(agent_id=0)
        state = ((0, 0), ((1, 1),), (4, 4))
        next_state = ((0, 1), ((1, 1),), (4, 4))

        td_error = agent.update(state, Action.RIGHT, -1.0, next_state, False)
        self.assertIsInstance(td_error, float)

    def test_epsilon_decay(self):
        agent = QLearningAgent(agent_id=0)
        initial_epsilon = agent.epsilon
        agent.decay_epsilon()
        self.assertLess(agent.epsilon, initial_epsilon)


class TestRSCTGatekeeper(unittest.TestCase):
    """Test RSCT gatekeeper."""

    def test_initialization(self):
        gate = RSCTGatekeeper()
        self.assertEqual(gate.total_checks, 0)
        self.assertEqual(gate.total_blocks, 0)

    def test_safe_action_approved(self):
        gate = RSCTGatekeeper(GatekeeperConfig(grid_size=5, kappa_min=0.0))
        positions = [(0, 0), (4, 4)]
        actions = [Action.RIGHT, Action.LEFT]
        goals = [(4, 4), (0, 0)]

        decision = gate.check_and_gate(positions, actions, goals)
        self.assertFalse(decision.was_blocked)
        self.assertEqual(decision.approved_actions, actions)

    def test_collision_blocked(self):
        gate = RSCTGatekeeper(GatekeeperConfig(grid_size=5, kappa_min=0.0))
        positions = [(2, 2), (2, 3)]
        actions = [Action.RIGHT, Action.LEFT]  # Would collide at (2, 3) or (2, 2)
        goals = [(4, 4), (0, 0)]

        decision = gate.check_and_gate(positions, actions, goals)
        # Should be blocked since they would collide
        self.assertTrue(decision.was_blocked)

    def test_stay_is_safe(self):
        gate = RSCTGatekeeper(GatekeeperConfig(grid_size=5, kappa_min=0.0))
        positions = [(0, 0), (4, 4)]
        actions = [Action.STAY, Action.STAY]
        goals = [(4, 4), (0, 0)]

        decision = gate.check_and_gate(positions, actions, goals)
        self.assertFalse(decision.was_blocked)


class TestTheorems(unittest.TestCase):
    """Test formal theorem verification."""

    def test_soundness_small_grid(self):
        """Verify soundness on 3x3 grid (fast)."""
        from src.theory.proofs import SoundnessTheorem
        theorem = SoundnessTheorem(grid_size=3, num_agents=2)
        result = theorem.verify_exhaustive()
        self.assertTrue(result.holds, f"Soundness failed: {result.counterexample}")

    def test_completeness_small_grid(self):
        """Verify completeness on 3x3 grid (fast)."""
        from src.theory.proofs import CompletenessTheorem
        theorem = CompletenessTheorem(grid_size=3, num_agents=2)
        result = theorem.verify_exhaustive()
        self.assertTrue(result.holds, f"Completeness failed: {result.counterexample}")


class TestIntegration(unittest.TestCase):
    """Integration tests."""

    def test_full_episode_marl(self):
        """Run a full episode under MARL regime."""
        env = MultiAgentGridworld()
        agents = [QLearningAgent(i) for i in range(2)]

        positions = env.reset()
        for _ in range(10):
            states = [env.get_state(i) for i in range(2)]
            actions = [agents[i].act(states[i]) for i in range(2)]
            result = env.step(actions)
            if result.done:
                break

    def test_full_episode_rsct_gated(self):
        """Run a full episode under RSCT-gated regime."""
        env = MultiAgentGridworld()
        agents = [QLearningAgent(i) for i in range(2)]
        gate = RSCTGatekeeper(GatekeeperConfig(grid_size=5))

        positions = env.reset()
        collisions = 0

        for _ in range(30):
            states = [env.get_state(i) for i in range(2)]
            proposed = [agents[i].act(states[i]) for i in range(2)]

            decision = gate.check_and_gate(
                env.positions, proposed, env.goals, env.obstacles
            )

            result = env.step(decision.approved_actions)
            if result.info["collision"]:
                collisions += 1
            if result.done:
                break

        # With gate, should have zero collisions
        self.assertEqual(collisions, 0, "RSCT gate failed to prevent collisions")


if __name__ == "__main__":
    unittest.main()
