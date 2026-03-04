"""
Tabular Q-Learning Agent for Multi-Agent Gridworld

Independent learner: each agent maintains its own Q-table and learns
based on its own state-action-reward experience.
"""

from dataclasses import dataclass
from typing import Dict, Tuple, Any, Optional
import numpy as np
from collections import defaultdict

from ..environment.gridworld import Action


@dataclass
class AgentConfig:
    """Configuration for Q-learning agent."""

    # Learning parameters
    learning_rate: float = 0.1
    discount_factor: float = 0.95

    # Exploration parameters
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995  # Multiplicative decay per episode

    # Q-table initialization
    initial_q_value: float = 0.0


class QLearningAgent:
    """
    Tabular Q-learning agent.

    Uses epsilon-greedy exploration and maintains a Q-table
    mapping state-action pairs to expected returns.
    """

    def __init__(
        self,
        agent_id: int,
        num_actions: int = 5,
        config: Optional[AgentConfig] = None,
    ):
        self.agent_id = agent_id
        self.num_actions = num_actions
        self.config = config or AgentConfig()

        # Q-table: maps (state, action) -> Q-value
        # Using defaultdict for automatic initialization
        self.q_table: Dict[Tuple, np.ndarray] = defaultdict(
            lambda: np.full(num_actions, self.config.initial_q_value)
        )

        # Current exploration rate
        self.epsilon = self.config.epsilon_start

        # Statistics
        self.total_updates = 0
        self.states_visited = set()

    def _state_to_key(self, state: Any) -> Tuple:
        """Convert state to hashable key for Q-table."""
        # State should already be a tuple from the environment
        return state if isinstance(state, tuple) else tuple(state)

    def act(self, state: Any, greedy: bool = False) -> Action:
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Current state observation
            greedy: If True, always select best action (no exploration)

        Returns:
            Selected action
        """
        state_key = self._state_to_key(state)
        self.states_visited.add(state_key)

        # Epsilon-greedy action selection
        if not greedy and np.random.random() < self.epsilon:
            return Action(np.random.randint(self.num_actions))

        # Greedy action selection
        q_values = self.q_table[state_key]

        # Break ties randomly
        max_q = np.max(q_values)
        best_actions = np.where(q_values == max_q)[0]
        return Action(np.random.choice(best_actions))

    def update(
        self,
        state: Any,
        action: Action,
        reward: float,
        next_state: Any,
        done: bool,
    ) -> float:
        """
        Update Q-value using temporal difference learning.

        Q(s,a) <- Q(s,a) + alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))

        Args:
            state: State where action was taken
            action: Action taken
            reward: Reward received
            next_state: Resulting state
            done: Whether episode terminated

        Returns:
            TD error (for monitoring)
        """
        state_key = self._state_to_key(state)
        next_state_key = self._state_to_key(next_state)

        # Current Q-value
        current_q = self.q_table[state_key][action]

        # Target Q-value
        if done:
            target_q = reward
        else:
            max_next_q = np.max(self.q_table[next_state_key])
            target_q = reward + self.config.discount_factor * max_next_q

        # TD error
        td_error = target_q - current_q

        # Update Q-value
        self.q_table[state_key][action] += self.config.learning_rate * td_error

        self.total_updates += 1

        return td_error

    def decay_epsilon(self):
        """Decay exploration rate (call at end of each episode)."""
        self.epsilon = max(
            self.config.epsilon_end,
            self.epsilon * self.config.epsilon_decay
        )

    def get_q_values(self, state: Any) -> np.ndarray:
        """Get Q-values for all actions in given state."""
        state_key = self._state_to_key(state)
        return self.q_table[state_key].copy()

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            "agent_id": self.agent_id,
            "epsilon": self.epsilon,
            "total_updates": self.total_updates,
            "unique_states_visited": len(self.states_visited),
            "q_table_size": len(self.q_table),
        }

    def reset_episode(self):
        """Reset episode-specific state (call at episode start)."""
        pass  # No episode-specific state currently

    def save(self, filepath: str):
        """Save Q-table to file."""
        import pickle
        with open(filepath, "wb") as f:
            pickle.dump({
                "q_table": dict(self.q_table),
                "epsilon": self.epsilon,
                "config": self.config,
                "stats": self.get_stats(),
            }, f)

    @classmethod
    def load(cls, filepath: str, agent_id: int) -> "QLearningAgent":
        """Load Q-table from file."""
        import pickle
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        agent = cls(agent_id=agent_id, config=data["config"])
        agent.q_table = defaultdict(
            lambda: np.full(agent.num_actions, agent.config.initial_q_value),
            data["q_table"]
        )
        agent.epsilon = data["epsilon"]
        return agent
