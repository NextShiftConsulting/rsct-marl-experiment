"""
RSCT-Style Gatekeeper for Multi-Agent Collision Prevention

This module implements a static compatibility gate that certifies joint actions
as safe BEFORE execution. No learning required - safety is enforced by
geometric constraints on the representation space (positions).

Key insight: Safety is not something to be learned through trial and error.
It can be certified statically when you have the right representation.

This is the discrete analogue of RSCT's continuous compatibility certificates.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

from ..environment.gridworld import Action, ACTION_EFFECTS


class BlockingStrategy(Enum):
    """Strategy for handling blocked actions."""
    BOTH_STAY = "both_stay"  # Force all colliding agents to stay
    PRIORITY_BY_DISTANCE = "priority_by_distance"  # Agent further from goal moves
    PRIORITY_BY_ID = "priority_by_id"  # Lower ID agent has priority


@dataclass
class GatekeeperConfig:
    """Configuration for RSCT gatekeeper."""

    # Minimum compatibility score (kappa) required for action approval
    kappa_min: float = 0.2

    # Blocking strategy when actions are incompatible
    blocking_strategy: BlockingStrategy = BlockingStrategy.PRIORITY_BY_DISTANCE

    # Grid parameters (for normalization)
    grid_size: int = 5

    @property
    def max_distance(self) -> float:
        """Maximum possible Manhattan distance on grid."""
        return 2 * (self.grid_size - 1)


@dataclass
class GateDecision:
    """Result of gatekeeper compatibility check."""

    original_actions: List[Action]
    approved_actions: List[Action]
    was_blocked: bool
    blocked_agents: List[int]
    compatibility_score: float
    collision_type: Optional[str]  # "same_cell", "swap", or None
    details: Dict[str, Any]


class RSCTGatekeeper:
    """
    RSCT-style compatibility gatekeeper.

    Implements static safety certification for joint actions in multi-agent systems.
    The gate checks geometric compatibility between agent positions and proposed
    actions, blocking unsafe combinations before they execute.

    This is NOT learned - it's a pure verification layer based on representation geometry.

    Mathematical formulation:
    - Let p_i be position of agent i, a_i be proposed action
    - Let p'_i = T(p_i, a_i) be the resulting position after action
    - Compatibility score: kappa = min_{i != j} d(p'_i, p'_j) / d_max
    - Joint action is certified iff:
        1. No collisions: p'_i != p'_j for all i != j
        2. No swaps: not(p'_i = p_j and p'_j = p_i) for all i != j
        3. kappa >= kappa_min (sufficient separation)
    """

    def __init__(self, config: Optional[GatekeeperConfig] = None):
        self.config = config or GatekeeperConfig()

        # Statistics
        self.total_checks = 0
        self.total_blocks = 0
        self.block_reasons: Dict[str, int] = {
            "same_cell": 0,
            "swap": 0,
            "kappa_violation": 0,
        }

    def _compute_next_position(
        self,
        current: Tuple[int, int],
        action: Action,
        obstacles: set,
    ) -> Tuple[int, int]:
        """Compute next position, respecting grid bounds and obstacles."""
        dr, dc = ACTION_EFFECTS[action]
        next_pos = (current[0] + dr, current[1] + dc)

        # Check bounds
        r, c = next_pos
        if r < 0 or r >= self.config.grid_size or c < 0 or c >= self.config.grid_size:
            return current

        # Check obstacles
        if next_pos in obstacles:
            return current

        return next_pos

    def _manhattan_distance(
        self, pos1: Tuple[int, int], pos2: Tuple[int, int]
    ) -> int:
        """Compute Manhattan distance between positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _check_same_cell_collision(
        self, next_positions: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        """Check for same-cell collisions. Returns list of (i, j) pairs."""
        collisions = []
        n = len(next_positions)
        for i in range(n):
            for j in range(i + 1, n):
                if next_positions[i] == next_positions[j]:
                    collisions.append((i, j))
        return collisions

    def _check_swap_collision(
        self,
        current_positions: List[Tuple[int, int]],
        next_positions: List[Tuple[int, int]],
    ) -> List[Tuple[int, int]]:
        """Check for swap collisions (agents crossing). Returns list of (i, j) pairs."""
        collisions = []
        n = len(current_positions)
        for i in range(n):
            for j in range(i + 1, n):
                if (
                    next_positions[i] == current_positions[j]
                    and next_positions[j] == current_positions[i]
                ):
                    collisions.append((i, j))
        return collisions

    def _compute_compatibility_score(
        self, next_positions: List[Tuple[int, int]]
    ) -> float:
        """
        Compute compatibility score (kappa) for proposed configuration.

        kappa = min pairwise distance / max possible distance

        Higher kappa = more separation = safer
        """
        n = len(next_positions)
        if n < 2:
            return 1.0

        min_dist = float("inf")
        for i in range(n):
            for j in range(i + 1, n):
                dist = self._manhattan_distance(next_positions[i], next_positions[j])
                min_dist = min(min_dist, dist)

        return min_dist / self.config.max_distance

    def _resolve_collision(
        self,
        current_positions: List[Tuple[int, int]],
        actions: List[Action],
        colliding_pairs: List[Tuple[int, int]],
        goals: List[Tuple[int, int]],
        obstacles: set = None,
    ) -> List[Action]:
        """
        Resolve collisions by modifying actions according to blocking strategy.

        Returns modified action list where colliding agents are blocked.

        IMPORTANT: This uses iterative resolution - after blocking one agent,
        we recompute to ensure the new configuration is still safe.
        """
        obstacles = obstacles or set()
        modified_actions = list(actions)
        max_iterations = len(actions) * 2  # Prevent infinite loops

        for _ in range(max_iterations):
            # Compute next positions with current modified actions
            next_positions = [
                self._compute_next_position(current_positions[i], modified_actions[i], obstacles)
                for i in range(len(current_positions))
            ]

            # Check for remaining collisions
            same_cell = self._check_same_cell_collision(next_positions)
            swap = self._check_swap_collision(current_positions, next_positions)
            all_collisions = same_cell + swap

            if not all_collisions:
                break  # No more collisions, we're done

            # Resolve one collision at a time
            i, j = all_collisions[0]

            if self.config.blocking_strategy == BlockingStrategy.BOTH_STAY:
                modified_actions[i] = Action.STAY
                modified_actions[j] = Action.STAY

            elif self.config.blocking_strategy == BlockingStrategy.PRIORITY_BY_DISTANCE:
                dist_i = self._manhattan_distance(current_positions[i], goals[i])
                dist_j = self._manhattan_distance(current_positions[j], goals[j])

                # Determine which agent is moving INTO the other's current cell
                # That agent should be blocked
                next_i = next_positions[i]
                next_j = next_positions[j]

                i_invading_j = (next_i == current_positions[j])
                j_invading_i = (next_j == current_positions[i])

                if i_invading_j and not j_invading_i:
                    # i is moving into j's cell, block i
                    modified_actions[i] = Action.STAY
                elif j_invading_i and not i_invading_j:
                    # j is moving into i's cell, block j
                    modified_actions[j] = Action.STAY
                elif dist_i > dist_j:
                    # Agent i is further from goal, let it move, block j
                    modified_actions[j] = Action.STAY
                elif dist_j > dist_i:
                    # Agent j is further from goal, let it move, block i
                    modified_actions[i] = Action.STAY
                else:
                    # Tie: block the one with higher ID
                    modified_actions[max(i, j)] = Action.STAY

            elif self.config.blocking_strategy == BlockingStrategy.PRIORITY_BY_ID:
                modified_actions[max(i, j)] = Action.STAY

        return modified_actions

    def check_and_gate(
        self,
        current_positions: List[Tuple[int, int]],
        proposed_actions: List[Action],
        goals: List[Tuple[int, int]],
        obstacles: set = None,
    ) -> GateDecision:
        """
        Check joint action compatibility and gate if unsafe.

        This is the main entry point for the RSCT gatekeeper.

        Args:
            current_positions: Current position of each agent
            proposed_actions: Proposed action for each agent
            goals: Goal position for each agent
            obstacles: Set of obstacle positions

        Returns:
            GateDecision with approved actions and diagnostic information
        """
        self.total_checks += 1
        obstacles = obstacles or set()

        # Compute proposed next positions
        next_positions = [
            self._compute_next_position(current_positions[i], proposed_actions[i], obstacles)
            for i in range(len(current_positions))
        ]

        # Check for collisions
        same_cell_collisions = self._check_same_cell_collision(next_positions)
        swap_collisions = self._check_swap_collision(current_positions, next_positions)

        # Compute compatibility score
        kappa = self._compute_compatibility_score(next_positions)

        # Determine if blocking is needed
        all_collisions = same_cell_collisions + swap_collisions
        kappa_violation = kappa < self.config.kappa_min and len(current_positions) > 1

        needs_blocking = len(all_collisions) > 0 or kappa_violation

        # Determine collision type for reporting
        collision_type = None
        if same_cell_collisions:
            collision_type = "same_cell"
        elif swap_collisions:
            collision_type = "swap"
        elif kappa_violation:
            collision_type = "kappa_violation"

        # Resolve if needed
        if needs_blocking:
            self.total_blocks += 1
            if same_cell_collisions:
                self.block_reasons["same_cell"] += 1
            if swap_collisions:
                self.block_reasons["swap"] += 1
            if kappa_violation and not all_collisions:
                self.block_reasons["kappa_violation"] += 1

            approved_actions = self._resolve_collision(
                current_positions,
                proposed_actions,
                all_collisions if all_collisions else [(0, 1)],  # kappa violation defaults to first pair
                goals,
                obstacles,
            )

            # Identify which agents were blocked
            blocked_agents = [
                i for i in range(len(proposed_actions))
                if approved_actions[i] != proposed_actions[i]
            ]
        else:
            approved_actions = list(proposed_actions)
            blocked_agents = []

        return GateDecision(
            original_actions=list(proposed_actions),
            approved_actions=approved_actions,
            was_blocked=needs_blocking,
            blocked_agents=blocked_agents,
            compatibility_score=kappa,
            collision_type=collision_type,
            details={
                "same_cell_collisions": same_cell_collisions,
                "swap_collisions": swap_collisions,
                "kappa_violation": kappa_violation,
                "next_positions_proposed": next_positions,
            },
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get gatekeeper statistics."""
        return {
            "total_checks": self.total_checks,
            "total_blocks": self.total_blocks,
            "block_rate": self.total_blocks / max(1, self.total_checks),
            "block_reasons": self.block_reasons.copy(),
        }

    def reset_stats(self):
        """Reset statistics (e.g., between experiments)."""
        self.total_checks = 0
        self.total_blocks = 0
        self.block_reasons = {
            "same_cell": 0,
            "swap": 0,
            "kappa_violation": 0,
        }


# Theorem verification functions (for theory module)

def verify_soundness(
    gatekeeper: RSCTGatekeeper,
    positions: List[Tuple[int, int]],
    actions: List[Action],
    goals: List[Tuple[int, int]],
    obstacles: set = None,
) -> bool:
    """
    Verify soundness: if gatekeeper approves, no collision will occur.

    Soundness theorem: For all approved joint actions,
    the resulting state has no collisions.

    This function verifies the property holds for a specific instance.
    """
    obstacles = obstacles or set()
    decision = gatekeeper.check_and_gate(positions, actions, goals, obstacles)

    # Compute actual next positions with approved actions
    next_positions = [
        gatekeeper._compute_next_position(positions[i], decision.approved_actions[i], obstacles)
        for i in range(len(positions))
    ]

    # Check no collisions
    same_cell = gatekeeper._check_same_cell_collision(next_positions)
    swap = gatekeeper._check_swap_collision(positions, next_positions)

    return len(same_cell) == 0 and len(swap) == 0


def verify_completeness_condition(
    gatekeeper: RSCTGatekeeper,
    positions: List[Tuple[int, int]],
    goals: List[Tuple[int, int]],
    obstacles: set = None,
) -> bool:
    """
    Check if completeness condition holds: there exists at least one
    safe joint action from current state.

    Completeness is guaranteed when:
    1. Agents are not already in collision
    2. There exists a path to goals that doesn't require simultaneous
       occupation of the same cell

    This is a necessary (not sufficient) check.
    """
    obstacles = obstacles or set()

    # Check current state is valid (no collision)
    if len(gatekeeper._check_same_cell_collision(positions)) > 0:
        return False

    # STAY is always safe if current positions are valid
    stay_actions = [Action.STAY] * len(positions)
    decision = gatekeeper.check_and_gate(positions, stay_actions, goals, obstacles)

    return not decision.was_blocked
