"""
Formal Theoretical Analysis of RSCT Gatekeeper

This module provides:
1. Soundness theorem: If gate approves, no collision occurs
2. Completeness conditions: When liveness is guaranteed
3. Connection to barrier certificates (Lyapunov-like analysis)
4. Formal verification functions

Mathematical Framework:
-----------------------
Let S = Z^2 × Z^2 × ... (n copies) be the joint state space (positions)
Let A = {UP, DOWN, LEFT, RIGHT, STAY}^n be the joint action space
Let T: S × A → S be the transition function
Let C ⊂ S be the collision set: {s | ∃i≠j: s_i = s_j}

The gatekeeper G: S × A → A is a function that:
- G(s, a) = a if T(s, a) ∉ C  (approve safe actions)
- G(s, a) = a' where T(s, a') ∉ C  (substitute unsafe actions)

Theorems:
---------
Soundness: ∀s ∈ S, ∀a ∈ A: T(s, G(s, a)) ∉ C
Completeness: ∀s ∉ C: ∃a ∈ A: G(s, a) = a (identity on at least one action)
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Set, Optional
import itertools
import numpy as np

from ..environment.gridworld import Action, ACTION_EFFECTS
from ..gatekeeper.rsct_gate import RSCTGatekeeper, GatekeeperConfig


@dataclass
class TheoremResult:
    """Result of theorem verification."""
    theorem_name: str
    holds: bool
    counterexample: Optional[Any] = None
    num_states_checked: int = 0
    details: Dict[str, Any] = None


class SoundnessTheorem:
    """
    Soundness Theorem for RSCT Gatekeeper

    Statement: For all states s and all proposed joint actions a,
    the gated action G(s,a) results in a collision-free next state.

    Formally: ∀s ∈ S, ∀a ∈ A: T(s, G(s, a)) ∉ C

    Proof sketch:
    1. The gatekeeper checks all collision conditions before approval
    2. If any collision would occur, actions are modified to STAY
    3. STAY from a non-collision state cannot create a collision
    4. Therefore, gated actions never lead to collisions

    This class provides exhaustive verification for finite grids.
    """

    def __init__(self, grid_size: int = 5, num_agents: int = 2):
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.gatekeeper = RSCTGatekeeper(
            GatekeeperConfig(grid_size=grid_size, kappa_min=0.0)
        )

    def _all_positions(self) -> List[Tuple[int, int]]:
        """Generate all valid positions on grid."""
        return [(r, c) for r in range(self.grid_size) for c in range(self.grid_size)]

    def _is_collision_state(self, positions: List[Tuple[int, int]]) -> bool:
        """Check if positions constitute a collision."""
        return len(positions) != len(set(positions))

    def _compute_next_positions(
        self,
        positions: List[Tuple[int, int]],
        actions: List[Action],
    ) -> List[Tuple[int, int]]:
        """Compute next positions given current positions and actions."""
        next_pos = []
        for i, (pos, action) in enumerate(zip(positions, actions)):
            dr, dc = ACTION_EFFECTS[action]
            new_r, new_c = pos[0] + dr, pos[1] + dc
            # Clip to grid bounds
            new_r = max(0, min(self.grid_size - 1, new_r))
            new_c = max(0, min(self.grid_size - 1, new_c))
            next_pos.append((new_r, new_c))
        return next_pos

    def _check_swap_collision(
        self,
        current: List[Tuple[int, int]],
        next_pos: List[Tuple[int, int]],
    ) -> bool:
        """Check for swap collisions."""
        n = len(current)
        for i in range(n):
            for j in range(i + 1, n):
                if current[i] == next_pos[j] and current[j] == next_pos[i]:
                    return True
        return False

    def verify_exhaustive(self, sample_size: Optional[int] = None) -> TheoremResult:
        """
        Exhaustively verify soundness for all reachable states.

        For small grids (5x5, 2 agents), this is tractable:
        - Positions: 25^2 = 625 states
        - Actions: 5^2 = 25 per state
        - Total: 15,625 state-action pairs

        Args:
            sample_size: If set, randomly sample instead of exhaustive check

        Returns:
            TheoremResult with verification outcome
        """
        all_positions = self._all_positions()
        all_actions = list(Action)

        # Generate all valid (non-collision) starting states
        if self.num_agents == 2:
            all_states = [
                [p1, p2]
                for p1 in all_positions
                for p2 in all_positions
                if p1 != p2  # Start from non-collision states
            ]
        else:
            # For more agents, use combinations
            all_states = [
                list(positions)
                for positions in itertools.permutations(all_positions, self.num_agents)
                if len(set(positions)) == self.num_agents
            ]

        # Generate all action combinations
        all_action_combos = list(itertools.product(all_actions, repeat=self.num_agents))

        # Sample if requested
        if sample_size and len(all_states) * len(all_action_combos) > sample_size:
            import random
            state_action_pairs = [
                (random.choice(all_states), random.choice(all_action_combos))
                for _ in range(sample_size)
            ]
        else:
            state_action_pairs = [
                (state, actions)
                for state in all_states
                for actions in all_action_combos
            ]

        num_checked = 0
        counterexample = None

        # Dummy goals (not used in collision detection)
        dummy_goals = [(self.grid_size - 1, self.grid_size - 1)] * self.num_agents

        for positions, actions in state_action_pairs:
            num_checked += 1

            # Get gated actions
            decision = self.gatekeeper.check_and_gate(
                current_positions=positions,
                proposed_actions=list(actions),
                goals=dummy_goals,
            )

            # Compute resulting state with gated actions
            next_positions = self._compute_next_positions(
                positions, decision.approved_actions
            )

            # Check for collisions
            same_cell = self._is_collision_state(next_positions)
            swap = self._check_swap_collision(positions, next_positions)

            if same_cell or swap:
                counterexample = {
                    "initial_positions": positions,
                    "proposed_actions": actions,
                    "gated_actions": decision.approved_actions,
                    "resulting_positions": next_positions,
                    "collision_type": "same_cell" if same_cell else "swap",
                }
                return TheoremResult(
                    theorem_name="Soundness",
                    holds=False,
                    counterexample=counterexample,
                    num_states_checked=num_checked,
                )

        return TheoremResult(
            theorem_name="Soundness",
            holds=True,
            counterexample=None,
            num_states_checked=num_checked,
            details={
                "grid_size": self.grid_size,
                "num_agents": self.num_agents,
                "verification_type": "exhaustive" if sample_size is None else "sampled",
            },
        )


class CompletenessTheorem:
    """
    Completeness Theorem for RSCT Gatekeeper

    Statement: From any non-collision state, there exists at least one
    joint action that the gatekeeper approves without modification.

    This guarantees liveness: agents can always make progress.

    Formally: ∀s ∉ C: ∃a ∈ A: G(s, a) = a

    Proof sketch:
    1. If agents are not in collision, they occupy distinct cells
    2. The action (STAY, STAY, ..., STAY) keeps all agents in place
    3. Since cells are already distinct, no collision occurs
    4. Therefore, the gatekeeper approves this action
    5. QED: At least one identity-approved action exists

    Note: This proves existence, not that all useful actions are approved.
    """

    def __init__(self, grid_size: int = 5, num_agents: int = 2):
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.gatekeeper = RSCTGatekeeper(
            GatekeeperConfig(grid_size=grid_size, kappa_min=0.0)
        )

    def verify_exhaustive(self) -> TheoremResult:
        """
        Verify that every non-collision state has at least one
        approved action (identity mapping through gate).
        """
        all_positions = [(r, c) for r in range(self.grid_size) for c in range(self.grid_size)]
        all_actions = list(Action)

        # Generate non-collision states
        if self.num_agents == 2:
            all_states = [
                [p1, p2]
                for p1 in all_positions
                for p2 in all_positions
                if p1 != p2
            ]
        else:
            all_states = [
                list(positions)
                for positions in itertools.permutations(all_positions, self.num_agents)
                if len(set(positions)) == self.num_agents
            ]

        all_action_combos = list(itertools.product(all_actions, repeat=self.num_agents))
        dummy_goals = [(self.grid_size - 1, self.grid_size - 1)] * self.num_agents

        num_checked = 0

        for positions in all_states:
            num_checked += 1
            found_identity = False

            for actions in all_action_combos:
                decision = self.gatekeeper.check_and_gate(
                    current_positions=positions,
                    proposed_actions=list(actions),
                    goals=dummy_goals,
                )

                # Check if gatekeeper returned actions unchanged
                if list(actions) == decision.approved_actions:
                    found_identity = True
                    break

            if not found_identity:
                return TheoremResult(
                    theorem_name="Completeness",
                    holds=False,
                    counterexample={"positions": positions},
                    num_states_checked=num_checked,
                )

        return TheoremResult(
            theorem_name="Completeness",
            holds=True,
            num_states_checked=num_checked,
            details={
                "grid_size": self.grid_size,
                "num_agents": self.num_agents,
            },
        )


class BarrierCertificate:
    """
    Barrier Certificate Analysis

    Connection to Control-Barrier Functions (Ames et al.):

    Define h(s) = min_{i≠j} ||s_i - s_j|| as the minimum pairwise distance.
    This is a barrier function where:
    - h(s) > 0 implies safe (no collision)
    - h(s) = 0 implies collision

    The RSCT gatekeeper enforces a discrete barrier condition:
    h(T(s, G(s,a))) ≥ h_min > 0 for all approved actions

    This is the discrete analogue of the continuous CBF condition:
    ḣ(s) + α(h(s)) ≥ 0

    The gatekeeper's kappa_min parameter directly controls h_min.
    """

    def __init__(self, grid_size: int = 5, num_agents: int = 2):
        self.grid_size = grid_size
        self.num_agents = num_agents

    def h(self, positions: List[Tuple[int, int]]) -> float:
        """
        Barrier function: minimum pairwise Manhattan distance.

        h(s) > 0 => safe state
        h(s) = 0 => collision state
        """
        if len(positions) < 2:
            return float("inf")

        min_dist = float("inf")
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                dist = abs(positions[i][0] - positions[j][0]) + \
                       abs(positions[i][1] - positions[j][1])
                min_dist = min(min_dist, dist)

        return min_dist

    def verify_barrier_invariant(
        self,
        gatekeeper: RSCTGatekeeper,
        h_min: float = 1.0,
        sample_size: int = 1000,
    ) -> TheoremResult:
        """
        Verify that the barrier invariant h(s') >= h_min is maintained.

        This shows the gatekeeper enforces a safety margin, not just
        collision avoidance.
        """
        all_positions = [(r, c) for r in range(self.grid_size) for c in range(self.grid_size)]
        all_actions = list(Action)

        import random
        violations = []
        num_checked = 0

        for _ in range(sample_size):
            # Random non-collision state with h >= h_min
            while True:
                positions = random.sample(all_positions, self.num_agents)
                if self.h(positions) >= h_min:
                    break

            # Random action
            actions = [random.choice(all_actions) for _ in range(self.num_agents)]
            dummy_goals = [(self.grid_size - 1, self.grid_size - 1)] * self.num_agents

            decision = gatekeeper.check_and_gate(positions, actions, dummy_goals)

            # Compute next state
            next_positions = []
            for i, (pos, action) in enumerate(zip(positions, decision.approved_actions)):
                dr, dc = ACTION_EFFECTS[action]
                new_r = max(0, min(self.grid_size - 1, pos[0] + dr))
                new_c = max(0, min(self.grid_size - 1, pos[1] + dc))
                next_positions.append((new_r, new_c))

            h_next = self.h(next_positions)
            num_checked += 1

            if h_next < h_min:
                violations.append({
                    "positions": positions,
                    "actions": actions,
                    "gated_actions": decision.approved_actions,
                    "next_positions": next_positions,
                    "h_before": self.h(positions),
                    "h_after": h_next,
                })

        return TheoremResult(
            theorem_name=f"Barrier Invariant (h >= {h_min})",
            holds=len(violations) == 0,
            counterexample=violations[0] if violations else None,
            num_states_checked=num_checked,
            details={
                "h_min": h_min,
                "num_violations": len(violations),
            },
        )


def verify_all_theorems(
    grid_size: int = 5,
    num_agents: int = 2,
    verbose: bool = True,
) -> Dict[str, TheoremResult]:
    """
    Run all theorem verifications.

    Args:
        grid_size: Size of grid for verification
        num_agents: Number of agents
        verbose: Print results

    Returns:
        Dictionary of theorem names to results
    """
    results = {}

    if verbose:
        print("\n" + "="*60)
        print("RSCT GATEKEEPER FORMAL VERIFICATION")
        print("="*60)
        print(f"Grid: {grid_size}x{grid_size}, Agents: {num_agents}")
        print("-"*60)

    # Soundness
    soundness = SoundnessTheorem(grid_size, num_agents)
    result = soundness.verify_exhaustive()
    results["soundness"] = result
    if verbose:
        status = "✓ HOLDS" if result.holds else "✗ FAILS"
        print(f"\nSoundness Theorem: {status}")
        print(f"  States checked: {result.num_states_checked}")
        if result.counterexample:
            print(f"  Counterexample: {result.counterexample}")

    # Completeness
    completeness = CompletenessTheorem(grid_size, num_agents)
    result = completeness.verify_exhaustive()
    results["completeness"] = result
    if verbose:
        status = "✓ HOLDS" if result.holds else "✗ FAILS"
        print(f"\nCompleteness Theorem: {status}")
        print(f"  States checked: {result.num_states_checked}")

    # Barrier certificate (with default kappa)
    barrier = BarrierCertificate(grid_size, num_agents)
    gatekeeper = RSCTGatekeeper(GatekeeperConfig(grid_size=grid_size, kappa_min=0.2))
    result = barrier.verify_barrier_invariant(gatekeeper, h_min=1.0)
    results["barrier"] = result
    if verbose:
        status = "✓ HOLDS" if result.holds else "✗ FAILS"
        print(f"\nBarrier Certificate (h >= 1.0): {status}")
        print(f"  States checked: {result.num_states_checked}")

    if verbose:
        print("\n" + "="*60)
        all_hold = all(r.holds for r in results.values())
        print(f"VERIFICATION COMPLETE: {'ALL THEOREMS HOLD' if all_hold else 'SOME THEOREMS FAIL'}")
        print("="*60)

    return results
