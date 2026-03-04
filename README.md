# RSCT vs MARL Gridworld Experiment

**Experiment ID:** EXP-RSCT-001: Gated vs MARL Navigation

A rigorous comparison of **learned coordination** (Multi-Agent Reinforcement Learning) versus **certified coordination** (RSCT-style static safety gating) in a multi-agent gridworld environment.

## Research Question

> Does adding an RSCT-style compatibility gate eliminate collisions *without* any extra training, while a pure MARL baseline needs many episodes and still exhibits residual unsafe behavior?

## Key Insight

**Safety is not something to be learned through trial and error. It can be certified statically when you have the right representation geometry.**

The RSCT gatekeeper enforces collision avoidance as a *verification layer*, not a learned policy. This provides:
- **Zero collisions from episode 1** (by construction)
- **No training required** for safety
- **Formal guarantees** via soundness theorem

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Quick test (100 episodes, ~30 seconds)
python run_experiment.py --quick

# Full experiment (500 episodes)
python run_experiment.py

# Formal verification only
python run_experiment.py --verify

# Larger scale experiment
python run_experiment.py --scale medium

# Real MAPF benchmarks (Berlin city map)
python run_mapf_experiment.py --agents 8 --episodes 50
```

## Interactive Demo

```bash
# Install demo dependencies
pip install streamlit pandas

# Launch interactive web demo
streamlit run demo/app.py
```

The demo lets you:
- Adjust grid size (5×5, 7×7, 10×10)
- Change number of agents (2-4)
- Watch MARL vs RSCT side-by-side in real-time
- See cumulative collision charts

## Jupyter Notebook

```bash
# Launch analysis notebook
jupyter notebook notebooks/analysis.ipynb
```

Generates publication-ready figures and LaTeX tables.

## Results

### Expected Outcomes

| Metric | MARL | RSCT-Gated |
|--------|------|------------|
| Collisions (training) | ~50-200 | **0** |
| Time to zero collisions | ~200 episodes | **Episode 1** |
| Success rate | 70-90% | 80-95% |

### Key Visualization: Collision Comparison

The "money shot" figure shows:
- **MARL**: Decaying but noisy collision curve, never truly zero
- **RSCT-Gated**: Flat line at zero from start

## Architecture

```
src/
├── environment/      # Multi-agent gridworld
│   ├── gridworld.py  # Core environment with collision detection
│   └── config.py     # Configurable grid sizes, agents, obstacles
├── agents/           # Q-learning implementation
│   └── q_learning.py # Tabular Q-learning with ε-greedy
├── gatekeeper/       # RSCT safety gate
│   └── rsct_gate.py  # Static compatibility checking
├── experiments/      # Experiment runner
│   ├── runner.py     # Controlled comparison framework
│   └── metrics.py    # Metrics collection and analysis
├── visualization/    # Publication-quality figures
│   ├── plots.py      # Heatmaps, learning curves
│   └── animation.py  # Trajectory animations
└── theory/           # Formal analysis
    └── proofs.py     # Soundness & completeness theorems
```

## Theoretical Foundations

### Soundness Theorem

> For all states s and all proposed joint actions a, the gated action G(s,a) results in a collision-free next state.

**Formally:** ∀s ∈ S, ∀a ∈ A: T(s, G(s, a)) ∉ C

This is verified exhaustively for finite grids (15,625 state-action pairs for 5×5, 2 agents).

### Completeness Theorem

> From any non-collision state, there exists at least one joint action that the gatekeeper approves without modification.

This guarantees liveness: agents can always make progress.

### Connection to Barrier Certificates

The RSCT gatekeeper implements a discrete barrier certificate:
- h(s) = min pairwise distance (barrier function)
- Gatekeeper ensures h(s') ≥ h_min for all approved transitions
- This is the discrete analogue of Control Barrier Functions (Ames et al.)

## Experiment Design

### Environment
- **Grid:** 5×5 (default), scalable to 20×20
- **Agents:** 2 (default), scalable to 8+
- **Obstacles:** Force paths to cross in center
- **Episode length:** 30 steps max

### Agents
- **Algorithm:** Tabular Q-learning (independent learners)
- **State:** (own_pos, other_pos, own_goal)
- **Exploration:** ε-greedy, decaying from 1.0 → 0.05

### RSCT Gate
- **Compatibility check:** Collision detection + κ threshold
- **Blocking strategy:** Priority by goal distance
- **Zero training:** Pure verification layer

### Metrics
- **Safety:** Collision rate, time-to-zero-collisions
- **Performance:** Success rate, steps-to-goal, return
- **Stability:** Variance, near-misses

## Citation

If you use this code, please cite:

```bibtex
@misc{rsct-marl-experiment,
  title={RSCT vs MARL: Certified Coordination in Multi-Agent Systems},
  author={NextShift Consulting},
  year={2024},
  url={https://github.com/nextshiftconsulting/rsct-marl-experiment}
}
```

## License

Apache 2.0 - See LICENSE file.
