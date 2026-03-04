"""
RSCT vs MARL: Interactive Demo

An interactive visualization comparing learned coordination (MARL)
against certified coordination (RSCT-gated) in multi-agent systems.

Run with: streamlit run demo/app.py
"""

import streamlit as st
import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment import MultiAgentGridworld, GridworldConfig, Action
from src.agents import QLearningAgent, AgentConfig
from src.gatekeeper import RSCTGatekeeper, GatekeeperConfig, BlockingStrategy

# Page config
st.set_page_config(
    page_title="RSCT vs MARL Demo",
    page_icon="🤖",
    layout="wide",
)

# Custom CSS
st.markdown("""
<style>
    .big-font {
        font-size: 24px !important;
        font-weight: bold;
    }
    .collision-bad {
        color: #d62728;
        font-size: 32px;
        font-weight: bold;
    }
    .collision-good {
        color: #2ca02c;
        font-size: 32px;
        font-weight: bold;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("🤖 RSCT vs MARL: Collision Avoidance Demo")
st.markdown("""
**Compare learned coordination (MARL) against certified coordination (RSCT-gated)**

- **MARL**: Agents learn to avoid collisions through trial and error
- **RSCT**: A static safety gate prevents collisions by construction
""")

# Sidebar controls
st.sidebar.header("⚙️ Settings")

grid_size = st.sidebar.selectbox(
    "Grid Size",
    options=[5, 7, 10],
    index=0,
    help="Size of the grid world"
)

num_agents = st.sidebar.selectbox(
    "Number of Agents",
    options=[2, 3, 4],
    index=0,
    help="Number of agents in the environment"
)

num_episodes = st.sidebar.slider(
    "Episodes to Run",
    min_value=10,
    max_value=200,
    value=50,
    step=10,
    help="Number of training episodes"
)

speed = st.sidebar.select_slider(
    "Animation Speed",
    options=["Slow", "Medium", "Fast", "Instant"],
    value="Fast",
    help="Speed of episode visualization"
)

speed_map = {"Slow": 0.5, "Medium": 0.2, "Fast": 0.05, "Instant": 0}

# Initialize session state
if 'marl_collisions' not in st.session_state:
    st.session_state.marl_collisions = []
    st.session_state.rsct_collisions = []
    st.session_state.running = False
    st.session_state.episode = 0


def create_grid_display(positions, goals, grid_size, collisions, obstacles=None):
    """Create a text-based grid display."""
    grid = [['⬜' for _ in range(grid_size)] for _ in range(grid_size)]

    # Mark obstacles
    if obstacles:
        for r, c in obstacles:
            grid[r][c] = '⬛'

    # Mark goals
    goal_symbols = ['🎯', '🏁', '⭐', '🔷']
    for i, (r, c) in enumerate(goals):
        if i < len(goal_symbols):
            grid[r][c] = goal_symbols[i]

    # Mark agents (check for collisions)
    agent_symbols = ['🔵', '🟢', '🟠', '🟣']
    pos_count = {}
    for pos in positions:
        pos_count[pos] = pos_count.get(pos, 0) + 1

    for i, (r, c) in enumerate(positions):
        if pos_count[(r, c)] > 1:
            grid[r][c] = '💥'  # Collision!
        elif i < len(agent_symbols):
            grid[r][c] = agent_symbols[i]

    # Convert to string
    return '\n'.join([''.join(row) for row in grid])


def run_episode(env, agents, gatekeeper=None, max_steps=30):
    """Run a single episode, return collision count."""
    env.reset()
    total_collisions = 0

    for step in range(max_steps):
        states = [env.get_state(i) for i in range(len(agents))]
        actions = [agents[i].act(states[i]) for i in range(len(agents))]

        if gatekeeper:
            decision = gatekeeper.check_and_gate(
                env.positions, actions, env.goals, env.obstacles
            )
            actions = decision.approved_actions

        result = env.step(actions)

        # Update agents
        next_states = [env.get_state(i) for i in range(len(agents))]
        for i in range(len(agents)):
            agents[i].update(states[i], actions[i], result.rewards[i], next_states[i], result.done)

        if result.info['collision']:
            total_collisions += 1

        if result.done:
            break

    for agent in agents:
        agent.decay_epsilon()

    return total_collisions


# Main content
col1, col2 = st.columns(2)

with col1:
    st.subheader("🎓 MARL (Learned)")
    marl_grid = st.empty()
    marl_metric = st.empty()

with col2:
    st.subheader("🛡️ RSCT-Gated (Certified)")
    rsct_grid = st.empty()
    rsct_metric = st.empty()

# Control buttons
col_btn1, col_btn2, col_btn3 = st.columns(3)

with col_btn1:
    run_button = st.button("▶️ Run Experiment", type="primary", use_container_width=True)

with col_btn2:
    reset_button = st.button("🔄 Reset", use_container_width=True)

with col_btn3:
    stop_button = st.button("⏹️ Stop", use_container_width=True)

if reset_button:
    st.session_state.marl_collisions = []
    st.session_state.rsct_collisions = []
    st.session_state.episode = 0
    st.rerun()

if stop_button:
    st.session_state.running = False

# Progress bar
progress_bar = st.progress(0)
status_text = st.empty()

# Results chart
chart_placeholder = st.empty()

if run_button:
    st.session_state.running = True
    st.session_state.marl_collisions = []
    st.session_state.rsct_collisions = []

    # Create environments and agents
    agent_configs = []
    if num_agents == 2:
        agent_configs = [((0, 0), (grid_size-1, grid_size-1)),
                         ((grid_size-1, 0), (0, grid_size-1))]
    elif num_agents == 3:
        agent_configs = [((0, 0), (grid_size-1, grid_size-1)),
                         ((grid_size-1, 0), (0, grid_size-1)),
                         ((0, grid_size-1), (grid_size-1, 0))]
    else:
        agent_configs = [((0, 0), (grid_size-1, grid_size-1)),
                         ((grid_size-1, 0), (0, grid_size-1)),
                         ((0, grid_size-1), (grid_size-1, 0)),
                         ((grid_size-1, grid_size-1), (0, 0))]

    config = GridworldConfig(
        grid_size=grid_size,
        agent_configs=agent_configs,
        obstacles=[],
        max_steps=30,
    )

    # MARL setup
    marl_env = MultiAgentGridworld(config)
    marl_agents = [QLearningAgent(i, config=AgentConfig(epsilon_decay=0.98))
                   for i in range(num_agents)]

    # RSCT setup
    rsct_env = MultiAgentGridworld(config)
    rsct_agents = [QLearningAgent(i, config=AgentConfig(epsilon_decay=0.98))
                   for i in range(num_agents)]
    rsct_gate = RSCTGatekeeper(GatekeeperConfig(
        grid_size=grid_size,
        kappa_min=0.0,
        blocking_strategy=BlockingStrategy.PRIORITY_BY_DISTANCE,
    ))

    # Run episodes
    for episode in range(num_episodes):
        if not st.session_state.running:
            break

        # Run MARL episode
        marl_collisions = run_episode(marl_env, marl_agents, gatekeeper=None)
        st.session_state.marl_collisions.append(marl_collisions)

        # Run RSCT episode
        rsct_collisions = run_episode(rsct_env, rsct_agents, gatekeeper=rsct_gate)
        st.session_state.rsct_collisions.append(rsct_collisions)

        # Update displays
        marl_grid.code(create_grid_display(
            marl_env.positions, marl_env.goals, grid_size, marl_collisions
        ))
        rsct_grid.code(create_grid_display(
            rsct_env.positions, rsct_env.goals, grid_size, rsct_collisions
        ))

        total_marl = sum(st.session_state.marl_collisions)
        total_rsct = sum(st.session_state.rsct_collisions)

        marl_metric.markdown(
            f'<p class="collision-bad">Collisions: {total_marl}</p>',
            unsafe_allow_html=True
        )
        rsct_metric.markdown(
            f'<p class="collision-good">Collisions: {total_rsct}</p>',
            unsafe_allow_html=True
        )

        # Update progress
        progress_bar.progress((episode + 1) / num_episodes)
        status_text.text(f"Episode {episode + 1}/{num_episodes}")

        # Update chart
        if len(st.session_state.marl_collisions) > 1:
            import pandas as pd
            chart_data = pd.DataFrame({
                'MARL': np.cumsum(st.session_state.marl_collisions),
                'RSCT-Gated': np.cumsum(st.session_state.rsct_collisions),
            })
            chart_placeholder.line_chart(chart_data, use_container_width=True)

        time.sleep(speed_map[speed])

    st.session_state.running = False
    status_text.text("✅ Complete!")

    # Final summary
    st.success(f"""
    **Results Summary:**
    - MARL Total Collisions: {sum(st.session_state.marl_collisions)}
    - RSCT Total Collisions: {sum(st.session_state.rsct_collisions)}
    - Collision Reduction: {sum(st.session_state.marl_collisions) - sum(st.session_state.rsct_collisions)} fewer collisions with RSCT
    """)

# Display existing data
if st.session_state.marl_collisions and not st.session_state.running:
    import pandas as pd
    chart_data = pd.DataFrame({
        'MARL': np.cumsum(st.session_state.marl_collisions),
        'RSCT-Gated': np.cumsum(st.session_state.rsct_collisions),
    })
    chart_placeholder.line_chart(chart_data, use_container_width=True)

    total_marl = sum(st.session_state.marl_collisions)
    total_rsct = sum(st.session_state.rsct_collisions)

    marl_metric.markdown(
        f'<p class="collision-bad">Collisions: {total_marl}</p>',
        unsafe_allow_html=True
    )
    rsct_metric.markdown(
        f'<p class="collision-good">Collisions: {total_rsct}</p>',
        unsafe_allow_html=True
    )

# Footer
st.markdown("---")
st.markdown("""
**How it works:**
- **MARL** agents learn via Q-learning with collision penalties (-20 reward)
- **RSCT Gate** checks each proposed action and blocks unsafe moves before execution
- The gate requires **zero training** - safety is guaranteed from episode 1

[View on GitHub](https://github.com/NextShiftConsulting/rsct-marl-experiment)
""")
