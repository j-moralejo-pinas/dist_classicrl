============
Environments
============

This section covers the environments supported by dist_classicrl, including built-in environments, integration with popular frameworks, and guidance for creating custom environments.

Overview
========

dist_classicrl supports multiple types of environments:

- **Built-in environments**: Custom implementations optimized for the library
- **Gymnasium integration**: Compatibility with the standard RL environment interface
- **PettingZoo integration**: Multi-agent environment support
- **Custom environments**: Framework for creating domain-specific environments

All environments follow consistent interfaces that work seamlessly with all algorithm implementations and execution modes.

Built-in Environments
=====================

TicTacToe Environment
---------------------

**Module**: ``dist_classicrl.environments.tiktaktoe_mod``

A modified Tic-Tac-Toe environment optimized for reinforcement learning research and benchmarking.

**Features**:

- Deterministic game rules
- Multi-agent support (2 players)
- Efficient state representation
- Built-in performance optimizations

**State Space**: 512 possible board configurations (3^9)
**Action Space**: 9 discrete actions (one for each board position)
**Observation**: Flattened board state as integer

**Example Usage**:

.. code-block:: python

    from dist_classicrl.environments.tiktaktoe_mod import TicTacToeEnv

    env = TicTacToeEnv()

    # Reset environment
    observation, info = env.reset()
    print(f"Initial state: {observation}")

    # Take an action
    action = 4  # Center position
    observation, reward, terminated, truncated, info = env.step(action)

    print(f"Reward: {reward}")
    print(f"Game over: {terminated or truncated}")

**Configuration Options**:

.. code-block:: python

    env = TicTacToeEnv(
        render_mode="human",  # Visualization mode
        player_symbols=['X', 'O'],  # Custom symbols
        reward_win=1.0,      # Reward for winning
        reward_lose=-1.0,    # Reward for losing
        reward_tie=0.0,      # Reward for tie
        reward_invalid=-0.1  # Penalty for invalid moves
    )

Custom Base Environment
-----------------------

**Module**: ``dist_classicrl.environments.custom_env``

A base class for implementing custom environments with dist_classicrl compatibility.

**Key Features**:

- Abstract interface for easy implementation
- Built-in support for multi-agent scenarios
- Performance optimization hooks
- Automatic compatibility with all algorithm modes

**Example Implementation**:

.. code-block:: python

    from dist_classicrl.environments.custom_env import DistClassicRLEnv
    import numpy as np

    class GridWorldEnv(DistClassicRLEnv):
        def __init__(self, width=5, height=5):
            super().__init__()
            self.width = width
            self.height = height
            self.num_agents = 1

            # Define action and observation spaces
            self.action_space_size = 4  # up, down, left, right
            self.observation_space_size = width * height

            # Initialize state
            self.agent_pos = [0, 0]
            self.goal_pos = [width-1, height-1]

        def reset(self, seed=None, options=None):
            self.agent_pos = [0, 0]
            observation = self._get_observation()
            info = {"agent_pos": self.agent_pos}
            return observation, info

        def step(self, actions):
            # Handle single agent case
            if not isinstance(actions, list):
                actions = [actions]

            action = actions[0]  # Single agent

            # Move agent
            if action == 0 and self.agent_pos[1] > 0:  # up
                self.agent_pos[1] -= 1
            elif action == 1 and self.agent_pos[1] < self.height-1:  # down
                self.agent_pos[1] += 1
            elif action == 2 and self.agent_pos[0] > 0:  # left
                self.agent_pos[0] -= 1
            elif action == 3 and self.agent_pos[0] < self.width-1:  # right
                self.agent_pos[0] += 1

            # Calculate reward
            if self.agent_pos == self.goal_pos:
                reward = 1.0
                terminated = True
            else:
                reward = -0.01  # Small penalty for each step
                terminated = False

            observation = self._get_observation()
            info = {"agent_pos": self.agent_pos}

            return observation, [reward], [terminated], [False], [info]

        def _get_observation(self):
            return self.agent_pos[1] * self.width + self.agent_pos[0]

Environment Wrappers
====================

FlattenMultiDiscrete Wrapper
-----------------------------

**Module**: ``dist_classicrl.wrappers.flatten_multidiscrete_wrapper``

A wrapper that flattens multi-discrete action spaces into single discrete actions, enabling compatibility with Q-Learning algorithms.

**Use Cases**:

- Multi-agent environments with discrete actions
- Environments with complex action spaces
- Simplifying action space for algorithm compatibility

**Example**:

.. code-block:: python

    from dist_classicrl.wrappers.flatten_multidiscrete_wrapper import FlattenMultiDiscreteWrapper
    from some_multi_agent_env import MultiAgentEnv

    base_env = MultiAgentEnv(num_agents=3, actions_per_agent=4)
    wrapped_env = FlattenMultiDiscreteWrapper(base_env)

    # Now the environment has a flattened action space
    print(f"Original action space: {base_env.action_space}")
    print(f"Flattened action space: {wrapped_env.action_space}")

External Environment Integration
================================

Gymnasium Environments
-----------------------

dist_classicrl can work with most Gymnasium environments through adapter patterns:

.. code-block:: python

    import gymnasium as gym
    from dist_classicrl.algorithms.runtime.q_learning_single_thread import SingleThreadQLearning

    # Create Gymnasium environment
    gym_env = gym.make('FrozenLake-v1', is_slippery=False)

    # Extract space information
    state_size = gym_env.observation_space.n
    action_size = gym_env.action_space.n

    # Create agent
    agent = SingleThreadQLearning(
        state_size=state_size,
        action_size=action_size
    )

    # Training loop
    for episode in range(1000):
        state, _ = gym_env.reset()
        terminated = False

        while not terminated:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = gym_env.step(action)

            agent.update(state, action, reward, next_state, terminated or truncated)
            state = next_state

PettingZoo Environments
-----------------------

For multi-agent environments, dist_classicrl integrates with PettingZoo:

.. code-block:: python

    from pettingzoo.classic import tictactoe_v3
    from dist_classicrl.algorithms.runtime.q_learning_single_thread import SingleThreadQLearning

    # Create PettingZoo environment
    env = tictactoe_v3.env()

    # Create agents for each player
    agents = {}
    for agent_name in env.possible_agents:
        agents[agent_name] = SingleThreadQLearning(
            state_size=512,  # TicTacToe state space
            action_size=9    # TicTacToe action space
        )

    # Training loop
    env.reset()
    for agent_name in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            action = None
        else:
            action = agents[agent_name].select_action(observation)

        env.step(action)

Custom Environment Development
==============================

Interface Requirements
-----------------------

All environments must implement the following interface:

.. code-block:: python

    class MyEnvironment:
        def reset(self, seed=None, options=None):
            """Reset environment to initial state.

            Returns:
                observation: Initial observation
                info: Additional information dictionary
            """
            pass

        def step(self, actions):
            """Execute one step in the environment.

            Args:
                actions: Action or list of actions for multi-agent

            Returns:
                observation: Next observation
                rewards: Reward or list of rewards
                terminated: Episode termination flag(s)
                truncated: Episode truncation flag(s)
                info: Additional information
            """
            pass

        @property
        def num_agents(self):
            """Number of agents in the environment."""
            return 1

Multi-Agent Considerations
--------------------------

For multi-agent environments, ensure consistent interfaces:

.. code-block:: python

    class MultiAgentEnvironment(DistClassicRLEnv):
        def __init__(self, num_agents):
            super().__init__()
            self.num_agents = num_agents

        def step(self, actions):
            # actions should be a list with length num_agents
            assert len(actions) == self.num_agents

            # Process actions for each agent
            observations = []
            rewards = []
            terminated = []
            truncated = []
            infos = []

            for agent_id in range(self.num_agents):
                # Process each agent's action
                obs, reward, term, trunc, info = self._step_agent(agent_id, actions[agent_id])
                observations.append(obs)
                rewards.append(reward)
                terminated.append(term)
                truncated.append(trunc)
                infos.append(info)

            return observations, rewards, terminated, truncated, infos

Performance Optimization
------------------------

For high-performance environments, consider these optimizations:

**1. Efficient State Representation**:

.. code-block:: python

    class OptimizedEnv(DistClassicRLEnv):
        def __init__(self):
            super().__init__()
            # Pre-allocate arrays
            self._state_buffer = np.zeros(self.state_size, dtype=np.int32)
            self._reward_buffer = np.zeros(self.num_agents, dtype=np.float32)

        def _get_observation(self):
            # Use pre-allocated buffer
            self._compute_state(self._state_buffer)
            return self._state_buffer.copy()

**2. Vectorized Operations**:

.. code-block:: python

    def _update_positions(self, actions):
        # Vectorized position updates
        actions = np.array(actions)
        self.positions += self.action_effects[actions]

        # Clip to bounds
        self.positions = np.clip(self.positions, 0, self.grid_size - 1)

**3. Caching Common Computations**:

.. code-block:: python

    class CachedEnv(DistClassicRLEnv):
        def __init__(self):
            super().__init__()
            self._observation_cache = {}

        def _get_observation(self):
            state_key = tuple(self.state)
            if state_key not in self._observation_cache:
                self._observation_cache[state_key] = self._compute_observation()
            return self._observation_cache[state_key]

Environment Testing
===================

Testing Framework
-----------------

Use the provided testing utilities to validate your environment:

.. code-block:: python

    from dist_classicrl.utils import validate_environment

    def test_my_environment():
        env = MyEnvironment()

        # Validate interface compliance
        validate_environment(env)

        # Test episode completion
        obs, info = env.reset()
        for _ in range(100):  # Max episode length
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                break

        assert terminated or truncated, "Episode should terminate"

Common Test Cases
-----------------

Implement these standard tests for your environment:

**1. Reset Functionality**:

.. code-block:: python

    def test_reset():
        env = MyEnvironment()

        # Initial reset
        obs1, info1 = env.reset()

        # Take some actions
        for _ in range(10):
            obs, _, _, _, _ = env.step(env.action_space.sample())

        # Reset again
        obs2, info2 = env.reset()

        # Should return to initial state
        assert obs1 == obs2

**2. Action Space Validation**:

.. code-block:: python

    def test_action_space():
        env = MyEnvironment()
        env.reset()

        # Test all valid actions
        for action in range(env.action_space_size):
            obs, reward, terminated, truncated, info = env.step(action)
            assert isinstance(reward, (int, float))
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)

**3. Episode Termination**:

.. code-block:: python

    def test_termination():
        env = MyEnvironment()
        env.reset()

        # Force terminal condition
        # (implementation depends on environment)
        env._force_terminal_state()

        obs, reward, terminated, truncated, info = env.step(0)
        assert terminated or truncated

Debugging Tools
===============

Environment Visualization
--------------------------

Add visualization capabilities for debugging:

.. code-block:: python

    class VisualizableEnv(DistClassicRLEnv):
        def render(self, mode='human'):
            if mode == 'human':
                self._render_to_screen()
            elif mode == 'rgb_array':
                return self._render_to_array()

        def _render_to_screen(self):
            # ASCII visualization
            print(f"State: {self.state}")
            print(f"Agent positions: {self.agent_positions}")

State Space Analysis
--------------------

Analyze your environment's state space:

.. code-block:: python

    def analyze_state_space(env, num_episodes=1000):
        visited_states = set()

        for episode in range(num_episodes):
            obs, _ = env.reset()
            visited_states.add(obs)

            terminated = False
            while not terminated:
                action = env.action_space.sample()
                obs, _, terminated, truncated, _ = env.step(action)
                visited_states.add(obs)
                terminated = terminated or truncated

        print(f"Visited {len(visited_states)} unique states")
        print(f"State space coverage: {len(visited_states) / env.observation_space_size:.2%}")

Best Practices
==============

Environment Design
------------------

1. **Clear Reward Structure**: Design rewards that guide learning effectively
2. **Appropriate Episode Length**: Balance exploration with computational efficiency
3. **Deterministic Behavior**: Ensure reproducible results with fixed seeds
4. **Efficient Implementation**: Optimize for the expected usage patterns

Integration Guidelines
----------------------

1. **Consistent Interfaces**: Follow the established patterns for seamless integration
2. **Error Handling**: Provide clear error messages for invalid actions or states
3. **Documentation**: Include clear documentation with examples
4. **Testing**: Comprehensive test coverage for all functionality

Performance Considerations
--------------------------

1. **Memory Usage**: Minimize memory allocation in performance-critical paths
2. **Computation Efficiency**: Use vectorized operations where possible
3. **Caching**: Cache expensive computations when appropriate
4. **Profiling**: Regular performance profiling to identify bottlenecks

See Also
========

- :doc:`algorithms`: Algorithm implementations that work with these environments
- :doc:`performance`: Performance optimization techniques
- :doc:`../tutorials`: Step-by-step environment creation tutorial
- :doc:`../development/testing`: Testing framework details
