============
Architecture
============

This section provides a comprehensive overview of the dist_classicrl architecture, including design principles, component interactions, and implementation details.

Overview
========

dist_classicrl is designed with the following core principles:

- **Modularity**: Clear separation of concerns with pluggable components
- **Performance**: Optimized implementations for different execution contexts
- **Scalability**: Support for single-core to large-scale distributed deployment
- **Extensibility**: Easy to extend with new algorithms and environments
- **Maintainability**: Clean code structure with comprehensive testing

The architecture follows a layered design pattern with clear interfaces between components.

High-Level Architecture
=======================

System Components
-----------------

.. code-block::

    ┌─────────────────────────────────────────────────────────────┐
    │                    Application Layer                        │
    │  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
    │  │   User Scripts  │  │   Benchmarks    │  │    Tests     │ │
    │  └─────────────────┘  └─────────────────┘  └──────────────┘ │
    └─────────────────────────────────────────────────────────────┘
                                    │
    ┌─────────────────────────────────────────────────────────────┐
    │                    Runtime Layer                            │
    │  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
    │  │ Single-threaded │  │    Parallel     │  │ Distributed  │ │
    │  │    Runtime      │  │    Runtime      │  │   Runtime    │ │
    │  └─────────────────┘  └─────────────────┘  └──────────────┘ │
    └─────────────────────────────────────────────────────────────┘
                                    │
    ┌─────────────────────────────────────────────────────────────┐
    │                   Algorithm Layer                           │
    │  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
    │  │  Q-Learning     │  │   Q-Learning    │  │  Q-Learning  │ │
    │  │    Optimal      │  │     List        │  │    NumPy     │ │
    │  └─────────────────┘  └─────────────────┘  └──────────────┘ │
    └─────────────────────────────────────────────────────────────┘
                                    │
    ┌─────────────────────────────────────────────────────────────┐
    │                 Environment Layer                           │
    │  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
    │  │   Built-in      │  │    Gymnasium    │  │  PettingZoo  │ │
    │  │ Environments    │  │  Integration    │  │ Integration  │ │
    │  └─────────────────┘  └─────────────────┘  └──────────────┘ │
    └─────────────────────────────────────────────────────────────┘
                                    │
    ┌─────────────────────────────────────────────────────────────┐
    │                    Utility Layer                            │
    │  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
    │  │   Wrappers      │  │    Buffers      │  │  Utilities   │ │
    │  └─────────────────┘  └─────────────────┘  └──────────────┘ │
    └─────────────────────────────────────────────────────────────┘

Layer Responsibilities
----------------------

**Application Layer**:
- User-facing scripts and applications
- Benchmarking and performance testing
- Unit and integration tests

**Runtime Layer**:
- Execution mode management (single/parallel/distributed)
- Resource allocation and scheduling
- Communication coordination

**Algorithm Layer**:
- Core reinforcement learning algorithms
- Different Q-Learning implementations
- Algorithm-specific optimizations

**Environment Layer**:
- Environment interfaces and implementations
- External framework integrations
- Environment wrappers and utilities

**Utility Layer**:
- Shared utilities and helper functions
- Data structures and buffers
- Common functionality

Core Components
===============

Base Algorithm Design
---------------------

All algorithms inherit from a common base class to ensure consistent interfaces:

.. code-block:: python

    from abc import ABC, abstractmethod
    import numpy as np

    class BaseAlgorithm(ABC):
        """Abstract base class for all RL algorithms."""

        def __init__(self, state_size, action_size, learning_rate=0.1,
                     discount_factor=0.99, epsilon=0.1, epsilon_decay=0.995,
                     epsilon_min=0.01):
            self.state_size = state_size
            self.action_size = action_size
            self.learning_rate = learning_rate
            self.discount_factor = discount_factor
            self.epsilon = epsilon
            self.epsilon_decay = epsilon_decay
            self.epsilon_min = epsilon_min

            # Initialize Q-table
            self.q_table = self._initialize_q_table()

            # Training statistics
            self.step_count = 0
            self.episode_count = 0

        @abstractmethod
        def _initialize_q_table(self):
            """Initialize the Q-table structure."""
            pass

        @abstractmethod
        def update(self, state, action, reward, next_state, done):
            """Update the algorithm with a single experience."""
            pass

        @abstractmethod
        def select_action(self, state):
            """Select an action given a state."""
            pass

        def train_episode(self, env):
            """Train for one complete episode."""
            observation, _ = env.reset()
            terminated = False
            total_reward = 0

            while not terminated:
                action = self.select_action(observation)
                next_observation, reward, terminated, truncated, _ = env.step(action)

                self.update(observation, action, reward, next_observation,
                           terminated or truncated)

                observation = next_observation
                total_reward += reward
                self.step_count += 1

            self.episode_count += 1
            self._update_epsilon()

            return total_reward

        def _update_epsilon(self):
            """Update exploration rate."""
            self.epsilon = max(self.epsilon_min,
                             self.epsilon * self.epsilon_decay)

Q-Learning Implementations
--------------------------

The library provides three main Q-Learning implementations, each optimized for different use cases:

**1. Optimal Q-Learning** (``q_learning_optimal.py``):

.. code-block:: python

    class OptimalQLearning(BaseAlgorithm):
        """Balanced Q-Learning implementation with multiple optimization strategies."""

        def _initialize_q_table(self):
            return np.zeros((self.state_size, self.action_size), dtype=np.float32)

        def update(self, state, action, reward, next_state, done):
            if not done:
                next_q_max = np.max(self.q_table[next_state])
            else:
                next_q_max = 0

            target = reward + self.discount_factor * next_q_max
            current_q = self.q_table[state, action]

            # Q-Learning update with optimized computation
            self.q_table[state, action] = current_q + self.learning_rate * (target - current_q)

        def select_action(self, state):
            if np.random.random() < self.epsilon:
                return np.random.randint(self.action_size)
            else:
                return np.argmax(self.q_table[state])

**2. List-Based Q-Learning** (``q_learning_list.py``):

.. code-block:: python

    class ListQLearning(BaseAlgorithm):
        """Memory-efficient Q-Learning using Python lists."""

        def _initialize_q_table(self):
            return [[0.0] * self.action_size for _ in range(self.state_size)]

        def update(self, state, action, reward, next_state, done):
            if not done:
                next_q_max = max(self.q_table[next_state])
            else:
                next_q_max = 0

            target = reward + self.discount_factor * next_q_max
            current_q = self.q_table[state][action]

            self.q_table[state][action] = current_q + self.learning_rate * (target - current_q)

        def select_action(self, state):
            if random.random() < self.epsilon:
                return random.randint(0, self.action_size - 1)
            else:
                return self.q_table[state].index(max(self.q_table[state]))

**3. NumPy-Based Q-Learning** (``q_learning_numpy.py``):

.. code-block:: python

    class NumpyQLearning(BaseAlgorithm):
        """High-performance Q-Learning using vectorized NumPy operations."""

        def _initialize_q_table(self):
            return np.zeros((self.state_size, self.action_size), dtype=np.float32)

        def update(self, state, action, reward, next_state, done):
            # Vectorized update computation
            next_q_max = 0 if done else np.max(self.q_table[next_state])
            target = reward + self.discount_factor * next_q_max

            # In-place update for memory efficiency
            self.q_table[state, action] += self.learning_rate * (target - self.q_table[state, action])

        def select_action(self, state):
            if np.random.random() < self.epsilon:
                return np.random.randint(self.action_size)
            else:
                return np.argmax(self.q_table[state])

        def batch_update(self, experiences):
            """Vectorized batch update for improved performance."""
            states, actions, rewards, next_states, dones = zip(*experiences)

            states = np.array(states)
            actions = np.array(actions)
            rewards = np.array(rewards)
            next_states = np.array(next_states)
            dones = np.array(dones)

            # Vectorized computation
            next_q_max = np.max(self.q_table[next_states], axis=1)
            next_q_max[dones] = 0

            targets = rewards + self.discount_factor * next_q_max
            current_q = self.q_table[states, actions]

            # Batch update
            self.q_table[states, actions] += self.learning_rate * (targets - current_q)

Runtime Architecture
====================

Execution Modes
----------------

The runtime layer provides different execution modes that wrap the base algorithms:

**Single-Threaded Runtime** (``q_learning_single_thread.py``):

.. code-block:: python

    class SingleThreadQLearning:
        """Single-threaded runtime wrapper for Q-Learning algorithms."""

        def __init__(self, algorithm_class=OptimalQLearning, **kwargs):
            self.algorithm = algorithm_class(**kwargs)
            self.training_stats = TrainingStatistics()

        def train(self, env, episodes=None, steps=None):
            """Train the agent with specified episodes or steps."""
            if episodes:
                for episode in range(episodes):
                    reward = self.algorithm.train_episode(env)
                    self.training_stats.record_episode(reward)
            elif steps:
                current_steps = 0
                while current_steps < steps:
                    reward = self.algorithm.train_episode(env)
                    current_steps = self.algorithm.step_count
                    self.training_stats.record_episode(reward)

**Parallel Runtime** (``q_learning_parallel.py``):

.. code-block:: python

    import multiprocessing as mp
    from multiprocessing import shared_memory

    class ParallelQLearning:
        """Parallel runtime using multiprocessing."""

        def __init__(self, algorithm_class=OptimalQLearning, num_processes=None, **kwargs):
            self.algorithm_class = algorithm_class
            self.num_processes = num_processes or mp.cpu_count()
            self.kwargs = kwargs

            # Shared memory for Q-table
            self.shared_q_table = None
            self.processes = []

        def _create_shared_q_table(self, state_size, action_size):
            """Create shared memory Q-table for inter-process communication."""
            q_table_size = state_size * action_size * 4  # float32
            self.shared_memory = shared_memory.SharedMemory(create=True, size=q_table_size)

            # Create numpy array backed by shared memory
            self.shared_q_table = np.ndarray(
                (state_size, action_size),
                dtype=np.float32,
                buffer=self.shared_memory.buf
            )
            self.shared_q_table[:] = 0

        def _worker_process(self, worker_id, env_factory, steps_per_worker, shared_memory_name):
            """Worker process for parallel training."""
            # Attach to shared memory
            existing_shm = shared_memory.SharedMemory(name=shared_memory_name)
            q_table = np.ndarray(
                (self.kwargs['state_size'], self.kwargs['action_size']),
                dtype=np.float32,
                buffer=existing_shm.buf
            )

            # Create local algorithm instance
            algorithm = self.algorithm_class(**self.kwargs)
            algorithm.q_table = q_table

            # Create environment
            env = env_factory()

            # Train for specified steps
            current_steps = 0
            while current_steps < steps_per_worker:
                algorithm.train_episode(env)
                current_steps = algorithm.step_count

        def train(self, env_factory, steps, steps_per_sync=1000):
            """Train using multiple parallel processes."""
            self._create_shared_q_table(self.kwargs['state_size'], self.kwargs['action_size'])

            steps_per_worker = steps // self.num_processes

            # Start worker processes
            for worker_id in range(self.num_processes):
                process = mp.Process(
                    target=self._worker_process,
                    args=(worker_id, env_factory, steps_per_worker, self.shared_memory.name)
                )
                process.start()
                self.processes.append(process)

            # Wait for all processes to complete
            for process in self.processes:
                process.join()

**Distributed Runtime** (``q_learning_async_dist.py``):

.. code-block:: python

    from mpi4py import MPI
    import numpy as np

    class DistAsyncQLearning:
        """Distributed runtime using MPI for cluster deployment."""

        def __init__(self, algorithm_class=OptimalQLearning, batch_size=32,
                     sync_frequency=100, **kwargs):
            self.algorithm_class = algorithm_class
            self.batch_size = batch_size
            self.sync_frequency = sync_frequency
            self.kwargs = kwargs

            # MPI setup
            self.comm = MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()

            # Create local algorithm instance
            self.algorithm = algorithm_class(**kwargs)

            # Communication buffers
            self.update_buffer = []
            self.sync_counter = 0

        def train(self, env, steps):
            """Distributed training with asynchronous parameter updates."""
            current_steps = 0

            while current_steps < steps:
                # Train one episode
                reward = self.algorithm.train_episode(env)
                current_steps = self.algorithm.step_count

                # Collect updates for batching
                if len(self.update_buffer) >= self.batch_size:
                    self._sync_parameters()
                    self.update_buffer.clear()

                # Periodic synchronization
                if self.sync_counter % self.sync_frequency == 0:
                    self._global_sync()

                self.sync_counter += 1

        def _sync_parameters(self):
            """Synchronize parameters across all workers."""
            if self.rank == 0:  # Parameter server
                # Collect updates from all workers
                all_updates = self.comm.gather(self.update_buffer, root=0)

                # Apply aggregated updates
                self._apply_updates(all_updates)

                # Broadcast updated Q-table
                self.comm.bcast(self.algorithm.q_table, root=0)
            else:  # Worker
                # Send updates to parameter server
                self.comm.gather(self.update_buffer, root=0)

                # Receive updated Q-table
                self.algorithm.q_table = self.comm.bcast(None, root=0)

        def _global_sync(self):
            """Global synchronization using all-reduce."""
            # All-reduce Q-table across all workers
            self.comm.Allreduce(MPI.IN_PLACE, self.algorithm.q_table, op=MPI.SUM)
            self.algorithm.q_table /= self.size

Environment Architecture
========================

Environment Interface
----------------------

All environments implement a consistent interface based on the Gymnasium standard:

.. code-block:: python

    from abc import ABC, abstractmethod

    class DistClassicRLEnv(ABC):
        """Base environment class for dist_classicrl."""

        def __init__(self):
            self.num_agents = 1
            self.current_step = 0
            self.max_episode_steps = 1000

        @abstractmethod
        def reset(self, seed=None, options=None):
            """Reset environment to initial state."""
            pass

        @abstractmethod
        def step(self, actions):
            """Execute one step in the environment."""
            pass

        @property
        @abstractmethod
        def observation_space_size(self):
            """Size of the observation space."""
            pass

        @property
        @abstractmethod
        def action_space_size(self):
            """Size of the action space."""
            pass

        def render(self, mode='human'):
            """Render the environment (optional)."""
            pass

        def close(self):
            """Clean up environment resources."""
            pass

Built-in Environment Implementation
-----------------------------------

The TicTacToe environment demonstrates the implementation pattern:

.. code-block:: python

    class TicTacToeEnv(DistClassicRLEnv):
        """Optimized TicTacToe environment for RL training."""

        def __init__(self, render_mode=None):
            super().__init__()
            self.render_mode = render_mode
            self.num_agents = 2

            # Game state
            self.board = np.zeros(9, dtype=np.int8)
            self.current_player = 1
            self.game_over = False

            # Pre-computed winning conditions
            self.winning_conditions = [
                [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
                [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
                [0, 4, 8], [2, 4, 6]              # Diagonals
            ]

        def reset(self, seed=None, options=None):
            if seed is not None:
                np.random.seed(seed)

            self.board.fill(0)
            self.current_player = 1
            self.game_over = False
            self.current_step = 0

            return self._get_observation(), {}

        def step(self, action):
            if self.game_over:
                return self._get_observation(), 0, True, False, {}

            # Validate action
            if not self._is_valid_action(action):
                return self._get_observation(), -1, True, False, {"invalid_move": True}

            # Apply action
            self.board[action] = self.current_player
            self.current_step += 1

            # Check for win/draw
            reward, terminated = self._check_game_end()

            # Switch player
            self.current_player = 3 - self.current_player  # Switch between 1 and 2

            observation = self._get_observation()
            info = {"current_player": self.current_player}

            return observation, reward, terminated, False, info

        def _get_observation(self):
            """Convert board state to integer observation."""
            # Convert 3x3 board to base-3 integer
            observation = 0
            for i, cell in enumerate(self.board):
                observation += cell * (3 ** i)
            return observation

        def _is_valid_action(self, action):
            """Check if action is valid."""
            return 0 <= action < 9 and self.board[action] == 0

        def _check_game_end(self):
            """Check if game has ended and calculate reward."""
            # Check for win
            for condition in self.winning_conditions:
                if (self.board[condition[0]] == self.board[condition[1]] ==
                    self.board[condition[2]] != 0):
                    winner = self.board[condition[0]]
                    reward = 1 if winner == self.current_player else -1
                    self.game_over = True
                    return reward, True

            # Check for draw
            if np.all(self.board != 0):
                self.game_over = True
                return 0, True

            return 0, False

        @property
        def observation_space_size(self):
            return 3 ** 9  # 19683 possible board states

        @property
        def action_space_size(self):
            return 9  # 9 possible moves

Wrapper Architecture
====================

Environment wrappers provide additional functionality without modifying the base environment:

.. code-block:: python

    class FlattenMultiDiscreteWrapper:
        """Wrapper to flatten multi-discrete action spaces."""

        def __init__(self, env):
            self.env = env
            self.wrapped_env = env

            # Calculate flattened action space
            if hasattr(env, 'action_space_sizes'):
                self.action_space_size = np.prod(env.action_space_sizes)
                self.action_space_sizes = env.action_space_sizes
            else:
                self.action_space_size = env.action_space_size
                self.action_space_sizes = [env.action_space_size]

        def reset(self, **kwargs):
            return self.env.reset(**kwargs)

        def step(self, action):
            # Convert flattened action to multi-discrete
            if len(self.action_space_sizes) > 1:
                multi_action = self._unflatten_action(action)
            else:
                multi_action = action

            return self.env.step(multi_action)

        def _unflatten_action(self, flat_action):
            """Convert flat action to multi-discrete action."""
            actions = []
            remaining = flat_action

            for size in reversed(self.action_space_sizes):
                actions.append(remaining % size)
                remaining //= size

            return list(reversed(actions))

        def __getattr__(self, name):
            """Delegate attribute access to wrapped environment."""
            return getattr(self.env, name)

Utility Components
==================

Buffer Management
-----------------

Efficient buffer implementations for experience storage:

.. code-block:: python

    from collections import deque
    import numpy as np

    class ExperienceBuffer:
        """Circular buffer for storing training experiences."""

        def __init__(self, capacity=10000):
            self.capacity = capacity
            self.buffer = deque(maxlen=capacity)
            self.position = 0

        def push(self, state, action, reward, next_state, done):
            """Add an experience to the buffer."""
            experience = (state, action, reward, next_state, done)
            self.buffer.append(experience)

        def sample(self, batch_size):
            """Sample a batch of experiences."""
            if len(self.buffer) < batch_size:
                return list(self.buffer)

            indices = np.random.choice(len(self.buffer), batch_size, replace=False)
            return [self.buffer[i] for i in indices]

        def __len__(self):
            return len(self.buffer)

Training Statistics
-------------------

Comprehensive training statistics collection:

.. code-block:: python

    import time
    from collections import defaultdict

    class TrainingStatistics:
        """Collect and manage training statistics."""

        def __init__(self):
            self.start_time = time.time()
            self.episode_rewards = []
            self.episode_lengths = []
            self.step_times = []
            self.metrics = defaultdict(list)

        def record_episode(self, reward, length=None):
            """Record episode statistics."""
            self.episode_rewards.append(reward)
            if length is not None:
                self.episode_lengths.append(length)

        def record_step_time(self, step_time):
            """Record step execution time."""
            self.step_times.append(step_time)

        def record_metric(self, name, value):
            """Record custom metric."""
            self.metrics[name].append(value)

        def get_summary(self):
            """Get training summary statistics."""
            if not self.episode_rewards:
                return {}

            return {
                'total_episodes': len(self.episode_rewards),
                'mean_reward': np.mean(self.episode_rewards),
                'std_reward': np.std(self.episode_rewards),
                'min_reward': np.min(self.episode_rewards),
                'max_reward': np.max(self.episode_rewards),
                'mean_episode_length': np.mean(self.episode_lengths) if self.episode_lengths else None,
                'total_training_time': time.time() - self.start_time,
                'mean_step_time': np.mean(self.step_times) if self.step_times else None
            }

Performance Optimizations
=========================

Memory Management
-----------------

The architecture includes several memory optimization strategies:

**1. Pre-allocation**: Pre-allocate arrays to avoid repeated memory allocation
**2. In-place Operations**: Use in-place operations to minimize memory copying
**3. Shared Memory**: Use shared memory for multi-process communication
**4. Memory Pooling**: Reuse memory buffers across operations

**Example Implementation**:

.. code-block:: python

    class MemoryOptimizedQLearning(BaseAlgorithm):
        """Q-Learning with memory optimizations."""

        def __init__(self, **kwargs):
            super().__init__(**kwargs)

            # Pre-allocate temporary arrays
            self._temp_q_values = np.zeros(self.action_size, dtype=np.float32)
            self._temp_next_q = np.zeros(1, dtype=np.float32)

            # Memory pool for batch operations
            self.memory_pool = MemoryPool()

        def select_action(self, state):
            # Use pre-allocated array
            np.copyto(self._temp_q_values, self.q_table[state])

            if np.random.random() < self.epsilon:
                return np.random.randint(self.action_size)
            else:
                return np.argmax(self._temp_q_values)

CPU Optimizations
-----------------

CPU-specific optimizations for different algorithms:

**1. Vectorization**: Use NumPy's vectorized operations
**2. Cache Optimization**: Optimize memory access patterns
**3. SIMD Instructions**: Leverage CPU SIMD capabilities
**4. Compiler Optimizations**: Use Numba for JIT compilation

**Example with Numba**:

.. code-block:: python

    from numba import jit

    @jit(nopython=True)
    def vectorized_q_update(q_table, states, actions, rewards, next_states,
                           dones, learning_rate, discount_factor):
        """Vectorized Q-learning update with JIT compilation."""
        for i in range(len(states)):
            state = states[i]
            action = actions[i]
            reward = rewards[i]
            next_state = next_states[i]
            done = dones[i]

            if done:
                next_q_max = 0.0
            else:
                next_q_max = np.max(q_table[next_state])

            target = reward + discount_factor * next_q_max
            current_q = q_table[state, action]
            q_table[state, action] = current_q + learning_rate * (target - current_q)

Testing Architecture
====================

The testing architecture ensures reliability and performance:

**Unit Tests**: Test individual components in isolation
**Integration Tests**: Test component interactions
**Performance Tests**: Benchmark performance characteristics
**Regression Tests**: Ensure changes don't break existing functionality

**Example Test Structure**:

.. code-block:: python

    import unittest
    import numpy as np
    from dist_classicrl.algorithms.base_algorithms.q_learning_optimal import OptimalQLearning

    class TestOptimalQLearning(unittest.TestCase):
        """Test suite for OptimalQLearning algorithm."""

        def setUp(self):
            """Set up test fixtures."""
            self.agent = OptimalQLearning(state_size=10, action_size=4)
            self.env = MockEnvironment()

        def test_initialization(self):
            """Test proper initialization."""
            self.assertEqual(self.agent.q_table.shape, (10, 4))
            self.assertTrue(np.all(self.agent.q_table == 0))

        def test_action_selection(self):
            """Test action selection logic."""
            # Test random action during exploration
            self.agent.epsilon = 1.0
            action = self.agent.select_action(0)
            self.assertIn(action, range(4))

            # Test greedy action during exploitation
            self.agent.epsilon = 0.0
            self.agent.q_table[0] = [1, 2, 3, 0]
            action = self.agent.select_action(0)
            self.assertEqual(action, 2)  # Should select action with highest Q-value

        def test_q_learning_update(self):
            """Test Q-learning update rule."""
            initial_q = self.agent.q_table[0, 1]
            self.agent.update(state=0, action=1, reward=1, next_state=1, done=False)
            updated_q = self.agent.q_table[0, 1]

            # Q-value should have increased
            self.assertGreater(updated_q, initial_q)

Extension Points
================

The architecture provides clear extension points for adding new functionality:

**1. New Algorithms**: Inherit from BaseAlgorithm
**2. New Environments**: Inherit from DistClassicRLEnv
**3. New Runtime Modes**: Implement runtime interface
**4. New Wrappers**: Follow wrapper pattern
**5. New Utilities**: Add to utility modules

This modular design ensures that the library can evolve while maintaining backward compatibility and clear interfaces.

See Also
========

- :doc:`testing`: Testing framework and guidelines
- :doc:`../user_guide/algorithms`: User guide for algorithms
- :doc:`../user_guide/performance`: Performance optimization techniques
- :doc:`../user_guide/distributed`: Distributed training architecture
