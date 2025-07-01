=======
Testing
=======

This section covers the comprehensive testing framework used in dist_classicrl, including unit tests, integration tests, performance tests, and distributed system testing.

Overview
========

dist_classicrl employs a multi-layered testing strategy to ensure:

- **Correctness**: All algorithms and components work as expected
- **Performance**: Performance characteristics meet requirements
- **Reliability**: System handles edge cases and failures gracefully
- **Scalability**: Components scale properly across different configurations
- **Compatibility**: Integration with external libraries works correctly

The testing framework includes:

- Unit tests for individual components
- Integration tests for component interactions
- Performance benchmarks and regression tests
- Distributed system tests with MPI
- Continuous integration and automated testing

Test Structure
==============

Directory Organization
----------------------

.. code-block::

    tests/
    ├── conftest.py                 # Pytest configuration and fixtures
    ├── dist_classicrl/
    │   ├── algorithms/
    │   │   ├── base_algorithms/    # Base algorithm tests
    │   │   └── runtime/           # Runtime execution tests
    │   └── environment/           # Environment tests
    └── utils/
        ├── mock_env.py            # Mock environments for testing
        └── test_utilities.py      # Testing utilities and helpers

Test Categories
---------------

**Unit Tests** (``test_*.py``):
- Test individual functions and classes in isolation
- Fast execution (< 1 second per test)
- No external dependencies or side effects

**Integration Tests** (``test_integration_*.py``):
- Test component interactions
- Environment and algorithm combinations
- Multi-component workflows

**Performance Tests** (``test_performance_*.py``):
- Benchmark execution speed and memory usage
- Performance regression detection
- Scalability validation

**Distributed Tests** (``test_distributed_*.py``):
- MPI-based distributed functionality
- Multi-node communication testing
- Fault tolerance validation

Running Tests
=============

Basic Test Execution
--------------------

.. code-block:: bash

    # Run all tests
    pytest

    # Run with verbose output
    pytest -v

    # Run specific test file
    pytest tests/dist_classicrl/algorithms/test_q_learning_optimal.py

    # Run specific test function
    pytest tests/dist_classicrl/algorithms/test_q_learning_optimal.py::TestOptimalQLearning::test_initialization

    # Run tests with coverage
    pytest --cov=dist_classicrl --cov-report=html

Test Configuration
------------------

.. code-block:: bash

    # Run with specific markers
    pytest -m "not slow"           # Skip slow tests
    pytest -m "performance"       # Run only performance tests
    pytest -m "distributed"       # Run only distributed tests

    # Run with parallel execution
    pytest -n 4                   # Use 4 parallel workers

    # Run with specific timeout
    pytest --timeout=300           # 5-minute timeout per test

Performance Testing
-------------------

.. code-block:: bash

    # Run performance benchmarks
    pytest tests/dist_classicrl/algorithms/test_performance_*.py -v

    # Generate performance report
    pytest --benchmark-only --benchmark-sort=mean

    # Compare with baseline
    pytest --benchmark-compare=baseline.json

Distributed Testing
-------------------

.. code-block:: bash

    # Run distributed tests with MPI
    mpirun -n 4 pytest tests/dist_classicrl/test_distributed_*.py

    # Run on cluster
    srun -n 8 --mpi=pmix pytest tests/dist_classicrl/test_distributed_*.py

Unit Testing
============

Algorithm Testing
-----------------

Example unit test for Q-Learning algorithms:

.. code-block:: python

    import unittest
    import numpy as np
    from dist_classicrl.algorithms.base_algorithms.q_learning_optimal import OptimalQLearning
    from tests.utils.mock_env import MockDiscreteEnv

    class TestOptimalQLearning(unittest.TestCase):
        """Comprehensive test suite for OptimalQLearning."""

        def setUp(self):
            """Set up test fixtures before each test method."""
            self.state_size = 10
            self.action_size = 4
            self.agent = OptimalQLearning(
                state_size=self.state_size,
                action_size=self.action_size,
                learning_rate=0.1,
                discount_factor=0.9,
                epsilon=0.1
            )
            self.env = MockDiscreteEnv(self.state_size, self.action_size)

        def test_initialization(self):
            """Test proper initialization of Q-Learning agent."""
            # Check Q-table shape
            self.assertEqual(self.agent.q_table.shape, (self.state_size, self.action_size))

            # Check Q-table initialization (should be zeros)
            self.assertTrue(np.all(self.agent.q_table == 0))

            # Check hyperparameters
            self.assertEqual(self.agent.learning_rate, 0.1)
            self.assertEqual(self.agent.discount_factor, 0.9)
            self.assertEqual(self.agent.epsilon, 0.1)

        def test_action_selection_exploration(self):
            """Test action selection during exploration."""
            # Force exploration
            self.agent.epsilon = 1.0

            # Should select random actions
            actions = [self.agent.select_action(0) for _ in range(100)]
            unique_actions = set(actions)

            # Should explore multiple actions
            self.assertGreater(len(unique_actions), 1)

            # All actions should be valid
            for action in actions:
                self.assertIn(action, range(self.action_size))

        def test_action_selection_exploitation(self):
            """Test action selection during exploitation."""
            # Force exploitation
            self.agent.epsilon = 0.0

            # Set up Q-values with clear best action
            self.agent.q_table[0] = [1, 3, 2, 0]  # Action 1 is best

            # Should consistently select best action
            for _ in range(10):
                action = self.agent.select_action(0)
                self.assertEqual(action, 1)

        def test_q_learning_update(self):
            """Test Q-learning update rule."""
            # Initial state
            state = 0
            action = 1
            reward = 1.0
            next_state = 2
            done = False

            # Set up next state Q-values
            self.agent.q_table[next_state] = [0, 2, 1, 0]  # Max = 2

            initial_q = self.agent.q_table[state, action]

            # Perform update
            self.agent.update(state, action, reward, next_state, done)

            # Calculate expected Q-value
            expected_q = initial_q + self.agent.learning_rate * (
                reward + self.agent.discount_factor * 2 - initial_q
            )

            # Check if Q-value updated correctly
            self.assertAlmostEqual(self.agent.q_table[state, action], expected_q, places=6)

        def test_terminal_state_update(self):
            """Test Q-learning update for terminal states."""
            state = 0
            action = 1
            reward = 1.0
            next_state = 2
            done = True

            initial_q = self.agent.q_table[state, action]

            # Perform update
            self.agent.update(state, action, reward, next_state, done)

            # Expected Q-value (no future reward for terminal states)
            expected_q = initial_q + self.agent.learning_rate * (reward - initial_q)

            self.assertAlmostEqual(self.agent.q_table[state, action], expected_q, places=6)

        def test_epsilon_decay(self):
            """Test epsilon decay functionality."""
            initial_epsilon = self.agent.epsilon

            # Train for multiple episodes
            for _ in range(10):
                self.agent.train_episode(self.env)

            # Epsilon should have decayed
            self.assertLess(self.agent.epsilon, initial_epsilon)

            # Epsilon should not go below minimum
            self.assertGreaterEqual(self.agent.epsilon, self.agent.epsilon_min)

        def test_training_statistics(self):
            """Test training statistics collection."""
            # Train for several episodes
            episode_count = 5
            for _ in range(episode_count):
                self.agent.train_episode(self.env)

            # Check episode count
            self.assertEqual(self.agent.episode_count, episode_count)

            # Check step count
            self.assertGreater(self.agent.step_count, 0)

Environment Testing
-------------------

Example test for environment implementations:

.. code-block:: python

    import unittest
    import numpy as np
    from dist_classicrl.environments.tiktaktoe_mod import TicTacToeEnv

    class TestTicTacToeEnv(unittest.TestCase):
        """Test suite for TicTacToe environment."""

        def setUp(self):
            """Set up test environment."""
            self.env = TicTacToeEnv()

        def test_reset(self):
            """Test environment reset functionality."""
            observation, info = self.env.reset()

            # Check return types
            self.assertIsInstance(observation, (int, np.integer))
            self.assertIsInstance(info, dict)

            # Check initial state
            self.assertEqual(observation, 0)  # Empty board
            self.assertFalse(self.env.game_over)
            self.assertEqual(self.env.current_player, 1)

        def test_valid_moves(self):
            """Test valid move execution."""
            self.env.reset()

            # Make a valid move
            observation, reward, terminated, truncated, info = self.env.step(4)  # Center

            # Check return types
            self.assertIsInstance(observation, (int, np.integer))
            self.assertIsInstance(reward, (int, float))
            self.assertIsInstance(terminated, bool)
            self.assertIsInstance(truncated, bool)
            self.assertIsInstance(info, dict)

            # Move should be accepted
            self.assertFalse(terminated)  # Game shouldn't end immediately
            self.assertEqual(self.env.board[4], 1)  # Position should be marked

        def test_invalid_moves(self):
            """Test invalid move handling."""
            self.env.reset()

            # Make a move
            self.env.step(4)

            # Try to make move in occupied position
            observation, reward, terminated, truncated, info = self.env.step(4)

            # Should return negative reward and terminate
            self.assertEqual(reward, -1)
            self.assertTrue(terminated)
            self.assertTrue(info.get("invalid_move", False))

        def test_winning_condition(self):
            """Test winning condition detection."""
            self.env.reset()

            # Set up winning condition (top row)
            moves = [0, 3, 1, 4, 2]  # Player 1 gets top row

            for i, move in enumerate(moves):
                observation, reward, terminated, truncated, info = self.env.step(move)

                if i == 4:  # Last move should win
                    self.assertEqual(reward, 1)
                    self.assertTrue(terminated)
                else:
                    self.assertFalse(terminated)

        def test_draw_condition(self):
            """Test draw condition detection."""
            self.env.reset()

            # Set up draw scenario
            moves = [0, 1, 2, 6, 3, 4, 7, 5, 8]  # Results in draw

            for i, move in enumerate(moves):
                observation, reward, terminated, truncated, info = self.env.step(move)

                if i == 8:  # Last move should result in draw
                    self.assertEqual(reward, 0)
                    self.assertTrue(terminated)

        def test_observation_space(self):
            """Test observation space properties."""
            self.assertEqual(self.env.observation_space_size, 3**9)

            # Test observation encoding
            self.env.reset()
            self.env.board = np.array([1, 2, 0, 2, 1, 0, 0, 0, 1])

            observation = self.env._get_observation()
            self.assertIsInstance(observation, (int, np.integer))
            self.assertGreaterEqual(observation, 0)
            self.assertLess(observation, 3**9)

        def test_action_space(self):
            """Test action space properties."""
            self.assertEqual(self.env.action_space_size, 9)

            # Test all actions are valid initially
            self.env.reset()
            for action in range(9):
                self.assertTrue(self.env._is_valid_action(action))

Integration Testing
===================

Algorithm-Environment Integration
---------------------------------

Test that algorithms work correctly with different environments:

.. code-block:: python

    import unittest
    from dist_classicrl.algorithms.runtime.q_learning_single_thread import SingleThreadQLearning
    from dist_classicrl.environments.tiktaktoe_mod import TicTacToeEnv
    from tests.utils.mock_env import MockDiscreteEnv

    class TestAlgorithmEnvironmentIntegration(unittest.TestCase):
        """Test algorithm-environment integration."""

        def test_training_convergence(self):
            """Test that training converges to better performance."""
            env = TicTacToeEnv()
            agent = SingleThreadQLearning(
                state_size=512,
                action_size=9,
                learning_rate=0.3,
                epsilon_decay=0.99
            )

            # Train for multiple episodes
            initial_performance = self._evaluate_agent(agent, env, episodes=10)

            # Training
            for _ in range(100):
                agent.train_episode(env)

            # Evaluate after training
            final_performance = self._evaluate_agent(agent, env, episodes=10)

            # Performance should improve (or at least not degrade significantly)
            self.assertGreaterEqual(final_performance, initial_performance - 0.1)

        def test_multiple_environments(self):
            """Test agent with different environment configurations."""
            environments = [
                TicTacToeEnv(),
                MockDiscreteEnv(state_size=100, action_size=4),
                MockDiscreteEnv(state_size=25, action_size=5)
            ]

            for env in environments:
                agent = SingleThreadQLearning(
                    state_size=env.observation_space_size,
                    action_size=env.action_space_size
                )

                # Should be able to train without errors
                try:
                    for _ in range(10):
                        agent.train_episode(env)
                except Exception as e:
                    self.fail(f"Training failed on environment {type(env).__name__}: {e}")

        def _evaluate_agent(self, agent, env, episodes=10):
            """Evaluate agent performance."""
            total_reward = 0
            original_epsilon = agent.epsilon
            agent.epsilon = 0  # No exploration during evaluation

            for _ in range(episodes):
                observation, _ = env.reset()
                episode_reward = 0
                terminated = False

                while not terminated:
                    action = agent.select_action(observation)
                    observation, reward, terminated, truncated, _ = env.step(action)
                    episode_reward += reward
                    terminated = terminated or truncated

                total_reward += episode_reward

            agent.epsilon = original_epsilon  # Restore original epsilon
            return total_reward / episodes

Runtime Integration Testing
---------------------------

Test different runtime modes:

.. code-block:: python

    import unittest
    import multiprocessing as mp
    from dist_classicrl.algorithms.runtime.q_learning_single_thread import SingleThreadQLearning
    from dist_classicrl.algorithms.runtime.q_learning_parallel import ParallelQLearning
    from dist_classicrl.environments.tiktaktoe_mod import TicTacToeEnv

    class TestRuntimeIntegration(unittest.TestCase):
        """Test different runtime execution modes."""

        def test_single_thread_runtime(self):
            """Test single-threaded execution."""
            env = TicTacToeEnv()
            agent = SingleThreadQLearning(state_size=512, action_size=9)

            # Should complete without errors
            agent.train(env=env, episodes=10)

            # Check that training occurred
            self.assertGreater(agent.algorithm.step_count, 0)
            self.assertEqual(agent.algorithm.episode_count, 10)

        @unittest.skipIf(mp.cpu_count() < 2, "Requires multiple CPU cores")
        def test_parallel_runtime(self):
            """Test parallel execution."""
            def env_factory():
                return TicTacToeEnv()

            agent = ParallelQLearning(
                state_size=512,
                action_size=9,
                num_processes=2
            )

            # Should complete without errors
            try:
                agent.train(env_factory=env_factory, steps=1000)
            except Exception as e:
                self.fail(f"Parallel training failed: {e}")

        def test_runtime_consistency(self):
            """Test that different runtimes produce consistent results."""
            # This is a simplified test - in practice, you'd need more sophisticated
            # comparison methods due to the stochastic nature of training

            env1 = TicTacToeEnv()
            env2 = TicTacToeEnv()

            # Set same random seed for reproducibility
            np.random.seed(42)
            agent1 = SingleThreadQLearning(state_size=512, action_size=9, epsilon=0)

            np.random.seed(42)
            agent2 = SingleThreadQLearning(state_size=512, action_size=9, epsilon=0)

            # Train both agents
            agent1.train(env=env1, episodes=5)
            agent2.train(env=env2, episodes=5)

            # With same seed and no exploration, results should be identical
            np.testing.assert_array_equal(agent1.algorithm.q_table, agent2.algorithm.q_table)

Performance Testing
===================

Benchmark Framework
-------------------

Use pytest-benchmark for performance testing:

.. code-block:: python

    import pytest
    import numpy as np
    from dist_classicrl.algorithms.base_algorithms.q_learning_optimal import OptimalQLearning
    from dist_classicrl.algorithms.base_algorithms.q_learning_numpy import NumpyQLearning
    from dist_classicrl.algorithms.base_algorithms.q_learning_list import ListQLearning
    from tests.utils.mock_env import MockDiscreteEnv

    class TestPerformanceBenchmarks:
        """Performance benchmarks for different components."""

        @pytest.fixture
        def small_env(self):
            """Small environment for quick benchmarks."""
            return MockDiscreteEnv(state_size=100, action_size=4)

        @pytest.fixture
        def large_env(self):
            """Large environment for scaling benchmarks."""
            return MockDiscreteEnv(state_size=10000, action_size=20)

        @pytest.mark.benchmark(group="algorithm_initialization")
        def test_optimal_q_learning_init(self, benchmark, large_env):
            """Benchmark OptimalQLearning initialization."""
            def init_agent():
                return OptimalQLearning(
                    state_size=large_env.observation_space_size,
                    action_size=large_env.action_space_size
                )

            agent = benchmark(init_agent)
            assert agent.q_table.shape == (10000, 20)

        @pytest.mark.benchmark(group="algorithm_initialization")
        def test_numpy_q_learning_init(self, benchmark, large_env):
            """Benchmark NumpyQLearning initialization."""
            def init_agent():
                return NumpyQLearning(
                    state_size=large_env.observation_space_size,
                    action_size=large_env.action_space_size
                )

            agent = benchmark(init_agent)
            assert agent.q_table.shape == (10000, 20)

        @pytest.mark.benchmark(group="algorithm_initialization")
        def test_list_q_learning_init(self, benchmark, large_env):
            """Benchmark ListQLearning initialization."""
            def init_agent():
                return ListQLearning(
                    state_size=large_env.observation_space_size,
                    action_size=large_env.action_space_size
                )

            agent = benchmark(init_agent)
            assert len(agent.q_table) == 10000

        @pytest.mark.benchmark(group="training_performance")
        def test_training_throughput(self, benchmark, small_env):
            """Benchmark training throughput."""
            agent = OptimalQLearning(
                state_size=small_env.observation_space_size,
                action_size=small_env.action_space_size
            )

            def train_episodes():
                for _ in range(100):
                    agent.train_episode(small_env)

            benchmark(train_episodes)

        @pytest.mark.benchmark(group="memory_usage")
        def test_memory_efficiency(self, benchmark, large_env):
            """Benchmark memory usage patterns."""
            import psutil
            import os

            def create_and_train():
                process = psutil.Process(os.getpid())
                initial_memory = process.memory_info().rss

                agent = OptimalQLearning(
                    state_size=large_env.observation_space_size,
                    action_size=large_env.action_space_size
                )

                # Train for a while
                for _ in range(50):
                    agent.train_episode(large_env)

                final_memory = process.memory_info().rss
                return final_memory - initial_memory

            memory_used = benchmark(create_and_train)

            # Memory usage should be reasonable (less than 100MB for this test)
            assert memory_used < 100 * 1024 * 1024

Performance Regression Testing
------------------------------

Track performance over time:

.. code-block:: python

    import json
    import os
    from datetime import datetime

    class PerformanceRegression:
        """Track performance metrics over time."""

        def __init__(self, baseline_file="performance_baseline.json"):
            self.baseline_file = baseline_file
            self.baseline = self._load_baseline()

        def _load_baseline(self):
            """Load performance baseline if it exists."""
            if os.path.exists(self.baseline_file):
                with open(self.baseline_file, 'r') as f:
                    return json.load(f)
            return {}

        def record_performance(self, test_name, duration, memory_used=None):
            """Record performance metrics."""
            timestamp = datetime.now().isoformat()

            if test_name not in self.baseline:
                self.baseline[test_name] = []

            self.baseline[test_name].append({
                'timestamp': timestamp,
                'duration': duration,
                'memory_used': memory_used
            })

            # Keep only last 100 entries
            if len(self.baseline[test_name]) > 100:
                self.baseline[test_name] = self.baseline[test_name][-100:]

            self._save_baseline()

        def check_regression(self, test_name, current_duration, threshold=1.2):
            """Check if current performance indicates regression."""
            if test_name not in self.baseline or len(self.baseline[test_name]) < 5:
                return False, "Insufficient baseline data"

            recent_durations = [entry['duration'] for entry in self.baseline[test_name][-10:]]
            avg_baseline = sum(recent_durations) / len(recent_durations)

            if current_duration > avg_baseline * threshold:
                return True, f"Performance regression: {current_duration:.3f}s vs {avg_baseline:.3f}s baseline"

            return False, "Performance within acceptable range"

        def _save_baseline(self):
            """Save baseline to file."""
            with open(self.baseline_file, 'w') as f:
                json.dump(self.baseline, f, indent=2)

Distributed Testing
===================

MPI Test Framework
------------------

Test distributed functionality with MPI:

.. code-block:: python

    import unittest
    import sys
    import os

    # Add path for distributed imports
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

    try:
        from mpi4py import MPI
        MPI_AVAILABLE = True
    except ImportError:
        MPI_AVAILABLE = False

    @unittest.skipUnless(MPI_AVAILABLE, "MPI not available")
    class TestDistributedTraining(unittest.TestCase):
        """Test distributed training functionality."""

        def setUp(self):
            """Set up MPI environment."""
            self.comm = MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()

        def test_mpi_communication(self):
            """Test basic MPI communication."""
            # Test broadcast
            if self.rank == 0:
                data = {"test": "data", "value": 42}
            else:
                data = None

            data = self.comm.bcast(data, root=0)

            self.assertEqual(data["test"], "data")
            self.assertEqual(data["value"], 42)

        def test_distributed_q_learning(self):
            """Test distributed Q-learning functionality."""
            from dist_classicrl.algorithms.runtime.q_learning_async_dist import DistAsyncQLearning
            from dist_classicrl.environments.tiktaktoe_mod import TicTacToeEnv

            # Create agent and environment
            agent = DistAsyncQLearning(state_size=512, action_size=9)
            env = TicTacToeEnv()

            # Short training run
            try:
                agent.train(env=env, steps=100)
            except Exception as e:
                self.fail(f"Distributed training failed on rank {self.rank}: {e}")

        def test_parameter_synchronization(self):
            """Test parameter synchronization across nodes."""
            import numpy as np

            # Create test data
            test_data = np.random.randn(10, 4) if self.rank == 0 else np.zeros((10, 4))

            # Broadcast from rank 0
            self.comm.Bcast(test_data, root=0)

            # All ranks should have the same data
            if self.rank == 0:
                self.original_data = test_data.copy()

            # Verify data consistency
            gathered_data = self.comm.gather(test_data, root=0)

            if self.rank == 0:
                for rank_data in gathered_data:
                    np.testing.assert_array_equal(rank_data, self.original_data)

Fault Tolerance Testing
-----------------------

Test system behavior under failure conditions:

.. code-block:: python

    import unittest
    import signal
    import time
    import multiprocessing as mp

    class TestFaultTolerance(unittest.TestCase):
        """Test fault tolerance mechanisms."""

        def test_checkpoint_recovery(self):
            """Test checkpoint and recovery functionality."""
            from dist_classicrl.algorithms.runtime.q_learning_single_thread import SingleThreadQLearning
            from dist_classicrl.environments.tiktaktoe_mod import TicTacToeEnv

            # Train agent and save checkpoint
            env = TicTacToeEnv()
            agent1 = SingleThreadQLearning(state_size=512, action_size=9)

            # Train for some episodes
            for _ in range(10):
                agent1.train_episode(env)

            # Save state
            q_table_backup = agent1.algorithm.q_table.copy()
            epsilon_backup = agent1.algorithm.epsilon
            step_count_backup = agent1.algorithm.step_count

            # Create new agent and restore state
            agent2 = SingleThreadQLearning(state_size=512, action_size=9)
            agent2.algorithm.q_table = q_table_backup
            agent2.algorithm.epsilon = epsilon_backup
            agent2.algorithm.step_count = step_count_backup

            # Verify restoration
            np.testing.assert_array_equal(agent1.algorithm.q_table, agent2.algorithm.q_table)
            self.assertEqual(agent1.algorithm.epsilon, agent2.algorithm.epsilon)
            self.assertEqual(agent1.algorithm.step_count, agent2.algorithm.step_count)

        def test_process_failure_simulation(self):
            """Test behavior when worker processes fail."""

            def failing_worker():
                """Worker that fails after some time."""
                time.sleep(0.1)
                raise RuntimeError("Simulated worker failure")

            def robust_worker():
                """Worker that handles failures gracefully."""
                try:
                    time.sleep(0.1)
                    # Simulate some work
                    return "success"
                except Exception as e:
                    return f"error: {e}"

            # Test that robust worker handles failures
            result = robust_worker()
            self.assertEqual(result, "success")

Test Utilities
==============

Mock Environments
-----------------

Create mock environments for controlled testing:

.. code-block:: python

    import numpy as np
    from dist_classicrl.environments.custom_env import DistClassicRLEnv

    class MockDiscreteEnv(DistClassicRLEnv):
        """Mock discrete environment for testing."""

        def __init__(self, state_size=10, action_size=4, episode_length=20):
            super().__init__()
            self._state_size = state_size
            self._action_size = action_size
            self.episode_length = episode_length

            self.current_state = 0
            self.step_count = 0

        def reset(self, seed=None, options=None):
            if seed is not None:
                np.random.seed(seed)

            self.current_state = np.random.randint(self._state_size)
            self.step_count = 0
            return self.current_state, {}

        def step(self, action):
            # Validate action
            if not (0 <= action < self._action_size):
                return self.current_state, -1, True, False, {"invalid_action": True}

            # Random state transition
            self.current_state = np.random.randint(self._state_size)
            self.step_count += 1

            # Random reward
            reward = np.random.normal(0, 1)

            # Episode termination
            terminated = self.step_count >= self.episode_length

            return self.current_state, reward, terminated, False, {}

        @property
        def observation_space_size(self):
            return self._state_size

        @property
        def action_space_size(self):
            return self._action_size

    class DeterministicMockEnv(DistClassicRLEnv):
        """Deterministic mock environment for reproducible testing."""

        def __init__(self, rewards_sequence=None):
            super().__init__()
            self.rewards_sequence = rewards_sequence or [1, 0, -1, 1]
            self.step_count = 0

        def reset(self, seed=None, options=None):
            self.step_count = 0
            return 0, {}

        def step(self, action):
            reward = self.rewards_sequence[self.step_count % len(self.rewards_sequence)]
            self.step_count += 1

            # Terminate after sequence completes
            terminated = self.step_count >= len(self.rewards_sequence)

            return self.step_count % 4, reward, terminated, False, {}

        @property
        def observation_space_size(self):
            return 4

        @property
        def action_space_size(self):
            return 2

Test Configuration
==================

Pytest Configuration
---------------------

Configure pytest with ``conftest.py``:

.. code-block:: python

    import pytest
    import numpy as np
    import tempfile
    import shutil
    from pathlib import Path

    @pytest.fixture
    def temp_dir():
        """Create temporary directory for tests."""
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        shutil.rmtree(temp_path)

    @pytest.fixture
    def random_seed():
        """Set consistent random seed for reproducible tests."""
        seed = 42
        np.random.seed(seed)
        return seed

    @pytest.fixture
    def small_q_table():
        """Small Q-table for testing."""
        return np.random.randn(10, 4)

    @pytest.fixture
    def large_q_table():
        """Large Q-table for performance testing."""
        return np.random.randn(1000, 20)

    def pytest_configure(config):
        """Configure pytest markers."""
        config.addinivalue_line("markers", "slow: marks tests as slow")
        config.addinivalue_line("markers", "distributed: marks tests requiring MPI")
        config.addinivalue_line("markers", "performance: marks performance benchmarks")
        config.addinivalue_line("markers", "integration: marks integration tests")

    def pytest_runtest_setup(item):
        """Setup for individual tests."""
        # Skip distributed tests if MPI not available
        if "distributed" in item.keywords:
            try:
                import mpi4py
            except ImportError:
                pytest.skip("MPI not available")

Continuous Integration
======================

GitHub Actions Configuration
-----------------------------

Example CI configuration (``.github/workflows/test.yml``):

.. code-block:: yaml

    name: Tests

    on:
      push:
        branches: [ main, develop ]
      pull_request:
        branches: [ main ]

    jobs:
      test:
        runs-on: ubuntu-latest
        strategy:
          matrix:
            python-version: [3.8, 3.9, "3.10", "3.11"]

        steps:
        - uses: actions/checkout@v3

        - name: Set up Python ${{ matrix.python-version }}
          uses: actions/setup-python@v3
          with:
            python-version: ${{ matrix.python-version }}

        - name: Install system dependencies
          run: |
            sudo apt-get update
            sudo apt-get install -y openmpi-bin openmpi-common libopenmpi-dev

        - name: Install Python dependencies
          run: |
            python -m pip install --upgrade pip
            pip install -e .[dev,test]

        - name: Run unit tests
          run: |
            pytest tests/ -v --cov=dist_classicrl --cov-report=xml

        - name: Run performance tests
          run: |
            pytest tests/ -m performance --benchmark-only

        - name: Run distributed tests
          run: |
            mpirun -n 2 pytest tests/ -m distributed

        - name: Upload coverage to Codecov
          uses: codecov/codecov-action@v3
          with:
            file: ./coverage.xml

Test Documentation
==================

Writing Good Tests
------------------

**Guidelines for effective testing**:

1. **Test Behavior, Not Implementation**: Focus on what the code should do, not how it does it
2. **Use Descriptive Names**: Test names should clearly indicate what is being tested
3. **One Concept Per Test**: Each test should focus on a single concept or behavior
4. **Arrange-Act-Assert**: Structure tests with clear setup, execution, and verification phases
5. **Make Tests Independent**: Tests should not depend on each other
6. **Use Fixtures Wisely**: Share setup code through fixtures, but avoid complex dependencies

**Example of a well-structured test**:

.. code-block:: python

    def test_q_learning_converges_to_optimal_policy_in_simple_environment(self):
        """Test that Q-learning converges to optimal policy in a simple deterministic environment."""
        # Arrange
        env = DeterministicMockEnv(rewards_sequence=[1, 1, 1, 0])  # Clear optimal path
        agent = OptimalQLearning(state_size=4, action_size=2, learning_rate=0.5, epsilon=0.1)

        # Act - Train for sufficient episodes
        for _ in range(100):
            agent.train_episode(env)

        # Assert - Check that agent learned optimal policy
        optimal_actions = [self._get_optimal_action(state) for state in range(4)]
        learned_actions = [agent.select_action(state) for state in range(4)]

        # Allow for some exploration, but most actions should be optimal
        correct_actions = sum(1 for opt, learned in zip(optimal_actions, learned_actions)
                            if opt == learned)
        self.assertGreater(correct_actions / len(optimal_actions), 0.8)

Best Practices
==============

Development Workflow
--------------------

1. **Test-Driven Development**: Write tests before implementing features
2. **Continuous Testing**: Run tests frequently during development
3. **Test Coverage**: Aim for high test coverage, but focus on quality over quantity
4. **Performance Monitoring**: Regular performance testing to catch regressions
5. **Documentation**: Keep test documentation up to date

Testing Strategy
----------------

1. **Unit Tests**: Test individual components thoroughly
2. **Integration Tests**: Test component interactions
3. **End-to-End Tests**: Test complete workflows
4. **Performance Tests**: Monitor performance characteristics
5. **Stress Tests**: Test system limits and failure modes

See Also
========

- :doc:`architecture`: Understanding the codebase architecture
- :doc:`../user_guide/performance`: Performance optimization and benchmarking
- :doc:`../user_guide/distributed`: Distributed system testing considerations
