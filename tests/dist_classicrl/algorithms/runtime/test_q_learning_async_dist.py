"""
Unit tests for the DistAsyncQLearning class.

This module contains comprehensive tests for all methods and functionality
of the DistAsyncQLearning class from the q_learning_async_dist module.

Note: Some tests require MPI and should be run with mpirun.
For MPI tests, use:
mpirun -n 3 python -m pytest test_q_learning_async_dist.py::TestDistAsyncQLearningMPI
For non-MPI tests, use:
pytest test_q_learning_async_dist.py::TestDistAsyncQLearning
"""

import queue
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from dist_classicrl.algorithms.runtime.q_learning_async_dist import (
    MASTER_RANK,
    NUM_NODES,
    RANK,
    comm,
)
from dist_classicrl.algorithms.runtime.q_learning_single_thread import (
    OptimalQLearningBase,
)
from tests.utils.mock_env import MockEnvironment

try:
    from mpi4py import MPI  # noqa: F401

    MPI_AVAILABLE = True
except ImportError:
    MPI_AVAILABLE = False

from dist_classicrl.algorithms.runtime.q_learning_async_dist import DistAsyncQLearning


class TestDistAsyncQLearning:
    """Test class for DistAsyncQLearning (non-MPI tests)."""

    def setup_method(self) -> None:
        """Set up test fixtures before each test method."""
        self.state_size = 10
        self.action_size = 3
        self.agent = DistAsyncQLearning(
            state_size=self.state_size,
            action_size=self.action_size,
            learning_rate=0.1,
            discount_factor=0.9,
            exploration_rate=0.1,
            exploration_decay=0.99,
            min_exploration_rate=0.01,
        )

    def test_initialization(self) -> None:
        """Test proper initialization of the DistAsyncQLearning class."""
        assert self.agent.state_size == 10
        assert self.agent.action_size == 3
        assert self.agent.learning_rate == 0.1
        assert self.agent.discount_factor == 0.9
        assert self.agent.exploration_rate == 0.1
        assert self.agent.exploration_decay == 0.99
        assert self.agent.min_exploration_rate == 0.01
        assert self.agent.q_table.shape == (10, 3)
        assert np.all(self.agent.q_table == 0)

    def test_inheritance_from_optimal_q_learning_base(self) -> None:
        """Test that DistAsyncQLearning properly inherits from OptimalQLearningBase."""
        assert isinstance(self.agent, OptimalQLearningBase)

        # Test that inherited methods are available
        assert hasattr(self.agent, "choose_actions")
        assert hasattr(self.agent, "learn")
        assert hasattr(self.agent, "get_q_value")
        assert hasattr(self.agent, "set_q_value")

    def test_update_q_table_method_exists(self) -> None:
        """Test that update_q_table method is available."""
        assert hasattr(self.agent, "update_q_table")
        assert callable(self.agent.update_q_table)

    def test_communicate_master_method_exists(self) -> None:
        """Test that communicate_master method is available."""
        assert hasattr(self.agent, "communicate_master")
        assert callable(self.agent.communicate_master)

    def test_run_environment_method_exists(self) -> None:
        """Test that run_environment method is available."""
        assert hasattr(self.agent, "run_environment")
        assert callable(self.agent.run_environment)

    def test_evaluate_steps_simple_env(self) -> None:
        """Test evaluation with steps on a simple environment."""
        env = MockEnvironment(num_envs=2, return_dict=False)

        with patch.object(self.agent, "choose_actions", return_value=np.array([0, 1])):
            total_rewards, reward_history = self.agent.evaluate_steps(env, steps=5)

            assert isinstance(total_rewards, (int, float, np.floating))
            assert isinstance(reward_history, list)

    def test_evaluate_steps_dict_observation_env(self) -> None:
        """Test evaluation with steps on a dict observation environment."""
        env = MockEnvironment(num_envs=2, return_dict=True)

        with patch.object(self.agent, "choose_actions", return_value=np.array([0, 1])):
            total_rewards, reward_history = self.agent.evaluate_steps(env, steps=5)

            assert isinstance(total_rewards, (int, float, np.floating))
            assert isinstance(reward_history, list)

    def test_evaluate_episodes_simple_env(self) -> None:
        """Test evaluation with episodes on a simple environment."""
        env = MockEnvironment(num_envs=2, return_dict=False)

        with patch.object(self.agent, "choose_actions", return_value=np.array([0, 1])):
            total_rewards, reward_history = self.agent.evaluate_episodes(env, episodes=2)

            assert isinstance(total_rewards, (int, float, np.floating))
            assert isinstance(reward_history, list)

    def test_evaluate_episodes_dict_observation_env(self) -> None:
        """Test evaluation with episodes on a dict observation environment."""
        env = MockEnvironment(num_envs=2, return_dict=True)

        with patch.object(self.agent, "choose_actions", return_value=np.array([0, 1])):
            total_rewards, reward_history = self.agent.evaluate_episodes(env, episodes=2)

            assert isinstance(total_rewards, (int, float, np.floating))
            assert isinstance(reward_history, list)

    @patch("queue.Queue")
    def test_update_q_table_with_mock_queue(self, mock_queue_class) -> None:
        """Test update_q_table method with mocked queue."""
        # Create a mock queue instance
        mock_queue = MagicMock()
        mock_queue_class.return_value = mock_queue

        # Mock queue.get to return some experiences and then None to stop
        experience = (0, 1, 1.0, 1, False)  # state  # action  # reward  # next_state  # terminated
        mock_queue.get.side_effect = [experience, experience, None]

        self.agent.experience_queue = mock_queue
        self.agent.batch_size = 2

        val_env = MockEnvironment(num_envs=1, return_dict=False)

        with (
            patch.object(self.agent, "learn"),
            patch.object(self.agent, "evaluate_steps", return_value=(5.0, [2.0, 3.0])),
            patch("builtins.print"),
        ):
            # This should exit quickly due to our mocked None
            self.agent.update_q_table(
                val_env=val_env, val_every_n_steps=5, val_steps=2, val_episodes=None
            )

    def test_train_method_validation_params(self) -> None:
        """Test that train method validates parameters correctly."""
        MockEnvironment(num_envs=1, return_dict=False)
        MockEnvironment(num_envs=1, return_dict=False)

        # This test just checks that the method exists and can handle validation
        # without actually running MPI code
        assert hasattr(self.agent, "train")
        assert callable(self.agent.train)


@pytest.mark.skipif(not MPI_AVAILABLE, reason="MPI not available")
class TestDistAsyncQLearningMPI:
    """Test class for DistAsyncQLearning with MPI functionality."""

    def setup_method(self) -> None:
        """Set up test fixtures before each test method."""
        self.state_size = 10
        self.action_size = 3
        self.agent = DistAsyncQLearning(
            state_size=self.state_size,
            action_size=self.action_size,
            learning_rate=0.1,
            discount_factor=0.9,
            exploration_rate=0.1,
            exploration_decay=0.99,
            min_exploration_rate=0.01,
        )

    def test_mpi_constants(self) -> None:
        """Test that MPI constants are properly set."""
        assert comm is not None
        assert isinstance(RANK, int)
        assert isinstance(NUM_NODES, int)
        assert MASTER_RANK == 0
        assert RANK >= 0
        assert RANK < NUM_NODES

    @patch("threading.Thread")
    @patch("queue.Queue")
    def test_train_master_node_setup(self, mock_queue, mock_thread) -> None:
        """Test training setup for master node."""
        if RANK == MASTER_RANK:
            env = MockEnvironment(num_envs=1, return_dict=False)
            val_env = MockEnvironment(num_envs=1, return_dict=False)

            # Mock the queue
            mock_queue_instance = MagicMock()
            mock_queue.return_value = mock_queue_instance

            # Mock the thread
            mock_thread_instance = MagicMock()
            mock_thread.return_value = mock_thread_instance

            with patch.object(self.agent, "communicate_master", return_value=[]):
                self.agent.train(
                    env=env,
                    steps=5,
                    val_env=val_env,
                    val_every_n_steps=3,
                    val_steps=2,
                    val_episodes=None,
                    batch_size=2,
                )

                # Verify thread was created and started
                mock_thread.assert_called_once()
                mock_thread_instance.start.assert_called_once()
                mock_thread_instance.join.assert_called_once()

    def test_train_worker_node_setup(self) -> None:
        """Test training setup for worker nodes."""
        if RANK != MASTER_RANK:
            env = MockEnvironment(num_envs=1, return_dict=False)
            val_env = MockEnvironment(num_envs=1, return_dict=False)

            with patch.object(self.agent, "run_environment"):
                self.agent.train(
                    env=env,
                    steps=5,
                    val_env=val_env,
                    val_every_n_steps=3,
                    val_steps=2,
                    val_episodes=None,
                    batch_size=2,
                )

    @patch(
        "dist_classicrl.algorithms.runtime.q_learning_async_dist.NUM_NODES", 2
    )  # 1 master + 1 worker
    @patch("dist_classicrl.algorithms.runtime.q_learning_async_dist.comm")
    def test_communicate_master_mock(self, mock_comm) -> None:
        """Test communicate_master method with mocked MPI communication."""
        if RANK == MASTER_RANK:
            # Mock MPI communication with proper step progression
            mock_request = MagicMock()

            # Create multiple successful calls to reach the step limit (2 steps)
            mock_request.test.side_effect = [
                # First step
                (
                    True,
                    (
                        np.array([1, 2], dtype=np.int32),  # next_states
                        np.array([1.0, 1.0], dtype=np.float32),  # rewards
                        np.array([False, False], dtype=bool),  # terminated
                        np.array([False, False], dtype=bool),  # truncated
                        [{}, {}],  # infos
                        np.array([True, True], dtype=bool),  # firsts
                    ),
                ),
                # Second step
                (
                    True,
                    (
                        np.array([3, 4], dtype=np.int32),  # next_states
                        np.array([0.5, 0.5], dtype=np.float32),  # rewards
                        np.array([True, True], dtype=bool),  # terminated (episodes end)
                        np.array([False, False], dtype=bool),  # truncated
                        [{}, {}],  # infos
                        np.array([False, False], dtype=bool),  # firsts
                    ),
                ),
                # Subsequent calls return False (no data available)
                (False, None),
            ]

            mock_comm.irecv.return_value = mock_request
            mock_comm.isend = MagicMock()

            self.agent.experience_queue = queue.Queue()

            with patch.object(self.agent, "choose_actions", return_value=np.array([0, 1])):
                reward_history = self.agent.communicate_master(steps=2)

                assert isinstance(reward_history, list)
                # Should have collected 2 rewards when episodes terminated
                assert len(reward_history) == 2

    @patch("dist_classicrl.algorithms.runtime.q_learning_async_dist.comm")
    def test_run_environment_mock(self, mock_comm) -> None:
        """Test run_environment method with mocked MPI communication."""
        if RANK != MASTER_RANK:
            env = MockEnvironment(num_envs=2, return_dict=False)

            # Mock MPI communication
            mock_status = MagicMock()
            mock_status.tag = 1  # Continue tag
            mock_comm.Probe = MagicMock()
            mock_comm.recv.side_effect = [
                np.array([0, 1], dtype=np.int32),  # First actions
                None,  # Stop signal
            ]
            mock_comm.send = MagicMock()

            # Mock the status to change tags to trigger loop exit
            def probe_side_effect(*args, **kwargs) -> None:
                if mock_comm.recv.call_count >= 2:
                    mock_status.tag = 0  # Stop tag
                kwargs["status"].tag = mock_status.tag

            mock_comm.Probe.side_effect = probe_side_effect

            self.agent.run_environment(env)

            # Verify that communication occurred
            assert mock_comm.send.call_count >= 1


# Standalone MPI test script for integration testing
def run_mpi_integration_test() -> None:
    """
    Integration test that can be run with mpirun.

    Usage: mpirun -n 3 python test_q_learning_async_dist.py.
    """
    if not MPI_AVAILABLE:
        return

    # Simple integration test
    agent = DistAsyncQLearning(
        state_size=5,
        action_size=2,
        learning_rate=0.1,
        discount_factor=0.9,
        exploration_rate=0.1,
    )

    env = MockEnvironment(num_envs=2, return_dict=False)
    val_env = MockEnvironment(num_envs=1, return_dict=False)

    agent.train(
        env=env,
        steps=10,
        val_env=val_env,
        val_every_n_steps=5,
        val_steps=3,
        val_episodes=None,
        batch_size=2,
    )


if __name__ == "__main__":
    # Run the MPI integration test if called directly
    run_mpi_integration_test()
