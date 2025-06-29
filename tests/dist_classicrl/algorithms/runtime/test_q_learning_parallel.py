"""
Unit tests for the ParallelQLearning class.

This module contains comprehensive tests for all methods and functionality
of the ParallelQLearning class from the q_learning_parallel module.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from dist_classicrl.algorithms.base_algorithms.q_learning_optimal import (
    OptimalQLearningBase,
)
from dist_classicrl.algorithms.runtime.q_learning_parallel import ParallelQLearning
from tests.utils.mock_env import MockEnvironment


class TestParallelQLearning:
    """Test class for ParallelQLearning."""

    def setup_method(self) -> None:
        """Set up test fixtures before each test method."""
        self.state_size = 10
        self.action_size = 3

        # Patch shared memory creation to avoid conflicts in testing
        with patch("multiprocessing.shared_memory.SharedMemory"), patch(
            "multiprocessing.Lock"
        ), patch("multiprocessing.Value"):
            self.agent = ParallelQLearning(
                state_size=self.state_size,
                action_size=self.action_size,
                learning_rate=0.1,
                discount_factor=0.9,
                exploration_rate=0.1,
                exploration_decay=0.99,
                min_exploration_rate=0.01,
            )

    def test_initialization(self) -> None:
        """Test proper initialization of the ParallelQLearning class."""
        assert self.agent.state_size == 10
        assert self.agent.action_size == 3
        assert self.agent.learning_rate == 0.1
        assert self.agent.discount_factor == 0.9
        assert self.agent.exploration_rate == 0.1
        assert self.agent.exploration_decay == 0.99
        assert self.agent.min_exploration_rate == 0.01
        assert self.agent.q_table.shape == (10, 3)

    def test_shared_memory_initialization(self) -> None:
        """Test that shared memory is properly initialized."""
        # This test requires more sophisticated mocking
        # We'll test the pattern instead of the actual shared memory
        assert hasattr(self.agent, "sm_name")
        assert self.agent.sm_name == "q_table"

    @patch("multiprocessing.Process")
    @patch("multiprocessing.Pipe")
    @patch("multiprocessing.Queue")
    def test_train_with_multiple_environments(self, mock_queue, mock_pipe, mock_process) -> None:
        """Test training with multiple environments in parallel."""
        # Mock the pipe communication
        mock_parent_conn = MagicMock()
        mock_child_conn = MagicMock()
        mock_pipe.return_value = (mock_parent_conn, mock_child_conn)

        # Mock the process
        mock_proc = MagicMock()
        mock_process.return_value = mock_proc

        # Mock the queue
        mock_reward_queue = MagicMock()
        mock_queue.return_value = mock_reward_queue
        mock_reward_queue.empty.return_value = True

        # Mock the pipe receive to return environment state
        mock_parent_conn.recv.return_value = {
            "env": MockEnvironment(num_envs=1),
            "states": np.array([0], dtype=np.int32),
            "infos": [{}],
            "agent_rewards": np.array([0.0], dtype=np.float32),
        }

        envs = [MockEnvironment(num_envs=1) for _ in range(2)]
        val_env = MockEnvironment(num_envs=1)

        with patch.object(
            self.agent, "evaluate_steps", return_value=(5.0, [2.0, 3.0])
        ), patch.object(self.agent, "sm") as mock_sm:
            mock_sm.close = MagicMock()
            mock_sm.unlink = MagicMock()

            self.agent.train(
                envs=envs,
                steps=10,
                val_env=val_env,
                val_every_n_steps=5,
                val_steps=3,
                val_episodes=None,
            )

            # Verify that processes were created
            assert mock_process.call_count == len(envs)

    def test_train_invalid_validation_params(self) -> None:
        """Test that training raises assertion error with invalid validation params."""
        envs = [MockEnvironment(num_envs=1)]
        val_env = MockEnvironment(num_envs=1)

        # Both val_steps and val_episodes provided
        with patch.object(self.agent, "sm") as mock_sm:
            mock_sm.close = MagicMock()
            mock_sm.unlink = MagicMock()

            with pytest.raises(AssertionError):
                self.agent.train(
                    envs=envs,
                    steps=10,
                    val_env=val_env,
                    val_every_n_steps=5,
                    val_steps=3,
                    val_episodes=2,
                )

        # Neither val_steps nor val_episodes provided
        with patch.object(self.agent, "sm") as mock_sm:
            mock_sm.close = MagicMock()
            mock_sm.unlink = MagicMock()

            with pytest.raises(AssertionError):
                self.agent.train(
                    envs=envs,
                    steps=10,
                    val_env=val_env,
                    val_every_n_steps=5,
                    val_steps=None,
                    val_episodes=None,
                )

    def test_update_explore_rate(self) -> None:
        """Test exploration rate update with thread safety."""
        # Mock the synchronized value
        mock_value = MagicMock()
        mock_value.value = 0.5
        self.agent._exploration_rate = mock_value

        self.agent.update_explore_rate()

        # Verify that the value was updated
        expected_new_rate = max(0.5 * 0.99, 0.01)
        assert self.agent.exploration_rate == expected_new_rate

    @patch("multiprocessing.shared_memory.SharedMemory")
    def test_run_steps_simple_env(self, mock_shared_memory) -> None:
        """Test run_steps method with a simple environment."""
        env = MockEnvironment(num_envs=2, return_dict=False)
        rewards_queue = MagicMock()
        sm_lock = MagicMock()
        exploration_rate_value = MagicMock()
        exploration_rate_value.value = 0.1
        curr_state_pipe = MagicMock()

        # Mock shared memory
        mock_sm = MagicMock()
        mock_shared_memory.return_value = mock_sm
        mock_sm.buf = bytearray(self.agent.q_table.nbytes)

        with patch.object(
            self.agent, "choose_actions", return_value=np.array([0, 1])
        ), patch.object(self.agent, "learn"):
            self.agent.run_steps(
                env=env,
                num_steps=3,
                rewards_queue=rewards_queue,
                sm_lock=sm_lock,
                exploration_rate_value=exploration_rate_value,
                curr_state_pipe=curr_state_pipe,
            )

            # Verify that the pipe was used to send final state
            curr_state_pipe.send.assert_called_once()
            curr_state_pipe.close.assert_called_once()

    @patch("multiprocessing.shared_memory.SharedMemory")
    def test_run_steps_dict_observation_env(self, mock_shared_memory) -> None:
        """Test run_steps method with dict observation environment."""
        env = MockEnvironment(num_envs=2, return_dict=True)
        rewards_queue = MagicMock()
        sm_lock = MagicMock()
        exploration_rate_value = MagicMock()
        exploration_rate_value.value = 0.1
        curr_state_pipe = MagicMock()

        # Mock shared memory
        mock_sm = MagicMock()
        mock_shared_memory.return_value = mock_sm
        mock_sm.buf = bytearray(self.agent.q_table.nbytes)

        with patch.object(
            self.agent, "choose_actions", return_value=np.array([0, 1])
        ), patch.object(self.agent, "learn"):
            self.agent.run_steps(
                env=env,
                num_steps=3,
                rewards_queue=rewards_queue,
                sm_lock=sm_lock,
                exploration_rate_value=exploration_rate_value,
                curr_state_pipe=curr_state_pipe,
            )

    def test_evaluate_steps_simple_env(self) -> None:
        """Test evaluation with steps on a simple environment."""
        env = MockEnvironment(num_envs=2, return_dict=False)

        with patch.object(self.agent, "choose_actions", return_value=np.array([0, 1])):
            total_rewards, reward_history = self.agent.evaluate_steps(env, steps=5)

            assert isinstance(total_rewards, float)
            assert isinstance(reward_history, list)

    def test_evaluate_steps_dict_observation_env(self) -> None:
        """Test evaluation with steps on a dict observation environment."""
        env = MockEnvironment(num_envs=2, return_dict=True)

        with patch.object(self.agent, "choose_actions", return_value=np.array([0, 1])):
            total_rewards, reward_history = self.agent.evaluate_steps(env, steps=5)

            assert isinstance(total_rewards, float)
            assert isinstance(reward_history, list)

    def test_evaluate_episodes_simple_env(self) -> None:
        """Test evaluation with episodes on a simple environment."""
        env = MockEnvironment(num_envs=2, return_dict=False)

        with patch.object(self.agent, "choose_actions", return_value=np.array([0, 1])):
            total_rewards, reward_history = self.agent.evaluate_episodes(env, episodes=2)

            assert isinstance(total_rewards, float)
            assert isinstance(reward_history, list)

    def test_evaluate_episodes_dict_observation_env(self) -> None:
        """Test evaluation with episodes on a dict observation environment."""
        env = MockEnvironment(num_envs=2, return_dict=True)

        with patch.object(self.agent, "choose_actions", return_value=np.array([0, 1])):
            total_rewards, reward_history = self.agent.evaluate_episodes(env, episodes=2)

            assert isinstance(total_rewards, float)
            assert isinstance(reward_history, list)

    def test_inheritance_from_optimal_q_learning_base(self) -> None:
        """Test that ParallelQLearning properly inherits from OptimalQLearningBase."""
        assert isinstance(self.agent, OptimalQLearningBase)

        # Test that inherited methods are available
        assert hasattr(self.agent, "choose_actions")
        assert hasattr(self.agent, "learn")
        assert hasattr(self.agent, "get_q_value")
        assert hasattr(self.agent, "set_q_value")

    @patch("multiprocessing.Process")
    @patch("multiprocessing.Pipe")
    @patch("multiprocessing.Queue")
    def test_reward_collection_during_training(self, mock_queue, mock_pipe, mock_process) -> None:
        """Test that rewards are properly collected during parallel training."""
        # Mock the pipe communication
        mock_parent_conn = MagicMock()
        mock_child_conn = MagicMock()
        mock_pipe.return_value = (mock_parent_conn, mock_child_conn)

        # Mock the process
        mock_proc = MagicMock()
        mock_process.return_value = mock_proc

        # Mock the queue with some rewards
        mock_reward_queue = MagicMock()
        mock_queue.return_value = mock_reward_queue
        mock_reward_queue.empty.side_effect = [False, False, True]  # Two rewards, then empty
        mock_reward_queue.get.side_effect = [5.0, 3.0]  # Two reward values

        # Mock the pipe receive
        mock_parent_conn.recv.return_value = {
            "env": MockEnvironment(num_envs=1),
            "states": np.array([0], dtype=np.int32),
            "infos": [{}],
            "agent_rewards": np.array([0.0], dtype=np.float32),
        }

        envs = [MockEnvironment(num_envs=1)]
        val_env = MockEnvironment(num_envs=1)

        with patch.object(
            self.agent, "evaluate_steps", return_value=(5.0, [2.0, 3.0])
        ), patch.object(self.agent, "sm") as mock_sm:
            mock_sm.close = MagicMock()
            mock_sm.unlink = MagicMock()

            self.agent.train(
                envs=envs,
                steps=10,
                val_env=val_env,
                val_every_n_steps=5,
                val_steps=3,
                val_episodes=None,
            )

            # Verify that rewards were collected from queue
            assert mock_reward_queue.get.call_count == 2

    def test_shared_memory_cleanup(self) -> None:
        """Test that shared memory is properly cleaned up."""
        envs = [MockEnvironment(num_envs=1)]
        val_env = MockEnvironment(num_envs=1)

        with patch.object(self.agent, "sm") as mock_sm:
            mock_sm.close = MagicMock()
            mock_sm.unlink = MagicMock()

            with patch("multiprocessing.Process"), patch("multiprocessing.Pipe"), patch(
                "multiprocessing.Queue"
            ), patch.object(self.agent, "evaluate_steps", return_value=(5.0, [2.0, 3.0])):
                self.agent.train(
                    envs=envs,
                    steps=10,
                    val_env=val_env,
                    val_every_n_steps=5,
                    val_steps=3,
                    val_episodes=None,
                )

        # Verify cleanup was called
        mock_sm.close.assert_called_once()
        mock_sm.unlink.assert_called_once()
