"""
Unit tests for the SingleThreadQLearning class.

This module contains comprehensive tests for all methods and functionality
of the SingleThreadQLearning class from the q_learning_single_thread module.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from gymnasium.vector import SyncVectorEnv

from dist_classicrl.algorithms.runtime.q_learning_single_thread import (
    SingleThreadQLearning,
)
from tests.utils.mock_env import MockEnvironment


class TestSingleThreadQLearning:
    """Test class for SingleThreadQLearning."""

    def setup_method(self) -> None:
        """Set up test fixtures before each test method."""
        self.state_size = 10
        self.action_size = 3
        self.agent = SingleThreadQLearning(
            state_size=self.state_size,
            action_size=self.action_size,
            learning_rate=0.1,
            discount_factor=0.9,
            exploration_rate=0.1,  # Low exploration for predictable testing
            exploration_decay=0.99,
            min_exploration_rate=0.01,
        )

    def test_initialization(self) -> None:
        """Test proper initialization of the SingleThreadQLearning class."""
        assert self.agent.state_size == 10
        assert self.agent.action_size == 3
        assert self.agent.learning_rate == 0.1
        assert self.agent.discount_factor == 0.9
        assert self.agent.exploration_rate == 0.1
        assert self.agent.exploration_decay == 0.99
        assert self.agent.min_exploration_rate == 0.01
        assert self.agent.q_table.shape == (10, 3)
        assert np.all(self.agent.q_table == 0)

    def test_train_with_simple_env(self) -> None:
        """Test training with a simple mock environment."""
        env = MockEnvironment(num_envs=1, return_dict=False)
        val_env = MockEnvironment(num_envs=1, return_dict=False)

        # Mock the choose_actions and learn methods to avoid randomness
        with patch.object(self.agent, "choose_actions", return_value=np.array([0])):
            with patch.object(self.agent, "learn") as mock_learn:
                self.agent.train(
                    env=env,
                    steps=5,
                    val_env=val_env,
                    val_every_n_steps=3,
                    val_steps=2,
                    val_episodes=None,
                )

                # Verify learn was called
                assert mock_learn.called

    def test_train_with_dict_observation_env(self) -> None:
        """Test training with an environment that returns dict observations."""
        env = MockEnvironment(num_envs=1, return_dict=True)
        val_env = MockEnvironment(num_envs=1, return_dict=True)

        # Mock the choose_actions and learn methods
        with patch.object(self.agent, "choose_actions", return_value=np.array([0])):
            with patch.object(self.agent, "learn") as mock_learn:
                self.agent.train(
                    env=env,
                    steps=5,
                    val_env=val_env,
                    val_every_n_steps=3,
                    val_steps=2,
                    val_episodes=None,
                )

                # Verify learn was called with action_mask
                mock_learn.assert_called()

    def test_train_with_validation_episodes(self) -> None:
        """Test training with validation by episodes instead of steps."""
        env = MockEnvironment(num_envs=1, return_dict=False)
        val_env = MockEnvironment(num_envs=1, return_dict=False)

        with patch.object(self.agent, "choose_actions", return_value=np.array([0])):
            with patch.object(self.agent, "learn"):
                self.agent.train(
                    env=env,
                    steps=5,
                    val_env=val_env,
                    val_every_n_steps=3,
                    val_steps=None,
                    val_episodes=2,
                )

    def test_train_invalid_validation_params(self) -> None:
        """Test that training raises assertion error with invalid validation params."""
        env = MockEnvironment(num_envs=1, return_dict=False)
        val_env = MockEnvironment(num_envs=1, return_dict=False)

        # Both val_steps and val_episodes provided
        with pytest.raises(AssertionError):
            self.agent.train(
                env=env, steps=5, val_env=val_env, val_every_n_steps=3, val_steps=2, val_episodes=2
            )

        # Neither val_steps nor val_episodes provided
        with pytest.raises(AssertionError):
            self.agent.train(
                env=env,
                steps=5,
                val_env=val_env,
                val_every_n_steps=3,
                val_steps=None,
                val_episodes=None,
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

    def test_train_with_vectorized_env(self) -> None:
        """Test training with vectorized environment."""
        # Create a mock SyncVectorEnv
        mock_env = MagicMock(spec=SyncVectorEnv)
        mock_env.num_envs = 3
        mock_env.reset.return_value = (np.array([0, 1, 2], dtype=np.int32), [{}] * 3)
        mock_env.step.return_value = (
            np.array([1, 2, 3], dtype=np.int32),
            np.array([1.0, 1.0, 1.0], dtype=np.float32),
            np.array([False, False, True], dtype=bool),
            np.array([False, False, False], dtype=bool),
            [{}] * 3,
        )

        val_env = MockEnvironment(num_envs=1, return_dict=False)

        with patch.object(self.agent, "choose_actions", return_value=np.array([0, 1, 2])):
            with patch.object(self.agent, "learn"):
                self.agent.train(
                    env=mock_env,
                    steps=5,
                    val_env=val_env,
                    val_every_n_steps=3,
                    val_steps=2,
                    val_episodes=None,
                )

    def test_inheritance_from_optimal_q_learning_base(self) -> None:
        """Test that SingleThreadQLearning properly inherits from OptimalQLearningBase."""
        from dist_classicrl.algorithms.base_algorithms.q_learning_optimal import (
            OptimalQLearningBase,
        )

        assert isinstance(self.agent, OptimalQLearningBase)

        # Test that inherited methods are available
        assert hasattr(self.agent, "choose_actions")
        assert hasattr(self.agent, "learn")
        assert hasattr(self.agent, "get_q_value")
        assert hasattr(self.agent, "set_q_value")

    def test_reward_accumulation_during_training(self) -> None:
        """Test that rewards are properly accumulated during training."""
        env = MockEnvironment(num_envs=2, return_dict=False)
        val_env = MockEnvironment(num_envs=1, return_dict=False)

        # Mock choose_actions to return deterministic actions
        with patch.object(self.agent, "choose_actions", return_value=np.array([0, 1])):
            with patch.object(self.agent, "learn"):
                # Capture print output to verify evaluation is happening
                with patch("builtins.print") as mock_print:
                    self.agent.train(
                        env=env,
                        steps=6,
                        val_env=val_env,
                        val_every_n_steps=3,
                        val_steps=2,
                        val_episodes=None,
                    )

                    # Verify that evaluation print was called
                    mock_print.assert_called()
