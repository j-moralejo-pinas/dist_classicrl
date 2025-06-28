"""
Unit tests for the OptimalQLearningBase class.

This module contains comprehensive tests for all methods and functionality
of the OptimalQLearningBase class from the q_learning_optimal module.
"""

from unittest.mock import patch

import numpy as np
import pytest

from dist_classicrl.algorithms.base_algorithms.q_learning_optimal import (
    OptimalQLearningBase,
)


class TestOptimalQLearningBase:
    """Test class for OptimalQLearningBase."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.state_size = 4
        self.action_size = 3
        self.agent = OptimalQLearningBase(
            state_size=self.state_size,
            action_size=self.action_size,
            learning_rate=0.1,
            discount_factor=0.9,
            exploration_rate=1.0,
            exploration_decay=0.99,
            min_exploration_rate=0.01,
        )

    def test_initialization(self):
        """Test proper initialization of the OptimalQLearningBase class."""
        assert self.agent.state_size == 4
        assert self.agent.action_size == 3
        assert self.agent.learning_rate == 0.1
        assert self.agent.discount_factor == 0.9
        assert self.agent.exploration_rate == 1.0
        assert self.agent.exploration_decay == 0.99
        assert self.agent.min_exploration_rate == 0.01
        assert self.agent.q_table.shape == (4, 3)
        assert np.all(self.agent.q_table == 0)

    def test_initialization_with_numpy_integers(self):
        """Test initialization with numpy integer types."""
        agent = OptimalQLearningBase(state_size=np.int32(4), action_size=np.int64(3))
        assert agent.state_size == 4
        assert agent.action_size == 3
        assert agent.q_table.shape == (4, 3)

    def test_get_q_value(self):
        """Test getting Q-value for a state-action pair."""
        # Set a specific Q-value
        self.agent.q_table[2, 1] = 0.5

        # Test retrieval
        q_value = self.agent.get_q_value(2, 1)
        assert q_value == 0.5

        # Test default value (should be 0)
        q_value = self.agent.get_q_value(0, 0)
        assert q_value == 0.0

    def test_get_q_values(self):
        """Test getting Q-values for multiple state-action pairs."""
        # Set specific Q-values
        self.agent.q_table[1, 0] = 0.3
        self.agent.q_table[2, 1] = 0.7
        self.agent.q_table[3, 2] = 0.9

        states = np.array([1, 2, 3], dtype=np.int32)
        actions = np.array([0, 1, 2], dtype=np.int32)

        q_values = self.agent.get_q_values(states, actions)
        expected = np.array([0.3, 0.7, 0.9])

        assert (q_values == expected).all(), "Q-values do not match expected values."

    def test_get_state_q_values(self):
        """Test getting all Q-values for a specific state."""
        # Set Q-values for state 2
        self.agent.q_table[2] = [0.1, 0.5, 0.3]

        q_values = self.agent.get_state_q_values(2)
        expected = np.array([0.1, 0.5, 0.3])

        assert (q_values == expected).all(), "Q-values do not match expected values."

    def test_get_states_q_values(self):
        """Test getting Q-values for multiple states."""
        # Set Q-values for states 1 and 3
        self.agent.q_table[1] = [0.1, 0.2, 0.3]
        self.agent.q_table[3] = [0.5, 0.6, 0.7]

        states = np.array([1, 3], dtype=np.int32)
        q_values = self.agent.get_states_q_values(states)

        expected = np.array([[0.1, 0.2, 0.3], [0.5, 0.6, 0.7]])

        assert (q_values == expected).all(), "Q-values do not match expected values."

    def test_set_q_value(self):
        """Test setting Q-value for a state-action pair."""
        self.agent.set_q_value(1, 2, 0.75)
        assert self.agent.q_table[1, 2] == 0.75

    def test_add_q_value(self):
        """Test adding to Q-value for a state-action pair."""
        self.agent.q_table[1, 2] = 0.5
        self.agent.add_q_value(1, 2, 0.25)
        assert self.agent.q_table[1, 2] == 0.75

    def test_add_q_values(self):
        """Test adding Q-values for multiple state-action pairs."""
        # Set initial values
        self.agent.q_table[1, 0] = 0.2
        self.agent.q_table[2, 1] = 0.4

        states = np.array([1, 2], dtype=np.int32)
        actions = np.array([0, 1], dtype=np.int32)
        values = np.array([0.3, 0.1], dtype=np.float64)

        self.agent.add_q_values(states, actions, values)

        assert self.agent.q_table[1, 0] == 0.5
        assert self.agent.q_table[2, 1] == 0.5

    # def test_save_and_load(self):
    #     """Test saving Q-table to file."""
    #     # Set some Q-values
    #     self.agent.q_table[0, 0] = 0.1
    #     self.agent.q_table[1, 1] = 0.2

    #     # Save to temporary file
    #     with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp:
    #         tmp_name = tmp.name

    #     try:
    #         self.agent.save(tmp_name.replace(".npy", ""))

    #         # Load and verify
    #         loaded_q_table = np.load(tmp_name)
    #         assert np.allclose(loaded_q_table, self.agent.q_table)
    #     finally:
    #         if os.path.exists(tmp_name):
    #             os.unlink(tmp_name)

    def test_choose_action_deterministic(self):
        """Test deterministic action selection."""
        # Set Q-values where action 2 is best for state 0
        self.agent.q_table[0] = [0.1, 0.2, 0.8]

        action = self.agent.choose_action(0, deterministic=True)
        assert action == 2

    def test_choose_action_tie_breaking(self):
        """Test tie breaking in deterministic action selection."""
        # Set Q-values where actions 1 and 3 are tied for best
        self.agent.q_table[0] = [0.1, 0.8, 0.8]

        # Run multiple times to check randomness in tie breaking
        actions = [self.agent.choose_action(0, deterministic=True) for _ in range(100)]

        # Should only choose actions 1 or 2
        assert all(action in [1, 2] for action in actions)
        # Should use both actions (with high probability)
        assert len(set(actions)) > 1

    def test_choose_action_exploration(self):
        """Test exploratory action selection."""
        # Set exploration rate to 1.0 (always explore)
        self.agent.exploration_rate = 1.0

        # Run multiple times and check that different actions are chosen
        actions = [self.agent.choose_action(0, deterministic=False) for _ in range(100)]

        # Should explore different actions
        assert len(set(actions)) > 1
        # All actions should be valid
        assert all(0 <= action < self.action_size for action in actions)

    def test_choose_action_no_exploration(self):
        """Test action selection with no exploration."""
        # Set exploration rate to 0 (never explore)
        self.agent.exploration_rate = 0.0
        self.agent.q_table[0] = [0.1, 0.2, 0.8]

        # Should always choose best action (action 2)
        for _ in range(10):
            action = self.agent.choose_action(0, deterministic=False)
            assert action == 2

    def test_choose_masked_action(self):
        """Test masked action selection."""
        # Set Q-values
        self.agent.q_table[0] = [0.1, 0.8, 0.3]

        # Mask out actions 1 (best action)
        action_mask = [1, 0, 1]

        action = self.agent.choose_masked_action(0, action_mask, deterministic=True)
        # Should choose action 2 (best among available)
        assert action == 2

    def test_choose_masked_action_exploration(self):
        """Test masked action selection with exploration."""
        self.agent.exploration_rate = 1.0
        action_mask = [1, 0, 1]  # Only actions 0 and 2 are available

        actions = [
            self.agent.choose_masked_action(0, action_mask, deterministic=False) for _ in range(100)
        ]

        # Should only choose from available actions
        assert all(action in [0, 2] for action in actions)

    def test_choose_masked_action_invalid_mask(self):
        """Test error handling for invalid action mask."""
        invalid_mask = [1, 0]  # Wrong size

        with pytest.raises(AssertionError):
            self.agent.choose_masked_action(0, invalid_mask, deterministic=True)

    def test_choose_actions_iter(self):
        """Test iterative action selection for multiple states."""
        # Set Q-values
        self.agent.q_table[0] = [0.1, 0.8, 0.3]
        self.agent.q_table[1] = [0.5, 0.2, 0.9]

        states = np.array([0, 1], dtype=np.int32)
        actions = self.agent.choose_actions_iter(states, deterministic=True)

        # Should choose best actions for each state
        assert actions[0] == 1  # Best for state 0
        assert actions[1] == 2  # Best for state 1

    def test_choose_actions_iter_with_masks(self):
        """Test iterative action selection with action masks."""
        self.agent.q_table[0] = [0.1, 0.8, 0.3]

        states = np.array([0], dtype=np.int32)
        action_masks = np.array([[1, 0, 1]], dtype=np.int32)

        actions = self.agent.choose_actions_iter(
            states, deterministic=True, action_masks=action_masks
        )

        # Should choose action 2 (best among available: 0, 2)
        assert actions[0] == 2

    def test_choose_action_vec(self):
        """Test vectorized single action selection."""
        # Set Q-values where action 2 is best
        self.agent.q_table[0] = [0.1, 0.2, 0.8]

        action = self.agent.choose_action_vec(0, deterministic=True)
        assert action == 2

    def test_choose_masked_action_vec(self):
        """Test vectorized masked action selection."""
        # Set Q-values
        self.agent.q_table[0] = [0.1, 0.8, 0.3]

        # Mask out actions 1 and 3
        action_mask = [1, 0, 1]

        action = self.agent.choose_masked_action_vec(0, action_mask, deterministic=True)
        assert action == 2

    def test_choose_masked_action_vec_invalid_mask(self):
        """Test error handling for invalid action mask in vectorized version."""
        invalid_mask = [1, 0]  # Wrong size

        with pytest.raises(AssertionError):
            self.agent.choose_masked_action_vec(0, invalid_mask, deterministic=True)

    def test_choose_actions_vec_iter(self):
        """Test vectorized iterative action selection."""
        self.agent.q_table[0] = [0.1, 0.8, 0.3]
        self.agent.q_table[1] = [0.5, 0.2, 0.9]
        states = np.array([0, 1], dtype=np.int32)
        actions = self.agent.choose_actions_vec_iter(states, deterministic=True)

        assert actions[0] == 1
        assert actions[1] == 2

    def test_choose_actions_vec(self):
        """Test fully vectorized action selection."""
        self.agent.q_table[0] = [0.1, 0.8, 0.3]
        self.agent.q_table[1] = [0.5, 0.2, 0.9]
        self.agent.q_table[2] = [0.2, 0.2, 0.2]  # All equal

        states = np.array([0, 1, 2], dtype=np.int32)
        actions = self.agent.choose_actions_vec(states, deterministic=True)

        assert actions[0] == 1
        assert actions[1] == 2
        # For state 2, any action is valid due to tie
        assert 0 <= actions[2] < 3

    def test_choose_masked_actions_vec(self):
        """Test fully vectorized masked action selection."""
        self.agent.q_table[0] = [0.1, 0.8, 0.3]
        self.agent.q_table[1] = [0.5, 0.2, 0.9]

        states = np.array([0, 1], dtype=np.int32)
        action_masks = np.array(
            [
                [1, 0, 1],  # Actions 0, 2 available for state 0
                [1, 1, 0],  # Actions 0, 1 available for state 1
            ],
            dtype=np.int32,
        )

        actions = self.agent.choose_masked_actions_vec(states, action_masks, deterministic=True)

        assert actions[0] == 2  # Best available for state 0
        assert actions[1] == 0  # Best available for state 1

    def test_choose_masked_actions_vec_invalid_shape(self):
        """Test error handling for invalid action mask shape."""
        states = np.array([0, 1], dtype=np.int32)
        invalid_masks = np.array([[1, 0]], dtype=np.int32)  # Wrong shape

        with pytest.raises(AssertionError):
            self.agent.choose_masked_actions_vec(states, invalid_masks, deterministic=True)

    def test_choose_actions_dispatch(self):
        """Test that choose_actions properly dispatches to optimal method."""
        states = np.array([0], dtype=np.int32)

        # Test without masks
        actions = self.agent.choose_actions(states, deterministic=True)
        assert len(actions) == 1

        # Test with masks
        action_masks = np.array([[1, 1, 1]], dtype=np.int32)
        actions = self.agent.choose_actions(states, deterministic=True, action_masks=action_masks)
        assert len(actions) == 1

    def test_update_explore_rate(self):
        """Test exploration rate decay."""
        initial_rate = self.agent.exploration_rate
        self.agent.update_explore_rate()

        expected_rate = max(
            self.agent.min_exploration_rate, initial_rate * self.agent.exploration_decay
        )

        assert self.agent.exploration_rate == expected_rate

    def test_update_explore_rate_minimum(self):
        """Test exploration rate doesn't go below minimum."""
        self.agent.exploration_rate = self.agent.min_exploration_rate
        self.agent.update_explore_rate()

        assert self.agent.exploration_rate == self.agent.min_exploration_rate

    def test_single_learn_not_terminated(self):
        """Test single learning step when episode is not terminated."""
        # Set up initial Q-value
        self.agent.q_table[0, 1] = 0.5
        self.agent.q_table[2] = [0.1, 0.3, 0.8]  # Max is 0.8

        # Learn from experience
        self.agent.single_learn(state=0, action=1, reward=1.0, next_state=2, terminated=False)

        # Calculate expected Q-value
        max_next_q = 0.8
        target = 1.0 + self.agent.discount_factor * max_next_q
        prediction = 0.5
        expected_q = 0.5 + self.agent.learning_rate * (target - prediction)

        assert self.agent.q_table[0, 1] == expected_q

    def test_single_learn_terminated(self):
        """Test single learning step when episode is terminated."""
        self.agent.q_table[0, 1] = 0.5

        self.agent.single_learn(state=0, action=1, reward=1.0, next_state=2, terminated=True)

        # When terminated, next state Q-value should be 0
        target = 1.0  # reward + 0 (no future reward)
        prediction = 0.5
        expected_q = 0.5 + self.agent.learning_rate * (target - prediction)

        assert self.agent.q_table[0, 1] == expected_q

    def test_single_learn_with_mask(self):
        """Test single learning step with action mask for next state."""
        self.agent.q_table[0, 1] = 0.5
        self.agent.q_table[2] = [0.1, 0.3, 0.8]

        # Mask out the best action (index 2)
        next_action_mask = np.array([1, 1, 0], dtype=np.int32)

        self.agent.single_learn(
            state=0,
            action=1,
            reward=1.0,
            next_state=2,
            terminated=False,
            next_action_mask=next_action_mask,
        )

        # Max among available actions is 0.3
        max_next_q = 0.3
        target = 1.0 + self.agent.discount_factor * max_next_q
        prediction = 0.5
        expected_q = 0.5 + self.agent.learning_rate * (target - prediction)

        assert self.agent.q_table[0, 1] == expected_q

    def test_learn_iter(self):
        """Test iterative learning from multiple experiences."""
        # Set initial Q-values
        self.agent.q_table[0, 0] = 0.2
        self.agent.q_table[1, 1] = 0.3

        states = np.array([0, 1], dtype=np.int32)
        actions = np.array([0, 1], dtype=np.int32)
        rewards = np.array([0.5, 1.0], dtype=np.float32)
        next_states = np.array([2, 3], dtype=np.int32)
        terminated = np.array([False, True], dtype=np.bool)

        # Set Q-values for next states
        self.agent.q_table[2] = [0.1, 0.2, 0.6]  # Max is 0.6
        self.agent.q_table[3] = [0.1, 0.1, 0.1]  # Doesn't matter (terminated)

        prediction_0 = self.agent.q_table[0, 0]
        prediction_1 = self.agent.q_table[1, 1]

        max_next_q_2 = 0.6  # For state 2
        target_0 = 0.5 + self.agent.discount_factor * max_next_q_2
        expected_q_00 = prediction_0 + self.agent.learning_rate * (target_0 - prediction_0)
        target_1 = 1.0
        expected_q_11 = prediction_1 + self.agent.learning_rate * (target_1 - prediction_1)

        self.agent.learn_iter(states, actions, rewards, next_states, terminated)

        # Verify Q-values were updated
        assert self.agent.q_table[0, 0] == expected_q_00
        assert self.agent.q_table[1, 1] == expected_q_11

    def test_learn_vec(self):
        """Test vectorized learning from multiple experiences."""
        states = np.array([0, 1], dtype=np.int32)
        actions = np.array([0, 1], dtype=np.int32)
        rewards = np.array([0.5, 1.0], dtype=np.float32)
        next_states = np.array([2, 3], dtype=np.int32)
        terminated = np.array([False, True], dtype=np.bool)

        # Set Q-values for next states
        self.agent.q_table[2] = [0.1, 0.2, 0.6]

        initial_q_values = self.agent.q_table.copy()

        prediction_0 = self.agent.q_table[0, 0]
        prediction_1 = self.agent.q_table[1, 1]
        max_next_q_2 = 0.6  # For state 2
        target_0 = 0.5 + self.agent.discount_factor * max_next_q_2
        expected_q_00 = prediction_0 + self.agent.learning_rate * (target_0 - prediction_0)
        target_1 = 1.0
        expected_q_11 = prediction_1 + self.agent.learning_rate * (target_1 - prediction_1)

        self.agent.learn_vec(states, actions, rewards, next_states, terminated)

        # Verify Q-values were updated
        assert self.agent.q_table[0, 0] == expected_q_00
        assert self.agent.q_table[1, 1] == expected_q_11

    def test_learn_dispatch(self):
        """Test that learn properly dispatches to appropriate method."""
        states = np.array([0], dtype=np.int32)
        actions = np.array([0], dtype=np.int32)
        rewards = np.array([1.0], dtype=np.float32)
        next_states = np.array([1], dtype=np.int32)
        terminated = np.array([False], dtype=np.bool)

        initial_exploration_rate = self.agent.exploration_rate

        self.agent.learn(states, actions, rewards, next_states, terminated)

        assert self.agent.exploration_rate == max(
            initial_exploration_rate * self.agent.exploration_decay, self.agent.min_exploration_rate
        )

    def test_learn_with_large_batch(self):
        """Test learning with large batch (should use vectorized version)."""
        batch_size = 15  # > 10, should trigger vectorized learning
        states = np.random.randint(0, self.state_size, batch_size, dtype=np.int32)
        actions = np.random.randint(0, self.action_size, batch_size, dtype=np.int32)
        rewards = np.random.rand(batch_size).astype(np.float32)
        next_states = np.random.randint(0, self.state_size, batch_size, dtype=np.int32)
        terminated = np.random.choice([True, False], batch_size).astype(np.bool)

        initial_exploration_rate = self.agent.exploration_rate

        self.agent.learn(states, actions, rewards, next_states, terminated)

        # Should update exploration rate
        assert self.agent.exploration_rate < initial_exploration_rate

    @patch("random.uniform")
    @patch("random.randint")
    def test_choose_action_randomness(self, mock_randint, mock_uniform):
        """Test randomness in action selection."""
        # Test exploration
        mock_uniform.return_value = 0.5  # Less than exploration_rate (1.0)
        mock_randint.return_value = 2

        action = self.agent.choose_action(0, deterministic=False)
        assert action == 2
        mock_randint.assert_called_with(0, self.action_size - 1)

    @patch("random.choice")
    def test_choose_action_tie_breaking_randomness(self, mock_choice):
        """Test random tie breaking in action selection."""
        # Set tied Q-values
        self.agent.q_table[0] = [0.5, 0.5, 0.5]
        mock_choice.return_value = 2

        action = self.agent.choose_action(0, deterministic=True)
        assert action == 2

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Test with state_size = 1, action_size = 1
        small_agent = OptimalQLearningBase(state_size=1, action_size=1)
        action = small_agent.choose_action(0, deterministic=True)
        assert action == 0

        # Test with very small learning rate
        small_lr_agent = OptimalQLearningBase(state_size=5, action_size=3, learning_rate=1e-10)
        initial_q = small_lr_agent.q_table[0, 0]
        small_lr_agent.single_learn(0, 0, 1.0, 1, False)
        # Q-value should barely change
        assert abs(small_lr_agent.q_table[0, 0] - initial_q) < 1e-8

    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        # Test with very large Q-values
        self.agent.q_table[0] = [1e6, -1e6, 1e6]
        action = self.agent.choose_action(0, deterministic=True)
        assert action in [0, 2]  # Should handle large values correctly

        # Test with very large rewards
        self.agent.single_learn(0, 0, 1e6, 1, False)
        assert not np.isnan(self.agent.q_table[0, 0])
        assert not np.isinf(self.agent.q_table[0, 0])

    def test_consistency_between_methods(self):
        """Test consistency between different implementation methods."""
        # Set same Q-values
        self.agent.q_table[0] = [0.1, 0.8, 0.3]
        self.agent.q_table[1] = [0.5, 0.2, 0.9]

        states = np.array([0, 1], dtype=np.int32)

        # Test consistency between different action selection methods
        # (when deterministic and no exploration)
        self.agent.exploration_rate = 0.0

        actions_iter = self.agent.choose_actions_iter(states, deterministic=True)
        actions_vec_iter = self.agent.choose_actions_vec_iter(states, deterministic=True)
        actions_vec = self.agent.choose_actions_vec(states, deterministic=True)

        # All methods should give same results for deterministic selection
        assert np.array_equal(actions_iter, actions_vec_iter)
        assert np.array_equal(actions_iter, actions_vec)

    def test_q_table_operations_consistency(self):
        """Test consistency of Q-table operations."""
        # Test that get/set operations are consistent
        test_value = 0.123456
        self.agent.set_q_value(3, 2, test_value)
        retrieved_value = self.agent.get_q_value(3, 2)
        assert retrieved_value == test_value

        # Test vectorized operations consistency
        states = np.array([1, 2], dtype=np.int32)
        actions = np.array([0, 1], dtype=np.int32)
        values = np.array([0.5, 0.7], dtype=np.float64)

        # Set values individually
        self.agent.set_q_value(1, 0, 0.5)
        self.agent.set_q_value(2, 1, 0.7)

        # Get values vectorized
        retrieved_values = self.agent.get_q_values(states, actions)
        assert (retrieved_values == values).all()


# Additional test class for parametrized tests
class TestOptimalQLearningParametrized:
    """Parametrized tests for OptimalQLearning with different configurations."""

    @pytest.mark.parametrize(
        "state_size,action_size",
        [
            (5, 3),
            (100, 10),
            (10, 100),
            (1000, 5),
        ],
    )
    def test_different_sizes(self, state_size, action_size):
        """Test OptimalQLearning with different state and action space sizes."""
        agent = OptimalQLearningBase(state_size=state_size, action_size=action_size)

        assert agent.state_size == state_size
        assert agent.action_size == action_size
        assert agent.q_table.shape == (state_size, action_size)

        # Test basic functionality
        if state_size > 0 and action_size > 0:
            action = agent.choose_action(0, deterministic=True)
            assert 0 <= action < action_size

    @pytest.mark.parametrize("learning_rate", [0.01, 0.1, 0.5, 1.0])
    def test_different_learning_rates(self, learning_rate):
        """Test learning with different learning rates."""
        agent = OptimalQLearningBase(state_size=5, action_size=3, learning_rate=learning_rate)

        initial_q = agent.q_table[0, 0]
        agent.single_learn(0, 0, 1.0, 1, True)

        # Change should be proportional to learning rate
        change = abs(agent.q_table[0, 0] - initial_q)
        expected_change = learning_rate * abs(1.0 - initial_q)
        assert change == expected_change, f"Expected change {expected_change}, got {change}"

    @pytest.mark.parametrize("exploration_rate", [0.0, 0.1, 0.5, 1.0])
    def test_exploration_rates(self, exploration_rate):
        """Test action selection with different exploration rates."""
        agent = OptimalQLearningBase(state_size=5, action_size=3, exploration_rate=exploration_rate)

        # Set clear best action
        agent.q_table[0] = [0.1, 0.9, 0.1]

        # Count how often best action (1) is chosen
        best_action_count = 0
        total_trials = 1000

        for _ in range(total_trials):
            action = agent.choose_action(0, deterministic=False)
            if action == 1:
                best_action_count += 1

        best_action_ratio = best_action_count / total_trials

        if exploration_rate == 0.0:
            # Should always choose best action
            assert best_action_ratio > 0.95
        elif exploration_rate == 1.0:
            # Should choose randomly (approximately 25% for best action)
            assert 0.15 < best_action_ratio < 0.35
        else:
            # Should be between random and greedy
            expected_ratio = (1 - exploration_rate) + exploration_rate * 0.25
            assert abs(best_action_ratio - expected_ratio) < 0.1
