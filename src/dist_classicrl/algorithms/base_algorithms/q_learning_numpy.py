"""Multi-agent Q-learning base implementation using NumPy."""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

NO_ACTION_MASKS_ACTION_SIZE_THRESHOLD = 100
ACTION_MASKS_NO_DETERMINISTIC_ACTION_SIZE_THRESHOLD = 10
ACTION_MASKS_DETERMINISTIC_ACTION_SIZE_THRESHOLD = 250


class MultiAgentQLearningNumpy:
    """
    Multi-agent Q-learning class.

    Attributes
    ----------
    num_agents : int
        Number of agents in the environment.
    state_size : int
        Size of the state space.
    action_size : int
        Size of the action space.
    learning_rate : float
        Learning rate for Q-learning.
    discount_factor : float
        Discount factor for future rewards.
    exploration_rate : float
        Initial exploration rate for epsilon-greedy policy.
    exploration_decay : float
        Decay rate for exploration rate.
    min_exploration_rate : float
        Minimum exploration rate.
    q_table : List[float]
        Q-table for the agents.
    _rng : np.random.Generator
        Random number generator for reproducibility.
    """

    num_agents: int
    state_size: int
    action_size: int
    learning_rate: float
    discount_factor: float
    exploration_rate: float
    exploration_decay: float
    min_exploration_rate: float
    q_table: NDArray[np.float64]
    _rng: np.random.Generator

    def __init__(
        self,
        num_agents: int,
        state_size: int,
        action_size: int,
        learning_rate: float = 0.1,
        discount_factor: float = 0.97,
        exploration_rate: float = 1.0,
        exploration_decay: float = 0.999,
        min_exploration_rate: float = 0.01,
        seed: int | None = None,
    ) -> None:
        """
        Initialize the MultiAgentQLearning class.

        Parameters
        ----------
        num_agents : int
            Number of agents in the environment.
        state_size : int
            Size of the state space.
        action_size : int
            Size of the action space.
        learning_rate : float, optional
            Learning rate for Q-learning, by default 0.1.
        discount_factor : float, optional
            Discount factor for future rewards, by default 0.99.
        exploration_rate : float, optional
            Initial exploration rate for epsilon-greedy policy, by default 1.0.
        exploration_decay : float, optional
            Decay rate for exploration rate, by default 0.995.
        min_exploration_rate : float, optional
            Minimum exploration rate, by default 0.01.
        seed : int, optional
            Random seed for reproducibility, by default None.
        """
        self.num_agents = num_agents
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate
        self.q_table = np.zeros((state_size, action_size))
        self._rng = np.random.default_rng(seed)

    def get_q_value(self, state: int, action: int) -> float:
        """
        Get the Q-value for a given state-action pair.

        Parameters
        ----------
        state : int
            State index.
        action : int
            Action index.

        Returns
        -------
        float
            Q-value for the state-action pair.
        """
        return self.q_table[state, action]

    def get_q_values(
        self, states: NDArray[np.int32], actions: NDArray[np.int32]
    ) -> NDArray[np.float64]:
        """
        Get the Q-values for a given list of state-action pairs.

        Parameters
        ----------
        states : NDArray[np.int32]
            List of state indices.
        actions : NDArray[np.int32]
            List of action indices.

        Returns
        -------
        NDArray[np.float64]
            Q-values for the state-action pairs.
        """
        return self.q_table[states, actions]

    def get_state_q_values(self, state: int) -> NDArray[np.float64]:
        """
        Get the Q-values for a given state.

        Parameters
        ----------
        state : int
            State index.

        Returns
        -------
        NDArray[np.float64]
            Q-values for the state.
        """
        return self.q_table[state]

    def get_states_q_values(
        self,
        states: NDArray[np.int32],
    ) -> NDArray[np.float64]:
        """
        Get the Q-values for a given list of states.

        Parameters
        ----------
        states : NDArray[np.int32]
            List of state indices.

        Returns
        -------
        NDArray[np.float64]
            Q-values for the states.
        """
        return self.q_table[states]

    def set_q_value(self, state: int, action: int, value: float) -> None:
        """
        Set the Q-value for a given state-action pair.

        Parameters
        ----------
        state : int
            State index.
        action : int
            Action index.
        value : float
            Q-value to set.
        """
        self.q_table[state, action] = value

    def add_q_values(
        self, states: NDArray[np.int32], actions: NDArray[np.int32], values: NDArray[np.float64]
    ) -> None:
        """
        Add Q-values for a given list of state-action pairs.

        Parameters
        ----------
        states : NDArray[np.int32]
            List of state indices.
        actions : NDArray[np.int32]
            List of action indices.
        values : NDArray[np.float64]
            List of Q-values to add.
        """
        self.q_table[states, actions] += values

    def save(self, filename: str) -> None:
        """
        Save the Q-table to a file.

        Parameters
        ----------
        filename : str
            File to save the Q-table to.
        """
        np.save(filename, self.q_table)

    def choose_action(  # noqa: C901 PLR0912
        self,
        state: int,
        *,
        deterministic: bool = False,
        action_mask: NDArray[np.int32] | None = None,
    ) -> int:
        """
        Choose an action based on the current state.

        Parameters
        ----------
        state : int
            Current state of the agent.
        deterministic : bool, optional
            Whether to choose the action deterministically, by default False.
        action_mask : Optional[List[int]], optional
            Mask for valid actions, by default None.

        Returns
        -------
        int
            Action chosen by the agent.
        """
        if action_mask is None:
            if not deterministic and random.random() < self.exploration_rate:
                return random.randint(0, self.action_size - 1)
            if self.action_size > NO_ACTION_MASKS_ACTION_SIZE_THRESHOLD:
                max_val = np.max(self.q_table[state])
                available_actions = np.where(self.q_table[state] == max_val)[0]
            else:
                max_val = -math.inf
                available_actions = []
                for idx, val in enumerate(self.q_table[state]):
                    if val > max_val:
                        max_val = val
                        available_actions = [idx]
                    elif val == max_val:
                        available_actions.append(idx)
            return random.choice(available_actions)

        assert action_mask.size == self.action_size, (
            "Action mask should have the same size as the action space."
        )

        if not deterministic and random.random() < self.exploration_rate:
            if self.action_size > ACTION_MASKS_NO_DETERMINISTIC_ACTION_SIZE_THRESHOLD:
                available_actions = np.where(action_mask == 1)[0]
            else:
                available_actions = []
                for i, val in enumerate(action_mask):
                    if val:
                        available_actions.append(i)
        elif self.action_size > ACTION_MASKS_DETERMINISTIC_ACTION_SIZE_THRESHOLD:
            masked_q_values = np.where(action_mask, self.q_table[state], -np.inf)
            max_val = np.max(masked_q_values)
            available_actions = np.where(masked_q_values == max_val)[0]
        else:
            max_val = 0
            available_actions = []
            for idx, (val, mask) in enumerate(zip(self.q_table[state], action_mask)):
                if mask:
                    if val > max_val:
                        max_val = val
                        available_actions = [idx]
                    elif val == max_val:
                        available_actions.append(idx)

        return random.choice(available_actions) if len(available_actions) > 0 else -1

    def choose_action_semi_vectorized(  # noqa: C901 PLR0912
        self,
        state: int,
        *,
        deterministic: bool = False,
        action_mask: NDArray[np.int32] | None = None,
    ) -> int:
        """
        Choose an action based on the current state.

        This method is a semi-vectorized version that handles action selection
        with or without an action mask. It uses NumPy operations for efficiency.

        Parameters
        ----------
        state : int
            Current state of the agent.
        deterministic : bool, optional
            Whether to choose the action deterministically, by default False.
        action_mask : Optional[List[int]], optional
            Mask for valid actions, by default None.

        Returns
        -------
        int
            Action chosen by the agent.
        """
        if action_mask is None:
            if not deterministic and random.random() < self.exploration_rate:
                return random.randint(0, self.action_size - 1)

            max_val = -math.inf
            available_actions = []
            for idx, val in enumerate(self.q_table[state]):
                if val > max_val:
                    max_val = val
                    available_actions = [idx]
                elif val == max_val:
                    available_actions.append(idx)
            return random.choice(available_actions)

        assert action_mask.size == self.action_size, (
            "Action mask should have the same size as the action space."
        )

        if not deterministic and random.random() < self.exploration_rate:
            available_actions = []
            for i, val in enumerate(action_mask):
                if val:
                    available_actions.append(i)
        else:
            max_val = 0
            available_actions = []
            for idx, (val, mask) in enumerate(zip(self.q_table[state], action_mask)):
                if mask:
                    if val > max_val:
                        max_val = val
                        available_actions = [idx]
                    elif val == max_val:
                        available_actions.append(idx)

        return random.choice(available_actions) if len(available_actions) > 0 else -1

    def choose_action_fully_vectorized(
        self,
        state: int,
        *,
        deterministic: bool = False,
        action_mask: NDArray[np.int32] | None = None,
    ) -> int:
        """
        Choose an action based on the current state.

        This method is a fully vectorized version that handles action selection
        with or without an action mask. It uses NumPy operations for efficiency.

        Parameters
        ----------
        state : int
            Current state of the agent.
        deterministic : bool, optional
            Whether to choose the action deterministically, by default False.
        action_mask : Optional[List[int]], optional
            Mask for valid actions, by default None.

        Returns
        -------
        int
            Action chosen by the agent.
        """
        if action_mask is None:
            if not deterministic and random.random() < self.exploration_rate:
                return random.randint(0, self.action_size - 1)

            max_val = np.max(self.q_table[state])
            available_actions = np.where(self.q_table[state] == max_val)[0]
            return random.choice(available_actions)

        assert action_mask.size == self.action_size, (
            "Action mask should have the same size as the action space."
        )

        if not deterministic and random.random() < self.exploration_rate:
            available_actions = np.where(action_mask == 1)[0]

        else:
            masked_q_values = np.where(action_mask, self.q_table[state], -np.inf)
            max_val = np.max(masked_q_values)
            available_actions = np.where(masked_q_values == max_val)[0]

        return random.choice(available_actions) if len(available_actions) > 0 else -1

    def choose_actions(
        self,
        states: NDArray[np.int32],
        *,
        deterministic: bool = False,
        action_masks: NDArray[np.int32] | None = None,
    ) -> NDArray[np.int32]:
        """
        Choose actions for all agents based on the current states.

        Parameters
        ----------
        states : List[int]
            Current states of all agents.
        deterministic : bool, optional
            Whether to choose the action deterministically, by default False.
        action_masks : Optional[List[List[int]]], optional
            Masks for valid actions, by default None.

        Returns
        -------
        List[int]
            Actions chosen for all agents.
        """
        if action_masks is None:
            return np.array(
                [self.choose_action(state, deterministic=deterministic) for state in states]
            )

        return np.array(
            [
                self.choose_action(state, deterministic=deterministic, action_mask=action_mask)
                for state, action_mask in zip(states, action_masks)
            ]
        )

    def choose_actions_semi_vectorized(
        self,
        states: NDArray[np.int32],
        *,
        deterministic: bool = False,
        action_masks: NDArray[np.int32] | None = None,
    ) -> NDArray[np.int32]:
        """
        Choose actions for all agents based on the current states.

        This method is an iterative version that calls the `choose_action` semi-vectorized method
        for each state and action mask pair.

        Parameters
        ----------
        states : NDArray[np.int32]
            Current states of all agents.
        deterministic : bool, optional
            Whether to choose the action deterministically, by default False.
        action_masks : NDArray[np.int32] | None, optional
            Masks for valid actions, by default None.

        Returns
        -------
        NDArray[np.int32]
            Actions chosen for all agents.
        """
        if action_masks is None:
            return np.array(
                [self.choose_action(state, deterministic=deterministic) for state in states]
            )

        return np.array(
            [
                self.choose_action_semi_vectorized(
                    state, deterministic=deterministic, action_mask=action_mask
                )
                for state, action_mask in zip(states, action_masks)
            ]
        )

    def choose_actions_iter_vectorized(
        self,
        states: NDArray[np.int32],
        *,
        deterministic: bool = False,
        action_masks: NDArray[np.int32] | None = None,
    ) -> NDArray[np.int32]:
        """
        Choose actions for all agents based on the current states.

        This method is an iterative version that calls the `choose_action` fully vectorized method
        for each state and action mask pair.

        Parameters
        ----------
        states : NDArray[np.int32]
            Current states of all agents.
        deterministic : bool, optional
            Whether to choose the action deterministically, by default False.
        action_masks : NDArray[np.int32] | None, optional
            Masks for valid actions, by default None.

        Returns
        -------
        NDArray[np.int32]
            Actions chosen for all agents.
        """
        if action_masks is None:
            return np.array(
                [self.choose_action(state, deterministic=deterministic) for state in states]
            )

        return np.array(
            [
                self.choose_action_fully_vectorized(
                    state, deterministic=deterministic, action_mask=action_mask
                )
                for state, action_mask in zip(states, action_masks)
            ]
        )

    def choose_actions_fully_vectorized(
        self,
        states: NDArray[np.int32],
        *,
        deterministic: bool = False,
        action_masks: NDArray[np.int32] | None = None,
    ) -> NDArray[np.int32]:
        """
        Choose actions for all agents based on the current states.

        This method is a fully vectorized version that handles action selection
        with or without an action mask. It uses NumPy operations for efficiency.

        Parameters
        ----------
        states : List[int]
            Current states of all agents.
        deterministic : bool, optional
            Whether to choose the action deterministically, by default False.
        action_masks : Optional[List[List[int]]], optional
            Masks for valid actions, by default None.

        Returns
        -------
        List[int]
            Actions chosen for all agents.
        """
        # Exploration: Randomly choose a valid action in a single step
        if deterministic:
            # If no mask is provided, allow all actions
            if action_masks is None:
                max_q_values = np.max(self.q_table[states], axis=1, keepdims=True)
                best_actions_per_state = np.array(
                    [
                        random.choice(np.where(q_value == max_q_value)[0])
                        for q_value, max_q_value in zip(self.q_table[states], max_q_values)
                    ]
                )
            else:
                # Ensure mask has correct shape
                assert action_masks.shape == (
                    states.size,
                    self.action_size,
                ), "Action masks must match the number of states and actions."

                # Exploitation: Get best actions based on Q-values
                masked_q_values = np.where(action_masks, self.q_table[states], -np.inf)
                max_q_values = np.max(
                    masked_q_values, axis=1, keepdims=True
                )  # Get max Q-value per state
                best_actions_per_state = np.array(
                    [
                        random.choice(np.where(masked_q_value == max_q_value)[0])
                        for masked_q_value, max_q_value in zip(masked_q_values, max_q_values)
                    ]
                )
            return best_actions_per_state

        explore_flags = self._rng.random(states.size) < self.exploration_rate

        # If no mask is provided, allow all actions
        if action_masks is None:
            exploratory_actions = self._rng.integers(self.action_size, size=states.size)
            max_q_values: NDArray[np.float64] = np.max(self.q_table[states], axis=1, keepdims=True)
            best_actions_per_state = np.array(
                [
                    (random.choice(np.where(q_value == max_q_value)[0]) if not explore_flag else -1)
                    for q_value, max_q_value, explore_flag in zip(
                        self.q_table[states], max_q_values, explore_flags
                    )
                ]
            )
        else:
            # Ensure mask has correct shape
            assert action_masks.shape == (
                states.size,
                self.action_size,
            ), "Action masks must match the number of states and actions."

            exploratory_actions = np.array(
                [
                    (
                        random.choice(valid_actions)
                        if (valid_actions := np.where(mask)[0]).size > 0
                        else -1
                    )
                    for mask in action_masks
                ]
            )

            # Exploitation: Get best actions based on Q-values
            masked_q_values = np.where(action_masks, self.q_table[states], -np.inf)
            max_q_values = np.max(
                masked_q_values, axis=1, keepdims=True
            )  # Get max Q-value per state
            best_actions_per_state = np.array(
                [
                    (
                        random.choice(np.where(masked_q_value == max_q_value)[0])
                        if not explore_flag
                        else -1
                    )
                    for masked_q_value, max_q_value, explore_flag in zip(
                        masked_q_values, max_q_values, explore_flags
                    )
                ]
            )

        # Select final actions: use exploration actions if exploring, otherwise exploitation
        return np.where(explore_flags, exploratory_actions, best_actions_per_state)

    def learn(
        self,
        states: NDArray[np.int32],
        actions: NDArray[np.int32],
        rewards: NDArray[np.float32],
        next_states: NDArray[np.int32],
    ) -> None:
        """
        Update Q-table based on the agents' experiences.

        Parameters
        ----------
        states : List[int]
            Current states of all agents.
        actions : List[int]
            Actions taken by all agents.
        rewards : List[float]
            Rewards received by all agents.
        next_states : List[int]
            Next states of all agents.
        """
        max_next_q_values = np.max(self.get_states_q_values(next_states), axis=1)
        targets = rewards + self.discount_factor * max_next_q_values
        predictions = self.get_q_values(states, actions)
        self.add_q_values(states, actions, self.learning_rate * (targets - predictions))
        self.exploration_rate = max(
            self.min_exploration_rate, self.exploration_rate * self.exploration_decay
        )
