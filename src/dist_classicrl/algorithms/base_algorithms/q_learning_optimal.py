"""Multi-agent Q-learning base implementation using lists or numpy arrays when appropriate."""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

NUM_STATES_LEARN_THRESHOLD = 10
NO_ACTION_MASKS_DETERMINISTIC_ACTION_SIZE_ITER = 10
NO_ACTION_MASKS_DETERMINISTIC_ACTION_SIZE_VEC_ITER = 1000
NO_ACTION_MASKS_NO_DETERMINISTIC_NUM_STATES_ITER = 100
ACTION_MASKS_DETERMINISTIC_ACTION_SIZE_ITER = 10
ACTION_MASKS_DETERMINISTIC_ACTION_SIZE_VEC_ITER = 1000
ACTION_MASKS_NO_DETERMINISTIC_ACTION_SIZE_ITER = 10


class OptimalQLearningBase:
    """
    Base Q-learning class that implements the Q-learning algorithm.

    It is implemented in different ways for different scenarios, giving the best performance
    at each case.

    Parameters
    ----------
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
    seed : int | None, optional
        Random seed for reproducibility, by default None.

    Attributes
    ----------
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
    """

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
        state_size: int | np.integer,
        action_size: int | np.integer,
        learning_rate: float = 0.1,
        discount_factor: float = 0.97,
        exploration_rate: float = 1.0,
        exploration_decay: float = 0.999,
        min_exploration_rate: float = 0.01,
        seed: int | None = None,
    ) -> None:
        self.state_size = int(state_size)
        self.action_size = int(action_size)
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

    def add_q_value(self, state: int, action: int, value: float) -> None:
        """
        Add a Q-value for a given state-action pair.

        Parameters
        ----------
        state : int
            State index.
        action : int
            Action index.
        value : float
            Q-value to add.
        """
        self.q_table[state, action] += value

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

    def choose_action(
        self,
        state: int,
        *,
        deterministic: bool = False,
    ) -> int:
        """
        Choose an action based on the current state.

        Parameters
        ----------
        state : int
            Current state of the agent.
        deterministic : bool, optional
            Whether to choose the action deterministically, by default False.

        Returns
        -------
        int
            Action chosen by the agent.
        """
        if not deterministic and random.uniform(0, 1) < self.exploration_rate:
            return random.randint(0, self.action_size - 1)
        q_values = self.get_state_q_values(state)
        max_val = -math.inf
        available_actions = []
        for i, v in enumerate(q_values):
            if v > max_val:
                max_val = v
                available_actions = [i]
            elif v == max_val:
                available_actions.append(i)

        if available_actions:
            return random.choice(available_actions)

        return -1

    def choose_masked_action(
        self, state: int, action_mask: list[int], *, deterministic: bool = False
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
        available_actions = []
        assert len(action_mask) == self.action_size, (
            "Action mask should have the same length as the action size."
        )
        if not deterministic and random.uniform(0, 1) < self.exploration_rate:
            available_actions = [a for a in range(self.action_size) if action_mask[a]]
        else:
            q_values = self.get_state_q_values(state)

            max_val = -math.inf
            for i, v in enumerate(q_values):
                if action_mask[i]:
                    if v > max_val:
                        max_val = v
                        available_actions = [i]
                    elif v == max_val:
                        available_actions.append(i)
        return random.choice(available_actions) if available_actions else -1

    def choose_actions_iter(
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
            return np.fromiter(
                (self.choose_action(state, deterministic=deterministic) for state in states),
                dtype=np.int32,
                count=len(states),
            )
        return np.fromiter(
            (
                self.choose_masked_action(state, action_mask, deterministic=deterministic)
                for state, action_mask in zip(states, action_masks)
            ),
            dtype=np.int32,
            count=len(states),
        )

    def choose_action_vec(
        self,
        state: int,
        *,
        deterministic: bool = False,
    ) -> int:
        """
        Choose an action based on the current state.

        Parameters
        ----------
        state : int
            Current state of the agent.
        deterministic : bool, optional
            Whether to choose the action deterministically, by default False.

        Returns
        -------
        int
            Action chosen by the agent.
        """
        if not deterministic and random.random() < self.exploration_rate:
            return random.randint(0, self.action_size - 1)

        max_val = np.max(self.q_table[state])
        return random.choice(np.where(self.q_table[state] == max_val)[0])

    def choose_masked_action_vec(
        self,
        state: int,
        action_mask: list[int],
        *,
        deterministic: bool = False,
    ) -> int:
        """
        Choose an action based on the current state.

        Parameters
        ----------
        state : int
            Current state of the agent.
        action_mask : NDArray[np.int32]
            Mask for valid actions.
        deterministic : bool, optional
            Whether to choose the action deterministically, by default False.

        Returns
        -------
        int
            Action chosen by the agent.
        """
        np_action_mask = np.fromiter(action_mask, dtype=np.int32, count=len(action_mask))
        assert np_action_mask.size == self.action_size, (
            "Action mask should have the same size as the action space."
        )

        if not deterministic and random.random() < self.exploration_rate:
            available_actions = np.where(np_action_mask)[0]
        else:
            masked_q_values = np.where(np_action_mask, self.q_table[state], -np.inf)
            max_val = np.max(masked_q_values)
            available_actions = np.where(masked_q_values == max_val)[0]
        return random.choice(available_actions)

    def choose_actions_vec_iter(
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
        states : NDArray[np.int32]
            Current states of all agents.
        deterministic : bool, optional
            Whether to choose the action deterministically, by default False.
        action_masks : Optional[NDArray[np.int32]], optional
            Masks for valid actions, by default None.

        Returns
        -------
        NDArray[np.int32]
            Actions chosen for all agents.
        """
        if action_masks is None:
            return np.fromiter(
                (self.choose_action_vec(state, deterministic=deterministic) for state in states),
                dtype=np.int32,
                count=len(states),
            )
        return np.fromiter(
            (
                self.choose_masked_action_vec(state, action_mask, deterministic=deterministic)
                for state, action_mask in zip(states, action_masks)
            ),
            dtype=np.int32,
            count=len(states),
        )

    def choose_actions_vec(
        self,
        states: NDArray[np.int32],
        *,
        deterministic: bool = False,
    ) -> NDArray[np.int32]:
        """
        Choose actions for all agents based on the current states.

        Parameters
        ----------
        states : NDArray[np.int32]
            Current states of all agents.
        deterministic : bool, optional
            Whether to choose the action deterministically, by default False.

        Returns
        -------
        NDArray[np.int32]
            Actions chosen for all agents.
        """
        max_q_values: NDArray[np.float32] = np.max(self.q_table[states], axis=1, keepdims=True)

        if not deterministic:
            explore_flags = self._rng.random(states.size) < self.exploration_rate
            exploratory_actions = self._rng.integers(self.action_size, size=states.size)
            return np.fromiter(
                (
                    (
                        random.choice(np.where(q_value == max_q_value)[0])
                        if not explore_flag
                        else exploratory_action
                    )
                    for q_value, max_q_value, explore_flag, exploratory_action in zip(
                        self.q_table[states], max_q_values, explore_flags, exploratory_actions
                    )
                ),
                dtype=np.int32,
                count=states.size,
            )

        return np.fromiter(
            (
                random.choice(np.where(q_value == max_q_value)[0])
                for q_value, max_q_value in zip(self.q_table[states], max_q_values)
            ),
            dtype=np.int32,
            count=states.size,
        )

    def choose_masked_actions_vec(
        self,
        states: NDArray[np.int32],
        action_masks: NDArray[np.int32],
        *,
        deterministic: bool = False,
    ) -> NDArray[np.int32]:
        """
        Choose actions for all agents based on the current states.

        Parameters
        ----------
        states : NDArray[np.int32]
            Current states of all agents.
        action_masks : NDArray[np.int32]
            Masks for valid actions.
        deterministic : bool, optional
            Whether to choose the action deterministically, by default False.

        Returns
        -------
        NDArray[np.int32]
            Actions chosen for all agents.
        """
        assert action_masks.shape == (
            states.size,
            self.action_size,
        ), "Action masks must match the number of states and actions."

        masked_q_values_vec = np.where(action_masks, self.q_table[states], -np.inf)
        max_q_values: NDArray[np.float32] = np.max(masked_q_values_vec, axis=1, keepdims=True)

        if not deterministic:
            explore_flags = self._rng.random(states.size) < self.exploration_rate
            return np.fromiter(
                (
                    (
                        random.choice(np.where(masked_q_value == max_q_value)[0])
                        if not explore_flag
                        else random.choice(np.where(mask)[0])
                    )
                    for masked_q_value, max_q_value, mask, explore_flag in zip(
                        masked_q_values_vec, max_q_values, action_masks, explore_flags
                    )
                ),
                dtype=np.int32,
                count=states.size,
            )

        return np.fromiter(
            (
                random.choice(np.where(masked_q_value == max_q_value)[0])
                for masked_q_value, max_q_value in zip(masked_q_values_vec, max_q_values)
            ),
            dtype=np.int32,
            count=states.size,
        )

    def choose_actions(  # noqa: PLR0911
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
        states : NDArray[np.int32]
            Current states of all agents.
        deterministic : bool, optional
            Whether to choose the action deterministically, by default False.
        action_masks : Optional[NDArray[np.int32]], optional
            Masks for valid actions, by default None.

        Returns
        -------
        NDArray[np.int32]
            Actions chosen for all agents.
        """
        if deterministic:
            if self.action_size <= NO_ACTION_MASKS_DETERMINISTIC_ACTION_SIZE_ITER:
                return self.choose_actions_iter(states, deterministic=deterministic)
            if self.action_size > NO_ACTION_MASKS_DETERMINISTIC_ACTION_SIZE_VEC_ITER:
                return self.choose_actions_vec_iter(states, deterministic=deterministic)
            if action_masks is not None:
                return self.choose_masked_actions_vec(
                    states, action_masks, deterministic=deterministic
                )
            return self.choose_actions_vec(states, deterministic=deterministic)

        if action_masks is None:
            if len(states) < NO_ACTION_MASKS_NO_DETERMINISTIC_NUM_STATES_ITER:
                return self.choose_actions_iter(states, deterministic=deterministic)
            return self.choose_actions_vec(states, deterministic=deterministic)

        if self.action_size <= ACTION_MASKS_NO_DETERMINISTIC_ACTION_SIZE_ITER:
            return self.choose_actions_iter(
                states, deterministic=deterministic, action_masks=action_masks
            )
        return self.choose_actions_vec_iter(
            states, deterministic=deterministic, action_masks=action_masks
        )

    def update_explore_rate(self) -> None:
        """Update the exploration rate."""
        self.exploration_rate = max(
            self.min_exploration_rate, self.exploration_rate * self.exploration_decay
        )

    def single_learn(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        terminated: bool,  # noqa: FBT001
        next_action_mask: NDArray[np.int32] | None = None,
    ) -> None:
        """
        Update Q-table based on the agent's experience.

        Parameters
        ----------
        state : int
            Current state of the agent.
        action : int
            Action taken by the agent.
        reward : float
            Reward received by the agent.
        next_state : int
            Next state of the agent.
        terminated : bool
            Whether the episode has terminated.
        """
        if next_action_mask is None:
            max_next_q_value = 0 if terminated else np.max(self.get_state_q_values(next_state))
        else:
            max_next_q_value = (
                0
                if terminated
                else np.max(self.get_state_q_values(next_state)[np.where(next_action_mask)])
            )
        target = reward + self.discount_factor * max_next_q_value
        prediction = self.get_q_value(state, action)
        self.add_q_value(state, action, self.learning_rate * (target - prediction))

    def learn_iter(
        self,
        states: NDArray[np.int32],
        actions: NDArray[np.int32],
        rewards: NDArray[np.float32],
        next_states: NDArray[np.int32],
        terminated: NDArray[np.bool],
        next_action_masks: NDArray[np.int32] | None = None,
    ) -> None:
        """
        Update Q-table based on the agents' experiences.

        Parameters
        ----------
        states : NDArray[np.int32]
            Current states of all agents.
        actions : NDArray[np.int32]
            Actions taken by all agents.
        rewards : NDArray[np.float32]
            Rewards received by all agents.
        next_states : NDArray[np.int32]
            Next states of all agents.
        """
        if next_action_masks is None:
            for state, action, reward, next_state, term in zip(
                states, actions, rewards, next_states, terminated
            ):
                self.single_learn(state, action, reward, next_state, term)
        else:
            for state, action, reward, next_state, term, next_action_mask in zip(
                states, actions, rewards, next_states, terminated, next_action_masks
            ):
                self.single_learn(
                    state,
                    action,
                    reward,
                    next_state,
                    term,
                    next_action_mask,
                )

    def learn_vec(
        self,
        states: NDArray[np.int32],
        actions: NDArray[np.int32],
        rewards: NDArray[np.float32],
        next_states: NDArray[np.int32],
        terminated: NDArray[np.bool],
        next_action_masks: NDArray[np.int32] | None = None,
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
        self._learn_vec(states, actions, rewards, next_states, terminated, next_action_masks)

    def _learn_vec(
        self,
        states: NDArray[np.int32],
        actions: NDArray[np.int32],
        rewards: NDArray[np.float32],
        next_states: NDArray[np.int32],
        terminated: NDArray[np.bool],
        next_action_masks: NDArray[np.int32] | None = None,
    ) -> None:
        """
        Update Q-table based on the agents' experiences.

        Parameters
        ----------
        states : NDArray[np.int32]
            Current states of all agents.
        actions : NDArray[np.int32]
            Actions taken by all agents.
        rewards : NDArray[np.float32]
            Rewards received by all agents.
        next_states : NDArray[np.int32]
            Next states of all agents.
        """
        if next_action_masks is None:
            max_next_q_values = np.max(self.get_states_q_values(next_states), axis=1)
        else:
            max_next_q_values = np.max(
                np.where(next_action_masks, self.get_states_q_values(next_states), -np.inf), axis=1
            )
        targets = rewards + self.discount_factor * max_next_q_values * (1 - terminated)
        predictions = self.get_q_values(states, actions)
        self.add_q_values(states, actions, self.learning_rate * (targets - predictions))

    def learn(
        self,
        states: NDArray[np.int32],
        actions: NDArray[np.int32],
        rewards: NDArray[np.float32],
        next_states: NDArray[np.int32],
        terminated: NDArray[np.bool],
        next_action_masks: NDArray[np.int32] | None = None,
    ) -> None:
        """
        Update Q-table based on the agents' experiences.

        Parameters
        ----------
        states : NDArray[np.int32]
            Current states of all agents.
        actions : NDArray[np.int32]
            Actions taken by all agents.
        rewards : NDArray[np.float32]
            Rewards received by all agents.
        next_states : NDArray[np.int32]
            Next states of all agents.
        """
        if len(states) > NUM_STATES_LEARN_THRESHOLD:
            self.learn_vec(states, actions, rewards, next_states, terminated, next_action_masks)
        else:
            self.learn_iter(states, actions, rewards, next_states, terminated, next_action_masks)
        self.update_explore_rate()
