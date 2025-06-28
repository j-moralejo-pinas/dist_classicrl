"""Multi-agent Q-learning  implementation using lists."""

import math
import random
from typing import List, Optional

import numpy as np


class MultiAgentQLearningLists:
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
    """

    num_agents: int
    state_size: int
    action_size: int
    learning_rate: float
    discount_factor: float
    exploration_rate: float
    exploration_decay: float
    min_exploration_rate: float
    q_table: List[float]

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
        """
        self.num_agents = num_agents
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate
        self.q_table = [0.0] * (state_size * action_size)

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
        return self.q_table[state * self.action_size + action]

    def get_q_values(self, states: List[int], actions: List[int]) -> List[float]:
        """
        Get the Q-values for a given list of state-action pairs.

        Parameters
        ----------
        states : List[int]
            List of state indices.
        actions : List[int]
            List of action indices.

        Returns
        -------
        List[float]
            Q-values for the state-action pairs.
        """
        return [self.get_q_value(state, action) for state, action in zip(states, actions)]

    def get_state_q_values(self, state: int) -> List[float]:
        """
        Get the Q-values for a given state.

        Parameters
        ----------
        state : int
            State index.

        Returns
        -------
        List[float]
            Q-values for the state.
        """
        return self.q_table[state * self.action_size : (state + 1) * self.action_size]

    def get_states_q_values(
        self,
        states: List[int],
    ) -> List[List[float]]:
        """
        Get the Q-values for a given list of states.

        Parameters
        ----------
        states : List[int]
            List of state indices.

        Returns
        -------
        List[List[float]]
            Q-values for the states.
        """
        return [self.get_state_q_values(state) for state in states]

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
        self.q_table[state * self.action_size + action] = value

    def add_q_value(self, state: int, action: int, value: float) -> None:
        """
        Add the Q-value for a given state-action pair.

        Parameters
        ----------
        state : int
            State index.
        action : int
            Action index.
        value : float
            Q-value to add.
        """
        self.q_table[state * self.action_size + action] += value

    def add_q_values(self, states: List[int], actions: List[int], values: List[float]) -> None:
        """
        Add Q-values for a given list of state-action pairs.

        Parameters
        ----------
        states : List[int]
            List of state indices.
        actions : List[int]
            List of action indices.
        values : List[float]
            List of Q-values to add.
        """
        for state, action, value in zip(states, actions, values):
            self.add_q_value(state, action, value)

    def save(self, filename: str) -> None:
        """
        Save the Q-table to a file.

        Parameters
        ----------
        filename : str
            File to save the Q-table to.
        """
        np.save(filename, np.array(self.q_table))

    def choose_action(
        self, state: int, deterministic: bool = False, action_mask: Optional[List[int]] = None
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
        if action_mask:
            assert (
                len(action_mask) == self.action_size
            ), "Action mask should have the same length as the action size."
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
        else:
            if not deterministic and random.uniform(0, 1) < self.exploration_rate:
                available_actions = range(self.action_size)
            else:
                q_values = self.get_state_q_values(state)
                max_val = -math.inf
                for i, v in enumerate(q_values):
                    if v > max_val:
                        max_val = v
                        available_actions = [i]
                    elif v == max_val:
                        available_actions.append(i)

        if available_actions:
            return random.choice(available_actions)

        return -1

    def choose_actions(
        self,
        states: List[int],
        deterministic: bool = False,
        action_masks: Optional[List[List[int]]] = None,
    ) -> List[int]:
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
        if action_masks:
            assert len(states) == len(
                action_masks
            ), "States list and action masks list should have the same length."
            return [
                self.choose_action(state, deterministic, action_mask)
                for state, action_mask in zip(states, action_masks)
            ]
        return [self.choose_action(state, deterministic) for state in states]

    def learn(
        self, states: List[int], actions: List[int], rewards: List[float], next_states: List[int]
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

        for state, action, reward, next_state in zip(states, actions, rewards, next_states):
            max_next_q_value = max(self.get_state_q_values(next_state))
            target = reward + self.discount_factor * max_next_q_value
            prediction = self.get_q_value(state, action)
            self.add_q_value(state, action, self.learning_rate * (target - prediction))

        self.exploration_rate = max(
            self.min_exploration_rate, self.exploration_rate * self.exploration_decay
        )
