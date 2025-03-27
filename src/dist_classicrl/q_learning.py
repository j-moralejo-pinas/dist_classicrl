"""This module contains the implementation of multi-agent Q-learning for the Repsol project."""

from re import L
import stat
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from pettingzoo import ParallelEnv
from gymnasium.vector import SyncVectorEnv
from dist_classicrl.custom_env import DistClassicRLEnv
from dist_classicrl.q_learning_optimal import OptimalQLearningBase


class SingleEnvQLearning(OptimalQLearningBase):
    """
    Single environment Q-learning agent.

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
    q_table : mp.Array
        Shared memory array for the Q-table.
    """

    state_size: int
    action_size: int
    learning_rate: float
    discount_factor: float
    exploration_rate: float
    exploration_decay: float
    min_exploration_rate: float
    q_table: NDArray[np.float32]

    def train(
        self,
        env: Union[DistClassicRLEnv, SyncVectorEnv],
        steps: int,
        val_env: Union[DistClassicRLEnv, SyncVectorEnv],
        val_every_n_steps: int,
        val_steps: Optional[int],
        val_episodes: Optional[int],
    ) -> None:
        """
        Train the agent in the environment for a given number of steps.

        Parameters
        ----------
        env : Env
            The environment to train.
        steps : int
            Number of steps to train.
        eval_env : Env
            The evaluation environment.
        eval_steps : int
            Number of steps to evaluate.
        eval_every_n_steps : int
            Evaluate the agent every n steps.
        """
        assert (val_steps is None) ^ (
            val_episodes is None
        ), "Either val_steps or val_episodes should be provided."

        states, infos = env.reset()
        reward_history = []
        val_reward_history = []
        val_agent_reward_history = []
        agent_reward_history = np.zeros(len(states), dtype=np.float32)
        for step in range(steps):
            if isinstance(states, dict):
                actions = self.choose_actions(
                    states=states["observation"], action_masks=states["action_mask"]
                )
            else:
                actions = self.choose_actions(states)

            next_states, rewards, terminateds, truncateds, infos = env.step(actions)

            agent_reward_history += rewards

            if isinstance(next_states, dict):
                assert isinstance(states, dict)
                self.learn(
                    states["observation"],
                    actions,
                    rewards,
                    next_states["observation"],
                    terminateds,
                    next_states["action_mask"],
                )
            else:
                assert not isinstance(states, dict)
                self.learn(states, actions, rewards, next_states, terminateds)
            states = next_states

            for i, (terminated, truncated) in enumerate(zip(terminateds, truncateds)):
                if terminated or truncated:
                    reward_history.append(agent_reward_history[i])
                    agent_reward_history[i] = 0

            if (step + 1) % val_every_n_steps == 0:
                val_total_rewards, val_agent_rewards = 0.0, {}
                if val_steps is not None:
                    val_total_rewards, val_agent_rewards = self.evaluate_steps(val_env, val_steps)
                elif val_episodes is not None:
                    val_total_rewards, val_agent_rewards = self.evaluate_episodes(
                        val_env, val_episodes
                    )

                val_reward_history.append(val_total_rewards)
                val_agent_reward_history.append(val_agent_rewards)
                print(f"Step {step + 1}, Eval total rewards: {val_total_rewards}")

    def evaluate_steps(
        self,
        env: Union[DistClassicRLEnv, SyncVectorEnv],
        steps: int,
    ) -> Tuple[float, List[float]]:
        """
        Evaluate the agent in the environment for a given number of steps.

        Parameters
        ----------
        env : Env
            The environment to evaluate.
        steps : int
            Number of steps to evaluate.

        Returns
        -------
        Tuple[float, Dict[Any, float]]
            Total rewards obtained by the agent and rewards for each agent.
        """

        states, infos = env.reset(seed=42)
        agent_rewards = np.zeros(len(states), dtype=np.float32)
        reward_history = []
        for _ in range(steps):
            if isinstance(states, dict):
                actions = self.choose_actions(
                    states=states["observation"],
                    action_masks=states["action_mask"],
                    deterministic=True,
                )
            else:
                actions = self.choose_actions(states, deterministic=True)
            next_states, rewards, terminateds, truncateds, infos = env.step(actions)
            agent_rewards += rewards
            states = next_states
            for i, (terminated, truncated) in enumerate(zip(terminateds, truncateds)):
                if terminated or truncated:
                    reward_history.append(agent_rewards[i])
                    agent_rewards[i] = 0
        return sum(reward_history), reward_history

    def evaluate_episodes(
        self,
        env: Union[DistClassicRLEnv, SyncVectorEnv],
        episodes: int,
    ) -> Tuple[float, List[float]]:
        """
        Evaluate the agent in the environment for a given number of episodes.

        Parameters
        ----------
        env : Env
            The environment to evaluate.
        episodes : int
            Number of episodes to evaluate.

        Returns
        -------
        Tuple[float, Dict[Any, float]]
            Total rewards obtained by the agent and rewards for each agent.
        """

        states, infos = env.reset(seed=42)
        agent_rewards = np.zeros(len(states), dtype=np.float32)
        reward_history = []
        episode = 0
        while episode < episodes:
            if isinstance(states, dict):
                actions = self.choose_actions(
                    states=states["observation"],
                    action_masks=states["action_mask"],
                    deterministic=True,
                )
            else:
                actions = self.choose_actions(states, deterministic=True)
            next_states, rewards, terminateds, truncateds, infos = env.step(actions)
            agent_rewards += rewards
            states = next_states
            for i, (terminated, truncated) in enumerate(zip(terminateds, truncateds)):
                if terminated or truncated:
                    episode += 1
                    reward_history.append(agent_rewards[i])
                    agent_rewards[i] = 0

        return sum(reward_history), reward_history
