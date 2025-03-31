"""This module contains the implementation of multi-agent Q-learning for the Repsol project."""

from typing import Any, Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from pettingzoo import ParallelEnv
from dist_classicrl.algorithms.base_algorithms.q_learning_optimal import OptimalQLearningBase


class SingleEnvQLearning(OptimalQLearningBase):
    """
    Single environment Q-learning agent.

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
    q_table : mp.Array
        Shared memory array for the Q-table.
    """

    num_agents: int
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
        env: ParallelEnv,
        steps: int,
        val_env: ParallelEnv,
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
        reward_history = []
        agent_reward_history = {}
        val_reward_history = []
        val_agent_reward_history = {}
        states, infos = env.reset()
        for step in range(steps):

            actions = self.choose_actions(states)
            next_states, rewards, terminated, truncated, infos = env.step(actions)

            for agent, reward in rewards.items():
                if agent not in agent_reward_history:
                    agent_reward_history[agent] = [0]
                agent_reward_history[agent][-1] += reward

            self.learn(states, actions, rewards, next_states, terminated)
            states = next_states

            if not states:
                states, infos = env.reset()
                reward_history.append(sum(agent_reward_history.values()))
                for agent, reward in agent_reward_history.items():
                    if agent not in val_agent_reward_history:
                        val_agent_reward_history[agent] = []
                    agent_reward_history[agent].append(0)
            else:
                for agent, reward in rewards.items():
                    agent_reward_history[agent][-1] += reward

            if (step + 1) % val_every_n_steps == 0:
                val_total_rewards, val_agent_rewards = 0.0, {}
                if val_steps is not None:
                    val_total_rewards, val_agent_rewards = self.evaluate_steps(val_env, val_steps)
                elif val_episodes is not None:
                    val_total_rewards, val_agent_rewards = self.evaluate_episodes(
                        val_env, val_episodes
                    )

                val_reward_history.append(val_total_rewards)
                for agent, reward in val_agent_rewards.items():
                    if agent not in val_agent_reward_history:
                        val_agent_reward_history[agent] = []
                    val_agent_reward_history[agent].append(reward)
                print(f"Step {step + 1}, Eval total rewards: {val_total_rewards}")

    def evaluate_steps(
        self,
        env: SyncVe,
        steps: int,
    ) -> Tuple[float, Dict[Any, float]]:
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
        agent_rewards = {}
        states, infos = env.reset()
        for _ in range(steps):
            actions = self.choose_actions(states, deterministic=True)
            next_states, rewards, terminated, truncated, infos = env.step(actions)
            for agent, reward in rewards.items():
                if agent not in agent_rewards:
                    agent_rewards[agent] = 0
                agent_rewards[agent] += reward
            states = next_states
            if not states:
                states, infos = env.reset()
        return sum(agent_rewards.values()), agent_rewards

    def evaluate_episodes(
        self,
        env: ParallelEnv,
        episodes: int,
    ) -> Tuple[float, Dict[Any, float]]:
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
        agent_rewards = {}
        states, infos = env.reset()
        episode = 0
        while episode < episodes:
            actions = self.choose_actions(states, deterministic=True)
            next_states, rewards, terminated, truncated, infos = env.step(actions)
            for agent, reward in rewards.items():
                if agent not in agent_rewards:
                    agent_rewards[agent] = 0
                agent_rewards[agent] += reward
            states = next_states
            if not states:
                episode += 1
                states, infos = env.reset()

        for agent in agent_rewards:
            agent_rewards[agent] /= episodes

        return sum(agent_rewards.values()), agent_rewards
