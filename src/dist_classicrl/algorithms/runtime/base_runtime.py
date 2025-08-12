"""Base runtime class for all reinforcement learning algorithms."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gymnasium.vector import VectorEnv

    from dist_classicrl.algorithms.base_algorithms.q_learning_optimal import OptimalQLearningBase
    from dist_classicrl.environments.custom_env import DistClassicRLEnv
    from dist_classicrl.schedules.base_schedules import BaseSchedule

logger = logging.getLogger(__name__)


class BaseRuntime(ABC):
    """
    Single environment Q-learning agent.

    Attributes
    ----------
    algorithm : OptimalQLearningBase
        The Q-learning algorithm to use.
    lr_schedule : BaseSchedule
        The learning rate schedule to use.
    exploration_rate_schedule : BaseSchedule
        The exploration rate schedule to use.
    """

    algorithm: OptimalQLearningBase
    lr_schedule: BaseSchedule
    exploration_rate_schedule: BaseSchedule

    def __init__(
        self,
        algorithm: OptimalQLearningBase,
        lr_schedule: BaseSchedule,
        exploration_rate_schedule: BaseSchedule,
    ) -> None:
        self.algorithm = algorithm
        self.lr_schedule = lr_schedule
        self.exploration_rate_schedule = exploration_rate_schedule

    @abstractmethod
    def init_training(self) -> None:
        """Initialize training."""

    @abstractmethod
    def run_steps(
        self,
        steps: int,
        env: DistClassicRLEnv | VectorEnv | Sequence[DistClassicRLEnv] | Sequence[VectorEnv],
        curr_state_dict: dict | list[dict] | None,
    ) -> tuple[
        float,
        list[float],
        DistClassicRLEnv | VectorEnv | Sequence[DistClassicRLEnv] | Sequence[VectorEnv],
        dict[str, Any],
    ]:
        """
        Run the training steps.

        Parameters
        ----------
        steps : int
            The number of steps to run.
        env : DistClassicRLEnv | VectorEnv
            The environment to train.
        states : Any
            The current states of the environments.
        infos : Any
            The current infos of the environments.
        agent_rewards : np.ndarray
            The rewards received by the agent.

        Returns
        -------
        float
            The average training reward.
        list[float]
            The list of episode rewards.
        DistClassicRLEnv | VectorEnv
            The current environments.
        dict[str, Any]
            The current state of the environments, including states, infos and episode rewards.
        """

    @abstractmethod
    def close_training(self) -> None:
        """Close training."""

    def train(
        self,
        env: DistClassicRLEnv | VectorEnv | Sequence[DistClassicRLEnv] | Sequence[VectorEnv],
        steps: int,
        val_env: DistClassicRLEnv | VectorEnv,
        val_every_n_steps: int,
        val_steps: int | None = None,
        val_episodes: int | None = None,
        curr_state_dict: dict | None = None,
    ) -> tuple[
        list[float],
        list[float],
        DistClassicRLEnv | VectorEnv | Sequence[DistClassicRLEnv] | Sequence[VectorEnv],
        dict[str, Any],
    ]:
        """
        Train the agent in the environment for a given number of steps.

        Parameters
        ----------
        env : DistClassicRLEnv | VectorEnv
            The environment to train.
        steps : int
            Number of steps to train.
        val_env : DistClassicRLEnv | VectorEnv
            The validation environment.
        val_every_n_steps : int
            Validate the agent every n steps.
        val_steps : int | None
            The number of steps to use for validation.
        val_episodes : int | None
            The number of episodes to use for validation.
        curr_state_dict : dict | None
            The current state of the environments.

        Return
        ------
        List[float]
            The reward history during training.
        List[float]
            The validation reward history.
        DistClassicRLEnv | VectorEnv | list[DistClassicRLEnv] | list[VectorEnv]
            The current environments.
        dict[str, Any]
            The current state of the environments, including states, infos and episode rewards.
        """
        assert (val_steps is None) ^ (val_episodes is None), (
            "Exactly one of val_steps or val_episodes must be specified."
        )

        self.init_training()

        reward_history = []
        episode_reward_history = []
        val_reward_history = []
        val_agent_reward_history = []

        for step in range(0, steps, val_every_n_steps):
            _, episode_reward_history, env, state_dict = self.run_steps(
                steps=min(val_every_n_steps, steps - step),
                env=env,
                curr_state_dict=curr_state_dict,
            )

            reward_history.extend(episode_reward_history)

            val_total_rewards, val_agent_rewards = 0.0, {}
            if val_steps is not None:
                val_total_rewards, val_agent_rewards = self.evaluate_steps(val_env, val_steps)
            elif val_episodes is not None:
                val_total_rewards, val_agent_rewards = self.evaluate_episodes(val_env, val_episodes)

            val_reward_history.append(val_total_rewards)
            val_agent_reward_history.append(val_agent_rewards)
            logger.debug("Step %d, Eval total rewards: %s", step + 1, val_total_rewards)

        self.close_training()

        return (
            reward_history,
            val_reward_history,
            env,
            state_dict,
        )

    def run_single_step(
        self,
        env: DistClassicRLEnv | VectorEnv,
        states: Any,
        agent_rewards: Any,
        reward_history: list[Any],
    ) -> tuple[Any, Any]:
        """
        Run a single step in the environment.

        Parameters
        ----------
        states : Any
            The current states of the environments.
        agent_rewards : Any
            The rewards received by the agent.
        reward_history : list[Any]
            The history of rewards for the agent.

        Returns
        -------
        tuple[Any, Any]
            The next states and infos from the environment.
        """
        actions = self._choose_actions(states)

        next_states, rewards, terminateds, truncateds, infos = env.step(actions)

        agent_rewards += rewards

        self._learn(states, actions, rewards, next_states, terminateds)

        states = next_states

        for i, (terminated, truncated) in enumerate(zip(terminateds, truncateds, strict=True)):
            if terminated or truncated:
                reward_history.append(agent_rewards[i])
                agent_rewards[i] = 0
        return states, infos

    def _learn(
        self, states: Any, actions: Any, rewards: Any, next_states: Any, terminateds: Any
    ) -> None:
        """
        Learn from the given experience.

        Parameters
        ----------
        states : Any
            The current states of the environments.
        actions : Any
            The actions taken by the agent.
        rewards : Any
            The rewards received by the agent.
        next_states : Any
            The next states of the environments.
        terminateds : Any
            The termination flags for the environments.
        """
        if isinstance(next_states, dict):
            assert isinstance(states, dict)
            self.algorithm.learn(
                states["observation"],
                actions,
                rewards,
                next_states["observation"],
                terminateds,
                self.lr_schedule.get_value(),
                next_states["action_mask"],
            )
            n_updates = len(states["observation"])
        else:
            assert not isinstance(states, dict)
            self.algorithm.learn(
                states, actions, rewards, next_states, terminateds, self.lr_schedule.get_value()
            )
            n_updates = len(states)

        self.lr_schedule.update(n_updates)
        self.exploration_rate_schedule.update(n_updates)

    def _choose_actions(self, states: Any) -> Any:
        """
        Choose actions for the given states using the exploration rate and deterministic flag.

        Parameters
        ----------
        states : Any
            The current states of the environments.
        exploration_rate : float
            The exploration rate for the epsilon-greedy policy.
        deterministic : bool
            Whether to use a deterministic policy.

        Returns
        -------
        Any
            The chosen actions.
        """
        if isinstance(states, dict):
            return self.algorithm.choose_actions(
                states=states["observation"],
                action_masks=states["action_mask"],
                exploration_rate=self.exploration_rate_schedule.get_value(),
            )
        return self.algorithm.choose_actions(
            states, exploration_rate=self.exploration_rate_schedule.get_value()
        )

    def evaluate_steps(
        self,
        env: DistClassicRLEnv | VectorEnv,
        steps: int,
    ) -> tuple[float, list[float]]:
        """
        Evaluate the agent in the environment for a given number of steps.

        Parameters
        ----------
        env : DistClassicRLEnv | VectorEnv
            The environment to evaluate.
        steps : int
            Number of steps to evaluate.

        Returns
        -------
        tuple[float, list[float]]
            Total rewards obtained by the agent and rewards for each agent.
        """
        states, infos = env.reset(seed=42)
        n_agents = len(states["observation"]) if isinstance(states, dict) else len(states)
        agent_rewards = np.zeros(n_agents, dtype=np.float32)
        reward_history = []
        for _ in range(steps):
            if isinstance(states, dict):
                actions = self.algorithm.choose_actions(
                    states=states["observation"],
                    action_masks=states["action_mask"],
                    exploration_rate=0.0,
                    deterministic=True,
                )
            else:
                actions = self.algorithm.choose_actions(
                    states, exploration_rate=0.0, deterministic=True
                )
            next_states, rewards, terminateds, truncateds, infos = env.step(actions)
            agent_rewards += rewards
            states = next_states
            for i, (terminated, truncated) in enumerate(zip(terminateds, truncateds, strict=False)):
                if terminated or truncated:
                    reward_history.append(agent_rewards[i])
                    agent_rewards[i] = 0
        return sum(reward_history), reward_history

    def evaluate_episodes(
        self,
        env: DistClassicRLEnv | VectorEnv,
        episodes: int,
    ) -> tuple[float, list[float]]:
        """
        Evaluate the agent in the environment for a given number of episodes.

        Parameters
        ----------
        env : DistClassicRLEnv | VectorEnv
            The environment to evaluate.
        episodes : int
            Number of episodes to evaluate.

        Returns
        -------
        tuple[float, list[float]]
            Total rewards obtained by the agent and rewards for each agent.
        """
        states, infos = env.reset(seed=42)
        n_agents = len(states["observation"]) if isinstance(states, dict) else len(states)
        agent_rewards = np.zeros(n_agents, dtype=np.float32)
        reward_history = []
        episode = 0
        while episode < episodes:
            if isinstance(states, dict):
                actions = self.algorithm.choose_actions(
                    states=states["observation"],
                    action_masks=states["action_mask"],
                    exploration_rate=0.0,
                    deterministic=True,
                )
            else:
                actions = self.algorithm.choose_actions(
                    states, exploration_rate=0.0, deterministic=True
                )
            next_states, rewards, terminateds, truncateds, infos = env.step(actions)
            agent_rewards += rewards
            states = next_states
            for i, (terminated, truncated) in enumerate(zip(terminateds, truncateds, strict=False)):
                if terminated or truncated:
                    episode += 1
                    reward_history.append(agent_rewards[i])
                    agent_rewards[i] = 0

        return sum(reward_history), reward_history
