"""Multi-agent Q-learning trainer implementation in a single process."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
from gymnasium.vector import SyncVectorEnv

from dist_classicrl.algorithms.runtime.base_runtime import BaseRuntime
from dist_classicrl.environments.custom_env import DistClassicRLEnv

if TYPE_CHECKING:
    from gymnasium.vector import SyncVectorEnv

    from dist_classicrl.environments.custom_env import DistClassicRLEnv

logger = logging.getLogger(__name__)


class SingleThreadQLearning(BaseRuntime):
    """Single environment Q-learning agent."""

    def init_training(self) -> None:
        """Initialize the training environment."""

    def run_steps(
        self,
        steps: int,
        env: DistClassicRLEnv | SyncVectorEnv,
        curr_state_dict: dict | None = None,
    ) -> tuple[float, list[float], DistClassicRLEnv | SyncVectorEnv, dict[str, Any]]:
        """
        Run the training steps.

        Parameters
        ----------
        steps : int
            The number of steps to run.
        env : DistClassicRLEnv | SyncVectorEnv
            The environment to train.
        curr_state_dict : dict | None
            The current state of the environments.

        Returns
        -------
        tuple[float, list[float], DistClassicRLEnv | SyncVectorEnv, dict[str, Any]]
            The average training reward, the list of episode rewards, the current environments,
            and the current state of the environments.
        """
        reward_history = []

        if curr_state_dict is None:
            states, infos = env.reset()
            n_agents = len(states["observation"]) if isinstance(states, dict) else len(states)
            agent_rewards = np.zeros(n_agents, dtype=np.float32)
        else:
            states = curr_state_dict["states"]
            infos = curr_state_dict["infos"]
            agent_rewards = curr_state_dict["rewards"]

        for _ in range(steps):
            states, infos = self.run_single_step(env, states, agent_rewards, reward_history)

        return (
            sum(reward_history) / len(reward_history),
            reward_history,
            env,
            {
                "states": states,
                "infos": infos,
                "rewards": agent_rewards,
                "episode_rewards": reward_history,
            },
        )

    def close_training(self) -> None:
        """Close the training environment."""
