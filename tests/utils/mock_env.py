"""Mock environment for testing purposes."""

from __future__ import annotations

from typing import Any

import numpy as np

from dist_classicrl.environments.custom_env import DistClassicRLEnv


class MockEnvironment(DistClassicRLEnv):
    """Mock environment for testing purposes."""

    def __init__(self, num_envs: int = 1, *, return_dict: bool = False) -> None:
        super().__init__()
        self.num_envs = num_envs
        self.return_dict = return_dict
        self._step_count = 0
        self._max_steps = 10

    def step(
        self,
    ) -> tuple[
        np.typing.NDArray[np.int32] | dict[str, np.typing.NDArray[np.int32]],
        np.typing.NDArray[np.float32],
        np.typing.NDArray[np.bool_],
        np.typing.NDArray[np.bool_],
        list[dict[str, Any]],
    ]:
        """Mock step function."""
        self._step_count += 1

        if self.return_dict:
            next_states = {
                "observation": np.array([self._step_count] * self.num_envs, dtype=np.int32),
                "action_mask": np.ones((self.num_envs, 3), dtype=np.int32),
            }
        else:
            next_states = np.array([self._step_count] * self.num_envs, dtype=np.int32)

        rewards = np.array([1.0] * self.num_envs, dtype=np.float32)
        terminated = np.array([self._step_count >= self._max_steps] * self.num_envs, dtype=bool)
        truncated = np.array([False] * self.num_envs, dtype=bool)
        infos = [{}] * self.num_envs

        return next_states, rewards, terminated, truncated, infos

    def reset(
        self,
    ) -> tuple[
        np.typing.NDArray[np.int32] | dict[str, np.typing.NDArray[np.int32]], list[dict[str, Any]]
    ]:
        """Mock reset function."""
        self._step_count = 0

        if self.return_dict:
            states = {
                "observation": np.array([0] * self.num_envs, dtype=np.int32),
                "action_mask": np.ones((self.num_envs, 3), dtype=np.int32),
            }
        else:
            states = np.array([0] * self.num_envs, dtype=np.int32)

        infos = [{}] * self.num_envs
        return states, infos

    def close(self) -> None:
        """Mock close function."""

    def render(self) -> None:
        """Mock render function."""

    def seed(self, seed: int = 0) -> None:
        """Mock seed function."""

    def get_env_info(self) -> dict[str, Any]:
        """Mock get_env_info function."""
        return {}

    def get_agent_info(self) -> dict[str, Any]:
        """Mock get_agent_info function."""
        return {}
