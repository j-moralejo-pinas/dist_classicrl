from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from dist_classicrl.environments.custom_env import DistClassicRLEnv


class MockEnvironment(DistClassicRLEnv):
    """Mock environment for testing purposes."""

    def __init__(self, num_envs=1, return_dict=False):
        super().__init__()
        self.num_envs = num_envs
        self.return_dict = return_dict
        self.observation_space = None
        self.action_space = None
        self._step_count = 0
        self._max_steps = 10

    def step(self, actions: np.typing.NDArray[np.int32]) -> Tuple[
        Union[np.typing.NDArray[np.int32], Dict[str, np.typing.NDArray[np.int32]]],
        np.typing.NDArray[np.float32],
        np.typing.NDArray[np.bool_],
        np.typing.NDArray[np.bool_],
        List[Dict[str, Any]],
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

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[
        Union[np.typing.NDArray[np.int32], Dict[str, np.typing.NDArray[np.int32]]],
        List[Dict[str, Any]],
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
        pass

    def render(self) -> None:
        """Mock render function."""
        pass

    def seed(self, seed: int = 0) -> None:
        """Mock seed function."""
        pass

    def get_env_info(self) -> Dict[str, Any]:
        """Mock get_env_info function."""
        return {}

    def get_agent_info(self) -> Dict[str, Any]:
        """Mock get_agent_info function."""
        return {}
