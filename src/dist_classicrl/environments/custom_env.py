import abc
from typing import Any, Dict, List, Optional, Tuple, Union
import gymnasium as gym
import numpy as np
from numpy.typing import NDArray


class DistClassicRLEnv(abc.ABC, gym.Env):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @abc.abstractmethod
    def step(self, actions: NDArray[np.int32]) -> Tuple[
        Union[Dict[str, NDArray[np.int32]], NDArray[np.int32]],
        NDArray[np.float32],
        NDArray[np.bool],
        NDArray[np.bool],
        List[Dict],
    ]:
        pass

    @abc.abstractmethod
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[
        Union[Dict[str, NDArray[np.int32]], NDArray[np.int32]],
        List[Dict],
    ]:
        pass

    @abc.abstractmethod
    def close(self):
        pass

    @abc.abstractmethod
    def render(self):
        pass

    @abc.abstractmethod
    def seed(self, seed):
        pass

    @abc.abstractmethod
    def get_env_info(self):
        pass

    @abc.abstractmethod
    def get_agent_info(self):
        pass
