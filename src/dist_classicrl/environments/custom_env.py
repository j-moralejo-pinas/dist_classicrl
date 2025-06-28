"""Custom environment interface for distributed classic RL environments."""

import abc
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray


class DistClassicRLEnv(abc.ABC, gym.Env):
    """
    Abstract base class for distributed classic reinforcement learning environments.

    This class extends gymnasium.Env to provide a standardized interface for
    multi-agent environments that can be used in distributed training scenarios.
    """

    def __init__(self, **kwargs):
        """
        Initialize the environment.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments passed to the parent class.
        """
        super().__init__(**kwargs)

    @abc.abstractmethod
    def step(self, actions: NDArray[np.int32]) -> Tuple[
        Union[Dict[str, NDArray[np.int32]], NDArray[np.int32]],
        NDArray[np.float32],
        NDArray[np.bool],
        NDArray[np.bool],
        List[Dict],
    ]:
        """
        Execute actions for all agents in the environment.

        Parameters
        ----------
        actions : NDArray[np.int32]
            Array of actions for each agent.

        Returns
        -------
        Tuple
            Tuple containing:
            - observations: Union[Dict[str, NDArray[np.int32]], NDArray[np.int32]]
              Next observations for all agents
            - rewards: NDArray[np.float32]
              Rewards for all agents
            - terminated: NDArray[np.bool]
              Terminated flags for all agents
            - truncated: NDArray[np.bool]
              Truncated flags for all agents
            - infos: List[Dict]
              Info dictionaries for all agents
        """
        pass

    @abc.abstractmethod
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[
        Union[Dict[str, NDArray[np.int32]], NDArray[np.int32]],
        List[Dict],
    ]:
        """
        Reset the environment to initial state.

        Parameters
        ----------
        seed : Optional[int], optional
            Random seed for environment reset.
        options : Optional[Dict[str, Any]], optional
            Additional options for environment reset.

        Returns
        -------
        Tuple
            Tuple containing:
            - observations: Union[Dict[str, NDArray[np.int32]], NDArray[np.int32]]
              Initial observations for all agents
            - infos: List[Dict]
              Info dictionaries for all agents
        """
        pass

    @abc.abstractmethod
    def close(self):
        """Close the environment and clean up resources."""

    @abc.abstractmethod
    def render(self):
        """Render the environment for visualization."""

    @abc.abstractmethod
    def seed(self, seed):
        """
        Set the random seed for the environment.

        Parameters
        ----------
        seed : int
            Random seed value.
        """

    @abc.abstractmethod
    def get_env_info(self):
        """
        Get environment information.

        Returns
        -------
        Dict
            Dictionary containing environment metadata.
        """

    @abc.abstractmethod
    def get_agent_info(self):
        """
        Get agent information.

        Returns
        -------
        Dict
            Dictionary containing agent metadata.
        """
