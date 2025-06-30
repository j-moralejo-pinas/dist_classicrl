"""Custom environment interface for distributed classic RL environments."""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any

import gymnasium as gym

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


class DistClassicRLEnv(abc.ABC, gym.Env):
    """
    Abstract base class for distributed classic reinforcement learning environments.

    This class extends gymnasium.Env to provide a standardized interface for
    multi-agent environments that can be used in distributed training scenarios.

    Attributes
    ----------
    num_agents : int
        Number of agents in the environment.
        This is an abstract attribute that must be defined in subclasses
    """

    num_agents: int

    @abc.abstractmethod
    def step(
        self, actions: NDArray[np.int32]
    ) -> tuple[
        dict[str, NDArray[np.int32]] | NDArray[np.int32],
        NDArray[np.float32],
        NDArray[np.bool],
        NDArray[np.bool],
        list[dict],
    ]:
        """
        Execute actions for all agents in the environment.

        Parameters
        ----------
        actions : NDArray[np.int32]
            Array of actions for each agent.

        Returns
        -------
        dict[str, NDArray[np.int32]] | NDArray[np.int32]
            Observations for each agent, either as a dictionary or a single array.
        NDArray[np.float32]
            Rewards for each agent.
        NDArray[np.bool]
            Termination flags for each agent.
        NDArray[np.bool]
            Truncation flags for each agent.
        list[dict]
            List of info dictionaries for each agent, containing additional information.
        """

    @abc.abstractmethod
    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[
        dict[str, NDArray[np.int32]] | NDArray[np.int32],
        list[dict],
    ]:
        """
        Reset the environment to initial state.

        Parameters
        ----------
        seed : int | None, optional
            Random seed for environment reset.
        options : dict[str, Any] | None, optional
            Additional options for environment reset.

        Returns
        -------
        tuple[dict[str, NDArray[np.int32]] | NDArray[np.int32], list[dict]]
            Tuple containing initial observations and info dictionaries.
        """

    @abc.abstractmethod
    def close(self) -> None:
        """Close the environment and clean up resources."""

    @abc.abstractmethod
    def render(self) -> None:
        """Render the environment for visualization."""

    @abc.abstractmethod
    def seed(self, seed: int) -> None:
        """
        Set the random seed for the environment.

        Parameters
        ----------
        seed : int
            Random seed value.
        """

    @abc.abstractmethod
    def get_env_info(self) -> dict[str, Any]:
        """
        Get environment information.

        Returns
        -------
        dict[str, Any]
            Dictionary containing environment metadata.
        """

    @abc.abstractmethod
    def get_agent_info(self) -> dict[str, Any]:
        """
        Get agent information.

        Returns
        -------
        dict[str, Any]
            Dictionary containing agent metadata.
        """
