"""Dummy vectorized environment wrapper for testing purposes."""


import numpy as np
from gymnasium import Env, Wrapper
from gymnasium.vector import VectorEnv
from numpy.typing import NDArray


class DummyVecWrapper(Wrapper, VectorEnv):
    """
    A dummy vectorized environment wrapper that adds no functionality.

    This wrapper is primarily used for testing purposes to ensure compatibility
    with vectorized environments.

    Attributes
    ----------
    envs : list[gymnasium.Env]
        The wrapped environments.
    """

    envs: list[Env]

    def __init__(self, envs: list[Env]) -> None:
        super().__init__(envs[0])
        self.envs = envs
        self.num_envs = len(envs)
        self.observation_space = envs[0].observation_space
        self.action_space = envs[0].action_space

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[NDArray, list[dict]]:
        """
        Reset all environments.

        Parameters
        ----------
        seed : int | None
            Optional seed for the environments' random number generators.
        options : dict | None
            Optional dictionary of additional options. Not used in this wrapper.

        Returns
        -------
        tuple[NDArray, list[dict]]
            A tuple containing the initial observations and a list of info dictionaries
            from each environment.
        """
        obs = []
        infos = []
        for env in self.envs:
            o, info = env.reset(seed=seed, options=options)
            obs.append(o)
            infos.append(info)
        return np.array(obs), infos

    def step(self, actions: NDArray) -> tuple[NDArray, NDArray, NDArray, NDArray, list[dict]]:
        """
        Take a step in all environments based on the given actions.

        Parameters
        ----------
        actions : NDArray
            An array of actions, one for each environment.

        Returns
        -------
        tuple[NDArray, NDArray, NDArray, NDArray, list[dict]]
            A tuple containing the next observations, rewards, terminated flags,
            truncated flags, and a list of info dictionaries from each environment.
        """
        obs = []
        rewards = []
        terminated = []
        truncated = []
        infos = []
        for env, action in zip(self.envs, actions, strict=True):
            o, r, t, tr, info = env.step(action)
            obs.append(o)
            rewards.append(r)
            terminated.append(t)
            truncated.append(tr)
            infos.append(info)
        return (
            np.array(obs),
            np.array(rewards),
            np.array(terminated),
            np.array(truncated),
            infos,
        )

    def close(self) -> None:
        """Close all environments."""
        for env in self.envs:
            env.close()

    def render(self) -> None:
        """Render all environments."""
        for env in self.envs:
            env.render()
