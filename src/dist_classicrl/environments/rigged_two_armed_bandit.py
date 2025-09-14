"""Rigged two-armed bandit environment."""

import gymnasium as gym


class RiggedTwoArmedBanditEnv(gym.Env):
    """
    A rigged two-armed bandit environment where the second action is optimal.

    The environment has a single state and two actions (0 and 1). Action
    0 yields a reward of 1.0, while action 1 yields a reward of 0.0. The episode
    terminates after a fixed number of steps.

    Attributes
    ----------
    action_space : gym.spaces.Discrete
        Discrete action space with 2 possible actions (0 and 1).
    observation_space : gym.spaces.Discrete
        Discrete observation space with a single state (0).
    episode_len : int
        The number of steps after which the episode terminates.
    _t : int
        Internal step counter.
    """

    def __init__(self, episode_len: int = 10) -> None:
        super().__init__()
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Discrete(1)
        self.episode_len = episode_len
        self._t = 0

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[int, dict]:  # noqa: ARG002
        """
        Reset the environment to the initial state.

        Parameters
        ----------
        seed : int | None
            Optional seed for the environment's random number generator.
        options : dict | None
            Optional dictionary of additional options. Not used in this environment.

        Returns
        -------
        tuple[int, dict]
            A tuple containing the initial observation (0) and an empty info dictionary.
        """
        super().reset(seed=seed)
        self._t = 0
        obs = 0
        info = {}
        return obs, info

    def step(self, action: int) -> tuple[int, float, bool, bool, dict]:
        """
        Take a step in the environment based on the given action.

        Parameters
        ----------
        action : int
            The action to take (0 or 1).

        Returns
        -------
        tuple[int, float, bool, bool, dict]
            A tuple containing the next observation (0), the reward, a boolean indicating
            if the episode has terminated, a boolean indicating if the episode was truncated,
            and an empty info dictionary.
        """
        assert self.action_space.contains(action), f"Invalid action: {action}"
        reward = action
        self._t += 1
        terminated = self._t >= self.episode_len
        truncated = False
        if terminated:
            self._t = 0
        obs = 0
        info = {}
        return obs, reward, terminated, truncated, info

    def render(self) -> None:
        """Render the environment. Not implemented."""


    def close(self) -> None:
        """Close the environment. Not implemented."""

