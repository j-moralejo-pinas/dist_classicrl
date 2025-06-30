"""Experience Replay buffer for storing and sampling experiences in reinforcement learning."""

from __future__ import annotations

import numpy as np


class ExperienceReplay:
    """
    Experience Replay buffer for storing and sampling experiences.

    Parameters
    ----------
    capacity : int
        Maximum number of experiences to store in the buffer.
    seed : int
        Random seed for reproducibility.

    Attributes
    ----------
    capacity : int
        Maximum number of experiences to store in the buffer.
    state_buffer : NDArray[np.int32]
        Buffer for storing states.
    action_buffer : NDArray[np.int32]
        Buffer for storing actions.
    reward_buffer : NDArray[np.float32]
        Buffer for storing rewards.
    next_state_buffer : NDArray[np.int32]
        Buffer for storing next states.
    done_buffer : NDArray[np.bool_]
        Buffer for storing done flags.
    position : int
        Current position in the buffer for the next experience.
    full : bool
        Flag indicating whether the buffer is full.
    """

    capacity: int
    state_buffer: np.typing.NDArray[np.int32]
    action_buffer: np.typing.NDArray[np.int32]
    reward_buffer: np.typing.NDArray[np.float32]
    next_state_buffer: np.typing.NDArray[np.int32]
    done_buffer: np.typing.NDArray[np.bool_]
    position: int
    full: bool
    rng: np.random.Generator

    def __init__(self, capacity: int, seed: int) -> None:
        self.capacity = capacity
        self.state_buffer = np.empty((capacity,), dtype=int)
        self.action_buffer = np.empty((capacity,), dtype=int)
        self.reward_buffer = np.empty((capacity,), dtype=float)
        self.next_state_buffer = np.empty((capacity,), dtype=int)
        self.done_buffer = np.empty((capacity,), dtype=bool)
        self.position = 0
        self.full = False
        self.rng = np.random.default_rng(seed)

    def push(self, experience: tuple[int, int, float, int, bool]) -> None:
        """
        Add a new experience to the buffer.

        Parameters
        ----------
        experience : Tuple[int, int, float, int, bool]
            A tuple containing (state, action, reward, next_state, done).
        """
        state, action, reward, next_state, done = experience
        self.state_buffer[self.position] = state
        self.action_buffer[self.position] = action
        self.reward_buffer[self.position] = reward
        self.next_state_buffer[self.position] = next_state
        self.done_buffer[self.position] = done
        self.position = (self.position + 1) % self.capacity
        self.full = self.full or self.position == 0

    def sample(self, batch_size: int) -> tuple[int, int, float, int, bool]:
        """
        Sample a batch of experiences from the buffer.

        Parameters
        ----------
        batch_size : int
            Number of experiences to sample.

        Returns
        -------
        Tuple[int, int, float, int, bool]
            A tuple containing arrays of states, actions, rewards,
            next states, and done flags.
        """
        indices = self.rng.choice(
            self.capacity if self.full else self.position, batch_size, replace=False
        )
        return (
            int(self.state_buffer[indices]),
            int(self.action_buffer[indices]),
            float(self.reward_buffer[indices]),
            int(self.next_state_buffer[indices]),
            bool(self.done_buffer[indices]),
        )

    def __len__(self) -> int:
        """
        Get the current number of experiences stored in the buffer.

        Returns
        -------
        int
            The current number of experiences stored in the buffer.
        """
        return self.capacity if self.full else self.position
