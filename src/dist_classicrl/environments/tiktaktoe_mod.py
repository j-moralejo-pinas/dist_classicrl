"""TicTacToe environment for reinforcement learning."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import gymnasium as gym
import numpy as np
from gymnasium.utils.seeding import np_random

if TYPE_CHECKING:
    from numpy.typing import NDArray


class TicTacToeEnv(gym.Env):
    """
    TicTacToe environment for reinforcement learning.

    Attributes
    ----------
    metadata : ClassVar[dict[str, Any]]
        Metadata for the environment, including supported rendering modes.
        Currently supports only "human" mode for rendering the board state.
    action_space : gym.spaces.Discrete
        Discrete action space with 9 possible moves (0-8).
    observation_space : gym.spaces.Dict
        Dictionary observation space containing:
        - 'observation': MultiDiscrete space representing the board state
            (3x3 grid flattened to a (9,) vector).
        - 'action_mask': MultiDiscrete space indicating valid moves
            (2 for valid, 0 for invalid) as a (9,) vector.
    agent_starts : bool
        Indicates if the agent starts first (True) or the machine starts first (False).
    agent_mark : int
        The mark for the agent (1 for 'X', 2 for 'O').
    machine_mark : int
        The mark for the machine (1 for 'X', 2 for 'O').
    board : NDArray[np.int8]
        The current state of the TicTacToe board as a 3x3 numpy array of integers.
        Values are:
        - 0: empty cell
        - 1: agent's mark ('X')
        - 2: machine's mark ('O')
    """

    metadata: ClassVar[dict[str, Any]] = {"render.modes": ["human"]}
    _np_random: np.random.Generator
    _np_random_seed: int | None = None
    action_space: gym.spaces.Discrete
    observation_space: gym.spaces.Dict
    agent_starts: bool
    agent_mark: int
    machine_mark: int
    board: NDArray[np.int8]

    def __init__(self) -> None:
        super().__init__()
        self.action_space = gym.spaces.Discrete(9)

        self.observation_space = gym.spaces.Dict(
            {
                "observation": gym.spaces.MultiDiscrete([3] * 9),
                "action_mask": gym.spaces.MultiDiscrete([2] * 9),
            }
        )

    def reset(
        self,
        *,
        seed: int | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """
        Reset the environment to an initial state.

        Parameters
        ----------
        seed : int | None, optional
            Random seed for reproducibility (default: None).

        Returns
        -------
        tuple[dict[str, np.ndarray], dict[str, Any]]
            Tuple containing:
            - observation: Initial board state and action mask
            - info: Additional information (empty dict)
        """
        if seed is None:
            if self._np_random_seed is None:
                self._np_random = np.random.default_rng()
            else:
                self._np_random, self._np_random_seed = np_random(self._np_random_seed)
        else:
            self._np_random, self._np_random_seed = np_random(seed)

        self.board = np.zeros((3, 3), dtype=np.int8)

        self.agent_starts = self._np_random.choice([True, False])

        if self.agent_starts:
            self.agent_mark = 1
            self.machine_mark = 2
        else:
            self.agent_mark = 2
            self.machine_mark = 1
            move = self._np_random.choice(range(9))
            self._apply_move(move, self.machine_mark)
        return self._get_obs(), {}

    def _apply_move(
        self, move: int, mark: int
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]] | None:
        """
        Apply a move to the board and check for game termination.

        Parameters
        ----------
        move : int
            Move index (0-8) representing the position on the board.
        mark : int
            Player mark (1 or 2) to place on the board.

        Returns
        -------
        tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]] | None
            If the game ends after this move, returns the step result tuple.
            Otherwise, returns None.
        """
        row, col = divmod(move, 3)
        assert self.board[row, col] == 0, "Invalid move."
        self.board[row, col] = mark
        if self._check_winner() == mark:
            reward = 1 if mark == self.agent_mark else -1
            return self._get_obs(), reward, True, False, {}
        if not self._get_valid_moves():
            return self._get_obs(), 0, True, False, {}
        return None

    def step(self, action: int) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """
        Execute one step in the environment with the given action.

        The agent makes a move, then the machine (if game continues) makes a move.

        Parameters
        ----------
        action : int
            Action index (0-8) representing the position to place the agent's mark.

        Returns
        -------
        tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]
            Tuple containing:
            - observation: Current board state and action mask
            - reward: Reward for the agent (1 for win, -1 for loss, 0 otherwise)
            - terminated: Whether the game has ended
            - truncated: Whether the game was truncated (always False)
            - info: Additional information (empty dict)
        """
        # Agent's move.
        result = self._apply_move(action, self.agent_mark)
        if result is not None:
            return result

        # Machine's move.
        machine_move = self._get_machine_move()
        result = self._apply_move(machine_move, self.machine_mark)
        if result is not None:
            return result

        return self._get_obs(), 0, False, False, {}

    def _get_machine_move(self) -> int:
        """
        Get a random valid move for the machine player.

        Returns
        -------
        int
            Index (0-8) of a valid move on the board.
        """
        # Get a random valid move for the machine.
        valid_moves = self._get_valid_moves()
        assert valid_moves, "There should be at least one valid move."
        return self._np_random.choice(valid_moves)

    def _get_valid_moves(self) -> list[int]:
        """
        Get a list of valid move indices where the board is empty.

        Returns
        -------
        list[int]
            List of indices (0-8) representing empty positions on the board.
        """
        # Return a list of indices (0-8) where the board is empty.
        return [i for i in range(9) if self.board.flat[i] == 0]

    def _get_obs(self) -> dict[str, np.ndarray]:
        """
        Get the current observation of the environment.

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary containing:
            - 'observation': Flattened board state as a (9,) array
            - 'action_mask': Binary mask indicating valid moves as a (9,) array
        """
        # Flatten the board to a (9,) vector.
        obs_board = self.board.flatten()
        # Create the action mask: 1 indicates an empty cell (legal move), 0 otherwise.
        action_mask = np.fromiter((obs_board[i] == 0 for i in range(9)), dtype=np.int8, count=9)
        return {"observation": obs_board, "action_mask": action_mask}

    def _check_winner(self) -> int | None:
        """
        Check if there is a winner on the current board.

        Returns
        -------
        int | None
            The mark (1 or 2) of the winning player, or None if no winner.
        """
        b = self.board
        # Check rows.
        for i in range(3):
            if b[i, 0] != 0 and b[i, 0] == b[i, 1] == b[i, 2]:
                return b[i, 0]
            if b[0, i] != 0 and b[0, i] == b[1, i] == b[2, i]:
                return b[0, i]
        # Check diagonals.
        if b[0, 0] != 0 and b[0, 0] == b[1, 1] == b[2, 2]:
            return b[0, 0]
        if b[0, 2] != 0 and b[0, 2] == b[1, 1] == b[2, 0]:
            return b[0, 2]
        return None

    def render(self) -> None:
        """Render the current state of the board."""
        symbol = {0: " ", 1: "X", 2: "O"}
        board_str = "\n".join("|".join(symbol[val] for val in row) for row in self.board)
        print(board_str)  # noqa: T201

    def close(self) -> None:
        """Close the environment and clean up resources."""
