from typing import Any, Dict, Optional
import gymnasium as gym
from gymnasium.utils.seeding import np_random
import numpy as np


class TicTacToeEnv(gym.Env):
    metadata = {"render.modes": ["human"]}
    _np_random: np.random.Generator
    board: np.ndarray
    agent_starts: bool
    agent_mark: int
    machine_mark: int

    def __init__(self):
        super(TicTacToeEnv, self).__init__()
        self.action_space = gym.spaces.Discrete(9)

        self.observation_space = gym.spaces.Dict(
            {
                "observation": gym.spaces.MultiDiscrete([3] * 9),
                "action_mask": gym.spaces.MultiDiscrete([2] * 9),
            }
        )
        self.reset()

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
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

    def _apply_move(self, move: int, mark: int):
        row, col = divmod(move, 3)
        assert self.board[row, col] == 0, "Invalid move."
        self.board[row, col] = mark
        if self._check_winner() == mark:
            reward = 1 if mark == self.agent_mark else -1
            return self._get_obs(), reward, True, False, {}
        if not self._get_valid_moves():
            return self._get_obs(), 0, True, False, {}
        return None

    def step(self, action: int):
        """ """
        # Agent's move.
        result = self._apply_move(action, self.agent_mark)
        if result is not None:
            return result

        # Machine's move.
        valid_moves = self._get_valid_moves()
        assert valid_moves, "There should be at least one valid move."
        machine_move = self._np_random.choice(valid_moves)
        result = self._apply_move(machine_move, self.machine_mark)
        if result is not None:
            return result

        return self._get_obs(), 0, False, False, {}

    def _get_valid_moves(self):
        # Return a list of indices (0-8) where the board is empty.
        return [i for i in range(9) if self.board.flat[i] == 0]

    def _get_obs(self):
        # Flatten the board to a (9,) vector.
        obs_board = self.board.flatten()
        # Create the action mask: 1 indicates an empty cell (legal move), 0 otherwise.
        action_mask = np.fromiter((obs_board[i] == 0 for i in range(9)), dtype=np.int8, count=9)
        return {"observation": obs_board, "action_mask": action_mask}

    def _check_winner(self):
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

    def render(self, mode="human"):
        symbol = {0: " ", 1: "X", 2: "O"}
        board_str = "\n".join("|".join(symbol[val] for val in row) for row in self.board)
        print(board_str)

    def close(self):
        pass
