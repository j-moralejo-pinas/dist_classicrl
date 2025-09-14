"""Unit tests for :class:`dist_classicrl.environments.tiktaktoe_mod.TicTacToeEnv`."""

import random

import numpy as np
import pytest

from dist_classicrl.environments.tiktaktoe_mod import TicTacToeEnv


def test_check_winner() -> None:
    """Test the winner detection functionality for various board configurations."""
    env = TicTacToeEnv()
    env.board = np.zeros((3, 3), dtype=np.int8)
    assert env._check_winner() is None, "No winner should be detected on an empty board."
    env.board = np.array(
        [
            [1, 1, 1],
            [0, 2, 0],
            [0, 0, 2],
        ]
    )
    assert env._check_winner() == 1, "Player 1 should win with a horizontal line."
    env.board = np.array(
        [
            [2, 2, 0],
            [0, 2, 0],
            [0, 2, 1],
        ]
    )
    assert env._check_winner() == 2, "Player 2 should win with a horizontal line."
    env.board = np.array(
        [
            [1, 0, 2],
            [1, 1, 2],
            [0, 0, 1],
        ]
    )
    assert env._check_winner() == 1, "Player 1 should win with a vertical line."


def test_get_valid_moves() -> None:
    """Test the valid moves detection for different board states."""
    env = TicTacToeEnv()
    env.board = np.zeros((3, 3), dtype=np.int8)
    valid_moves = env._get_valid_moves()
    assert len(valid_moves) == 9, "All moves should be valid on an empty board."
    assert all(move in valid_moves for move in range(9)), (
        "All positions should be valid moves on an empty board."
    )

    # Make some moves
    env.board[0, 0] = 1
    env.board[1, 0] = 2
    valid_moves = env._get_valid_moves()
    assert len(valid_moves) == 7, "There should be 7 valid moves after two moves are made."
    assert all(move in valid_moves for move in [1, 2, 4, 5, 6, 7, 8]), (
        "Valid moves should exclude occupied positions."
    )

    env.board = np.array(
        [
            [1, 1, 1],
            [2, 2, 2],
            [1, 1, 2],
        ]
    )
    valid_moves = env._get_valid_moves()
    assert len(valid_moves) == 0, "No valid moves should be available when the board is full."


def test_get_obs() -> None:
    """Test observation generation including board state and action mask."""
    env = TicTacToeEnv()
    env.board = np.zeros((3, 3), dtype=np.int8)
    obs = env._get_obs()

    assert "observation" in obs, "Observation should contain the board state."
    assert "action_mask" in obs, "Observation should contain the action mask."
    assert obs["observation"].shape == (9,), "Observation should be a flat array of size 9."
    assert obs["action_mask"].shape == (9,), "Action mask should be a flat array of size 9."
    assert np.all(obs["observation"] == 0), "Observation should be all zeros on an empty board."
    assert np.all(obs["action_mask"] == 1), (
        "Action mask should indicate all moves are valid on an empty board."
    )

    new_board = np.array(
        [
            [1, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]
    )
    env.board = np.copy(new_board)
    obs = env._get_obs()
    assert np.array_equal(obs["observation"], new_board.flatten()), (
        "Observation should match the current board state."
    )
    assert obs["action_mask"][0] == 0, (
        "Action mask should indicate the first move is no longer valid."
    )
    assert np.all(obs["action_mask"][1:] == 1), "All other moves should still be valid."

    new_board = np.array(
        [
            [1, 2, 2],
            [1, 1, 2],
            [2, 2, 1],
        ]
    )
    env.board = np.copy(new_board)
    obs = env._get_obs()
    assert np.array_equal(obs["observation"], new_board.flatten()), (
        "Observation should match the current board state."
    )
    assert np.all(obs["action_mask"] == 0), (
        "Action mask should indicate no valid moves when the board is full."
    )


def test_get_machine_move() -> None:
    """Test machine move selection logic and edge cases."""
    env = TicTacToeEnv()
    env.board = np.zeros((3, 3), dtype=np.int8)
    env._np_random = np.random.default_rng(seed=42)  # Set a seed for reproducibility
    move = env._get_machine_move()
    assert move in range(9), "Machine move should be a valid position on the board."

    env.board = np.array(
        [
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
        ]
    )
    move = env._get_machine_move()
    assert move == 4, "Machine should choose the center position when available."

    env.board = np.array(
        [
            [1, 2, 1],
            [1, 2, 2],
            [2, 1, 1],
        ]
    )
    with pytest.raises(AssertionError, match="There should be at least one valid move\\."):
        env._get_machine_move()


def test_apply_move() -> None:
    """Test move application, win detection, and game termination conditions."""
    env = TicTacToeEnv()
    env.board = np.zeros((3, 3), dtype=np.int8)
    env.agent_mark = 1
    env.machine_mark = 2
    return_value = env._apply_move(0, env.agent_mark)
    assert return_value is None, "No winner or draw should be detected after the first move."
    assert np.array_equal(
        env.board,
        np.array(
            [
                [env.agent_mark, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ]
        ),
    ), "Move should be applied correctly."

    env.board = np.array(
        [
            [0, env.agent_mark, 0],
            [0, env.agent_mark, 0],
            [0, 0, 0],
        ]
    )
    return_value = env._apply_move(7, env.agent_mark)
    assert return_value is not None, "Agent should win with a vertical line."
    obs, rew, term, trunc, info = return_value
    assert np.array_equal(
        obs["observation"],
        np.array([0, env.agent_mark, 0, 0, env.agent_mark, 0, 0, env.agent_mark, 0]),
    ), "Move should be applied correctly."
    assert np.array_equal(
        env.board,
        np.array(
            [
                [0, env.agent_mark, 0],
                [0, env.agent_mark, 0],
                [0, env.agent_mark, 0],
            ]
        ),
    ), "Board should reflect the winning move."
    assert rew == 1, "Agent should receive a reward of 1 for winning."
    assert term, "Game should be terminated after a win."
    assert not trunc, "Game should not be truncated after a win."

    env.board = np.array(
        [
            [env.machine_mark, 0, 0],
            [0, env.machine_mark, 0],
            [0, 0, 0],
        ]
    )
    return_value = env._apply_move(8, env.machine_mark)
    assert return_value is not None, "Machine should win with a diagonal line."
    obs, rew, term, trunc, _ = return_value
    assert np.array_equal(
        obs["observation"],
        np.array([env.machine_mark, 0, 0, 0, env.machine_mark, 0, 0, 0, env.machine_mark]),
    ), "Move should be applied correctly."
    assert np.array_equal(
        env.board,
        np.array(
            [
                [env.machine_mark, 0, 0],
                [0, env.machine_mark, 0],
                [0, 0, env.machine_mark],
            ]
        ),
    ), "Board should reflect the winning move."
    assert rew == -1, "Machine should receive a reward of -1 for winning."
    assert term, "Game should be terminated after a win."
    assert not trunc, "Game should not be truncated after a win."

    env.board = np.array(
        [
            [env.agent_mark, env.machine_mark, env.agent_mark],
            [env.machine_mark, env.agent_mark, env.machine_mark],
            [env.machine_mark, env.agent_mark, 0],
        ]
    )
    return_value = env._apply_move(8, env.machine_mark)
    assert return_value is not None, "Game should end in a draw."
    obs, rew, term, trunc, _info = return_value
    assert np.array_equal(
        obs["observation"],
        np.array(
            [
                env.agent_mark,
                env.machine_mark,
                env.agent_mark,
                env.machine_mark,
                env.agent_mark,
                env.machine_mark,
                env.machine_mark,
                env.agent_mark,
                env.machine_mark,
            ]
        ),
    ), "Move should be applied correctly."

    assert rew == 0, "Reward should be 0 for a draw."
    assert term, "Game should be terminated after a draw."
    assert not trunc, "Game should not be truncated after a draw."

    env.board = np.array(
        [
            [env.agent_mark, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]
    )
    with pytest.raises(AssertionError, match="Invalid move\\."):
        env._apply_move(0, env.machine_mark)


def test_reset() -> None:
    """Test environment reset functionality and initial state validation."""
    env = TicTacToeEnv()
    obs, _info = env.reset()
    assert "observation" in obs
    assert "action_mask" in obs
    assert obs["observation"].shape == (9,)
    assert obs["action_mask"].shape == (9,)
    assert np.where(obs["action_mask"])[0].size > 0, "No valid moves available at reset."


def test_step() -> None:
    """Test the step function with agent and machine moves."""
    env = TicTacToeEnv()
    env._np_random = np.random.default_rng(seed=42)
    env.board = np.zeros((3, 3), dtype=np.int8)
    env.agent_mark = 1
    env.machine_mark = 2
    obs, _reward, _terminated, _truncated, _info = env.step(0)
    assert sum(obs["action_mask"]) == 7, (
        "There should be 7 valid moves after the first agent and machine moves."
    )

    assert obs["action_mask"][0] == 0, "First move should not be valid anymore."
    assert env.board[0, 0] == env.agent_mark, "Board should reflect the agent's move."
    assert len(np.where(env.board.flatten() == 0)[0]) == 7, (
        "There should be 7 valid moves after the first move."
    )


def test_running_environment() -> None:
    """Test running complete games to ensure environment stability and correctness."""
    env = TicTacToeEnv()

    for i in range(100):
        obs, _ = env.reset(seed=i)
        env.render()

        terminated = False
        while not terminated:
            valid_moves = np.where(obs["action_mask"])[0]
            if valid_moves.size == 0:
                break
            # For demonstration, choose the first valid move.
            action = random.choice(valid_moves)
            obs, _reward, terminated, _truncated, _info = env.step(action)
            env.render()
