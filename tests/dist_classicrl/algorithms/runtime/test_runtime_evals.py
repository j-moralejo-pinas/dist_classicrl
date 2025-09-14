"""Evaluation tests for SingleThreadQLearning with a rigged two-armed bandit."""

from typing import TYPE_CHECKING

import numpy as np

from dist_classicrl.algorithms.base_algorithms.q_learning_optimal import OptimalQLearningBase
from dist_classicrl.algorithms.runtime.single_thread_runtime import SingleThreadQLearning
from dist_classicrl.environments.rigged_two_armed_bandit import RiggedTwoArmedBanditEnv
from dist_classicrl.schedules.constant_schedule import ConstantSchedule
from dist_classicrl.wrappers.dummy_vec_wrapper import DummyVecWrapper

if TYPE_CHECKING:
    import gymnasium as gym
    from gymnasium.vector import VectorEnv


def _make_runtime() -> SingleThreadQLearning:
    """Create a deterministic runtime preferring action 1 for state 0.

    The Q-table for state 0 is set to [0.0, 1.0], and both learning rate and
    exploration rate schedules are constant zeros to avoid any learning or
    exploration during evaluation.

    Returns
    -------
    SingleThreadQLearning
        Configured runtime ready for evaluation.
    """
    # Configure algorithm to prefer action 1 deterministically for state 0
    algo = OptimalQLearningBase(state_size=1, action_size=2, discount_factor=0.99, seed=0)
    algo.q_table[0] = np.array([0.0, 1.0], dtype=np.float64)
    lr = ConstantSchedule(0.0)
    eps = ConstantSchedule(0.0)
    return SingleThreadQLearning(algorithm=algo, lr_schedule=lr, exploration_rate_schedule=eps)


def _make_vec_env(n_envs: int, episode_len: int = 10) -> DummyVecWrapper:
    """Create a vectorized environment of identical rigged bandits.

    Parameters
    ----------
    n_envs : int
        Number of parallel environments.
    episode_len : int, optional
        Length of each episode in steps, by default 10.

    Returns
    -------
    DummyVecWrapper
        Vectorized wrapper containing ``n_envs`` bandit environments.
    """
    envs: list[gym.Env] = [RiggedTwoArmedBanditEnv(episode_len=episode_len) for _ in range(n_envs)]
    return DummyVecWrapper(envs)


def _assert_evaluate_steps(
    runtime: SingleThreadQLearning, n_envs: int, episode_len: int, steps: int
) -> None:
    """Assert evaluate_steps returns totals and history for full episodes only.

    The function converts the requested agent ``steps`` to vector steps
    (``steps // n_envs``), counts how many full episodes can be completed,
    and checks both the total return and the per-episode history.
    """
    env: VectorEnv = _make_vec_env(n_envs=n_envs, episode_len=episode_len)
    total, history = runtime.evaluate_steps(env, steps=steps)
    # Steps are agent-steps; evaluate_steps iterates in chunks of n_envs (vector steps)
    vector_steps = steps // n_envs
    full_episode_blocks = vector_steps // episode_len
    expected_len = full_episode_blocks * n_envs
    expected_history = [float(episode_len)] * expected_len
    assert history == expected_history
    assert total == float(episode_len * expected_len)


def _assert_evaluate_episodes(
    runtime: SingleThreadQLearning, n_envs: int, episode_len: int, episodes: int
) -> None:
    """Assert evaluate_episodes returns totals and history for exact episodes.

    The vectorized environment runs until the requested number of episodes is
    completed across parallel envs; the total is the sum of per-episode
    returns, and the history lists each episode return.
    """
    env = _make_vec_env(n_envs=n_envs, episode_len=episode_len)
    total, history = runtime.evaluate_episodes(env, episodes=episodes)
    assert history == [float(episode_len)] * episodes
    assert total == float(episode_len * episodes)


def test_vec_single_env_evaluate_steps_and_episodes() -> None:
    """Single-env vector wrapper: steps and episodes evaluation are consistent."""
    runtime = _make_runtime()
    episode_len = 10
    # Single vectorized env (n_envs=1)
    _assert_evaluate_steps(runtime, n_envs=1, episode_len=episode_len, steps=10)
    _assert_evaluate_episodes(runtime, n_envs=1, episode_len=episode_len, episodes=3)


def test_vec_evaluate_steps_exact_one_episode_per_env() -> None:
    """With n_envs=3, steps cover exactly one episode per env across vector steps."""
    runtime = _make_runtime()
    n_envs, episode_len = 3, 10
    # With n_envs=3, after 10 vector steps, all 3 envs terminate once
    _assert_evaluate_steps(
        runtime, n_envs=n_envs, episode_len=episode_len, steps=episode_len * n_envs
    )


def test_vec_evaluate_episodes_multiple_total_across_envs() -> None:
    """Episodes target equals a multiple of envs; total and history match exactly."""
    runtime = _make_runtime()
    n_envs, episode_len = 4, 10
    episodes = 2 * n_envs
    _assert_evaluate_episodes(runtime, n_envs=n_envs, episode_len=episode_len, episodes=episodes)
