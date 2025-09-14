"""Run-step tests for single-thread and parallel Q-learning runtimes."""

from collections.abc import Sequence

import numpy as np

from dist_classicrl.algorithms.base_algorithms.q_learning_optimal import OptimalQLearningBase
from dist_classicrl.algorithms.runtime.base_runtime import BaseRuntime
from dist_classicrl.algorithms.runtime.parallel_runtime import ParallelQLearning
from dist_classicrl.algorithms.runtime.single_thread_runtime import SingleThreadQLearning
from dist_classicrl.environments.rigged_two_armed_bandit import RiggedTwoArmedBanditEnv
from dist_classicrl.schedules.constant_schedule import ConstantSchedule
from dist_classicrl.schedules.linear_schedule import LinearSchedule
from dist_classicrl.utils import _make_dummy_vec_env


class DeterministicRNG:
    """Minimal RNG shim to deterministically exercise exploration paths.

    - ``uniform``/``random`` return 0.0 so epsilon-conditions always trigger.
    - ``randint`` returns 1 to force the exploratory action to be index 1.
    - ``choice`` prefers element 1 if present; otherwise returns the first.
    """

    def uniform(self, _a: float = 0.0, _b: float = 1.0) -> float:
        """Return 0.0 to ensure epsilon check passes."""
        return 0.0

    def random(self) -> float:
        """Return 0.0 to ensure epsilon check passes."""
        return 0.0

    def randint(self, _a: int, _b: int) -> int:
        """Return 1 to force action index 1 during exploration."""
        return 1

    def choice(self, seq: Sequence[int] | np.ndarray) -> int:
        """Prefer 1 if present, otherwise return the first element.

        Handles both Python sequences and NumPy arrays for convenience.
        """
        arr = np.asarray(seq)
        if (arr == 1).any():
            return 1
        return int(arr[0])


def _make_algo_and_runtime(
    lr0: float, eps0: float, runtime_cls: type[BaseRuntime]
) -> tuple[OptimalQLearningBase, BaseRuntime]:
    """Create an algorithm/runtime pair configured for deterministic exploration.

    Parameters
    ----------
    lr0 : float
        Initial learning rate for a ConstantSchedule.
    eps0 : float
        Initial epsilon for a LinearSchedule starting at 0, used to accumulate
        by the number of updates performed during ``run_steps``.
    runtime_cls : type[BaseRuntime]
        Runtime class to instantiate (single-thread or parallel).

    Returns
    -------
    tuple[OptimalQLearningBase, BaseRuntime]
        The configured algorithm and runtime.
    """
    algo = OptimalQLearningBase(state_size=1, action_size=2, discount_factor=1.0, seed=0)
    # Force random path and random action 1
    algo._rng = DeterministicRNG()  # type: ignore[assignment]
    lr = ConstantSchedule(lr0)
    eps = LinearSchedule(1.0, eps0)
    rt = runtime_cls(algorithm=algo, lr_schedule=lr, exploration_rate_schedule=eps)
    return algo, rt


def test_single_thread_run_steps_random_actions_and_updates() -> None:
    """Single-thread runtime performs updates and returns expected rewards/state."""
    # Single-thread runtime with 1 vector env, episode ends exactly once
    algo, rt = _make_algo_and_runtime(lr0=1.0, eps0=1.0, runtime_cls=SingleThreadQLearning)
    env = _make_dummy_vec_env(
        n_envs=1, env_class=RiggedTwoArmedBanditEnv, env_kwargs={"episode_len": 5}
    )

    avg, history, _env, state_dict = rt.run_steps(steps=5, env=env, curr_state_dict=None)

    # Rewards: always action 1 -> reward 1 per step, episode length 5
    assert history == [5.0]
    assert avg == 5.0
    # Q-table updated; last update is terminal so target=1
    assert algo.q_table.shape == (1, 2)
    assert algo.q_table[0, 1] == 1.0  # pytest.approx(1.0, rel=1e-7, abs=1e-7)
    # Schedules incremented by n_updates per step (n_envs=1)
    assert rt.lr_schedule.get_value() == 1.0
    assert rt.exploration_rate_schedule.get_value() == 6.0  # 1.0 + 5 steps
    # State dict propagated
    assert "states" in state_dict
    assert isinstance(state_dict["states"], np.ndarray)


def test_parallel_run_steps_random_actions_and_updates() -> None:
    """Parallel runtime (1 worker) updates Q and schedules and returns states."""
    # Parallel runtime with 1 worker, each worker has 1 vectorized env
    _, rt = _make_algo_and_runtime(lr0=1.0, eps0=1.0, runtime_cls=ParallelQLearning)
    envs = [
        _make_dummy_vec_env(
            n_envs=1, env_class=RiggedTwoArmedBanditEnv, env_kwargs={"episode_len": 5}
        )
    ]
    try:
        rt.init_training()
        avg, history, _returned_envs, states_list = rt.run_steps(
            steps=5, env=envs, curr_state_dict=None
        )
        # Rewards
        assert history == [5.0]
        assert avg == 5.0
        # Q-table updated in shared mem; after close_training it persists
        assert rt.algorithm.q_table.shape == (1, 2)
        # The last terminal update leaves q=1.0
        assert rt.algorithm.q_table[0, 1] == 1.0
        # Schedules updated by child (n_updates=1 per step)
        assert rt.lr_schedule.get_value() == 1.0
        assert rt.exploration_rate_schedule.get_value() == 6.0
        # State dicts returned
        assert len(states_list) == 1
        assert "states" in states_list[0]  # type: ignore[reportArgumentType]
    finally:
        rt.close_training()


def test_parallel_run_steps_two_envs_10_steps() -> None:
    """Two envs over 10 steps yield two full episodes with correct schedule updates."""
    # Two environments, 10 total steps split evenly -> 5 steps per env
    _algo, rt = _make_algo_and_runtime(lr0=1.0, eps0=1.0, runtime_cls=ParallelQLearning)
    envs = [
        _make_dummy_vec_env(
            n_envs=1, env_class=RiggedTwoArmedBanditEnv, env_kwargs={"episode_len": 5}
        )
        for _ in range(2)
    ]
    try:
        rt.init_training()
        avg, history, _returned_envs, states_list = rt.run_steps(
            steps=10, env=envs, curr_state_dict=None
        )
        # Each env completes exactly one episode of length 5 with reward 1 per step
        assert history == [5.0, 5.0]
        assert avg == 5.0
        # Final Q after the last terminal update is 1.0
        assert rt.algorithm.q_table.shape == (1, 2)
        assert rt.algorithm.q_table[0, 1] == 1.0
        # Schedules: lr constant=1.0, epsilon accumulated by total updates (10)
        assert rt.lr_schedule.get_value() == 1.0
        assert rt.exploration_rate_schedule.get_value() == 11.0
        # Returned states per env
        assert len(states_list) == 2
        assert all("states" in d for d in states_list)
    finally:
        rt.close_training()
