"""Distributed Q-learning tests using MPI ranks with deterministic exploration."""

from collections.abc import Sequence

import numpy as np
import pytest
from mpi4py import MPI

from dist_classicrl.algorithms.base_algorithms.q_learning_optimal import OptimalQLearningBase
from dist_classicrl.algorithms.runtime.base_runtime import BaseRuntime
from dist_classicrl.algorithms.runtime.q_learning_async_dist import DistAsyncQLearning
from dist_classicrl.environments.rigged_two_armed_bandit import RiggedTwoArmedBanditEnv
from dist_classicrl.schedules.constant_schedule import ConstantSchedule
from dist_classicrl.schedules.linear_schedule import LinearSchedule
from dist_classicrl.utils import _make_dummy_vec_env


def _make_algo_and_runtime(
    runtime_cls: type[BaseRuntime],
) -> tuple[OptimalQLearningBase, BaseRuntime]:
    """Create an algorithm/runtime pair for deterministic distributed training.

    Parameters
    ----------
    runtime_cls : type[BaseRuntime]
        Runtime class to instantiate (distributed async).

    Returns
    -------
    tuple[OptimalQLearningBase, BaseRuntime]
        The configured algorithm and runtime.
    """
    algo = OptimalQLearningBase(state_size=1, action_size=2, discount_factor=1.0, seed=0)
    # Force random path and random action 1
    algo._rng = DeterministicRNG()  # type: ignore[assignment]
    algo._np_rng = DeterministicNPRNG()  # type: ignore[assignment]
    lr = ConstantSchedule(1.0)
    eps = LinearSchedule(0, 1.0)
    rt = runtime_cls(algorithm=algo, lr_schedule=lr, exploration_rate_schedule=eps)
    return algo, rt


class DeterministicRNG:
    """RNG that forces exploration and always selects action 1 when needed.

    Methods return fixed values to ensure epsilon triggers and action 1 is
    chosen during exploration.
    """

    def uniform(self, _a: float = 0.0, _b: float = 1.0) -> float:
        """Return 0.0 to always satisfy epsilon checks."""
        return 0.0

    def random(self) -> float:
        """Return 0.0 to always satisfy epsilon checks."""
        return 0.0

    def randint(self, _a: int, _b: int) -> int:
        """Return 1 to force action index 1 when exploring."""
        return 1

    def choice(self, seq: Sequence[int] | np.ndarray) -> int:  # pragma: no cover
        """Prefer 1 if present in the sequence/array; otherwise return the first."""
        arr = np.asarray(seq)
        if (arr == 1).any():
            return 1
        return int(arr[0])


class DeterministicNPRNG:
    """NumPy-like RNG: zeros for random() and ones for integers()."""

    def random(self, size: int | tuple[int, ...] | None = None) -> float | np.ndarray:  # type: ignore[override]
        """Return 0.0 or an array of zeros matching the requested size."""
        if size is None:
            return 0.0
        return np.zeros(size, dtype=float)

    def integers(
        self,
        _low: int,
        _high: int | None = None,
        size: int | tuple[int, ...] | None = None,
        dtype: type = int,
        **_kwargs: object,
    ) -> int | np.ndarray:  # type: ignore[override]
        """Return 1 or an array of ones (within [low, high) semantics)."""
        if size is None:
            return 1
        return np.ones(size, dtype=dtype)


COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.Get_size()


@pytest.mark.skipif(SIZE < 2, reason="Distributed test requires at least 2 MPI ranks")
def test_distributed_train_skips_validation_and_updates_q_table() -> None:
    """Training with validation interval > steps skips validation and updates Q-table."""
    # Algorithm configured to always explore and pick action 1
    algo, agent = _make_algo_and_runtime(DistAsyncQLearning)
    # Training and validation setup
    steps = 5
    env = _make_dummy_vec_env(
        n_envs=1, env_class=RiggedTwoArmedBanditEnv, env_kwargs={"episode_len": 6}
    )
    val_env = _make_dummy_vec_env(
        n_envs=1, env_class=RiggedTwoArmedBanditEnv, env_kwargs={"episode_len": 6}
    )

    # Set validation interval and steps larger than training to effectively skip validation
    val_every_n_steps = steps + 1
    val_steps = steps + 1

    reward_history, val_history, ret_envs, curr_state = agent.train(
        env=env,
        steps=steps,
        val_env=val_env,
        val_every_n_steps=val_every_n_steps,
        val_steps=val_steps,
        val_episodes=None,
        curr_state_dict={},
        batch_size=8,  # type: ignore[reportCallIssue]
    )

    if RANK == 0:
        # Master: collects training rewards, no validation performed
        assert val_history == []
        assert isinstance(reward_history, list)
        # Q-table updated; optimal action 1 gets value 1.0 (terminal update)
        assert agent.algorithm.q_table.shape == (1, 2)
        assert agent.algorithm.q_table[0, 1] == 5.0
        # Schedules updated by number of experiences processed (5)
        assert agent.lr_schedule.get_value() == 1.0
        assert agent.exploration_rate_schedule.get_value() == 6.0
        # Envs/state not returned on master
        assert ret_envs is None
        assert curr_state is None
    else:
        # Workers: no histories; env and current state returned
        assert reward_history == []
        assert val_history == []
        assert isinstance(ret_envs, RiggedTwoArmedBanditEnv)
        assert isinstance(curr_state, dict)
        assert "states" in curr_state


@pytest.mark.skipif(SIZE < 2, reason="Distributed test requires at least 2 MPI ranks")
def test_distributed_train_with_validation_collects_val_history() -> None:
    """Training with validation collects validation reward history on master rank."""
    # Configure algorithm to always explore during training (action 1)
    algo, agent = _make_algo_and_runtime(DistAsyncQLearning)

    steps = 6
    env = _make_dummy_vec_env(
        n_envs=1, env_class=RiggedTwoArmedBanditEnv, env_kwargs={"episode_len": 5}
    )
    val_env = _make_dummy_vec_env(
        n_envs=1, env_class=RiggedTwoArmedBanditEnv, env_kwargs={"episode_len": 5}
    )

    # Trigger validation multiple times during training; use val_steps multiple of episode_len
    val_every_n_steps = 2
    val_steps = 5

    reward_history, val_history, ret_envs, curr_state = agent.train(
        env=env,
        steps=steps,
        val_env=val_env,
        val_every_n_steps=val_every_n_steps,
        val_steps=val_steps,
        val_episodes=None,
        curr_state_dict={},
        batch_size=8,  # type: ignore[reportCallIssue]
    )

    if RANK == 0:
        # Don't assert training reward_history (episode completion timing
        # is non-deterministic across workers)
        assert isinstance(reward_history, list)
        # Validation should have run >=1 times and each run gets total reward 5
        assert len(val_history) >= 1
        assert all(v == 5.0 for v in val_history)
        # Master returns no env/state
        assert ret_envs is None
        assert curr_state is None
    else:
        # Workers return env/state; no histories
        assert reward_history == []
        assert val_history == []
        assert isinstance(ret_envs, RiggedTwoArmedBanditEnv)
        assert isinstance(curr_state, dict)
        assert "states" in curr_state
