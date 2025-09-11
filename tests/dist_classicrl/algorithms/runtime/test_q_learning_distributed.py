import numpy as np
import numpy.typing as npt
import pytest

from mpi4py import MPI

from dist_classicrl.algorithms.base_algorithms.q_learning_optimal import OptimalQLearningBase
from dist_classicrl.algorithms.runtime.q_learning_async_dist import DistAsyncQLearning
from dist_classicrl.environments.custom_env import DistClassicRLEnv
from dist_classicrl.schedules.base_schedules import BaseSchedule


class ConstSchedule(BaseSchedule):
    def __init__(self, value: float) -> None:
        super().__init__(value=value, min_value=value)

    def update(self, steps: int) -> None:  # pragma: no cover - remains constant
        pass


class AccumSchedule(BaseSchedule):
    def __init__(self, value: float) -> None:
        super().__init__(value=value, min_value=-1e9)

    def update(self, steps: int) -> None:
        self.set_value(self.get_value() + steps)


class BanditDistEnv(DistClassicRLEnv):
    """Simple deterministic bandit DistClassicRLEnv used for distributed tests.

    - Single state, two actions (0, 1)
    - Reward is 1.0 if action == 1 else 0.0
    - Episode terminates every `episode_len` steps
    """

    num_agents: int

    def __init__(self, n_agents: int = 1, episode_len: int = 5) -> None:
        self.num_agents = n_agents
        self.episode_len = episode_len
        self._t = 0

    def reset(self, seed: int | None = None, options: dict | None = None):
        self._t = 0
        obs = np.zeros(self.num_agents, dtype=np.int32)
        infos: list[dict] = [{} for _ in range(self.num_agents)]
        return obs, infos

    def step(self, actions: npt.NDArray[np.int32]):
        rewards = (actions == 1).astype(np.float32)
        self._t += 1
        terminated = np.array([self._t % self.episode_len == 0] * self.num_agents, dtype=bool)
        truncated = np.zeros(self.num_agents, dtype=bool)
        if terminated.any():
            self._t = 0
        obs = np.zeros(self.num_agents, dtype=np.int32)
        infos: list[dict] = [{} for _ in range(self.num_agents)]
        return obs, rewards, terminated, truncated, infos

    def close(self) -> None:  # pragma: no cover
        pass

    def render(self) -> None:  # pragma: no cover
        pass

    def seed(self, seed: int) -> None:  # pragma: no cover
        pass

    def get_env_info(self) -> dict[str, object]:  # pragma: no cover
        return {"episode_len": self.episode_len, "num_agents": self.num_agents}

    def get_agent_info(self) -> dict[str, object]:  # pragma: no cover
        return {}


class DeterministicRNG:
    """RNG that forces exploration path and always selects action 1 when needed."""

    def uniform(self, a: float = 0.0, b: float = 1.0) -> float:
        return 0.0

    def random(self) -> float:
        return 0.0

    def randint(self, a: int, b: int) -> int:
        return 1

    def choice(self, seq):  # pragma: no cover
        try:
            if 1 in seq:
                return 1
        except Exception:
            arr = np.asarray(seq)
            if (arr == 1).any():
                return 1
        return seq[0]


class DeterministicNPRNG:
    """NumPy-like RNG that returns zeros for random() and ones for integers()."""

    def random(self, size=None):  # type: ignore[override]
        if size is None:
            return 0.0
        return np.zeros(size, dtype=float)

    def integers(self, low, high=None, size=None, dtype=int, endpoint=False):  # type: ignore[override]
        # Always return 1 within [low, high)
        if size is None:
            return 1
        return np.ones(size, dtype=dtype)


COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.Get_size()


@pytest.mark.skipif(SIZE < 2, reason="Distributed test requires at least 2 MPI ranks")
def test_distributed_train_skips_validation_and_updates_q_table():
    # Algorithm configured to always explore and pick action 1
    algo = OptimalQLearningBase(state_size=1, action_size=2, discount_factor=1.0, seed=0)
    algo._rng = DeterministicRNG()  # type: ignore[assignment]
    algo._np_rng = DeterministicNPRNG()  # type: ignore[assignment]

    lr = ConstSchedule(1.0)
    eps = AccumSchedule(1.0)
    agent = DistAsyncQLearning(algorithm=algo, lr_schedule=lr, exploration_rate_schedule=eps)

    # Training and validation setup
    steps = 5
    env = BanditDistEnv(n_agents=1, episode_len=6)
    val_env = BanditDistEnv(n_agents=1, episode_len=6)

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
        batch_size=8,
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
        assert ret_envs is None and curr_state is None
    else:
        # Workers: no histories; env and current state returned
        assert reward_history == [] and val_history == []
        assert isinstance(ret_envs, BanditDistEnv)
        assert isinstance(curr_state, dict) and "states" in curr_state


@pytest.mark.skipif(SIZE < 2, reason="Distributed test requires at least 2 MPI ranks")
def test_distributed_train_with_validation_collects_val_history():
    # Configure algorithm to always explore during training (action 1)
    algo = OptimalQLearningBase(state_size=1, action_size=2, discount_factor=1.0, seed=0)
    algo._rng = DeterministicRNG()  # type: ignore[assignment]
    algo._np_rng = DeterministicNPRNG()  # type: ignore[assignment]

    # Seed Q-table to make greedy validation deterministic (prefer action 1)
    algo.q_table[0, 1] = 1.0

    lr = ConstSchedule(1.0)
    eps = AccumSchedule(1.0)
    agent = DistAsyncQLearning(algorithm=algo, lr_schedule=lr, exploration_rate_schedule=eps)

    steps = 6
    env = BanditDistEnv(n_agents=1, episode_len=5)
    val_env = BanditDistEnv(n_agents=1, episode_len=5)

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
        batch_size=8,
    )

    if RANK == 0:
        # Don't assert training reward_history (episode completion timing is non-deterministic across workers)
        assert isinstance(reward_history, list)
        # Validation should have run >=1 times and each run gets total reward 5
        assert len(val_history) >= 1
        assert all(v == 5.0 for v in val_history)
        # Master returns no env/state
        assert ret_envs is None and curr_state is None
    else:
        # Workers return env/state; no histories
        assert reward_history == [] and val_history == []
        assert isinstance(ret_envs, BanditDistEnv)
        assert isinstance(curr_state, dict) and "states" in curr_state
