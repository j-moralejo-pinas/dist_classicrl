import numpy as np
import numpy.typing as npt
import pytest

from dist_classicrl.algorithms.base_algorithms.q_learning_optimal import OptimalQLearningBase
from dist_classicrl.algorithms.runtime.single_thread_runtime import SingleThreadQLearning
from dist_classicrl.algorithms.runtime.parallel_runtime import ParallelQLearning
from dist_classicrl.schedules.base_schedules import BaseSchedule
import gymnasium as gym
from gymnasium.vector import VectorEnv
from dist_classicrl.environments.custom_env import DistClassicRLEnv


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


class DummyVecEnv(VectorEnv):
    def __init__(self, n_envs: int, episode_len: int = 10) -> None:
        super().__init__()
        self.num_envs = n_envs
        self.observation_space = gym.spaces.Discrete(1)
        self.action_space = gym.spaces.Discrete(2)
        self.episode_len = episode_len
        self._t = np.zeros(n_envs, dtype=np.int32)

    def reset(self, seed: int | None = None):
        self._t[:] = 0
        obs = np.zeros(self.num_envs, dtype=np.int32)
        infos: list[dict] = [{} for _ in range(self.num_envs)]
        return obs, infos

    def step(self, actions: npt.NDArray[np.int32]):
        assert actions.shape == (self.num_envs,)
        rewards = (actions == 1).astype(np.float32)
        self._t += 1
        terminated = (self._t % self.episode_len == 0)
        # allow subsequent episodes
        self._t[terminated] = 0
        truncated = np.zeros(self.num_envs, dtype=bool)
        obs = np.zeros(self.num_envs, dtype=np.int32)
        infos: list[dict] = [{} for _ in range(self.num_envs)]
        return obs, rewards, terminated.astype(bool), truncated, infos


class BanditDistEnv(DistClassicRLEnv):
    """DistClassicRLEnv variant of the bandit for single-thread and parallel tests."""

    num_agents: int

    def __init__(self, n_agents: int = 1, episode_len: int = 10) -> None:
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
    def uniform(self, a: float = 0.0, b: float = 1.0) -> float:
        return 0.0

    def random(self) -> float:
        return 0.0

    def randint(self, a: int, b: int) -> int:
        # Always pick action index 1 when exploring (assuming action space {0,1})
        return 1

    def choice(self, seq):
        # For completeness: prefer 1 if present, else first
        try:
            if 1 in seq:
                return 1
        except Exception:  # pragma: no cover - seq may be numpy array
            arr = np.asarray(seq)
            if (arr == 1).any():
                return 1
        return seq[0]


def _make_algo_and_runtime_single(lr0: float, eps0: float) -> tuple[OptimalQLearningBase, SingleThreadQLearning]:
    algo = OptimalQLearningBase(state_size=1, action_size=2, discount_factor=1.0, seed=0)
    # Force random path and random action 1
    algo._rng = DeterministicRNG()  # type: ignore[assignment]
    lr = ConstSchedule(lr0)
    eps = AccumSchedule(eps0)
    rt = SingleThreadQLearning(algorithm=algo, lr_schedule=lr, exploration_rate_schedule=eps)
    return algo, rt


def _make_algo_and_runtime_parallel(lr0: float, eps0: float) -> tuple[OptimalQLearningBase, ParallelQLearning]:
    algo = OptimalQLearningBase(state_size=1, action_size=2, discount_factor=1.0, seed=0)
    algo._rng = DeterministicRNG()  # type: ignore[assignment]
    lr = ConstSchedule(lr0)
    eps = AccumSchedule(eps0)
    rt = ParallelQLearning(algorithm=algo, lr_schedule=lr, exploration_rate_schedule=eps)
    return algo, rt


def test_single_thread_run_steps_random_actions_and_updates() -> None:
    # Single-thread runtime with 1 vector env, episode ends exactly once
    algo, rt = _make_algo_and_runtime_single(lr0=1.0, eps0=1.0)
    env = BanditDistEnv(n_agents=1, episode_len=5)

    avg, history, _env, state_dict = rt.run_steps(steps=5, env=env, curr_state_dict=None)

    # Rewards: always action 1 -> reward 1 per step, episode length 5
    assert history == [5.0]
    assert avg == 5.0
    # Q-table updated; last update is terminal so target=1
    assert algo.q_table.shape == (1, 2)
    assert algo.q_table[0, 1] == 1.0 #pytest.approx(1.0, rel=1e-7, abs=1e-7)
    # Schedules incremented by n_updates per step (n_envs=1)
    assert rt.lr_schedule.get_value() == 1.0
    assert rt.exploration_rate_schedule.get_value() == 6.0  # 1.0 + 5 steps
    # State dict propagated
    assert "states" in state_dict and isinstance(state_dict["states"], np.ndarray)


def test_parallel_run_steps_random_actions_and_updates() -> None:
    # Parallel runtime with 1 worker, each worker has 1 vectorized env
    algo, rt = _make_algo_and_runtime_parallel(lr0=1.0, eps0=1.0)
    envs: list[DistClassicRLEnv] = [BanditDistEnv(n_agents=1, episode_len=5)]
    try:
        rt.init_training()
        avg, history, returned_envs, states_list = rt.run_steps(
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
        assert len(states_list) == 1 and "states" in states_list[0]
    finally:
        rt.close_training()


def test_parallel_run_steps_two_envs_10_steps() -> None:
    # Two environments, 10 total steps split evenly -> 5 steps per env
    algo, rt = _make_algo_and_runtime_parallel(lr0=1.0, eps0=1.0)
    envs: list[DistClassicRLEnv] = [BanditDistEnv(n_agents=1, episode_len=5), BanditDistEnv(n_agents=1, episode_len=5)]
    try:
        rt.init_training()
        avg, history, returned_envs, states_list = rt.run_steps(
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
        assert len(states_list) == 2 and all("states" in d for d in states_list)
    finally:
        rt.close_training()

