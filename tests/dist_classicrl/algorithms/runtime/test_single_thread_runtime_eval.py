import numpy as np
import numpy.typing as npt

from dist_classicrl.algorithms.runtime.single_thread_runtime import SingleThreadQLearning
from dist_classicrl.algorithms.base_algorithms.q_learning_optimal import OptimalQLearningBase
from dist_classicrl.schedules.base_schedules import BaseSchedule
import gymnasium as gym
from gymnasium.vector import VectorEnv


class DummySchedule(BaseSchedule):
    """Minimal schedule that returns a fixed value and no-ops on update."""

    def __init__(self, value: float = 0.0) -> None:
        super().__init__(value=value, min_value=value)

    def update(self, _: int) -> None:  # pragma: no cover - unused in these tests
        pass


class DummyVecEnv(VectorEnv):
    """
    Minimal vectorized environment for tests.

    - n_envs independent one-state bandits
    - reward = 1 if action == 1, else 0
    - each env terminates every `episode_len` steps
    - reset returns (obs, infos) with obs shape (n_envs,)
    - step consumes an array of actions shape (n_envs,)
    """

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
        # reset counters for those envs that terminated to allow subsequent episodes
        self._t[terminated] = 0
        truncated = np.zeros(self.num_envs, dtype=bool)
        obs = np.zeros(self.num_envs, dtype=np.int32)
        infos: list[dict] = [{} for _ in range(self.num_envs)]
        return obs, rewards, terminated.astype(bool), truncated, infos


def _make_runtime():
    # Configure algorithm to prefer action 1 deterministically for state 0
    algo = OptimalQLearningBase(state_size=1, action_size=2, discount_factor=0.99, seed=0)
    algo.q_table[0] = np.array([0.0, 1.0], dtype=np.float64)
    lr = DummySchedule(0.0)
    eps = DummySchedule(0.0)
    return SingleThreadQLearning(algorithm=algo, lr_schedule=lr, exploration_rate_schedule=eps)


def _make_vec_env(n_envs: int, episode_len: int = 10) -> DummyVecEnv:
    return DummyVecEnv(n_envs=n_envs, episode_len=episode_len)


def _assert_evaluate_steps(runtime: SingleThreadQLearning, n_envs: int, episode_len: int, steps: int):
    env = _make_vec_env(n_envs=n_envs, episode_len=episode_len)
    total, history = runtime.evaluate_steps(env, steps=steps)
    # Steps are agent-steps; evaluate_steps iterates in chunks of n_envs (vector steps)
    vector_steps = steps // n_envs
    full_episode_blocks = vector_steps // episode_len
    expected_len = full_episode_blocks * n_envs
    expected_history = [float(episode_len)] * expected_len
    assert history == expected_history
    assert total == float(episode_len * expected_len)


def _assert_evaluate_episodes(runtime: SingleThreadQLearning, n_envs: int, episode_len: int, episodes: int):
    env = _make_vec_env(n_envs=n_envs, episode_len=episode_len)
    total, history = runtime.evaluate_episodes(env, episodes=episodes)
    assert history == [float(episode_len)] * episodes
    assert total == float(episode_len * episodes)


def test_vec_single_env_evaluate_steps_and_episodes() -> None:
    runtime = _make_runtime()
    episode_len = 10
    # Single vectorized env (n_envs=1)
    _assert_evaluate_steps(runtime, n_envs=1, episode_len=episode_len, steps=10)
    _assert_evaluate_episodes(runtime, n_envs=1, episode_len=episode_len, episodes=3)


def test_vec_evaluate_steps_exact_one_episode_per_env() -> None:
    runtime = _make_runtime()
    n_envs, episode_len = 3, 10
    # With n_envs=3, after 10 vector steps, all 3 envs terminate once
    _assert_evaluate_steps(runtime, n_envs=n_envs, episode_len=episode_len, steps=episode_len * n_envs)


def test_vec_evaluate_episodes_multiple_total_across_envs() -> None:
    runtime = _make_runtime()
    n_envs, episode_len = 4, 10
    episodes = 2 * n_envs
    _assert_evaluate_episodes(runtime, n_envs=n_envs, episode_len=episode_len, episodes=episodes)
