"""Test Q-Learning algorithm."""

from gymnasium import spaces
from gymnasium.vector import SyncVectorEnv
from gymnasium.vector.vector_env import AutoresetMode

from dist_classicrl.algorithms.runtime.single_thread_runtime import (
    SingleThreadQLearning,
)
from dist_classicrl.environments.tiktaktoe_mod import TicTacToeEnv
from dist_classicrl.wrappers.flatten_multidiscrete_wrapper import (
    FlattenMultiDiscreteObservationsWrapper,
)


def make_env():
    env = TicTacToeEnv()
    return FlattenMultiDiscreteObservationsWrapper(env)


if __name__ == "__main__":
    num_agents = 10
    steps = int(100000 / num_agents)
    val_every_n_steps = int(1000 / num_agents)
    env = SyncVectorEnv(
        [make_env for _ in range(num_agents)], autoreset_mode=AutoresetMode.SAME_STEP
    )
    val_env = SyncVectorEnv([make_env for _ in range(1)], autoreset_mode=AutoresetMode.SAME_STEP)
    assert isinstance(env.single_action_space, spaces.Discrete)
    assert isinstance(env.single_observation_space, (spaces.Discrete, spaces.Dict))
    if isinstance(env.single_observation_space, spaces.Dict):
        assert "observation" in env.single_observation_space.spaces
        assert isinstance(env.single_observation_space.spaces["observation"], spaces.Discrete)
        agent = SingleThreadQLearning(
            state_size=env.single_observation_space.spaces["observation"].n,
            action_size=env.single_action_space.n,
            learning_rate=0.1,
            discount_factor=0.99,
            exploration_rate=1.0,
            exploration_decay=0.995,
            min_exploration_rate=0.01,
        )
    else:
        agent = SingleThreadQLearning(
            state_size=env.single_observation_space.n,
            action_size=env.single_action_space.n,
            learning_rate=0.1,
            discount_factor=0.99,
            exploration_rate=1.0,
            exploration_decay=0.995,
            min_exploration_rate=0.01,
        )

    agent.train(
        env,
        steps=steps,
        val_env=val_env,
        val_every_n_steps=val_every_n_steps,
        val_steps=None,
        val_episodes=10,
    )
