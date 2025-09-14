"""Test Q-Learning algorithm."""

from gymnasium import spaces
from gymnasium.vector import SyncVectorEnv
from gymnasium.vector.vector_env import AutoresetMode

from dist_classicrl.algorithms.runtime.parallel_runtime import ParallelQLearning
from dist_classicrl.environments.tiktaktoe_mod import TicTacToeEnv
from dist_classicrl.wrappers.flatten_multidiscrete_wrapper import (
    FlattenMultiDiscreteObservationsWrapper,
)


def make_env():
    env = TicTacToeEnv()
    return FlattenMultiDiscreteObservationsWrapper(env)


if __name__ == "__main__":
    num_agents = 10
    num_cores = 6
    steps = int(1000000 / (num_agents * num_cores))
    val_every_n_steps = int(1000 / (num_agents * num_cores))
    env = [
        SyncVectorEnv([make_env for _ in range(num_agents)], autoreset_mode=AutoresetMode.SAME_STEP)
        for _ in range(num_cores)
    ]
    val_env = SyncVectorEnv([make_env for _ in range(1)], autoreset_mode=AutoresetMode.SAME_STEP)
    assert isinstance(env[0].single_action_space, spaces.Discrete)
    assert isinstance(env[0].single_observation_space, spaces.Discrete) or isinstance(
        env[0].single_observation_space, spaces.Dict
    )
    if isinstance(env[0].single_observation_space, spaces.Dict):
        assert "observation" in env[0].single_observation_space.spaces
        assert isinstance(env[0].single_observation_space.spaces["observation"], spaces.Discrete)
        agent = ParallelQLearning(
            state_size=env[0].single_observation_space.spaces["observation"].n,
            action_size=env[0].single_action_space.n,
            learning_rate=0.01,
            discount_factor=0.99,
            exploration_rate=1.0,
            exploration_decay=0.995,
            min_exploration_rate=0.01,
        )
    else:
        agent = ParallelQLearning(
            state_size=env[0].single_observation_space.n,
            action_size=env[0].single_action_space.n,
            learning_rate=0.01,
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
