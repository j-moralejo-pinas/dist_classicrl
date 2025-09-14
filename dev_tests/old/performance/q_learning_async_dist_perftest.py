"""Test Q-Learning algorithm."""

import argparse

from gymnasium import spaces
from gymnasium.vector import SyncVectorEnv
from gymnasium.vector.vector_env import AutoresetMode

from dist_classicrl.algorithms.runtime.q_learning_async_dist import DistAsyncQLearning
from dist_classicrl.environments.tiktaktoe_mod import TicTacToeEnv
from dist_classicrl.wrappers.flatten_multidiscrete_wrapper import (
    FlattenMultiDiscreteObservationsWrapper,
)


def make_env():
    env = TicTacToeEnv()
    return FlattenMultiDiscreteObservationsWrapper(env)


if __name__ == "__main__":
    # Add command line argument parsing using argparse

    parser = argparse.ArgumentParser(description="Test DistAsyncQLearning on TicTacToeEnv.")

    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of parallel workers to use for the environment.",
    )
    parser.add_argument(
        "--num-agents",
        type=int,
        default=10,
        help="Number of agents to simulate in the environment.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training.",
    )

    num_agents = parser.parse_args().num_agents
    num_cores = parser.parse_args().num_workers
    batch_size = parser.parse_args().batch_size
    steps = int(1000000 / (num_agents * num_cores))
    val_every_n_steps = int(1000 / (num_agents * num_cores))
    env = SyncVectorEnv(
        [make_env for _ in range(num_agents)], autoreset_mode=AutoresetMode.SAME_STEP
    )
    val_env = SyncVectorEnv([make_env for _ in range(1)], autoreset_mode=AutoresetMode.SAME_STEP)
    assert isinstance(env.single_action_space, spaces.Discrete)
    assert isinstance(env.single_observation_space, (spaces.Discrete, spaces.Dict))
    if isinstance(env.single_observation_space, spaces.Dict):
        assert "observation" in env.single_observation_space.spaces
        assert isinstance(env.single_observation_space.spaces["observation"], spaces.Discrete)
        agent = DistAsyncQLearning(
            state_size=env.single_observation_space.spaces["observation"].n,
            action_size=env.single_action_space.n,
            learning_rate=0.01,
            discount_factor=0.99,
            exploration_rate=1.0,
            exploration_decay=0.995,
            min_exploration_rate=0.01,
        )
    else:
        agent = DistAsyncQLearning(
            state_size=env.single_observation_space.n,
            action_size=env.single_action_space.n,
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
        batch_size=batch_size,
    )
