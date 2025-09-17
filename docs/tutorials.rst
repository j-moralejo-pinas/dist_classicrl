========
Tutorial
========

This tutorial shows you how to write simple scripts to launch Q-learning training using different runtime modes: single-thread, parallel, or distributed.

.. contents:: Contents
    :local:
    :depth: 2

Single-Thread Training
======================

The simplest way to run Q-learning training on a single thread.

**Script: train_single.py**

.. code-block:: python

    """Single-thread Q-learning training script."""
    from gymnasium.vector import SyncVectorEnv
    from gymnasium.vector.vector_env import AutoresetMode

    from dist_classicrl.algorithms.base_algorithms.q_learning_optimal import OptimalQLearningBase
    from dist_classicrl.algorithms.runtime.single_thread_runtime import SingleThreadQLearning
    from dist_classicrl.environments.tiktaktoe_mod import TicTacToeEnv
    from dist_classicrl.schedules.exponential_schedule import ExponentialSchedule
    from dist_classicrl.wrappers.flatten_multidiscrete_wrapper import (
        FlattenMultiDiscreteObservationsWrapper,
    )


    def make_env():
        """Create a TicTacToe environment with wrapper."""
        env = TicTacToeEnv()
        return FlattenMultiDiscreteObservationsWrapper(env)


    def main():
        # Create vectorized environments
        num_agents = 4
        env = SyncVectorEnv(
            [make_env for _ in range(num_agents)],
            autoreset_mode=AutoresetMode.SAME_STEP
        )
        val_env = SyncVectorEnv([make_env], autoreset_mode=AutoresetMode.SAME_STEP)

        # Get environment dimensions
        state_size = env.single_observation_space.spaces["observation"].n
        action_size = env.single_action_space.n

        # Create Q-learning algorithm
        algo = OptimalQLearningBase(
            state_size=state_size,
            action_size=action_size,
            discount_factor=0.99,
        )

        # Create schedules
        lr_schedule = ExponentialSchedule(value=0.1, min_value=1e-5, decay_rate=0.995)
        exploration_schedule = ExponentialSchedule(value=1.0, min_value=0.01, decay_rate=0.995)

        # Create single-thread agent
        agent = SingleThreadQLearning(algo, lr_schedule, exploration_schedule)

        print("Starting single-thread training...")

        # Train the agent
        agent.train(
            env=env,
            steps=10000,
            val_env=val_env,
            val_every_n_steps=2000,
            val_episodes=10
        )

        print("Training completed!")


    if __name__ == "__main__":
        main()

**Run the script:**

.. code-block:: bash

    python train_single.py


Parallel Training
=================

Use multiple processes to speed up training with parallel environments.

**Script: train_parallel.py**

.. code-block:: python

    """Parallel Q-learning training script."""
    from gymnasium.vector import SyncVectorEnv
    from gymnasium.vector.vector_env import AutoresetMode

    from dist_classicrl.algorithms.base_algorithms.q_learning_optimal import OptimalQLearningBase
    from dist_classicrl.algorithms.runtime.parallel_runtime import ParallelQLearning
    from dist_classicrl.environments.tiktaktoe_mod import TicTacToeEnv
    from dist_classicrl.schedules.exponential_schedule import ExponentialSchedule
    from dist_classicrl.wrappers.flatten_multidiscrete_wrapper import (
        FlattenMultiDiscreteObservationsWrapper,
    )


    def make_env():
        """Create a TicTacToe environment with wrapper."""
        env = TicTacToeEnv()
        return FlattenMultiDiscreteObservationsWrapper(env)


    def main():
        # Create multiple vectorized environments for parallel training
        num_agents_per_process = 4
        num_processes = 3

        # Create list of environments for parallel processes
        envs = [
            SyncVectorEnv(
                [make_env for _ in range(num_agents_per_process)],
                autoreset_mode=AutoresetMode.SAME_STEP
            )
            for _ in range(num_processes)
        ]

        val_env = SyncVectorEnv([make_env], autoreset_mode=AutoresetMode.SAME_STEP)

        # Get environment dimensions from the first environment
        state_size = envs[0].single_observation_space.spaces["observation"].n
        action_size = envs[0].single_action_space.n

        # Create Q-learning algorithm
        algo = OptimalQLearningBase(
            state_size=state_size,
            action_size=action_size,
            discount_factor=0.99,
        )

        # Create schedules
        lr_schedule = ExponentialSchedule(value=0.1, min_value=1e-5, decay_rate=0.995)
        exploration_schedule = ExponentialSchedule(value=1.0, min_value=0.01, decay_rate=0.995)

        # Create parallel agent
        agent = ParallelQLearning(algo, lr_schedule, exploration_schedule)

        print(f"Starting parallel training with {num_processes} processes...")
        print(f"Total agents: {num_agents_per_process * num_processes}")

        # Train the agent
        agent.train(
            env=envs,  # List of environments for parallel training
            steps=10000,
            val_env=val_env,
            val_every_n_steps=2000,
            val_episodes=10
        )

        print("Parallel training completed!")


    if __name__ == "__main__":
        main()

**Run the script:**

.. code-block:: bash

    python train_parallel.py


Distributed Training
====================

Scale training across multiple nodes using MPI for distributed computing.

**Script: train_distributed.py**

.. code-block:: python

    """Distributed Q-learning training script using MPI."""
    from gymnasium.vector import SyncVectorEnv
    from gymnasium.vector.vector_env import AutoresetMode
    from mpi4py import MPI

    from dist_classicrl.algorithms.base_algorithms.q_learning_optimal import OptimalQLearningBase
    from dist_classicrl.algorithms.runtime.q_learning_async_dist import DistAsyncQLearning
    from dist_classicrl.environments.tiktaktoe_mod import TicTacToeEnv
    from dist_classicrl.schedules.exponential_schedule import ExponentialSchedule
    from dist_classicrl.wrappers.flatten_multidiscrete_wrapper import (
        FlattenMultiDiscreteObservationsWrapper,
    )


    def make_env():
        """Create a TicTacToe environment with wrapper."""
        env = TicTacToeEnv()
        return FlattenMultiDiscreteObservationsWrapper(env)


    def main():
        # MPI setup
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        print(f"Process {rank}/{size} starting...")

        # Create vectorized environments
        num_agents = 4
        env = SyncVectorEnv(
            [make_env for _ in range(num_agents)],
            autoreset_mode=AutoresetMode.SAME_STEP
        )
        val_env = SyncVectorEnv([make_env], autoreset_mode=AutoresetMode.SAME_STEP)

        # Get environment dimensions
        state_size = env.single_observation_space.spaces["observation"].n
        action_size = env.single_action_space.n

        # Create Q-learning algorithm
        algo = OptimalQLearningBase(
            state_size=state_size,
            action_size=action_size,
            discount_factor=0.99,
        )

        # Create schedules
        lr_schedule = ExponentialSchedule(value=0.1, min_value=1e-5, decay_rate=0.995)
        exploration_schedule = ExponentialSchedule(value=1.0, min_value=0.01, decay_rate=0.995)

        # Create distributed agent
        agent = DistAsyncQLearning(algo, lr_schedule, exploration_schedule)

        if rank == 0:
            print(f"Starting distributed training with {size} MPI processes...")
            print(f"Agents per process: {num_agents}")

        # Train the agent (distributed across all processes)
        agent.train(
            env=env,
            steps=20000,
            val_env=val_env,
            val_every_n_steps=4000,
            val_episodes=10
        )

        if rank == 0:
            print("Distributed training completed!")


    if __name__ == "__main__":
        main()

**Run the script:**

.. code-block:: bash

    # Run with 4 MPI processes
    mpirun -n 4 python train_distributed.py

    # Run on SLURM cluster (example)
    # srun --mpi=pmix -n 8 python train_distributed.py


Key Points
==========

**Environment Setup**: All examples use vectorized environments with the TicTacToe game and a flattening wrapper.

**Algorithm Configuration**: The Q-learning algorithm is configured with:

- Exponential learning rate schedule (starts at 0.1, decays to 1e-5)
- Exponential exploration schedule (starts at 1.0, decays to 0.01)
- Discount factor of 0.99

**Runtime Modes**:

- **Single-thread**: Uses one vectorized environment with multiple agents
- **Parallel**: Uses multiple vectorized environments across multiple processes
- **Distributed**: Uses MPI to coordinate training across multiple nodes

**Scaling**: Adjust ``num_agents``, ``num_processes``, or MPI ranks to scale training based on your computational resources.

See Also
========

- `Installation <docs/installation.rst>`_ - Installation instructions including MPI setup
- `Contributing <CONTRIBUTING.rst>`_ - Guidelines for contributing to the project
