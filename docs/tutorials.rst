=========
Tutorials
=========

This section provides step-by-step tutorials for getting started with dist_classicrl.
Each tutorial builds upon previous concepts and includes complete, runnable examples.

.. contents:: Tutorial Contents
   :local:
   :depth: 2

Tutorial 1: Basic Q-Learning
=============================

Let's start with a simple Q-learning example using the built-in TicTacToe environment.

**Goal**: Train an agent to play TicTacToe using single-threaded Q-learning.

**Prerequisites**: Basic understanding of reinforcement learning concepts.

Step 1: Import Required Modules
--------------------------------

.. code-block:: python

    from dist_classicrl.algorithms.runtime.q_learning_single_thread import SingleThreadQLearning
    from dist_classicrl.environments.tiktaktoe_mod import TicTacToeEnv
    import numpy as np

    # Set random seed for reproducibility
    np.random.seed(42)

Step 2: Create Environment and Agent
------------------------------------

.. code-block:: python

    # Create the TicTacToe environment
    env = TicTacToeEnv()

    # Create Q-learning agent
    agent = SingleThreadQLearning(
        state_size=512,      # 3^9 possible board states
        action_size=9,       # 9 possible moves (0-8)
        learning_rate=0.1,   # How fast the agent learns
        discount_factor=0.99, # Importance of future rewards
        exploration_rate=1.0, # Start with full exploration
        exploration_decay=0.995, # Gradually reduce exploration
        min_exploration_rate=0.01 # Minimum exploration to maintain
    )

Step 3: Train the Agent
-----------------------

.. code-block:: python

    # Train for 10,000 steps
    print("Starting training...")

    train_rewards, val_rewards = agent.train(
        env=env,
        steps=10000,
        val_env=env,  # Use same environment for validation
        val_every_n_steps=1000,  # Validate every 1000 steps
        val_episodes=100  # Run 100 episodes for each validation
    )

    print(f"Training completed!")
    print(f"Final training reward: {train_rewards[-1]:.3f}")
    print(f"Final validation reward: {val_rewards[-1]:.3f}")

Step 4: Test the Trained Agent
------------------------------

.. code-block:: python

    # Test the agent's performance
    test_rewards = []
    for episode in range(10):
        obs, info = env.reset()
        total_reward = 0
        terminated = False

        while not terminated:
            # Use deterministic policy (no exploration)
            action = agent.choose_action(obs["observation"], deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if terminated or truncated:
                break

        test_rewards.append(total_reward)
        print(f"Episode {episode + 1}: Reward = {total_reward}")

    print(f"Average test reward: {np.mean(test_rewards):.3f}")

**Expected Output**: The agent should learn to play TicTacToe reasonably well, with rewards improving over time.

Tutorial 2: Parallel Training
==============================

Now let's scale up training using multiple parallel environments.

**Goal**: Speed up training using multiprocessing with multiple environments.

Step 1: Setup Parallel Environments
-----------------------------------

.. code-block:: python

    from dist_classicrl.algorithms.runtime.q_learning_parallel import ParallelQLearning
    from dist_classicrl.environments.tiktaktoe_mod import TicTacToeEnv

    # Create multiple environment factories
    def make_env():
        return TicTacToeEnv()

    # Create list of environment factories for parallel training
    num_parallel_envs = 4
    envs = [make_env for _ in range(num_parallel_envs)]

    print(f"Created {num_parallel_envs} parallel environments")

Step 2: Create Parallel Agent
-----------------------------

.. code-block:: python

    # Create parallel Q-learning agent
    parallel_agent = ParallelQLearning(
        state_size=512,
        action_size=9,
        learning_rate=0.1,
        discount_factor=0.99,
        exploration_rate=1.0,
        exploration_decay=0.995,
        min_exploration_rate=0.01
    )

Step 3: Train with Parallel Environments
----------------------------------------

.. code-block:: python

    print("Starting parallel training...")

    # Train with parallel environments
    train_rewards, val_rewards = parallel_agent.train(
        envs=envs,  # List of environment factories
        steps=50000,  # More steps since we have more environments
        val_env=make_env(),  # Single environment for validation
        val_every_n_steps=5000,
        val_episodes=100
    )

    print("Parallel training completed!")
    print(f"Final validation reward: {val_rewards[-1]:.3f}")

**Performance Tip**: Parallel training should be significantly faster than single-threaded training.

Tutorial 3: Distributed Training with MPI
==========================================

For large-scale training, we can distribute across multiple nodes using MPI.

**Goal**: Scale training across multiple machines or nodes.

**Prerequisites**: MPI installed (see :doc:`installation` for setup instructions).

Step 1: Create Distributed Training Script
------------------------------------------

Save this as ``train_distributed.py``:

.. code-block:: python

    from dist_classicrl.algorithms.runtime.q_learning_async_dist import DistAsyncQLearning
    from dist_classicrl.environments.tiktaktoe_mod import TicTacToeEnv
    from mpi4py import MPI

    def main():
        # MPI setup is handled automatically by DistAsyncQLearning
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        print(f"Node {rank}/{size} starting...")

        # Create distributed agent
        dist_agent = DistAsyncQLearning(
            state_size=512,
            action_size=9,
            learning_rate=0.1,
            discount_factor=0.99,
            exploration_rate=1.0,
            exploration_decay=0.995,
            min_exploration_rate=0.01
        )

        # Create environments
        env = TicTacToeEnv()
        val_env = TicTacToeEnv()

        # Train distributedly
        # Only master node (rank 0) will return training history
        train_rewards, val_rewards = dist_agent.train(
            env=env,
            steps=100000,
            val_env=val_env,
            val_every_n_steps=10000,
            val_episodes=100,
            batch_size=32  # Synchronization batch size
        )

        if rank == 0:  # Master node
            print("Distributed training completed!")
            print(f"Final validation reward: {val_rewards[-1]:.3f}")

    if __name__ == "__main__":
        main()

Step 2: Run Distributed Training
--------------------------------

.. code-block:: bash

    # Run on 4 processes
    mpirun -n 4 python train_distributed.py

    # Run on cluster (example with SLURM)
    # srun --mpi=pmix -n 16 python train_distributed.py

**Note**: The master node (rank 0) coordinates training while worker nodes run environments.

Tutorial 4: Custom Environment
===============================

Learn how to create your own environment for use with dist_classicrl.

**Goal**: Implement a simple custom environment (GridWorld).

Step 1: Define Custom Environment
---------------------------------

.. code-block:: python

    import numpy as np
    import gymnasium as gym
    from dist_classicrl.environments.custom_env import DistClassicRLEnv

    class GridWorldEnv(DistClassicRLEnv):
        """Simple 4x4 grid world with goal at bottom-right corner."""

        def __init__(self, size=4):
            super().__init__()
            self.size = size
            self.num_agents = 1

            # Define action and observation spaces
            self.action_space = gym.spaces.Discrete(4)  # Up, Down, Left, Right
            self.observation_space = gym.spaces.Discrete(size * size)

            # Goal position
            self.goal_pos = (size - 1, size - 1)
            self.reset()

        def reset(self, seed=None, options=None):
            """Reset environment to initial state."""
            if seed is not None:
                np.random.seed(seed)

            # Start at top-left corner
            self.agent_pos = (0, 0)
            return self._get_obs(), {}

        def step(self, actions):
            """Execute one step in the environment."""
            action = actions[0] if isinstance(actions, (list, np.ndarray)) else actions

            # Move agent
            row, col = self.agent_pos
            if action == 0:  # Up
                row = max(0, row - 1)
            elif action == 1:  # Down
                row = min(self.size - 1, row + 1)
            elif action == 2:  # Left
                col = max(0, col - 1)
            elif action == 3:  # Right
                col = min(self.size - 1, col + 1)

            self.agent_pos = (row, col)

            # Calculate reward
            if self.agent_pos == self.goal_pos:
                reward = 1.0
                terminated = True
            else:
                reward = -0.01  # Small penalty for each step
                terminated = False

            return (
                self._get_obs(),
                np.array([reward], dtype=np.float32),
                np.array([terminated], dtype=bool),
                np.array([False], dtype=bool),  # truncated
                [{}]  # info
            )

        def _get_obs(self):
            """Convert 2D position to 1D observation."""
            return np.array([self.agent_pos[0] * self.size + self.agent_pos[1]], dtype=np.int32)

Step 2: Train Agent on Custom Environment
-----------------------------------------

.. code-block:: python

    # Create custom environment
    env = GridWorldEnv(size=4)

    # Create agent
    agent = SingleThreadQLearning(
        state_size=16,  # 4x4 = 16 states
        action_size=4,  # 4 actions
        learning_rate=0.1,
        discount_factor=0.95,
        exploration_rate=1.0,
        exploration_decay=0.995,
        min_exploration_rate=0.1
    )

    # Train
    print("Training on custom GridWorld environment...")
    train_rewards, val_rewards = agent.train(
        env=env,
        steps=5000,
        val_env=GridWorldEnv(size=4),
        val_every_n_steps=1000,
        val_episodes=50
    )

    print(f"Training completed! Final reward: {val_rewards[-1]:.3f}")

Step 3: Visualize Agent's Policy
--------------------------------

.. code-block:: python

    def visualize_policy(agent, env_size=4):
        """Visualize the learned policy."""
        print("Learned Policy (↑↓←→):")
        actions_symbols = ['↑', '↓', '←', '→']

        for row in range(env_size):
            for col in range(env_size):
                state = row * env_size + col
                action = agent.choose_action(state, deterministic=True)
                print(f"{actions_symbols[action]} ", end="")
            print()  # New line

    visualize_policy(agent)

**Expected Output**: The agent should learn to navigate toward the goal efficiently.

Tutorial 5: Performance Optimization
=====================================

Learn how to optimize training performance for large-scale problems.

**Goal**: Understand performance considerations and optimization techniques.

Performance Monitoring
----------------------

.. code-block:: python

    import time

    def benchmark_training(agent_class, env_factory, steps=10000):
        """Benchmark training performance."""
        env = env_factory()
        agent = agent_class(state_size=512, action_size=9)

        start_time = time.time()
        agent.train(env=env, steps=steps)
        end_time = time.time()

        elapsed = end_time - start_time
        steps_per_second = steps / elapsed

        print(f"{agent_class.__name__}:")
        print(f"  Time: {elapsed:.2f} seconds")
        print(f"  Performance: {steps_per_second:.1f} steps/second")
        return steps_per_second

    # Compare different implementations
    single_perf = benchmark_training(SingleThreadQLearning, TicTacToeEnv)
    # parallel_perf = benchmark_training(ParallelQLearning, TicTacToeEnv)

    # print(f"Parallel speedup: {parallel_perf / single_perf:.2f}x")

Memory Optimization
------------------

.. code-block:: python

    import psutil
    import os

    def monitor_memory_usage():
        """Monitor memory usage during training."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # MB

    # Example: Monitor memory during training
    print(f"Initial memory: {monitor_memory_usage():.1f} MB")

    # Train with memory monitoring
    agent = SingleThreadQLearning(state_size=1000, action_size=10)
    print(f"After agent creation: {monitor_memory_usage():.1f} MB")

Next Steps
==========

After completing these tutorials, you should be able to:

✅ Train basic Q-learning agents
✅ Use parallel training for speedup
✅ Deploy distributed training with MPI
✅ Create custom environments
✅ Optimize performance

**What's Next?**

- Explore :doc:`user_guide/algorithms` for advanced algorithm details
- Check :doc:`user_guide/performance` for detailed optimization guides
- See :doc:`autoapi/index` for complete API documentation
- Read :doc:`user_guide/distributed` for advanced distributed training techniques

**Need Help?**

- Check the :doc:`../README` for troubleshooting
- Look at the test files in the repository for more examples
- Open an issue on GitHub if you encounter problems
