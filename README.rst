.. image:: https://img.shields.io/pypi/v/dist_classicrl.svg
    :alt: PyPI-Server
    :target: https://pypi.org/project/dist_classicrl/
.. image:: https://img.shields.io/badge/license-MIT-blue.svg
    :alt: License
    :target: https://github.com/j-moralejo-pinas/dist_classicrl/blob/main/LICENSE.txt
.. image:: https://img.shields.io/badge/python-3.8+-blue.svg
    :alt: Python Version
    :target: https://www.python.org/downloads/

|

==============
dist_classicrl
==============

A Python package for distributed classical reinforcement learning algorithms.

**dist_classicrl** provides high-performance, scalable implementations of classic reinforcement learning algorithms with support for distributed training. The library focuses on Q-Learning with multiple execution modes: single-threaded, parallel (multiprocessing), and distributed (MPI) training.

**Key Features:**

- 🚀 **Multiple Execution Modes**: Single-threaded, parallel, and MPI-distributed training
- ⚡ **High Performance**: Optimized implementations with vectorized operations and performance benchmarking
- 🎮 **Multi-Agent Support**: Built-in support for multi-agent environments
- 🔧 **Flexible Architecture**: Abstract base classes for easy extension and custom environments
- 🌐 **Standards Compliant**: Compatible with Gymnasium and PettingZoo environments (coming soon)


Installation
============

Quick Start
-----------

Install dist_classicrl using pip:

.. code-block:: bash

    pip install dist_classicrl

From Source
-----------

For development or to get the latest features:

.. code-block:: bash

    git clone https://github.com/j-moralejo-pinas/dist_classicrl.git
    cd dist_classicrl
    pip install -e .

MPI Support (Optional)
----------------------

For distributed training with MPI, install mpi4py:

.. code-block:: bash

    # Ubuntu/Debian
    sudo apt-get install libopenmpi-dev
    pip install mpi4py

    # macOS with Homebrew
    brew install open-mpi
    pip install mpi4py

    # Conda
    conda install -c conda-forge mpi4py

Usage
=====

Basic Q-Learning
----------------

Single-threaded Q-Learning for simple environments:

.. code-block:: python

    from dist_classicrl.algorithms.runtime.q_learning_single_thread import SingleThreadQLearning
    from dist_classicrl.environments.tiktaktoe_mod import TicTacToeEnv

    # Create environment and agent
    env = TicTacToeEnv()
    agent = SingleThreadQLearning(
        state_size=512,  # 3^9 possible board states
        action_size=9,   # 9 possible moves
        learning_rate=0.1,
        discount_factor=0.99,
        exploration_rate=1.0,
        exploration_decay=0.995
    )

    # Train the agent
    agent.train(
        env=env,
        steps=10000,
        val_env=env,
        val_every_n_steps=1000,
        val_episodes=100
    )

Parallel Q-Learning
-------------------

Multi-process Q-Learning for faster training:

.. code-block:: python

    from dist_classicrl.algorithms.runtime.q_learning_parallel import ParallelQLearning
    from gymnasium.vector import AsyncVectorEnv

    # Create multiple environments
    def make_env():
        return TicTacToeEnv()

    envs = [make_env for _ in range(4)]  # 4 parallel environments

    # Create parallel agent
    agent = ParallelQLearning(
        state_size=512,
        action_size=9,
        learning_rate=0.1,
        discount_factor=0.99
    )

    # Train with parallel environments
    agent.train(
        envs=envs,
        steps=50000,
        val_env=make_env(),
        val_every_n_steps=5000,
        val_episodes=100
    )

Distributed Q-Learning with MPI
--------------------------------

Scale training across multiple nodes:

.. code-block:: python

    # Save as train_distributed.py
    from dist_classicrl.algorithms.runtime.q_learning_async_dist import DistAsyncQLearning

    agent = DistAsyncQLearning(
        state_size=512,
        action_size=9,
        learning_rate=0.1,
        discount_factor=0.99
    )

    # This will automatically handle master/worker coordination
    agent.train(
        env=TicTacToeEnv(),
        steps=100000,
        val_env=TicTacToeEnv(),
        val_every_n_steps=10000,
        val_episodes=100,
        batch_size=32
    )

Run with MPI:

.. code-block:: bash

    mpirun -n 4 python train_distributed.py

Custom Environments
-------------------

Create your own environment by inheriting from the base class:

.. code-block:: python

    import numpy as np
    from dist_classicrl.environments.custom_env import DistClassicRLEnv

    class MyCustomEnv(DistClassicRLEnv):
        def __init__(self):
            super().__init__()
            self.num_agents = 1
            self.action_space = gym.spaces.Discrete(4)
            self.observation_space = gym.spaces.Discrete(16)

        def step(self, actions):
            # Implement your environment logic
            next_state = self._get_next_state(actions)
            rewards = self._calculate_rewards(actions)
            terminated = self._check_termination()
            truncated = np.array([False])
            infos = [{}]

            return next_state, rewards, terminated, truncated, infos

        def reset(self, seed=None, options=None):
            # Reset environment to initial state
            return self._get_initial_state(), {}

Library Structure
=================

**Algorithms:**

- ``algorithms.base_algorithms``: Core Q-Learning implementations with different optimizations
- ``algorithms.runtime``: Execution modes (single-thread, parallel, distributed)
- ``algorithms.buffers``: Experience replay and buffer management (future feature)

**Environments:**

- ``environments.custom_env``: Abstract base class for custom environments
- ``environments.tiktaktoe_mod``: TicTacToe environment for testing and demos

**Utilities:**

- ``utils``: Helper functions for multi-discrete action spaces
- ``wrappers``: Environment wrappers for action/observation space transformations

Performance
===========

The library includes comprehensive performance benchmarking. Different algorithm implementations are optimized for various scenarios:

- **Small action spaces (< 100)**: Iterative methods perform best
- **Medium action spaces (100-1000)**: Vectorized methods show improvements
- **Large action spaces (> 1000)**: Fully vectorized implementations provide significant speedups
- **Multi-agent scenarios**: Parallel and distributed training scale effectively

Run performance tests:

.. code-block:: bash

    cd dev_tests
    python perf_test.py

Testing
=======

Run the complete test suite:

.. code-block:: bash

    pytest tests/

For runtime-specific tests:

.. code-block:: bash

    # Single-threaded and parallel tests
    pytest tests/dist_classicrl/algorithms/runtime/

    # MPI distributed tests (requires MPI)
    mpirun -n 3 python -m pytest tests/dist_classicrl/algorithms/runtime/test_q_learning_async_dist.py::TestDistAsyncQLearningMPI

Or use the provided test runner:

.. code-block:: bash

    bash tests/dist_classicrl/algorithms/runtime/run_runtime_tests.sh

Contributing
============

We welcome contributions! Please see our `Contributing Guide <CONTRIBUTING.rst>`_ for details.

**Quick Setup for Contributors:**

.. code-block:: bash

    git clone https://github.com/j-moralejo-pinas/dist_classicrl.git
    cd dist_classicrl
    pip install -e ".[dev]"
    pre-commit install

**Development Workflow:**

1. Install pre-commit hooks (handles linting, formatting)
2. Write your code and tests
3. Run tests: ``pytest tests/``
4. Submit a pull request

License
=======

This project is licensed under the MIT License - see the `LICENSE.txt <LICENSE.txt>`_ file for details.

Acknowledgments
===============

- Inspired by classical reinforcement learning research
- Performance optimization techniques from high-performance computing literature

Roadmap
=======

**Upcoming Features:**

- Experience replay buffers
- Additional RL algorithms (SARSA, Expected SARSA)
- Improved stability for large-scale distributed training

**Known Issues:**

- Large numbers of vectorized environments may cause training instability

For detailed version history, see `CHANGELOG.rst <CHANGELOG.rst>`_.
