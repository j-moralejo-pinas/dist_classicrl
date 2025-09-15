"""
Distributed Classic Reinforcement Learning Library.

This package provides high-performance, scalable implementations of classic reinforcement
learning algorithms with support for distributed training. The library focuses on Q-Learning
with multiple execution modes: single-threaded, parallel (multiprocessing), and distributed (MPI).

Key Features
------------
- Multiple execution modes: single-threaded, parallel, and MPI-distributed training
- High performance: optimized implementations with vectorized operations
- Multi-agent support: built-in support for multi-agent environments
- Flexible architecture: abstract base classes for easy extension
- Comprehensive testing: extensive test suite with performance profiling
- Standards compliant: compatible with Gymnasium and PettingZoo environments

Modules
-------
algorithms
    Core reinforcement learning algorithm implementations:

    * ``base_algorithms`` - Core Q-Learning implementations with optimizations
    * ``runtime`` - Execution modes (single-thread, parallel, distributed)
    * ``buffers`` - Experience replay and buffer management

environments
    Environment implementations and interfaces:

    * ``custom_env`` - Abstract base class for custom environments
    * ``tiktaktoe_mod`` - TicTacToe environment for testing and demos

wrappers
    Environment wrappers for action/observation space transformations.

utils
    Utility functions for multi-discrete action spaces.

Examples
--------
Basic Q-Learning:
    >>> from dist_classicrl.algorithms.runtime.q_learning_single_thread import SingleThreadQLearning
    >>> from dist_classicrl.environments.tiktaktoe_mod import TicTacToeEnv
    >>> env = TicTacToeEnv()
    >>> agent = SingleThreadQLearning(state_size=512, action_size=9)
    >>> agent.train(env=env, steps=10000)

Parallel Training:
    >>> from dist_classicrl.algorithms.runtime.q_learning_parallel import ParallelQLearning
    >>> agent = ParallelQLearning(state_size=512, action_size=9)
    >>> envs = [lambda: TicTacToeEnv() for _ in range(4)]
    >>> agent.train(envs=envs, steps=50000)

Distributed Training (requires MPI):
    >>> # Run with: mpirun -n 4 python script.py
    >>> from dist_classicrl.algorithms.runtime.q_learning_async_dist import DistAsyncQLearning
    >>> agent = DistAsyncQLearning(state_size=512, action_size=9)
    >>> agent.train(env=TicTacToeEnv(), steps=100000)
"""

try:
    from ._version import version as __version__
except ImportError:
    # 2) Fallback to installed package metadata
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:
        # Very old Python or exotic environment
        __version__ = "0.0.0"
    else:
        try:
            __version__ = version("dist_classicrl")
        except PackageNotFoundError:
            __version__ = "0.0.0"
        finally:
            # Avoid leaking names into the package namespace
            del version
            del PackageNotFoundError

