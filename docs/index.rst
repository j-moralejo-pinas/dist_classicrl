==============
dist_classicrl
==============

.. image:: https://img.shields.io/pypi/v/dist_classicrl.svg
    :alt: PyPI-Server
    :target: https://pypi.org/project/dist_classicrl/
.. image:: https://img.shields.io/badge/license-MIT-blue.svg
    :alt: License
    :target: https://github.com/j-moralejo-pinas/dist_classicrl/blob/main/LICENSE.txt
.. image:: https://img.shields.io/badge/python-3.13+-blue.svg
    :alt: Python Version

**A high-performance Python package for distributed classical reinforcement learning**

dist_classicrl provides scalable implementations of classic reinforcement learning algorithms
with support for single-threaded, parallel (multiprocessing), and distributed (MPI) training.
The library focuses on Q-Learning with optimized vectorized operations and comprehensive
performance benchmarking.

Quick Start
===========

Install the package:

.. code-block:: bash

    pip install dist_classicrl

Basic Q-Learning example:

.. code-block:: python

    from dist_classicrl.algorithms.runtime.q_learning_single_thread import SingleThreadQLearning
    from dist_classicrl.environments.tiktaktoe_mod import TicTacToeEnv

    # Create environment and agent
    env = TicTacToeEnv()
    agent = SingleThreadQLearning(
        state_size=512,  # 3^9 possible board states
        action_size=9,   # 9 possible moves
        learning_rate=0.1,
        discount_factor=0.99
    )

    # Train the agent
    agent.train(env=env, steps=10000)

Key Features
============

🚀 **Multiple Execution Modes**
    - Single-threaded for development and debugging
    - Parallel multiprocessing for local scaling
    - MPI distributed training for cluster deployment

⚡ **High Performance**
    - Vectorized NumPy operations
    - Optimized algorithm implementations
    - Performance benchmarking and profiling

🎮 **Multi-Agent Support**
    - Built-in support for multi-agent environments
    - Compatible with Gymnasium and PettingZoo (coming soon)

🔧 **Flexible Architecture**
    - Abstract base classes for easy extension
    - Modular design for algorithm composition
    - Custom environment support

Documentation Sections
=======================
========

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   Overview <readme>
   Installation & Quick Start <installation>
   Tutorials <tutorials>
   Performance Benchmarks <benchmarks>

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   Algorithms <user_guide/algorithms>
   Environments <user_guide/environments>
   Performance & Benchmarking <user_guide/performance>
   Distributed Training <user_guide/distributed>

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   API Documentation <autoapi/index>

.. toctree::
   :maxdepth: 2
   :caption: Development

   Contributing <contributing>
   Architecture <development/architecture>
   Testing <development/testing>

.. toctree::
   :maxdepth: 1
   :caption: Project Info

   License <license>
   Authors <authors>
   Changelog <changelog>


Examples
========

**Parallel Training:**

.. code-block:: python

    from dist_classicrl.algorithms.runtime.q_learning_parallel import ParallelQLearning

    # Create multiple environments for parallel training
    envs = [lambda: TicTacToeEnv() for _ in range(4)]

    agent = ParallelQLearning(state_size=512, action_size=9)
    agent.train(envs=envs, steps=50000)

**Distributed Training with MPI:**

.. code-block:: python

    # Save as train_distributed.py
    from dist_classicrl.algorithms.runtime.q_learning_async_dist import DistAsyncQLearning

    agent = DistAsyncQLearning(state_size=512, action_size=9)
    agent.train(env=TicTacToeEnv(), steps=100000, batch_size=32)

Run with MPI:

.. code-block:: bash

    mpirun -n 4 python train_distributed.py

**Custom Environment:**

.. code-block:: python

    import numpy as np
    from dist_classicrl.environments.custom_env import DistClassicRLEnv

    class MyEnv(DistClassicRLEnv):
        def __init__(self):
            super().__init__()
            self.num_agents = 1
            # Define action and observation spaces

        def step(self, actions):
            # Implement environment dynamics
            return next_state, rewards, terminated, truncated, infos

Performance Highlights
======================

The library includes extensive performance optimizations:

- **Vectorized Operations**: Up to 10x speedup for large action spaces
- **Memory Efficiency**: Optimized Q-table storage and access patterns
- **Parallel Scaling**: Near-linear speedup with multiple CPU cores
- **Distributed Scaling**: Efficient MPI communication patterns for large-scale training

Algorithm Implementations
=========================

**Q-Learning Variants:**

- **Optimal Q-Learning**: Base implementation with multiple optimization strategies
- **List-based Q-Learning**: List based Q-table for small state spaces
- **NumPy Q-Learning**: Vectorized Q-table using NumPy for large state spaces

**Execution Modes:**

- **Single-threaded**: ``q_learning_single_thread``
- **Parallel**: ``q_learning_parallel`` (multiprocessing)
- **Distributed**: ``q_learning_async_dist`` (MPI)

**Future Algorithms:**

- SARSA and Expected SARSA
- Deep Q-Learning integration
- Multi-agent coordination algorithms

Support and Community
=====================

- **GitHub Issues**: `Report bugs and request features <https://github.com/j-moralejo-pinas/dist_classicrl/issues>`_
- **Documentation**: You're reading it! 📖
- **Contributing**: See :doc:`contributing` for how to get involved

The dist_classicrl project welcomes contributions from the community. Whether you're fixing bugs,
adding features, improving documentation, or sharing your use cases, we'd love to hear from you!


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _toctree: https://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html
.. _reStructuredText: https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html
.. _references: https://www.sphinx-doc.org/en/stable/markup/inline.html
.. _Python domain syntax: https://www.sphinx-doc.org/en/master/usage/restructuredtext/domains.html#the-python-domain
.. _Sphinx: https://www.sphinx-doc.org/
.. _Python: https://docs.python.org/
.. _Numpy: https://numpy.org/doc/stable
.. _SciPy: https://docs.scipy.org/doc/scipy/reference/
.. _matplotlib: https://matplotlib.org/contents.html#
.. _Pandas: https://pandas.pydata.org/pandas-docs/stable
.. _Scikit-Learn: https://scikit-learn.org/stable
.. _autodoc: https://www.sphinx-doc.org/en/master/ext/autodoc.html
.. _Google style: https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings
.. _NumPy style: https://numpydoc.readthedocs.io/en/latest/format.html
.. _classical style: https://www.sphinx-doc.org/en/master/domains.html#info-field-lists
