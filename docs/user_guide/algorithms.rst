===========
Algorithms
===========

This section provides detailed information about the reinforcement learning algorithms implemented in dist_classicrl.

Overview
========

dist_classicrl focuses on classical reinforcement learning algorithms with an emphasis on Q-Learning variants. All algorithms are designed with performance and scalability in mind, offering multiple execution modes from single-threaded development to distributed cluster deployment.

Q-Learning Implementations
===========================

List-Based Q-Learning
---------------------

An implementation that uses Python lists for Q-table storage, suitable for environments with sparse state spaces or when memory usage is a concern.

**Module**: ``dist_classicrl.algorithms.base_algorithms.q_learning_list``

**Key Features**:

- Memory-efficient for sparse state spaces
- Dynamic Q-table growth
- Compatible with all execution modes

**When to Use**:

- Large state spaces with sparse visitation
- Memory-constrained environments
- Exploratory phases of development

NumPy-Based Q-Learning
----------------------

A vectorized implementation using NumPy arrays for optimal performance with dense state spaces and batch operations.

**Module**: ``dist_classicrl.algorithms.base_algorithms.q_learning_numpy``

**Key Features**:

- Vectorized operations for maximum performance
- Batch update support
- Optimized for dense state spaces
- SIMD acceleration where available

**When to Use**:

- Dense state spaces
- High-frequency training
- Performance-critical applications

Base Q-Learning (Optimal)
--------------------------

The optimal Q-Learning implementation serves as the foundation for all subsequent variants. It provides the core Q-Learning algorithm and switches between iterative and vectorized implementations depending on the problem conditions.

**Module**: ``dist_classicrl.algorithms.base_algorithms.q_learning_optimal``

**Key Features**:

- Epsilon-greedy exploration strategy
- Configurable learning rate and discount factor
- Support for both deterministic and stochastic environments
- Memory-efficient Q-table management

**Usage**:

.. code-block:: python

    from dist_classicrl.algorithms.base_algorithms.q_learning_optimal import OptimalQLearning

    agent = OptimalQLearning(
        state_size=512,
        action_size=9,
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon=0.1,
        epsilon_decay=0.995
    )

Runtime Execution Modes
========================

Single-Threaded Execution
--------------------------

**Module**: ``dist_classicrl.algorithms.runtime.q_learning_single_thread``

The single-threaded mode is ideal for development, debugging, and smaller-scale problems.

**Features**:

- Full control over execution flow
- Easier debugging and profiling
- Deterministic behavior
- Lower resource overhead

**Example**:

.. code-block:: python

    from dist_classicrl.algorithms.runtime.q_learning_single_thread import SingleThreadQLearning
    from dist_classicrl.environments.tiktaktoe_mod import TicTacToeEnv

    env = TicTacToeEnv()
    agent = SingleThreadQLearning(
        state_size=512,
        action_size=9,
        learning_rate=0.1,
        discount_factor=0.99
    )

    # Train with progress tracking
    for episode in range(1000):
        agent.train_episode(env)
        if episode % 100 == 0:
            print(f"Episode {episode}, Epsilon: {agent.epsilon:.3f}")

Parallel Execution
------------------

**Module**: ``dist_classicrl.algorithms.runtime.q_learning_parallel``

Parallel execution uses Python's multiprocessing to train multiple agents simultaneously, aggregating their experiences.

**Features**:

- Multi-process training on single machine
- Near-linear scaling with CPU cores
- Shared memory optimization
- Automatic load balancing

**Example**:

.. code-block:: python

    from dist_classicrl.algorithms.runtime.q_learning_parallel import ParallelQLearning
    from dist_classicrl.environments.tiktaktoe_mod import TicTacToeEnv

    # Create environment factory
    def env_factory():
        return TicTacToeEnv()

    agent = ParallelQLearning(
        state_size=512,
        action_size=9,
        num_processes=4,  # Use 4 CPU cores
        learning_rate=0.1,
        discount_factor=0.99
    )

    # Train with multiple environments
    agent.train(
        env_factory=env_factory,
        total_steps=100000,
        steps_per_sync=1000  # Synchronize every 1000 steps
    )

Distributed Execution (MPI)
----------------------------

**Module**: ``dist_classicrl.algorithms.runtime.q_learning_async_dist``

Distributed execution uses MPI (Message Passing Interface) for training across multiple machines in a cluster.

**Features**:

- Asynchronous parameter updates
- Fault tolerance
- Scalable to hundreds of nodes
- Efficient communication patterns

**Example**:

.. code-block:: python

    # train_distributed.py
    from dist_classicrl.algorithms.runtime.q_learning_async_dist import DistAsyncQLearning
    from dist_classicrl.environments.tiktaktoe_mod import TicTacToeEnv

    agent = DistAsyncQLearning(
        state_size=512,
        action_size=9,
        learning_rate=0.1,
        discount_factor=0.99,
        batch_size=32,
        sync_frequency=100  # Sync every 100 updates
    )

    env = TicTacToeEnv()
    agent.train(env=env, steps=1000000)

**Running**:

.. code-block:: bash

    # Local cluster
    mpirun -n 4 python train_distributed.py

    # SLURM cluster
    srun -n 16 --mpi=pmix python train_distributed.py

Algorithm Parameters
====================

Common Parameters
-----------------

All Q-Learning implementations share these core parameters:

- **state_size** (int): Number of possible states in the environment
- **action_size** (int): Number of possible actions
- **learning_rate** (float, default=0.1): How quickly the agent learns from new experiences
- **discount_factor** (float, default=0.99): Importance of future rewards vs immediate rewards
- **epsilon** (float, default=0.1): Exploration rate for epsilon-greedy policy
- **epsilon_decay** (float, default=0.995): Decay rate for epsilon over time
- **epsilon_min** (float, default=0.01): Minimum exploration rate

Execution-Specific Parameters
------------------------------

**Parallel Execution**:

- **num_processes** (int): Number of parallel processes to spawn
- **steps_per_sync** (int): Frequency of Q-table synchronization
- **shared_memory** (bool): Whether to use shared memory optimization

**Distributed Execution**:

- **batch_size** (int): Size of batches for parameter updates
- **sync_frequency** (int): Frequency of parameter synchronization
- **compression** (bool): Whether to compress communication
- **async_updates** (bool): Enable asynchronous parameter updates

Performance Considerations
===========================

Algorithm Selection
--------------------

Choose your algorithm implementation based on your use case:

.. list-table::
   :header-rows: 1
   :widths: 20 25 25 30

   * - Implementation
     - Best For
     - Memory Usage
     - Performance
   * - List-based
     - Sparse state spaces
     - Low to Medium
     - Good
   * - NumPy-based
     - Dense state spaces
     - Medium to High
     - Excellent
   * - Optimal
     - General purpose
     - Medium
     - Very Good

Execution Mode Selection
------------------------

Choose your execution mode based on available resources and requirements:

.. list-table::
   :header-rows: 1
   :widths: 20 25 25 30

   * - Execution Mode
     - Best For
     - Setup Complexity
     - Scalability
   * - Single-threaded
     - Development, debugging
     - None
     - Limited
   * - Parallel
     - Single machine scaling
     - Low
     - Good (up to CPU cores)
   * - Distributed
     - Cluster deployment
     - High
     - Excellent

Hyperparameter Tuning
======================

Learning Rate
-------------

The learning rate controls how quickly the agent updates its Q-values:

- **High learning rate (0.5-1.0)**: Fast learning, may be unstable
- **Medium learning rate (0.1-0.3)**: Balanced approach, good default
- **Low learning rate (0.01-0.05)**: Stable learning, slower convergence

**Recommendation**: Start with 0.1 and adjust based on convergence behavior.

Discount Factor
---------------

The discount factor determines the importance of future vs immediate rewards:

- **High discount (0.95-0.99)**: Values long-term rewards
- **Medium discount (0.8-0.9)**: Balanced time horizon
- **Low discount (0.5-0.7)**: Focuses on immediate rewards

**Recommendation**: Use 0.99 for most applications unless the environment has natural episode boundaries.

Exploration Strategy
--------------------

Epsilon-greedy exploration parameters:

- **Initial epsilon (0.1-1.0)**: Start with high exploration
- **Epsilon decay (0.99-0.9999)**: Gradually reduce exploration
- **Minimum epsilon (0.01-0.1)**: Maintain some exploration

**Recommendation**: Start with epsilon=1.0, decay=0.995, min=0.01 for most environments.

Advanced Topics
===============

Custom Algorithm Development
----------------------------

To implement a custom algorithm, inherit from the base classes:

.. code-block:: python

    from dist_classicrl.algorithms.base_algorithms.base_algorithm import BaseAlgorithm

    class MyCustomAlgorithm(BaseAlgorithm):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            # Custom initialization

        def update(self, state, action, reward, next_state, done):
            # Custom update logic
            pass

        def select_action(self, state):
            # Custom action selection
            pass

Algorithm Composition
---------------------

Combine multiple algorithms or add custom components:

.. code-block:: python

    from dist_classicrl.algorithms.runtime.q_learning_single_thread import SingleThreadQLearning

    class HybridAgent:
        def __init__(self):
            self.q_learning = SingleThreadQLearning(...)
            self.custom_component = MyCustomComponent()

        def train_step(self, experience):
            # Use both components
            self.q_learning.update(*experience)
            self.custom_component.process(experience)

See Also
========

- :doc:`environments`: Learn about supported environments
- :doc:`performance`: Performance optimization techniques
- :doc:`distributed`: Detailed guide to distributed training
- :doc:`../development/architecture`: Internal architecture details
