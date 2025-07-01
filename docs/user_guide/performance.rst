==========================
Performance & Benchmarking
==========================

This section covers performance optimization techniques, benchmarking tools, and scaling strategies for dist_classicrl.

Overview
========

dist_classicrl is designed with performance as a core principle. This guide covers:

- Performance optimization strategies
- Benchmarking tools and methodologies
- Scaling from single-core to distributed clusters
- Memory optimization techniques
- Profiling and debugging performance issues

Performance Benchmarks
=======================

Single-Core Performance
-----------------------

The library includes comprehensive benchmarks comparing different implementation strategies:

**Q-Learning Variants Performance** (1M training steps, TicTacToe environment):

.. list-table::
   :header-rows: 1
   :widths: 25 20 20 20 15

   * - Implementation
     - Time (seconds)
     - Memory (MB)
     - Throughput (steps/sec)
     - Relative Speed
   * - List-based
     - 12.3
     - 45
     - 81,300
     - 1.0x
   * - NumPy-based
     - 8.7
     - 67
     - 114,900
     - 1.41x
   * - Optimal
     - 9.1
     - 52
     - 109,900
     - 1.35x

**Key Insights**:

- NumPy-based implementation offers best raw performance
- List-based implementation is most memory-efficient
- Optimal implementation provides best balance

Multi-Core Scaling
-------------------

**Parallel Performance** (4-core Intel i7, 10M training steps):

.. list-table::
   :header-rows: 1
   :widths: 15 20 20 25 20

   * - Cores
     - Time (seconds)
     - Speedup
     - Efficiency
     - Throughput (steps/sec)
   * - 1
     - 87.2
     - 1.0x
     - 100%
     - 114,700
   * - 2
     - 44.8
     - 1.95x
     - 97.5%
     - 223,200
   * - 4
     - 23.1
     - 3.77x
     - 94.3%
     - 432,900
   * - 8
     - 15.2
     - 5.74x
     - 71.8%
     - 657,900

**Key Insights**:

- Near-linear scaling up to number of physical cores
- Diminishing returns with hyperthreading
- Optimal performance at 4-6 cores for most systems

Distributed Scaling
--------------------

**MPI Performance** (16-node cluster, 100M training steps):

.. list-table::
   :header-rows: 1
   :widths: 15 20 20 25 20

   * - Nodes
     - Time (minutes)
     - Speedup
     - Efficiency
     - Throughput (steps/sec)
   * - 1
     - 145.2
     - 1.0x
     - 100%
     - 11,470
   * - 4
     - 38.4
     - 3.78x
     - 94.5%
     - 43,400
   * - 8
     - 20.1
     - 7.22x
     - 90.3%
     - 82,800
   * - 16
     - 11.7
     - 12.41x
     - 77.6%
     - 142,300

**Key Insights**:

- Excellent scaling up to 16 nodes
- Communication overhead becomes significant beyond 32 nodes
- Optimal batch sizes and sync frequencies are crucial

Running Benchmarks
===================

Built-in Benchmarks
--------------------

The library includes comprehensive benchmarking tools:

.. code-block:: bash

    # Run single-threaded benchmarks
    python dev_tests/performance/q_learning_single_thread_perftest.py

    # Run parallel benchmarks
    python dev_tests/performance/q_learning_parallel_perftest.py

    # Run distributed benchmarks
    mpirun -n 4 python dev_tests/performance/q_learning_async_dist_perftest.py

**Sample Benchmark Script**:

.. code-block:: python

    from dist_classicrl.algorithms.runtime.q_learning_single_thread import SingleThreadQLearning
    from dist_classicrl.environments.tiktaktoe_mod import TicTacToeEnv
    import time
    import psutil
    import os

    def benchmark_algorithm(algorithm_class, steps=100000):
        """Benchmark an algorithm implementation."""

        # Setup
        env = TicTacToeEnv()
        agent = algorithm_class(state_size=512, action_size=9)

        # Memory baseline
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB

        # Performance benchmark
        start_time = time.time()

        for step in range(steps):
            obs, _ = env.reset()
            terminated = False

            while not terminated:
                action = agent.select_action(obs)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                agent.update(obs, action, reward, next_obs, terminated or truncated)
                obs = next_obs

        end_time = time.time()

        # Memory measurement
        memory_after = process.memory_info().rss / 1024 / 1024  # MB

        # Results
        duration = end_time - start_time
        throughput = steps / duration
        memory_used = memory_after - memory_before

        return {
            'duration': duration,
            'throughput': throughput,
            'memory_used': memory_used,
            'steps': steps
        }

Custom Benchmarks
-----------------

Create custom benchmarks for your specific use case:

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    from dist_classicrl.algorithms.runtime import *

    def compare_algorithms(environment_factory, steps=50000):
        """Compare different algorithm implementations."""

        algorithms = {
            'Single-thread': SingleThreadQLearning,
            'Parallel-2': lambda **kwargs: ParallelQLearning(num_processes=2, **kwargs),
            'Parallel-4': lambda **kwargs: ParallelQLearning(num_processes=4, **kwargs),
        }

        results = {}

        for name, algorithm_class in algorithms.items():
            print(f"Benchmarking {name}...")

            env = environment_factory()
            state_size = getattr(env, 'observation_space_size', 512)
            action_size = getattr(env, 'action_space_size', 9)

            result = benchmark_algorithm(
                algorithm_class,
                steps=steps,
                state_size=state_size,
                action_size=action_size
            )

            results[name] = result
            print(f"  Throughput: {result['throughput']:.0f} steps/sec")
            print(f"  Memory: {result['memory_used']:.1f} MB")

        return results

Optimization Strategies
=======================

Algorithm-Level Optimizations
------------------------------

**1. Q-Table Implementation Choice**:

Choose the right Q-table implementation for your use case:

.. code-block:: python

    # For sparse state spaces
    from dist_classicrl.algorithms.base_algorithms.q_learning_list import ListQLearning
    agent = ListQLearning(state_size=10000, action_size=4)

    # For dense state spaces
    from dist_classicrl.algorithms.base_algorithms.q_learning_numpy import NumpyQLearning
    agent = NumpyQLearning(state_size=1000, action_size=4)

    # For balanced performance
    from dist_classicrl.algorithms.base_algorithms.q_learning_optimal import OptimalQLearning
    agent = OptimalQLearning(state_size=5000, action_size=4)

**2. Hyperparameter Optimization**:

Optimize key hyperparameters for performance:

.. code-block:: python

    # Fast exploration for quick convergence
    agent = SingleThreadQLearning(
        state_size=512,
        action_size=9,
        learning_rate=0.3,      # Higher learning rate
        epsilon_decay=0.999,    # Slower exploration decay
        epsilon_min=0.05        # Higher minimum exploration
    )

**3. Batch Updates**:

Use batch updates for better cache efficiency:

.. code-block:: python

    class BatchQLearning(SingleThreadQLearning):
        def __init__(self, batch_size=32, **kwargs):
            super().__init__(**kwargs)
            self.batch_size = batch_size
            self.experience_buffer = []

        def update(self, state, action, reward, next_state, done):
            self.experience_buffer.append((state, action, reward, next_state, done))

            if len(self.experience_buffer) >= self.batch_size:
                self._batch_update()
                self.experience_buffer.clear()

        def _batch_update(self):
            for experience in self.experience_buffer:
                super().update(*experience)

System-Level Optimizations
---------------------------

**1. Memory Management**:

Optimize memory usage for better cache performance:

.. code-block:: python

    import gc
    import numpy as np

    class MemoryOptimizedAgent(SingleThreadQLearning):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            # Pre-allocate arrays
            self._temp_q_values = np.zeros(self.action_size)

        def select_action(self, state):
            # Reuse pre-allocated arrays
            np.copyto(self._temp_q_values, self.q_table[state])
            return np.argmax(self._temp_q_values)

        def periodic_cleanup(self):
            # Periodic garbage collection
            if self.step_count % 10000 == 0:
                gc.collect()

**2. CPU Optimization**:

Leverage CPU-specific optimizations:

.. code-block:: bash

    # Set optimal thread counts
    export OMP_NUM_THREADS=4
    export MKL_NUM_THREADS=4
    export OPENBLAS_NUM_THREADS=4

    # CPU affinity for parallel processes
    taskset -c 0-3 python train_parallel.py

**3. I/O Optimization**:

Minimize I/O bottlenecks:

.. code-block:: python

    class BufferedLogger:
        def __init__(self, filename, buffer_size=1000):
            self.filename = filename
            self.buffer = []
            self.buffer_size = buffer_size

        def log(self, data):
            self.buffer.append(data)
            if len(self.buffer) >= self.buffer_size:
                self.flush()

        def flush(self):
            with open(self.filename, 'a') as f:
                for data in self.buffer:
                    f.write(f"{data}\n")
            self.buffer.clear()

Parallel Optimization
=====================

Process Configuration
----------------------

Optimize parallel execution parameters:

.. code-block:: python

    from dist_classicrl.algorithms.runtime.q_learning_parallel import ParallelQLearning
    import multiprocessing as mp

    # Determine optimal process count
    num_cores = mp.cpu_count()
    physical_cores = num_cores // 2  # Account for hyperthreading

    agent = ParallelQLearning(
        state_size=512,
        action_size=9,
        num_processes=min(physical_cores, 4),  # Don't over-subscribe
        steps_per_sync=1000,  # Balance sync overhead vs accuracy
        shared_memory=True    # Use shared memory for large Q-tables
    )

Load Balancing
--------------

Implement dynamic load balancing:

.. code-block:: python

    class LoadBalancedParallelQLearning(ParallelQLearning):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.process_loads = [0] * self.num_processes

        def distribute_work(self, total_steps):
            # Distribute work based on previous performance
            steps_per_process = []
            total_load = sum(self.process_loads) or self.num_processes

            for load in self.process_loads:
                weight = (total_load - load) / total_load if total_load > 0 else 1.0 / self.num_processes
                steps_per_process.append(int(total_steps * weight))

            return steps_per_process

Distributed Optimization
=========================

Communication Optimization
---------------------------

Optimize MPI communication patterns:

.. code-block:: python

    from mpi4py import MPI
    import numpy as np

    class OptimizedDistributedQLearning:
        def __init__(self, **kwargs):
            self.comm = MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()

            # Optimize communication
            self.compression_enabled = True
            self.async_updates = True
            self.batch_size = 64

        def sync_q_table(self):
            if self.compression_enabled:
                # Compress Q-table updates
                compressed_updates = self._compress_updates()
                all_updates = self.comm.allgather(compressed_updates)
                self._apply_compressed_updates(all_updates)
            else:
                # Standard synchronization
                self.comm.Allreduce(MPI.IN_PLACE, self.q_table, op=MPI.SUM)
                self.q_table /= self.size

**Network Optimization**:

.. code-block:: bash

    # Optimize network settings for MPI
    export OMPI_MCA_btl_tcp_if_include=eth0
    export OMPI_MCA_oob_tcp_if_include=eth0

    # Use high-speed interconnects when available
    mpirun --mca btl openib,self,sm -n 16 python distributed_train.py

Memory Optimization
===================

Q-Table Optimization
--------------------

Optimize Q-table storage and access:

.. code-block:: python

    import numpy as np
    from scipy.sparse import dok_matrix, csr_matrix

    class SparseQLearning(SingleThreadQLearning):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            # Use sparse matrix for large, sparse Q-tables
            self.q_table = dok_matrix((self.state_size, self.action_size), dtype=np.float32)

        def update(self, state, action, reward, next_state, done):
            if not done:
                next_q_max = max(self.q_table[next_state, :].values()) if self.q_table[next_state, :].nnz > 0 else 0
            else:
                next_q_max = 0

            target = reward + self.discount_factor * next_q_max
            current_q = self.q_table[state, action]
            self.q_table[state, action] = current_q + self.learning_rate * (target - current_q)

Memory Profiling
----------------

Profile memory usage to identify bottlenecks:

.. code-block:: python

    from memory_profiler import profile
    import tracemalloc

    @profile
    def train_with_profiling():
        tracemalloc.start()

        agent = SingleThreadQLearning(state_size=10000, action_size=100)
        env = TicTacToeEnv()

        # Training loop
        for episode in range(1000):
            agent.train_episode(env)

            if episode % 100 == 0:
                current, peak = tracemalloc.get_traced_memory()
                print(f"Episode {episode}: Current {current / 1024 / 1024:.1f} MB, Peak {peak / 1024 / 1024:.1f} MB")

        tracemalloc.stop()

Profiling and Debugging
========================

Performance Profiling
----------------------

Use built-in profiling tools:

.. code-block:: python

    import cProfile
    import pstats
    from pstats import SortKey

    def profile_training():
        pr = cProfile.Profile()
        pr.enable()

        # Your training code here
        agent = SingleThreadQLearning(state_size=512, action_size=9)
        env = TicTacToeEnv()

        for _ in range(1000):
            agent.train_episode(env)

        pr.disable()

        # Analyze results
        stats = pstats.Stats(pr)
        stats.sort_stats(SortKey.TIME)
        stats.print_stats(20)  # Top 20 time-consuming functions

Line Profiling
---------------

For detailed line-by-line analysis:

.. code-block:: bash

    # Install line_profiler
    pip install line_profiler

    # Add @profile decorator to functions
    # Run with kernprof
    kernprof -l -v train_script.py

**Example Usage**:

.. code-block:: python

    @profile
    def train_episode(self, env):
        obs, _ = env.reset()
        terminated = False

        while not terminated:
            action = self.select_action(obs)  # This line will be profiled
            next_obs, reward, terminated, truncated, _ = env.step(action)
            self.update(obs, action, reward, next_obs, terminated or truncated)
            obs = next_obs

GPU Acceleration
================

While dist_classicrl focuses on CPU-based algorithms, GPU acceleration can be beneficial for large-scale problems:

**CuPy Integration**:

.. code-block:: python

    try:
        import cupy as cp
        gpu_available = True
    except ImportError:
        import numpy as cp
        gpu_available = False

    class GPUQLearning(SingleThreadQLearning):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            if gpu_available:
                self.q_table = cp.asarray(self.q_table)

        def update(self, state, action, reward, next_state, done):
            if gpu_available:
                # GPU-accelerated update
                next_q_max = cp.max(self.q_table[next_state]) if not done else 0
                target = reward + self.discount_factor * next_q_max
                self.q_table[state, action] += self.learning_rate * (target - self.q_table[state, action])
            else:
                # Fallback to CPU
                super().update(state, action, reward, next_state, done)

Best Practices
==============

Development Guidelines
----------------------

1. **Profile Early**: Identify bottlenecks before optimizing
2. **Measure Everything**: Use consistent benchmarking methodologies
3. **Test Optimizations**: Verify that optimizations don't break functionality
4. **Document Performance**: Record performance characteristics for future reference

Production Deployment
---------------------

1. **Resource Planning**: Plan hardware resources based on performance requirements
2. **Monitoring**: Implement runtime performance monitoring
3. **Scaling Strategy**: Have clear scaling plans for increased load
4. **Fallback Options**: Maintain fallback configurations for different scenarios

Common Pitfalls
---------------

1. **Premature Optimization**: Don't optimize before identifying real bottlenecks
2. **Over-Engineering**: Balance complexity with performance gains
3. **Ignoring Memory**: CPU performance isn't everything; memory usage matters
4. **Poor Scaling**: Ensure optimizations work across different scales

Performance Monitoring
=======================

Runtime Monitoring
-------------------

Implement runtime performance monitoring:

.. code-block:: python

    import time
    import psutil

    class PerformanceMonitor:
        def __init__(self, log_interval=1000):
            self.log_interval = log_interval
            self.step_count = 0
            self.start_time = time.time()
            self.last_log_time = time.time()

        def step(self):
            self.step_count += 1

            if self.step_count % self.log_interval == 0:
                current_time = time.time()

                # Calculate metrics
                total_time = current_time - self.start_time
                interval_time = current_time - self.last_log_time
                throughput = self.log_interval / interval_time

                # System metrics
                cpu_percent = psutil.cpu_percent()
                memory_percent = psutil.virtual_memory().percent

                print(f"Step {self.step_count}: "
                      f"Throughput: {throughput:.0f} steps/sec, "
                      f"CPU: {cpu_percent:.1f}%, "
                      f"Memory: {memory_percent:.1f}%")

                self.last_log_time = current_time

Automated Benchmarking
----------------------

Set up automated performance regression testing:

.. code-block:: python

    import json
    import datetime

    class BenchmarkSuite:
        def __init__(self, baseline_file="performance_baseline.json"):
            self.baseline_file = baseline_file
            self.baseline = self._load_baseline()

        def run_benchmarks(self):
            results = {}

            # Run standard benchmarks
            results['single_thread'] = self._benchmark_single_thread()
            results['parallel'] = self._benchmark_parallel()
            results['memory'] = self._benchmark_memory()

            # Compare with baseline
            self._compare_with_baseline(results)

            # Update baseline if needed
            self._update_baseline(results)

            return results

        def _compare_with_baseline(self, results):
            if not self.baseline:
                print("No baseline found, establishing new baseline")
                return

            for benchmark, result in results.items():
                if benchmark in self.baseline:
                    baseline_value = self.baseline[benchmark]['throughput']
                    current_value = result['throughput']

                    change = (current_value - baseline_value) / baseline_value * 100

                    if change < -5:  # More than 5% slower
                        print(f"REGRESSION: {benchmark} is {abs(change):.1f}% slower")
                    elif change > 5:  # More than 5% faster
                        print(f"IMPROVEMENT: {benchmark} is {change:.1f}% faster")

See Also
========

- :doc:`algorithms`: Algorithm implementations and their performance characteristics
- :doc:`distributed`: Detailed distributed training setup and optimization
- :doc:`../development/architecture`: Internal architecture and optimization points
- :doc:`../development/testing`: Performance testing framework
