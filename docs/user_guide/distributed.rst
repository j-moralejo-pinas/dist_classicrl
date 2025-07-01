==================
Distributed Training
==================

This section provides comprehensive guidance for setting up and running distributed reinforcement learning training using dist_classicrl with MPI (Message Passing Interface).

Overview
========

Distributed training in dist_classicrl enables scaling reinforcement learning algorithms across multiple machines in a cluster. The distributed implementation uses:

- **MPI (Message Passing Interface)** for inter-node communication
- **Asynchronous parameter updates** for optimal performance
- **Fault tolerance** mechanisms for reliable operation
- **Load balancing** for efficient resource utilization

The distributed training framework supports both homogeneous clusters (identical machines) and heterogeneous clusters (mixed hardware configurations).

Architecture
============

Distributed Training Paradigm
------------------------------

dist_classicrl uses a **parameter server** architecture with asynchronous updates:

.. code-block::

    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
    │   Worker 0  │    │   Worker 1  │    │   Worker n  │
    │ (Parameter  │    │             │    │             │
    │   Server)   │    │             │    │             │
    └─────────────┘    └─────────────┘    └─────────────┘
           │                   │                   │
           └───────────────────┼───────────────────┘
                               │
                    Asynchronous Updates
                    (Q-table parameters)

**Key Components**:

1. **Parameter Server** (Rank 0): Maintains the master Q-table and coordinates updates
2. **Workers** (Rank 1-N): Perform local training and send parameter updates
3. **Communication Layer**: Handles asynchronous message passing between nodes

Communication Patterns
-----------------------

The framework uses several communication patterns optimized for reinforcement learning:

**1. Broadcast**: Parameter server broadcasts updated Q-table to all workers
**2. Gather**: Workers send local updates to parameter server
**3. All-Reduce**: Collective operations for synchronization
**4. Point-to-Point**: Direct communication for specific updates

Installation and Setup
=======================

MPI Installation
----------------

**Ubuntu/Debian**:

.. code-block:: bash

    # Install OpenMPI
    sudo apt-get update
    sudo apt-get install openmpi-bin openmpi-common libopenmpi-dev

    # Install mpi4py
    pip install mpi4py

**CentOS/RHEL**:

.. code-block:: bash

    # Install OpenMPI
    sudo yum install openmpi openmpi-devel

    # Load MPI module (if using modules)
    module load mpi/openmpi-x86_64

    # Install mpi4py
    pip install mpi4py

**From Source** (for optimal performance):

.. code-block:: bash

    # Download and compile OpenMPI
    wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.4.tar.gz
    tar -xzf openmpi-4.1.4.tar.gz
    cd openmpi-4.1.4

    ./configure --prefix=/usr/local/openmpi --enable-mpi-cxx
    make -j $(nproc)
    sudo make install

    # Update environment
    export PATH=/usr/local/openmpi/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/openmpi/lib:$LD_LIBRARY_PATH

    # Install mpi4py
    pip install mpi4py

Cluster Configuration
---------------------

**SSH Setup**:

.. code-block:: bash

    # Generate SSH key (on head node)
    ssh-keygen -t rsa -b 4096

    # Copy public key to all compute nodes
    for node in node1 node2 node3; do
        ssh-copy-id user@$node
    done

    # Test passwordless SSH
    ssh node1 "echo 'Connection successful'"

**Hostfile Setup**:

Create a hostfile listing all available nodes:

.. code-block:: bash

    # /etc/openmpi/hostfile or ~/hostfile
    node1 slots=4
    node2 slots=4
    node3 slots=4
    node4 slots=4

**Network Configuration**:

.. code-block:: bash

    # Verify network connectivity
    mpirun -n 4 --hostfile hostfile hostname

    # Test MPI communication
    mpirun -n 4 --hostfile hostfile python -c "
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    print(f'Rank {comm.Get_rank()} of {comm.Get_size()} on {MPI.Get_processor_name()}')
    "

Basic Distributed Training
===========================

Simple Example
--------------

Create a basic distributed training script:

.. code-block:: python

    # train_distributed.py
    from mpi4py import MPI
    from dist_classicrl.algorithms.runtime.q_learning_async_dist import DistAsyncQLearning
    from dist_classicrl.environments.tiktaktoe_mod import TicTacToeEnv
    import numpy as np

    def main():
        # Initialize MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        print(f"Starting worker {rank} of {size}")

        # Create environment and agent
        env = TicTacToeEnv()
        agent = DistAsyncQLearning(
            state_size=512,
            action_size=9,
            learning_rate=0.1,
            discount_factor=0.99,
            batch_size=32,
            sync_frequency=100  # Sync every 100 updates
        )

        # Train the agent
        total_steps = 100000
        steps_per_worker = total_steps // size

        print(f"Worker {rank} training for {steps_per_worker} steps")
        agent.train(env=env, steps=steps_per_worker)

        # Synchronize final results
        if rank == 0:
            print("Training completed. Final Q-table synchronized.")
            # Save results
            agent.save_model(f"distributed_model_{size}_workers.pkl")

    if __name__ == "__main__":
        main()

**Running the Example**:

.. code-block:: bash

    # Local execution (4 processes)
    mpirun -n 4 python train_distributed.py

    # Cluster execution
    mpirun -n 16 --hostfile hostfile python train_distributed.py

    # With specific network interface
    mpirun -n 16 --hostfile hostfile --mca btl_tcp_if_include eth0 python train_distributed.py

Advanced Configuration
======================

Optimized Distributed Agent
----------------------------

Configure advanced parameters for optimal performance:

.. code-block:: python

    from dist_classicrl.algorithms.runtime.q_learning_async_dist import DistAsyncQLearning

    class OptimizedDistributedAgent(DistAsyncQLearning):
        def __init__(self, **kwargs):
            super().__init__(
                # Core parameters
                state_size=kwargs.get('state_size', 512),
                action_size=kwargs.get('action_size', 9),
                learning_rate=kwargs.get('learning_rate', 0.1),
                discount_factor=kwargs.get('discount_factor', 0.99),

                # Distributed-specific parameters
                batch_size=kwargs.get('batch_size', 64),
                sync_frequency=kwargs.get('sync_frequency', 200),
                compression_threshold=kwargs.get('compression_threshold', 0.01),
                async_updates=kwargs.get('async_updates', True),

                # Performance parameters
                buffer_size=kwargs.get('buffer_size', 10000),
                prefetch_factor=kwargs.get('prefetch_factor', 2),
                communication_backend='nccl' if kwargs.get('use_gpu', False) else 'mpi'
            )

Load Balancing
--------------

Implement dynamic load balancing for heterogeneous clusters:

.. code-block:: python

    import time
    from mpi4py import MPI

    class LoadBalancedTraining:
        def __init__(self, agent, env):
            self.agent = agent
            self.env = env
            self.comm = MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()

            # Performance tracking
            self.performance_history = []
            self.workload_factor = 1.0

        def adaptive_training(self, total_steps, rebalance_interval=1000):
            remaining_steps = total_steps

            while remaining_steps > 0:
                # Calculate steps for this worker
                steps_this_round = min(
                    remaining_steps // self.size,
                    int(rebalance_interval * self.workload_factor)
                )

                # Execute training
                start_time = time.time()
                self.agent.train(self.env, steps=steps_this_round)
                execution_time = time.time() - start_time

                # Calculate performance
                performance = steps_this_round / execution_time if execution_time > 0 else 0

                # Share performance data
                all_performances = self.comm.allgather(performance)

                # Update workload factor
                if self.rank == 0:
                    avg_performance = sum(all_performances) / len(all_performances)
                    workload_factors = [avg_performance / perf if perf > 0 else 1.0
                                      for perf in all_performances]
                else:
                    workload_factors = None

                # Broadcast new workload factors
                workload_factors = self.comm.bcast(workload_factors, root=0)
                self.workload_factor = workload_factors[self.rank]

                remaining_steps -= steps_this_round * self.size

                if self.rank == 0:
                    print(f"Completed {total_steps - remaining_steps}/{total_steps} steps")

Fault Tolerance
===============

Checkpoint and Recovery
-----------------------

Implement checkpointing for fault tolerance:

.. code-block:: python

    import pickle
    import os
    from pathlib import Path

    class FaultTolerantDistributedTraining:
        def __init__(self, agent, env, checkpoint_dir="checkpoints"):
            self.agent = agent
            self.env = env
            self.comm = MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()

            self.checkpoint_dir = Path(checkpoint_dir)
            self.checkpoint_dir.mkdir(exist_ok=True)

            # Checkpoint frequency (number of sync operations)
            self.checkpoint_frequency = 10
            self.sync_counter = 0

        def save_checkpoint(self, step_count):
            """Save checkpoint of current training state."""
            checkpoint_data = {
                'q_table': self.agent.q_table,
                'step_count': step_count,
                'epsilon': self.agent.epsilon,
                'sync_counter': self.sync_counter
            }

            checkpoint_file = self.checkpoint_dir / f"checkpoint_rank_{self.rank}_step_{step_count}.pkl"

            with open(checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint_data, f)

            # Keep only last 3 checkpoints
            self._cleanup_old_checkpoints()

        def load_checkpoint(self, checkpoint_file=None):
            """Load the most recent checkpoint."""
            if checkpoint_file is None:
                checkpoint_files = list(self.checkpoint_dir.glob(f"checkpoint_rank_{self.rank}_*.pkl"))
                if not checkpoint_files:
                    return None
                checkpoint_file = max(checkpoint_files, key=lambda x: x.stat().st_mtime)

            with open(checkpoint_file, 'rb') as f:
                checkpoint_data = pickle.load(f)

            self.agent.q_table = checkpoint_data['q_table']
            self.agent.epsilon = checkpoint_data['epsilon']
            self.sync_counter = checkpoint_data['sync_counter']

            return checkpoint_data['step_count']

        def train_with_checkpointing(self, total_steps, checkpoint_interval=5000):
            # Try to restore from checkpoint
            start_step = self.load_checkpoint() or 0

            if start_step > 0 and self.rank == 0:
                print(f"Resuming training from step {start_step}")

            current_step = start_step

            while current_step < total_steps:
                # Train for checkpoint interval
                steps_to_train = min(checkpoint_interval, total_steps - current_step)

                try:
                    self.agent.train(self.env, steps=steps_to_train)
                    current_step += steps_to_train

                    # Save checkpoint
                    if current_step % checkpoint_interval == 0:
                        self.save_checkpoint(current_step)

                        if self.rank == 0:
                            print(f"Checkpoint saved at step {current_step}")

                except Exception as e:
                    if self.rank == 0:
                        print(f"Error during training: {e}")
                        print("Attempting to restore from checkpoint...")

                    # Restore from checkpoint and continue
                    restored_step = self.load_checkpoint()
                    if restored_step:
                        current_step = restored_step
                        if self.rank == 0:
                            print(f"Restored from step {current_step}")
                    else:
                        raise e

Node Failure Detection
----------------------

Implement node failure detection and recovery:

.. code-block:: python

    import signal
    import time

    class FailureDetector:
        def __init__(self, comm, heartbeat_interval=30):
            self.comm = comm
            self.rank = comm.Get_rank()
            self.size = comm.Get_size()
            self.heartbeat_interval = heartbeat_interval
            self.last_heartbeat = time.time()
            self.failed_nodes = set()

        def send_heartbeat(self):
            """Send heartbeat to all other nodes."""
            heartbeat_data = {
                'rank': self.rank,
                'timestamp': time.time(),
                'status': 'alive'
            }

            # Broadcast heartbeat
            try:
                self.comm.bcast(heartbeat_data, root=self.rank)
            except Exception as e:
                print(f"Heartbeat failed from rank {self.rank}: {e}")

        def check_node_health(self):
            """Check health of all nodes."""
            current_time = time.time()

            for rank in range(self.size):
                if rank == self.rank:
                    continue

                try:
                    # Non-blocking check
                    status = self.comm.recv(source=rank, tag=99)

                    if current_time - status['timestamp'] > self.heartbeat_interval * 2:
                        if rank not in self.failed_nodes:
                            print(f"Node {rank} appears to have failed")
                            self.failed_nodes.add(rank)
                            self._handle_node_failure(rank)

                except Exception:
                    # Node is not responding
                    if rank not in self.failed_nodes:
                        print(f"Node {rank} is not responding")
                        self.failed_nodes.add(rank)
                        self._handle_node_failure(rank)

        def _handle_node_failure(self, failed_rank):
            """Handle node failure by redistributing work."""
            active_nodes = [i for i in range(self.size) if i not in self.failed_nodes]

            if self.rank == 0:  # Parameter server handles redistribution
                print(f"Redistributing work from failed node {failed_rank}")
                # Implement work redistribution logic
                pass

Performance Optimization
=========================

Communication Optimization
---------------------------

Optimize MPI communication for better performance:

.. code-block:: python

    from mpi4py import MPI
    import numpy as np

    class OptimizedCommunication:
        def __init__(self, comm):
            self.comm = comm
            self.rank = comm.Get_rank()
            self.size = comm.Get_size()

            # Communication optimizations
            self.compression_enabled = True
            self.batch_updates = True
            self.async_communication = True

        def compressed_allreduce(self, data, compression_ratio=0.1):
            """Perform compressed all-reduce operation."""
            if self.compression_enabled:
                # Compress data (keep only significant changes)
                threshold = np.std(data) * compression_ratio
                mask = np.abs(data) > threshold
                compressed_data = data * mask

                # All-reduce compressed data
                self.comm.Allreduce(MPI.IN_PLACE, compressed_data, op=MPI.SUM)
                return compressed_data / self.size
            else:
                # Standard all-reduce
                self.comm.Allreduce(MPI.IN_PLACE, data, op=MPI.SUM)
                return data / self.size

        def batched_parameter_sync(self, updates_buffer):
            """Synchronize parameters in batches."""
            if not self.batch_updates or len(updates_buffer) == 0:
                return

            # Batch multiple updates
            batched_updates = np.vstack(updates_buffer)

            # Perform communication
            if self.async_communication:
                request = self.comm.Iallreduce(MPI.IN_PLACE, batched_updates, op=MPI.SUM)
                return request  # Non-blocking
            else:
                self.comm.Allreduce(MPI.IN_PLACE, batched_updates, op=MPI.SUM)
                return batched_updates / self.size

Network Topology Optimization
------------------------------

Optimize communication patterns based on network topology:

.. code-block:: python

    class TopologyAwareTraining:
        def __init__(self, comm):
            self.comm = comm
            self.rank = comm.Get_rank()
            self.size = comm.Get_size()

            # Create topology-aware communicators
            self.node_comm = self._create_node_communicator()
            self.inter_node_comm = self._create_inter_node_communicator()

        def _create_node_communicator(self):
            """Create communicator for processes on the same node."""
            # Use hostname to group processes
            hostname = MPI.Get_processor_name()
            color = hash(hostname) % 1000  # Simple hash for grouping

            return self.comm.Split(color, self.rank)

        def _create_inter_node_communicator(self):
            """Create communicator between node leaders."""
            node_rank = self.node_comm.Get_rank()

            # Only rank 0 processes participate in inter-node communication
            if node_rank == 0:
                color = 0
            else:
                color = MPI.UNDEFINED

            return self.comm.Split(color, self.rank)

        def hierarchical_allreduce(self, data):
            """Perform hierarchical all-reduce: intra-node then inter-node."""
            # Step 1: All-reduce within each node
            self.node_comm.Allreduce(MPI.IN_PLACE, data, op=MPI.SUM)

            # Step 2: All-reduce between node leaders
            if self.inter_node_comm != MPI.COMM_NULL:
                self.inter_node_comm.Allreduce(MPI.IN_PLACE, data, op=MPI.SUM)

            # Step 3: Broadcast result within each node
            self.node_comm.Bcast(data, root=0)

            return data / self.size

Cluster Deployment
==================

SLURM Integration
-----------------

Deploy on SLURM-managed clusters:

.. code-block:: bash

    #!/bin/bash
    #SBATCH --job-name=dist_rl_training
    #SBATCH --nodes=4
    #SBATCH --ntasks-per-node=8
    #SBATCH --cpus-per-task=1
    #SBATCH --time=02:00:00
    #SBATCH --partition=compute
    #SBATCH --output=training_%j.out
    #SBATCH --error=training_%j.err

    # Load modules
    module load openmpi/4.1.4
    module load python/3.9

    # Activate virtual environment
    source venv/bin/activate

    # Set MPI parameters
    export OMPI_MCA_btl_tcp_if_include=ib0
    export OMPI_MCA_oob_tcp_if_include=ib0

    # Run distributed training
    srun python train_distributed.py \
        --state_size 10000 \
        --action_size 100 \
        --total_steps 1000000 \
        --checkpoint_interval 10000

**Submit the job**:

.. code-block:: bash

    sbatch slurm_job.sh

Kubernetes Deployment
----------------------

Deploy using Kubernetes with MPI Operator:

.. code-block:: yaml

    # mpi-job.yaml
    apiVersion: kubeflow.org/v1alpha2
    kind: MPIJob
    metadata:
      name: dist-classicrl-training
    spec:
      slotsPerWorker: 1
      runPolicy:
        cleanPodPolicy: Running
      mpiReplicaSpecs:
        Launcher:
          replicas: 1
          template:
            spec:
              containers:
              - image: dist-classicrl:latest
                name: mpi-launcher
                command:
                - mpirun
                - -n
                - "16"
                - python
                - train_distributed.py
                resources:
                  limits:
                    cpu: 1
                    memory: 2Gi
        Worker:
          replicas: 4
          template:
            spec:
              containers:
              - image: dist-classicrl:latest
                name: mpi-worker
                resources:
                  limits:
                    cpu: 4
                    memory: 8Gi

**Deploy the job**:

.. code-block:: bash

    kubectl apply -f mpi-job.yaml

Docker Container Setup
----------------------

Create optimized Docker containers for distributed training:

.. code-block:: dockerfile

    # Dockerfile.distributed
    FROM ubuntu:20.04

    # Install MPI and dependencies
    RUN apt-get update && apt-get install -y \
        openmpi-bin \
        openmpi-common \
        libopenmpi-dev \
        python3 \
        python3-pip \
        openssh-client \
        && rm -rf /var/lib/apt/lists/*

    # Install Python packages
    COPY requirements.txt .
    RUN pip3 install -r requirements.txt

    # Copy application code
    COPY . /app
    WORKDIR /app

    # Set MPI environment
    ENV OMPI_ALLOW_RUN_AS_ROOT=1
    ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

    ENTRYPOINT ["python3", "train_distributed.py"]

Monitoring and Debugging
=========================

Distributed Monitoring
----------------------

Monitor distributed training progress:

.. code-block:: python

    import matplotlib.pyplot as plt
    import numpy as np
    from mpi4py import MPI

    class DistributedMonitor:
        def __init__(self, comm, log_interval=1000):
            self.comm = comm
            self.rank = comm.Get_rank()
            self.size = comm.Get_size()
            self.log_interval = log_interval

            # Metrics storage
            self.metrics = {
                'steps': [],
                'rewards': [],
                'loss': [],
                'throughput': [],
                'communication_time': []
            }

        def log_metrics(self, step, reward, loss, throughput, comm_time):
            """Log metrics from each worker."""
            if step % self.log_interval == 0:
                # Gather metrics from all workers
                all_metrics = self.comm.gather({
                    'step': step,
                    'reward': reward,
                    'loss': loss,
                    'throughput': throughput,
                    'comm_time': comm_time,
                    'rank': self.rank
                }, root=0)

                if self.rank == 0:
                    self._process_global_metrics(all_metrics)

        def _process_global_metrics(self, all_metrics):
            """Process and display global metrics."""
            avg_reward = np.mean([m['reward'] for m in all_metrics])
            avg_loss = np.mean([m['loss'] for m in all_metrics])
            total_throughput = sum([m['throughput'] for m in all_metrics])
            avg_comm_time = np.mean([m['comm_time'] for m in all_metrics])

            print(f"Global metrics - "
                  f"Avg Reward: {avg_reward:.4f}, "
                  f"Avg Loss: {avg_loss:.4f}, "
                  f"Total Throughput: {total_throughput:.0f} steps/sec, "
                  f"Avg Comm Time: {avg_comm_time:.3f}s")

        def plot_training_curves(self):
            """Plot training curves (called by rank 0)."""
            if self.rank != 0:
                return

            fig, axes = plt.subplots(2, 2, figsize=(12, 8))

            # Plot reward curve
            axes[0, 0].plot(self.metrics['steps'], self.metrics['rewards'])
            axes[0, 0].set_title('Training Reward')
            axes[0, 0].set_xlabel('Steps')
            axes[0, 0].set_ylabel('Average Reward')

            # Plot loss curve
            axes[0, 1].plot(self.metrics['steps'], self.metrics['loss'])
            axes[0, 1].set_title('Training Loss')
            axes[0, 1].set_xlabel('Steps')
            axes[0, 1].set_ylabel('Loss')

            # Plot throughput
            axes[1, 0].plot(self.metrics['steps'], self.metrics['throughput'])
            axes[1, 0].set_title('Training Throughput')
            axes[1, 0].set_xlabel('Steps')
            axes[1, 0].set_ylabel('Steps/Second')

            # Plot communication overhead
            axes[1, 1].plot(self.metrics['steps'], self.metrics['communication_time'])
            axes[1, 1].set_title('Communication Overhead')
            axes[1, 1].set_xlabel('Steps')
            axes[1, 1].set_ylabel('Communication Time (s)')

            plt.tight_layout()
            plt.savefig('distributed_training_metrics.png')

Debugging Tools
---------------

Debug distributed training issues:

.. code-block:: python

    class DistributedDebugger:
        def __init__(self, comm):
            self.comm = comm
            self.rank = comm.Get_rank()
            self.size = comm.Get_size()

        def check_data_consistency(self, data, tolerance=1e-6):
            """Check if data is consistent across all processes."""
            all_data = self.comm.allgather(data)

            if self.rank == 0:
                reference = all_data[0]
                for i, worker_data in enumerate(all_data[1:], 1):
                    diff = np.abs(worker_data - reference).max()
                    if diff > tolerance:
                        print(f"WARNING: Data inconsistency detected between rank 0 and rank {i}")
                        print(f"Maximum difference: {diff}")
                        return False
                print("Data consistency check passed")

            return True

        def profile_communication(self, data_size=1000, iterations=100):
            """Profile communication performance."""
            test_data = np.random.randn(data_size)

            # Measure broadcast performance
            start_time = time.time()
            for _ in range(iterations):
                self.comm.Bcast(test_data, root=0)
            broadcast_time = (time.time() - start_time) / iterations

            # Measure allreduce performance
            start_time = time.time()
            for _ in range(iterations):
                self.comm.Allreduce(MPI.IN_PLACE, test_data, op=MPI.SUM)
            allreduce_time = (time.time() - start_time) / iterations

            if self.rank == 0:
                print(f"Communication Profile:")
                print(f"  Broadcast time: {broadcast_time*1000:.2f} ms")
                print(f"  Allreduce time: {allreduce_time*1000:.2f} ms")
                print(f"  Broadcast bandwidth: {data_size*8/broadcast_time/1e6:.2f} Mbps")
                print(f"  Allreduce bandwidth: {data_size*8/allreduce_time/1e6:.2f} Mbps")

Best Practices
==============

Development Guidelines
----------------------

1. **Start Small**: Begin with 2-4 nodes before scaling to large clusters
2. **Test Locally**: Use mpirun locally before deploying to clusters
3. **Monitor Performance**: Track communication overhead and scaling efficiency
4. **Handle Failures**: Implement robust error handling and recovery mechanisms

Production Deployment
---------------------

1. **Resource Planning**: Plan network bandwidth and compute requirements
2. **Security**: Use secure communication protocols in production environments
3. **Monitoring**: Implement comprehensive monitoring and alerting
4. **Backup Strategy**: Regular checkpointing and backup procedures

Common Issues and Solutions
===========================

**Issue**: Poor scaling performance
**Solution**: Check network bandwidth, reduce communication frequency, use compression

**Issue**: Node failures causing training crashes
**Solution**: Implement checkpointing and fault tolerance mechanisms

**Issue**: Memory issues on large clusters
**Solution**: Optimize batch sizes, use compression, implement memory monitoring

**Issue**: Slow convergence compared to single-node training
**Solution**: Tune learning rates, adjust synchronization frequency, check for race conditions

See Also
========

- :doc:`performance`: Performance optimization techniques
- :doc:`algorithms`: Algorithm implementations suitable for distributed training
- :doc:`../development/architecture`: Internal architecture details
- :doc:`../development/testing`: Testing distributed implementations
