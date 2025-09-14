# Benchmarks

This directory contains benchmarking utilities for the distributed classic reinforcement learning algorithms.

## Throughput Benchmark

The `throughput_benchmark.py` script measures the training throughput (steps per second) of different Q-learning runtime implementations.

### Features

- **Multiple Runtime Support**: Benchmarks single-thread, parallel, and distributed runtimes
- **Configurable Parameters**: Adjust number of agents, processes, and training steps
- **Automated Results Saving**: Results are automatically saved to JSON files with encoded experiment parameters
- **MPI Support**: Full support for distributed benchmarking with MPI
- **Comprehensive Metrics**: Measures effective throughput, elapsed time, and step multipliers

### Usage

#### Command Line Interface

```bash
# Single-thread benchmark
python throughput_benchmark.py --runtime single_thread --agents 10 --processes 1 --steps 100000

# Parallel benchmark
python throughput_benchmark.py --runtime parallel --agents 10 --processes 4 --steps 100000

# Distributed benchmark (requires MPI)
mpirun -n 4 python throughput_benchmark.py --runtime distributed --agents 10 --processes 4 --steps 100000
```

#### Parameters

- `--runtime`: Runtime implementation to benchmark (`single_thread`, `parallel`, `distributed`)
- `--agents`: Number of agents per vectorized environment (default: 10)
- `--processes`: Number of processes/cores to use (default: 1)
- `--steps`: Number of training steps (default: 100000)
- `--output-dir`: Directory to save results (default: benchmark_results)
- `--verbose`: Enable verbose logging

#### Example Usage Script

Use the provided `run_throughput_benchmarks.sh` script to run a comprehensive benchmark suite:

```bash
./run_throughput_benchmarks.sh
```

### Output

Results are saved as JSON files with names encoding the experiment parameters:
- `single_thread_nagents_10_nprocesses_1.json`
- `parallel_nagents_10_nprocesses_4.json`
- `distributed_nagents_10_nprocesses_4.json`

Each result file contains:
- Runtime configuration
- Total and effective steps
- Elapsed time
- Throughput (steps/second)
- MPI information (for distributed runs)
- Timestamp

### Example Results

```json
{
  "runtime": "parallel",
  "total_steps": 100000,
  "effective_steps": 1000000,
  "elapsed_time": 45.67,
  "throughput": 21906.34,
  "step_multiplier": 10,
  "num_agents": 10,
  "num_processes": 4,
  "timestamp": 1723456789.12
}
```

### Requirements

- Python environment with all project dependencies
- MPI (for distributed benchmarks): `mpi4py` and an MPI implementation (OpenMPI, MPICH, etc.)
- Sufficient computational resources for parallel/distributed testing

### Performance Considerations

- **Validation Disabled**: The benchmark disables validation by setting `val_every_n_steps` higher than total steps
- **Pure Training Throughput**: Measures only the training loop performance
- **Resource Scaling**: Results will vary based on available CPU cores and memory
- **Environment Complexity**: TicTacToe is used as a simple, fast environment for consistent benchmarking

### Troubleshooting

1. **MPI Issues**: Ensure MPI is properly installed and configured
2. **Memory Usage**: Large numbers of agents may require significant memory
3. **Process Limits**: Ensure your system can handle the requested number of processes
4. **Python Environment**: Ensure all dependencies are installed in the active environment
