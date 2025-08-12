"""
Throughput Benchmark Script for Q-Learning Algorithms.

This script benchmarks the throughput of different Q-learning runtime implementations
by measuring the time taken to complete a fixed number of training steps without validation.
Results are saved to files with names encoding the experiment parameters.

Usage:
    python throughput_benchmark.py --runtime single_thread --agents 10 --processes 1
    python throughput_benchmark.py --runtime parallel --agents 10 --processes 4
    mpirun -n 4 python throughput_benchmark.py --runtime distributed --agents 10 --processes 4
"""

import os

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import json
import logging
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from gymnasium import spaces
from gymnasium.vector import SyncVectorEnv, VectorEnv
from gymnasium.vector.vector_env import AutoresetMode
from mpi4py import MPI

from dist_classicrl.algorithms.base_algorithms.q_learning_optimal import OptimalQLearningBase
from dist_classicrl.algorithms.runtime.base_runtime import BaseRuntime
from dist_classicrl.algorithms.runtime.parallel_runtime import ParallelQLearning
from dist_classicrl.algorithms.runtime.q_learning_async_dist import DistAsyncQLearning
from dist_classicrl.algorithms.runtime.single_thread_runtime import (
    SingleThreadQLearning,
)
from dist_classicrl.environments.tiktaktoe_mod import TicTacToeEnv
from dist_classicrl.schedules.exponential_schedule import ExponentialSchedule
from dist_classicrl.wrappers.flatten_multidiscrete_wrapper import (
    FlattenMultiDiscreteObservationsWrapper,
)

# MPI setup
comm = MPI.COMM_WORLD
RANK = comm.Get_rank()
NUM_NODES = comm.Get_size()
MASTER_RANK = 0

# Configuration constants
DEFAULT_STEPS = 100000
DEFAULT_LEARNING_RATE = 0.1
DEFAULT_DISCOUNT_FACTOR = 0.99
DEFAULT_EXPLORATION_RATE = 1.0
DEFAULT_EXPLORATION_DECAY = 0.995
DEFAULT_MIN_EXPLORATION_RATE = 0.01
MIN_MPI_NODES_FOR_DISTRIBUTED = 2

logger = logging.getLogger(__name__)


def setup_logging(level: int = logging.INFO) -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s - Rank %(rank)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Add rank to all log messages
    old_factory = logging.getLogRecordFactory()

    def record_factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
        record = old_factory(*args, **kwargs)
        record.rank = RANK  # type: ignore[attr-defined]
        return record

    logging.setLogRecordFactory(record_factory)


def make_env() -> FlattenMultiDiscreteObservationsWrapper:
    """Create a single TicTacToe environment with wrapper."""
    env = TicTacToeEnv()
    return FlattenMultiDiscreteObservationsWrapper(env)


def check_print(runtime: str) -> bool:
    """
    Check if printing is allowed based on the runtime environment.

    Parameters
    ----------
    runtime : str
        The runtime environment string.

    Returns
    -------
    bool
        True if printing is allowed, False otherwise.
    """
    return runtime != "distributed" or RANK == MASTER_RANK


def create_environments(
    num_agents: int, num_processes: int = 1, runtime: str = "single_thread"
) -> tuple[VectorEnv | Sequence[VectorEnv], VectorEnv]:
    """Create training and validation environments based on runtime type."""
    val_env = SyncVectorEnv([make_env for _ in range(1)], autoreset_mode=AutoresetMode.SAME_STEP)

    if runtime == "parallel":
        # For parallel runtime, create list of environments for multiple processes
        env = [
            SyncVectorEnv(
                [make_env for _ in range(num_agents)], autoreset_mode=AutoresetMode.SAME_STEP
            )
            for _ in range(num_processes)
        ]
    else:
        # For single-thread and distributed runtime, create single vectorized environment
        env = SyncVectorEnv(
            [make_env for _ in range(num_agents)], autoreset_mode=AutoresetMode.SAME_STEP
        )

    return env, val_env


def initialize_agent(
    env: VectorEnv | Sequence[VectorEnv], runtime: str = "single_thread"
) -> BaseRuntime:
    """Initialize the Q-learning agent based on environment spaces and runtime type."""
    # For parallel runtime, use the first environment in the list
    ref_env = env[0] if isinstance(env, Sequence) else env

    # Verify environment spaces
    assert isinstance(ref_env.single_action_space, spaces.Discrete)
    assert isinstance(ref_env.single_observation_space, (spaces.Discrete, spaces.Dict))

    # Initialize agent based on observation space type
    if isinstance(ref_env.single_observation_space, spaces.Dict):
        assert "observation" in ref_env.single_observation_space.spaces
        assert isinstance(ref_env.single_observation_space.spaces["observation"], spaces.Discrete)
        state_size = ref_env.single_observation_space.spaces["observation"].n
    else:
        state_size = ref_env.single_observation_space.n

    action_size = ref_env.single_action_space.n

    # Create algorithm with modern interface
    algo = OptimalQLearningBase(
        state_size=state_size,
        action_size=action_size,
        discount_factor=DEFAULT_DISCOUNT_FACTOR,
    )

    # Create schedules
    lr_schedule = ExponentialSchedule(
        value=DEFAULT_LEARNING_RATE,
        min_value=1e-5,
        decay_rate=DEFAULT_EXPLORATION_DECAY,
    )
    exploration_rate_schedule = ExponentialSchedule(
        value=DEFAULT_EXPLORATION_RATE,
        min_value=DEFAULT_MIN_EXPLORATION_RATE,
        decay_rate=DEFAULT_EXPLORATION_DECAY,
    )

    # Create agent based on runtime type
    if runtime == "single_thread":
        agent = SingleThreadQLearning(algo, lr_schedule, exploration_rate_schedule)
    elif runtime == "parallel":
        agent = ParallelQLearning(algo, lr_schedule, exploration_rate_schedule)
    elif runtime == "distributed":
        agent = DistAsyncQLearning(algo, lr_schedule, exploration_rate_schedule)
    else:
        msg = f"Unknown runtime type: {runtime}"
        raise ValueError(msg)

    return agent


def run_benchmark(
    agent: BaseRuntime,
    env: VectorEnv | Sequence[VectorEnv],
    val_env: VectorEnv,
    total_steps: int,
    runtime: str,
) -> dict[str, Any]:
    """
    Run the benchmark and return timing results.

    Parameters
    ----------
    agent : BaseRuntime
        The Q-learning agent to benchmark.
    env : VectorEnv | Sequence[VectorEnv]
        The training environment(s).
    val_env : VectorEnv
        The validation environment (not used, but required by train method).
    total_steps : int
        Number of training steps to run.
    runtime : str
        The runtime type being benchmarked.

    Returns
    -------
    dict[str, Any]
        Dictionary containing benchmark results.
    """
    if check_print(runtime):
        logger.info("Starting benchmark...")
        logger.info("Runtime: %s", runtime)
        logger.info("Total steps: %d", total_steps)
        if isinstance(env, Sequence):
            logger.info("Number of parallel processes: %d", len(env))
            logger.info("Steps per process: %d", total_steps // len(env) if len(env) > 0 else 0)

    # Set validation interval higher than total steps to disable validation
    val_every_n_steps = total_steps * 2

    # Record start time
    start_time = time.perf_counter()

    # Run training - no validation will occur since val_every_n_steps > total_steps
    try:
        _, _, _, _ = agent.train(
            env=env,
            steps=total_steps,
            val_env=val_env,
            val_every_n_steps=val_every_n_steps,
            val_steps=None,
            val_episodes=10,
            curr_state_dict=None,
        )
    except Exception:
        if check_print(runtime):
            logger.exception("Training failed")
        raise

    # Record end time
    end_time = time.perf_counter()

    # Calculate results
    elapsed_time = end_time - start_time

    # Calculate effective throughput
    step_multiplier = env[0].num_envs if isinstance(env, Sequence) else env.num_envs
    effective_steps = total_steps * step_multiplier
    throughput = effective_steps / elapsed_time  # steps per second

    results = {
        "runtime": runtime,
        "total_steps": total_steps,
        "effective_steps": effective_steps,
        "elapsed_time": elapsed_time,
        "throughput": throughput,
        "step_multiplier": step_multiplier,
        "timestamp": time.time(),
    }

    if check_print(runtime):
        logger.info("Benchmark completed!")
        logger.info("Elapsed time: %.2f seconds", elapsed_time)
        logger.info("Effective steps: %d", effective_steps)
        logger.info("Throughput: %.2f steps/second", throughput)

    return results


def save_results(
    results: dict[str, Any],
    num_agents: int,
    num_processes: int,
    output_dir: str = "benchmark_results",
) -> str:
    """
    Save benchmark results to a JSON file.

    Parameters
    ----------
    results : dict[str, Any]
        The benchmark results to save.
    num_agents : int
        Number of agents per vectorized environment.
    num_processes : int
        Number of processes/cores used.
    output_dir : str
        Directory to save results in.

    Returns
    -------
    str
        Path to the saved results file.
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Create filename encoding the experiment
    runtime = results["runtime"]
    if runtime == "single_thread":
        filename = f"{runtime}_{num_agents}_agents.json"
    elif runtime == "parallel":
        filename = f"{runtime}_{num_agents}_agents_{num_processes}_processes.json"
    elif runtime == "distributed":
        filename = f"{runtime}_{num_agents}_agents_{NUM_NODES}_processes.json"
    filepath = output_path / filename

    # Add experiment metadata to results
    results.update(
        {
            "num_agents": num_agents,
            "num_processes": num_processes,
            "mpi_rank": RANK,
            "mpi_size": NUM_NODES,
        }
    )

    # Save results
    with filepath.open("w") as f:
        json.dump(results, f, indent=2)

    return str(filepath)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark throughput of Q-learning runtime implementations."
    )
    parser.add_argument(
        "--runtime",
        choices=["single_thread", "parallel", "distributed"],
        required=True,
        help="Runtime implementation to benchmark",
    )
    parser.add_argument(
        "--agents",
        type=int,
        default=10,
        help="Number of agents per vectorized environment (default: 10)",
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=1,
        help="Number of processes/cores to use (default: 1)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_STEPS,
        help=f"Number of training steps (default: {DEFAULT_STEPS})",
    )
    parser.add_argument(
        "--output-dir",
        default="benchmark_results",
        help="Directory to save results (default: benchmark_results)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


def main() -> None:
    """Run the main benchmark function."""
    args = parse_arguments()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level)

    # Validate arguments
    if (
        args.runtime == "distributed"
        and NUM_NODES < MIN_MPI_NODES_FOR_DISTRIBUTED
        and check_print(args.runtime)
    ):
        logger.warning(
            "Distributed runtime requires multiple MPI processes. "
            "Current MPI size: %d. Consider using mpirun.",
            NUM_NODES,
        )

    if args.runtime == "parallel" and args.processes == 1 and check_print(args.runtime):
        logger.warning(
            "Parallel runtime with 1 process may not show performance benefits. "
            "Consider increasing --processes."
        )

    try:
        # Create environments
        env, val_env = create_environments(args.agents, args.processes, args.runtime)

        # Initialize agent
        agent = initialize_agent(env, args.runtime)

        # Run benchmark
        results = run_benchmark(agent, env, val_env, args.steps, args.runtime)

        # Save results (only master rank in distributed mode)
        if check_print(args.runtime):
            filepath = save_results(results, args.agents, args.processes, args.output_dir)
            logger.info("Results saved to: %s", filepath)

    except Exception:
        if check_print(args.runtime):
            logger.exception("Benchmark failed")
        raise


if __name__ == "__main__":
    main()
