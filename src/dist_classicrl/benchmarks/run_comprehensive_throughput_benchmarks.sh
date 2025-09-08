#!/bin/bash

# Comprehensive Throughput Benchmark Script
# This script runs a comprehensive set of throughput benchmarks for Q-learning algorithms
# covering single-thread, parallel, and distributed runtime configurations

set -e  # Exit on any error

echo "Starting comprehensive throughput benchmarks for Q-learning algorithms..."

# Set up environment
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"

# Try to detect the appropriate Python command
if command -v python &> /dev/null; then
    PYTHON_CMD="python"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    echo "Error: No Python interpreter found"
    exit 1
fi

echo "Using Python command: $PYTHON_CMD"

# Configuration
STEPS=1000  # Number of training steps for each experiment
OUTPUT_DIR="benchmark_results"
VERBOSE="--verbose"

# Create benchmark results directory
mkdir -p $OUTPUT_DIR

# Function to run a benchmark with error handling
run_benchmark() {
    local runtime=$1
    local agents=$2
    local processes_or_ranks=$3
    local description=$4

    echo "----------------------------------------"
    echo "Running: $description"
    echo "Runtime: $runtime, Agents: $agents"
    if [ "$runtime" = "distributed" ]; then
        echo "MPI Ranks: $processes_or_ranks"
    else
        echo "Processes: $processes_or_ranks"
    fi
    echo "----------------------------------------"

    if [ "$runtime" = "distributed" ]; then
        # Use mpirun for distributed runtime with ranks
        if command -v mpirun &> /dev/null; then
            # For distributed, processes_or_ranks is actually the number of MPI ranks
            local mpi_ranks=$processes_or_ranks
            local actual_processes=$((mpi_ranks - 1))
            mpirun -n $mpi_ranks $PYTHON_CMD src/dist_classicrl/benchmarks/throughput_benchmark.py \
                --runtime $runtime \
                --agents $agents \
                --processes $actual_processes \
                --steps $STEPS \
                --output-dir $OUTPUT_DIR \
                $VERBOSE
        else
            echo "Warning: mpirun not found, skipping distributed benchmark"
            return 1
        fi
    else
        # Use regular python for single-thread and parallel
        $PYTHON_CMD src/dist_classicrl/benchmarks/throughput_benchmark.py \
            --runtime $runtime \
            --agents $agents \
            --processes $processes_or_ranks \
            --steps $STEPS \
            --output-dir $OUTPUT_DIR \
            $VERBOSE
    fi

    if [ $? -eq 0 ]; then
        echo "✓ Completed: $description"
    else
        echo "✗ Failed: $description"
        return 1
    fi

    echo ""
}

# Track total experiments and completed ones
total_experiments=0
completed_experiments=0
failed_experiments=0

# Start timestamp
start_time=$(date)
echo "Benchmark suite started at: $start_time"
echo ""

# ==========================================
# SINGLE THREAD EXPERIMENTS
# ==========================================
echo "=========================================="
echo "SINGLE THREAD EXPERIMENTS"
echo "=========================================="

n_agents=(1 2 4 8 16 32 64 128)

for agents in "${n_agents[@]}"; do
    total_experiments=$((total_experiments + 1))
    if run_benchmark "single_thread" $agents 1 "Single-thread with $agents agents"; then
        completed_experiments=$((completed_experiments + 1))
    else
        failed_experiments=$((failed_experiments + 1))
    fi
done

# ==========================================
# PARALLEL EXPERIMENTS - GRID SEARCH
# ==========================================
echo "=========================================="
echo "PARALLEL EXPERIMENTS - GRID SEARCH"
echo "=========================================="

# Define arrays for grid search
parallel_processes=(1 2 4 8 16)

echo "Testing all combinations of agents and processes for parallel runtime..."
echo "Agents: ${n_agents[*]}"
echo "Processes: ${parallel_processes[*]}"
echo "Total parallel combinations: $((${#n_agents[@]} * ${#parallel_processes[@]}))"
echo ""

for agents in "${n_agents[@]}"; do
    for processes in "${parallel_processes[@]}"; do
        total_experiments=$((total_experiments + 1))
        if run_benchmark "parallel" $agents $processes "Parallel with $agents agents, $processes processes"; then
            completed_experiments=$((completed_experiments + 1))
        else
            failed_experiments=$((failed_experiments + 1))
        fi
    done
done

# ==========================================
# DISTRIBUTED EXPERIMENTS - GRID SEARCH
# ==========================================
echo "=========================================="
echo "DISTRIBUTED EXPERIMENTS - GRID SEARCH"
echo "=========================================="

# Define arrays for grid search
distributed_ranks=(2 3 5 8)

echo "Testing all combinations of agents and MPI ranks for distributed runtime..."
echo "Agents: ${n_agents[*]}"
echo "MPI Ranks: ${distributed_ranks[*]}"
echo "Total distributed combinations: $((${#n_agents[@]} * ${#distributed_ranks[@]}))"
echo ""

for agents in "${n_agents[@]}"; do
    for ranks in "${distributed_ranks[@]}"; do
        processes=$((ranks - 1))
        total_experiments=$((total_experiments + 1))
        if run_benchmark "distributed" $agents $ranks "Distributed with $agents agents, $ranks MPI ranks ($processes processes)"; then
            completed_experiments=$((completed_experiments + 1))
        else
            failed_experiments=$((failed_experiments + 1))
        fi
    done
done

# ==========================================
# SUMMARY
# ==========================================
end_time=$(date)

echo "=========================================="
echo "BENCHMARK SUITE COMPLETED"
echo "=========================================="
echo "Started at:  $start_time"
echo "Finished at: $end_time"
echo ""
echo "EXPERIMENT SUMMARY:"
echo "Total experiments:     $total_experiments"
echo "Completed successfully: $completed_experiments"
echo "Failed:                $failed_experiments"
echo "Success rate:          $(echo "scale=2; $completed_experiments * 100 / $total_experiments" | bc -l)%"
echo ""

if [ $completed_experiments -gt 0 ]; then
    echo "Results saved in: $OUTPUT_DIR/"
    echo ""
    echo "Generated result files:"
    ls -la $OUTPUT_DIR/*.json 2>/dev/null | sort
    echo ""
    echo "Summary by runtime:"
    echo "Single-thread files: $(ls $OUTPUT_DIR/single_thread_*.json 2>/dev/null | wc -l)"
    echo "Parallel files:      $(ls $OUTPUT_DIR/parallel_*.json 2>/dev/null | wc -l)"
    echo "Distributed files:   $(ls $OUTPUT_DIR/distributed_*.json 2>/dev/null | wc -l)"
else
    echo "No experiments completed successfully."
fi

if [ $failed_experiments -gt 0 ]; then
    echo ""
    echo "⚠️  Some experiments failed. Check the output above for details."
    exit 1
else
    echo ""
    echo "✅ All experiments completed successfully!"
fi
