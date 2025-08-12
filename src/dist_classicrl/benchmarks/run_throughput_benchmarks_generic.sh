#!/bin/bash

# Throughput Benchmark Example Usage Script (Generic)
# This script demonstrates how to run the throughput benchmark for different configurations
# It automatically detects the appropriate Python command to use

echo "Running throughput benchmarks for Q-learning algorithms..."

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

# Create benchmark results directory
mkdir -p benchmark_results

# Single-thread benchmark
echo "Running single-thread benchmark..."
$PYTHON_CMD src/dist_classicrl/benchmarks/throughput_benchmark.py \
    --runtime single_thread \
    --agents 10 \
    --processes 1 \
    --steps 50000 \
    --output-dir benchmark_results \
    --verbose

# Parallel benchmark with 4 processes
echo "Running parallel benchmark with 4 processes..."
$PYTHON_CMD src/dist_classicrl/benchmarks/throughput_benchmark.py \
    --runtime parallel \
    --agents 10 \
    --processes 4 \
    --steps 50000 \
    --output-dir benchmark_results \
    --verbose

# Distributed benchmark (requires MPI)
echo "Running distributed benchmark with 4 MPI processes..."
mpirun -n 4 $PYTHON_CMD src/dist_classicrl/benchmarks/throughput_benchmark.py \
    --runtime distributed \
    --agents 10 \
    --processes 4 \
    --steps 50000 \
    --output-dir benchmark_results \
    --verbose

echo "Benchmarks completed! Results saved in benchmark_results/"
echo "Result files:"
ls -la benchmark_results/
