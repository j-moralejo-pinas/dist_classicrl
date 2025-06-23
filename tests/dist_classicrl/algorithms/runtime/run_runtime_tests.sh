#!/bin/bash

# Test runner script for Q-learning runtime algorithms
# This script runs both regular pytest tests and MPI tests

echo "=== Running Q-learning Runtime Tests ==="

# Set PYTHONPATH to include src directory
export PYTHONPATH=$(pwd)/src:$PYTHONPATH

echo "PYTHONPATH: $PYTHONPATH"
echo ""

# Run single-thread tests
echo "1. Running SingleThreadQLearning tests..."
python -m pytest tests/dist_classicrl/algorithms/runtime/test_q_learning_single_thread.py -v
if [ $? -ne 0 ]; then
    echo "SingleThreadQLearning tests failed!"
    exit 1
fi
echo ""

# Run parallel tests
echo "2. Running ParallelQLearning tests..."
python -m pytest tests/dist_classicrl/algorithms/runtime/test_q_learning_parallel.py -v
if [ $? -ne 0 ]; then
    echo "ParallelQLearning tests failed!"
    exit 1
fi
echo ""

# Run async distributed tests (non-MPI)
echo "3. Running DistAsyncQLearning tests (non-MPI)..."
python -m pytest tests/dist_classicrl/algorithms/runtime/test_q_learning_async_dist.py::TestDistAsyncQLearning -v
if [ $? -ne 0 ]; then
    echo "DistAsyncQLearning non-MPI tests failed!"
    exit 1
fi
echo ""

# Check if MPI is available
if command -v mpirun &> /dev/null; then
    echo "4. Running DistAsyncQLearning MPI tests..."

    # Run MPI-specific tests with pytest
    mpirun -n 3 python -m pytest tests/dist_classicrl/algorithms/runtime/test_q_learning_async_dist.py::TestDistAsyncQLearningMPI -v
    if [ $? -ne 0 ]; then
        echo "DistAsyncQLearning MPI tests failed!"
        exit 1
    fi

    echo ""
    echo "5. Running DistAsyncQLearning integration test..."

    # Run integration test
    mpirun -n 3 python tests/dist_classicrl/algorithms/runtime/test_q_learning_async_dist.py
    if [ $? -ne 0 ]; then
        echo "DistAsyncQLearning integration test failed!"
        exit 1
    fi
else
    echo "4. MPI not available, skipping MPI-specific tests"
fi

echo ""
echo "=== All tests completed successfully! ==="
