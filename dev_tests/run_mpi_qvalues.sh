#!/bin/bash
export PYTHONPATH=$(pwd)/src:$PYTHONPATH

num_workers=7
num_agents=1024
batch_size=1

mpirun -n $((num_workers+1)) python dev_tests/q_learning_qvalues_plot.py --num-workers $((num_workers)) --num-agents $((num_agents)) --batch-size $((batch_size))
