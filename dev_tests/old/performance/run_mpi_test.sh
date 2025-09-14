export PYTHONPATH=$(pwd)/src:$PYTHONPATH

num_workers=5
num_agents=10
batch_size=32

mpirun -n $((num_workers+1)) python tests/dist_classicrl/algorithms/runtime/test_async_dist_q_learning.py --num-workers $((num_workers)) --num-agents $((num_agents)) --batch-size $((batch_size))
