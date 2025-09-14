export PYTHONPATH=$(pwd)/src:$PYTHONPATH

pytest tests/dist_classicrl/test_tiktaktoe_mod.py

pytest tests/dist_classicrl/algorithms/base_algorithms/test_q_learning_optimal.py
