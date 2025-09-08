import random
import timeit

import numpy as np

# Define different selection methods
test_where_choice = """
indices = np.where(mask)[0]
index = random.choice(indices) if indices.size > 0 else -1
"""

test_where_choice_np = """
indices = np.where(mask)[0]
index = np.random.choice(indices) if indices.size > 0 else -1
"""

test_loop_random_choice = """
indices = []
for i, val in enumerate(mask.flat):
    if val:
        indices.append(i)
index = random.choice(indices) if indices else -1
"""

test_loop_np_choice = """
indices = []
for i, val in enumerate(mask.flat):
    if val:
        indices.append(i)
index = np.random.choice(indices) if indices else -1
"""

test_list_loop_random_choice = """
mask_list = mask.tolist()
indices = []
for i, val in enumerate(mask_list):
    if val:
        indices.append(i)
index = random.choice(indices) if indices else -1
"""

test_base_list_random_choice = """
indices = []
for i, val in enumerate(mask_list_2):
    if val == 1:
        indices.append(i)
index = random.choice(indices) if indices else -1
"""

# Setup code with a fixed seed
setup_code_mask = """
import numpy as np
import random
np.random.seed(42)
random.seed(42)
mask = np.random.choice([0, 1], size=10, p=[0.5, 0.5])
mask_list_2 = mask.tolist()
"""

# Set number of loops for timing
num_loops = 200000

# Run timeit for each approach
time_where_choice = timeit.timeit(test_where_choice, setup=setup_code_mask, number=num_loops)
time_where_choice_np = timeit.timeit(test_where_choice_np, setup=setup_code_mask, number=num_loops)
time_loop_random_choice = timeit.timeit(
    test_loop_random_choice, setup=setup_code_mask, number=num_loops
)
time_loop_np_choice = timeit.timeit(test_loop_np_choice, setup=setup_code_mask, number=num_loops)
time_list_loop_random_choice = timeit.timeit(
    test_list_loop_random_choice, setup=setup_code_mask, number=num_loops
)
time_base_list_random_choice = timeit.timeit(
    test_base_list_random_choice, setup=setup_code_mask, number=num_loops
)

# Store results in a dictionary
results_mask_selection = {
    "np.where + random.choice": time_where_choice,
    "np.where + np.random.choice": time_where_choice_np,
    "Loop + random.choice": time_loop_random_choice,
    "Loop + np.random.choice": time_loop_np_choice,
    "List conversion + loop + random.choice": time_list_loop_random_choice,
    "Base list + loop + random.choice": time_base_list_random_choice,
}

# Print results
print("\nMask-Based Random Selection Performance Results:")
for method, time_taken in results_mask_selection.items():
    print(f"{method}: {time_taken:.6f} seconds")
