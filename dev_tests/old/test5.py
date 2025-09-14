import timeit

# Define different selection methods
test_loop_max_track = """
max_value = float('-inf')
max_indices = []
for i, (val, m) in enumerate(zip(arr.flat, mask.flat)):
    if m:
        if val > max_value:
            max_value = val
            max_indices = [i]
        elif val == max_value:
            max_indices.append(i)
index = random.choice(max_indices) if max_indices else -1
"""

test_list_loop_max_track = """
arr_list = arr.tolist()
mask_list = mask.tolist()
max_value = float('-inf')
max_indices = []
for i, (val, m) in enumerate(zip(arr_list, mask_list)):
    if m:
        if val > max_value:
            max_value = val
            max_indices = [i]
        elif val == max_value:
            max_indices.append(i)
index = random.choice(max_indices) if max_indices else -1
"""

test_base_list_max_track = """
max_value = float('-inf')
max_indices = []
for i, (val, m) in enumerate(zip(arr_list, mask_list)):
    if m:
        if val > max_value:
            max_value = val
            max_indices = [i]
        elif val == max_value:
            max_indices.append(i)
index = random.choice(max_indices) if max_indices else -1
"""

test_np_where_max = """
masked_arr = np.where(mask, arr, -np.inf)
max_value = np.max(masked_arr)
indices = np.where(masked_arr == max_value)[0]
index = np.random.choice(indices) if indices.size > 0 else -1
"""

# Setup code with a fixed seed
setup_code_max = """
import numpy as np
import random
np.random.seed(42)
random.seed(42)
s = 250
arr = np.random.rand(s)  # Random values between 0 and 1
mask = np.random.choice([0, 1], size=s, p=[0.5, 0.5])
arr_list = list(arr)  # Pre-converted list for base list test
mask_list = list(mask)
"""

# Set number of loops for timing
num_loops_max = 20000

# Run timeit for each approach
time_loop_max_track = timeit.timeit(test_loop_max_track, setup=setup_code_max, number=num_loops_max)
time_list_loop_max_track = timeit.timeit(
    test_list_loop_max_track, setup=setup_code_max, number=num_loops_max
)
time_base_list_max_track = timeit.timeit(
    test_base_list_max_track, setup=setup_code_max, number=num_loops_max
)
time_np_where_max = timeit.timeit(test_np_where_max, setup=setup_code_max, number=num_loops_max)

# Store results in a dictionary
results_max_selection = {
    "Loop with max tracking + random.choice": time_loop_max_track,
    "List conversion + loop max tracking + random.choice": time_list_loop_max_track,
    "Base list + loop max tracking + random.choice": time_base_list_max_track,
    "np.max + np.where + np.random.choice": time_np_where_max,
}

# Print results
for _method, _time_taken in results_max_selection.items():
    pass
