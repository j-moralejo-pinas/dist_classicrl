import timeit

# Define the grid
action_sizes = [2, 10, 100, 1000, 10000]
num_agents_list = [1, 10, 100, 1000, 10000]

# Setup code (converted to a string)
setup_code_template = """
import numpy as np
import random

state_size = 1000
action_size = {action_size}
num_agents = {num_agents}

action_masks = [[random.randint(0, 1) for _ in range(action_size)] for _ in range(num_agents)]
states = [random.randint(0, state_size - 1) for _ in range(num_agents)]
actions = [random.randint(0, action_size - 1) for _ in range(num_agents)]
rewards = [random.random() for _ in range(num_agents)]
next_states = [random.randint(0, state_size - 1) for _ in range(num_agents)]

for action_mask in action_masks:
    action_mask[0] = 1

action_masks_np = np.array(action_masks, dtype=np.int32)

states_np = np.array(states, dtype=np.int32)
actions_np = np.array(actions, dtype=np.int32)
rewards_np = np.array(rewards, dtype=np.float32)
next_states_np = np.array(next_states, dtype=np.int32)

from q_learning_optimal import MultiAgentQLearning
ql = MultiAgentQLearning(num_agents, state_size, action_size)
ql.q_table = np.array([random.random() for _ in range(state_size * action_size)]).reshape(state_size, action_size)
"""


def get_code_from_file(file_path):
    """Reads and extracts code from a file."""
    with open(file_path) as file:
        return file.read()


def determine_iterations(action_size, num_agents):
    """Adjusts iterations based on input size."""
    base_iterations = 10000000  # For smallest case (2 actions, 1 agent)
    scaling_factor = action_size * num_agents
    return max(base_iterations // scaling_factor, 10)  # Ensure a minimum of 10 iterations


def benchmark_files(file_paths, output_file) -> None:
    """Benchmarks multiple Python files and saves the best-performing file index per test case."""
    num_tests = len(file_paths)
    results = {
        test_idx: {
            (num_agents, action_size): 0.0
            for num_agents in num_agents_list
            for action_size in action_sizes
        }
        for test_idx in range(num_tests)
    }

    for test_idx, file_path in enumerate(file_paths):
        file_code = get_code_from_file(file_path)

        for action_size in action_sizes:
            for num_agents in num_agents_list:
                setup_code = setup_code_template.format(
                    action_size=action_size, num_agents=num_agents
                )
                iterations = determine_iterations(action_size, num_agents)

                time_taken = timeit.timeit(stmt=file_code, setup=setup_code, number=iterations)

                results[test_idx][(num_agents, action_size)] = time_taken

    # Identify the best-performing test for each case
    best_tests = {
        (num_agents, action_size): min(
            results, key=lambda idx: results[idx][(num_agents, action_size)]
        )
        for num_agents in num_agents_list
        for action_size in action_sizes
    }

    # Save results to file
    with open(output_file, "w") as f:
        for test_idx in range(num_tests):
            # Write header for each test file
            f.write(f"Execution Times for Test {test_idx}:\n")
            f.write(f"{'Num Agents \\ Actions':<15}")
            for action_size in action_sizes:
                f.write(f"{action_size:<16}")
            f.write("\n" + "=" * 100 + "\n")

            for num_agents in num_agents_list:
                f.write(f"{num_agents:<15}")
                for action_size in action_sizes:
                    f.write(f"{results[test_idx][(num_agents, action_size)]:<16.6f}")
                f.write("\n")

            f.write("\n" + "=" * 100 + "\n\n")

        # Write best test indexes table
        f.write("Best Performing Test Index:\n")
        f.write(f"{'Num Agents \\ Actions':<15}")
        for action_size in action_sizes:
            f.write(f"{action_size:<16}")
        f.write("\n" + "=" * 100 + "\n")

        for num_agents in num_agents_list:
            f.write(f"{num_agents:<15}")
            for action_size in action_sizes:
                f.write(f"{best_tests[(num_agents, action_size)]:<16}")
            f.write("\n")


def benchmark_code_strings(code_strings, output_file) -> None:
    """Benchmarks multiple Python strings and saves the best-performing file index per test case."""
    num_tests = len(code_strings)
    results = {
        test_idx: {
            (num_agents, action_size): 0.0
            for num_agents in num_agents_list
            for action_size in action_sizes
        }
        for test_idx in range(num_tests)
    }

    for test_idx, code_string in enumerate(code_strings):
        for action_size in action_sizes:
            for num_agents in num_agents_list:
                setup_code = setup_code_template.format(
                    action_size=action_size, num_agents=num_agents
                )
                iterations = determine_iterations(action_size, num_agents)

                time_taken = timeit.timeit(stmt=code_string, setup=setup_code, number=iterations)

                results[test_idx][(num_agents, action_size)] = time_taken

    # Identify the best-performing test for each case
    best_tests = {
        (num_agents, action_size): min(
            results, key=lambda idx: results[idx][(num_agents, action_size)]
        )
        for num_agents in num_agents_list
        for action_size in action_sizes
    }

    # Save results to file
    with open(output_file, "w") as f:
        for test_idx in range(num_tests):
            # Write header for each test file
            f.write(f"Execution Times for Test {test_idx}:\n")
            f.write(f"{'Num Agents \\ Actions':<15}")
            for action_size in action_sizes:
                f.write(f"{action_size:<16}")
            f.write("\n" + "=" * 100 + "\n")

            for num_agents in num_agents_list:
                f.write(f"{num_agents:<15}")
                for action_size in action_sizes:
                    f.write(f"{results[test_idx][(num_agents, action_size)]:<16.6f}")
                f.write("\n")

            f.write("\n" + "=" * 100 + "\n\n")

        # Write best test indexes table
        f.write("Best Performing Test Index:\n")
        f.write(f"{'Num Agents \\ Actions':<15}")
        for action_size in action_sizes:
            f.write(f"{action_size:<16}")
        f.write("\n" + "=" * 100 + "\n")

        for num_agents in num_agents_list:
            f.write(f"{num_agents:<15}")
            for action_size in action_sizes:
                f.write(f"{best_tests[(num_agents, action_size)]:<16}")
            f.write("\n")


if __name__ == "__main__":
    output_filename = "det_true_mask.txt"

    code = [
        """
a = np.fromiter([ql.choose_masked_action(state, mask, deterministic=True) for state, mask in zip(states, action_masks)], dtype=np.int32)
""",
        """
a = np.fromiter([ql.choose_masked_action_vec(state, mask, deterministic=True) for state, mask in zip(states_np, action_masks_np)], dtype=np.int32)
""",
        """
a = ql.choose_masked_actions_vec(states_np, action_masks_np, deterministic=True)
""",
    ]

    code_2 = [
        """
ql.learn(states, actions, rewards, next_states)
""",
        """
ql.learn(states_np.tolist(), actions_np.tolist(), rewards_np.tolist(), next_states_np.tolist())
""",
        """
qn.learn(states_np, actions_np, rewards_np, next_states_np)
""",
    ]

    benchmark_code_strings(code, output_filename)
