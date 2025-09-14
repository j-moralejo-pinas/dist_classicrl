"""
Throughput Benchmark Analysis and Visualization Script.

This script analyzes benchmark results from JSON files and creates comprehensive
visualizations showing throughput performance across different runtime configurations.

Usage:
    python analyze_throughput_benchmarks.py --input-dir benchmark_results --output-dir plots
"""

import argparse
import json
import logging
import sys
import traceback
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Configure plotting style
plt.style.use("default")
plt.rcParams["figure.figsize"] = [10, 6]
plt.rcParams["font.size"] = 10
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Color palette for different runtimes
COLORS = {
    "single_thread": "#2E8B57",  # SeaGreen
    "parallel": "#4169E1",  # RoyalBlue
    "distributed": "#DC143C",  # Crimson
}


def load_benchmark_results(input_dir: Path) -> list[dict]:
    """Load all benchmark result JSON files from the input directory."""
    results = []
    json_files = list(input_dir.glob("*.json"))

    if not json_files:
        logger.warning("No JSON files found in %s", input_dir)
        return results

    logger.info("Found %d benchmark result files", len(json_files))

    for json_file in json_files:
        try:
            with json_file.open() as f:
                data = json.load(f)
                results.append(data)
                logger.debug("Loaded %s", json_file.name)
        except (OSError, json.JSONDecodeError):
            logger.exception("Error loading %s", json_file)

    logger.info("Successfully loaded %d benchmark results", len(results))
    return results


def organize_data(results: list[dict]) -> dict:
    """Organize benchmark results by runtime type."""
    organized = {"single_thread": [], "parallel": [], "distributed": []}

    for result in results:
        runtime = result.get("runtime", "unknown")
        if runtime in organized:
            organized[runtime].append(result)

    # Sort each runtime's results
    for runtime in organized.values():
        runtime.sort(key=lambda x: (x.get("num_agents", 0), x.get("num_processes", 0)))

    logger.info(
        "Organized data - Single-thread: %d, Parallel: %d, Distributed: %d",
        len(organized["single_thread"]),
        len(organized["parallel"]),
        len(organized["distributed"]),
    )

    return organized


def plot_single_thread_scaling(data: list[dict], output_dir: Path) -> None:
    """Plot throughput scaling for single-thread runtime vs number of agents."""
    if not data:
        logger.warning("No single-thread results found")
        return

    # Extract data
    agents = [result["num_agents"] for result in data]
    throughput = [result["throughput"] for result in data]

    # Create the plot
    _fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    ax.plot(
        agents,
        throughput,
        marker="o",
        linewidth=2,
        markersize=8,
        color=COLORS["single_thread"],
        label="Single-thread",
    )
    ax.set_xlabel("Number of Agents")
    ax.set_ylabel("Throughput (steps/second)")
    ax.set_title("Single-Thread: Throughput vs Number of Agents")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    # Add value annotations
    for x, y in zip(agents, throughput, strict=False):
        ax.annotate(
            f"{y:.0f}", (x, y), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9
        )

    ax.legend()
    plt.tight_layout()

    output_file = output_dir / "single_thread_scaling.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info("Single-thread scaling plot saved to %s", output_file)


def plot_parallel_scaling(data: list[dict], output_dir: Path) -> None:
    """Plot parallel runtime throughput with lines for each number of processes."""
    if not data:
        logger.warning("No parallel results found")
        return

    # Group data by number of processes
    process_groups = defaultdict(list)
    for result in data:
        process_groups[result["num_processes"]].append(result)

    # Sort each group by number of agents
    for processes in process_groups:
        process_groups[processes].sort(key=lambda x: x["num_agents"])

    # Create the plot
    _, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Generate colors for different process counts
    process_counts = sorted(process_groups.keys())
    colors = plt.cm.get_cmap("viridis")(np.linspace(0, 1, len(process_counts)))

    for process_count, color in zip(process_counts, colors, strict=False):
        results = process_groups[process_count]
        agents = [result["num_agents"] for result in results]
        throughput = [result["throughput"] for result in results]

        ax.plot(
            agents,
            throughput,
            marker="s",
            linewidth=2,
            markersize=6,
            label=f"{process_count} processes",
            color=color,
        )

    ax.set_xlabel("Number of Agents")
    ax.set_ylabel("Throughput (steps/second)")
    ax.set_title("Parallel Runtime: Throughput vs Agents (by Process Count)")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend()

    plt.tight_layout()

    output_file = output_dir / "parallel_scaling.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info("Parallel scaling plot saved to %s", output_file)
    """Plot parallel runtime throughput as a heatmap."""
    if not data:
        logger.warning("No parallel results found")
        return

    # Extract unique agents and processes
    agents_set = {result["num_agents"] for result in data}
    processes_set = {result["num_processes"] for result in data}

    agents_list = sorted(agents_set)
    processes_list = sorted(processes_set)

    # Create throughput matrix
    throughput_matrix = np.zeros((len(agents_list), len(processes_list)))

    for result in data:
        agent_idx = agents_list.index(result["num_agents"])
        process_idx = processes_list.index(result["num_processes"])
        throughput_matrix[agent_idx, process_idx] = result["throughput"]

    # Create the heatmap
    _fig, ax = plt.subplots(figsize=(12, 8))

    im = ax.imshow(throughput_matrix, cmap="viridis", aspect="auto")

    # Set ticks and labels
    ax.set_xticks(range(len(processes_list)))
    ax.set_yticks(range(len(agents_list)))
    ax.set_xticklabels(processes_list)
    ax.set_yticklabels(agents_list)

    ax.set_xlabel("Number of Processes")
    ax.set_ylabel("Number of Agents")
    ax.set_title("Parallel Runtime: Throughput Heatmap (steps/second)")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Throughput (steps/second)")

    # Add text annotations
    for i in range(len(agents_list)):
        for j in range(len(processes_list)):
            if throughput_matrix[i, j] > 0:  # Only annotate if we have data
                ax.text(
                    j,
                    i,
                    f"{throughput_matrix[i, j]:.0f}",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=8,
                )

    plt.tight_layout()

    output_file = output_dir / "parallel_heatmap.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info("Parallel heatmap saved to %s", output_file)


def plot_parallel_heatmap(data: list[dict], output_dir: Path) -> None:
    """Plot parallel runtime throughput as a heatmap."""
    if not data:
        logger.warning("No parallel results found")
        return

    # Extract unique agents and processes
    agents_set = {result["num_agents"] for result in data}
    processes_set = {result["num_processes"] for result in data}

    agents_list = sorted(agents_set)
    processes_list = sorted(processes_set)

    # Create throughput matrix
    throughput_matrix = np.zeros((len(agents_list), len(processes_list)))

    for result in data:
        agent_idx = agents_list.index(result["num_agents"])
        process_idx = processes_list.index(result["num_processes"])
        throughput_matrix[agent_idx, process_idx] = result["throughput"]

    # Create the heatmap
    _fig, ax = plt.subplots(figsize=(12, 8))

    im = ax.imshow(throughput_matrix, cmap="viridis", aspect="auto")

    # Set ticks and labels
    ax.set_xticks(range(len(processes_list)))
    ax.set_yticks(range(len(agents_list)))
    ax.set_xticklabels(processes_list)
    ax.set_yticklabels(agents_list)

    ax.set_xlabel("Number of Processes")
    ax.set_ylabel("Number of Agents")
    ax.set_title("Parallel Runtime: Throughput Heatmap (steps/second)")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Throughput (steps/second)")

    # Add text annotations
    for i in range(len(agents_list)):
        for j in range(len(processes_list)):
            if throughput_matrix[i, j] > 0:  # Only annotate if we have data
                ax.text(
                    j,
                    i,
                    f"{throughput_matrix[i, j]:.0f}",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=8,
                )

    plt.tight_layout()

    output_file = output_dir / "parallel_heatmap.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info("Parallel heatmap saved to %s", output_file)


def plot_distributed_scaling(data: list[dict], output_dir: Path) -> None:
    """Plot distributed runtime throughput with lines for each number of processes."""
    if not data:
        logger.warning("No distributed results found")
        return

    # Group data by number of processes
    process_groups = defaultdict(list)
    for result in data:
        process_groups[result["num_processes"]].append(result)

    # Sort each group by number of agents
    for processes in process_groups:
        process_groups[processes].sort(key=lambda x: x["num_agents"])

    # Create the plot
    _fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Generate colors for different process counts
    process_counts = sorted(process_groups.keys())
    colors = plt.cm.get_cmap("plasma")(np.linspace(0, 1, len(process_counts)))

    for process_count, color in zip(process_counts, colors, strict=True):
        results = process_groups[process_count]
        agents = [result["num_agents"] for result in results]
        throughput = [result["throughput"] for result in results]

        ax.plot(
            agents,
            throughput,
            marker="^",
            linewidth=2,
            markersize=6,
            label=f"{process_count} processes",
            color=color,
        )

    ax.set_xlabel("Number of Agents")
    ax.set_ylabel("Throughput (steps/second)")
    ax.set_title("Distributed Runtime: Throughput vs Agents (by Process Count)")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend()

    plt.tight_layout()

    output_file = output_dir / "distributed_scaling.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info("Distributed scaling plot saved to %s", output_file)


def plot_distributed_heatmap(data: list[dict], output_dir: Path) -> None:
    """Plot distributed runtime throughput as a heatmap."""
    if not data:
        logger.warning("No distributed results found")
        return

    # Extract unique agents and processes
    agents_set = {result["num_agents"] for result in data}
    processes_set = {result["num_processes"] for result in data}

    agents_list = sorted(agents_set)
    processes_list = sorted(processes_set)

    # Create throughput matrix
    throughput_matrix = np.zeros((len(agents_list), len(processes_list)))

    for result in data:
        agent_idx = agents_list.index(result["num_agents"])
        process_idx = processes_list.index(result["num_processes"])
        throughput_matrix[agent_idx, process_idx] = result["throughput"]

    # Create the heatmap
    _fig, ax = plt.subplots(figsize=(12, 8))

    im = ax.imshow(throughput_matrix, cmap="plasma", aspect="auto")

    # Set ticks and labels
    ax.set_xticks(range(len(processes_list)))
    ax.set_yticks(range(len(agents_list)))
    ax.set_xticklabels(processes_list)
    ax.set_yticklabels(agents_list)

    ax.set_xlabel("Number of Processes")
    ax.set_ylabel("Number of Agents")
    ax.set_title("Distributed Runtime: Throughput Heatmap (steps/second)")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Throughput (steps/second)")

    # Add text annotations
    for i in range(len(agents_list)):
        for j in range(len(processes_list)):
            if throughput_matrix[i, j] > 0:  # Only annotate if we have data
                ax.text(
                    j,
                    i,
                    f"{throughput_matrix[i, j]:.0f}",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=8,
                )

    plt.tight_layout()

    output_file = output_dir / "distributed_heatmap.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info("Distributed heatmap saved to %s", output_file)


def create_summary_table(organized_data: dict, output_dir: Path) -> None:
    """Create a summary table of benchmark results."""
    output_file = output_dir / "benchmark_summary.txt"

    with Path(output_file).open("w") as f:
        f.write("=== THROUGHPUT BENCHMARK SUMMARY ===\n\n")

        for runtime, data in organized_data.items():
            if not data:
                continue

            f.write(f"{runtime.upper().replace('_', ' ')} RUNTIME:\n")
            f.write("-" * 40 + "\n")

            if runtime == "single_thread":
                for result in data:
                    agents = result["num_agents"]
                    throughput = result["throughput"]
                    f.write(f"  {agents} agents: {throughput:.0f} steps/sec\n")
            else:
                # Group by agents
                agents_groups = defaultdict(list)
                for result in data:
                    agents_groups[result["num_agents"]].append(result)

                for agents in sorted(agents_groups.keys()):
                    f.write(f"  {agents} agents:\n")
                    for result in sorted(agents_groups[agents], key=lambda x: x["num_processes"]):
                        processes = result["num_processes"]
                        throughput = result["throughput"]
                        f.write(f"    {processes} processes: {throughput:.0f} steps/sec\n")
            f.write("\n")

        # Add statistics
        f.write("=== STATISTICS ===\n")
        for runtime, data in organized_data.items():
            if data:
                throughputs = [result["throughput"] for result in data]
                f.write(f"{runtime.replace('_', ' ').title()}:\n")
                f.write(f"  Count: {len(throughputs)}\n")
                f.write(f"  Mean: {np.mean(throughputs):.0f} steps/sec\n")
                f.write(f"  Max: {np.max(throughputs):.0f} steps/sec\n")
                f.write(f"  Min: {np.min(throughputs):.0f} steps/sec\n")
                f.write("\n")

    logger.info("Summary table saved to %s", output_file)


def main() -> int | None:
    """Run the benchmark analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze and visualize throughput benchmark results"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("benchmark_results"),
        help="Directory containing benchmark JSON files (default: benchmark_results)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots"),
        help="Directory to save plots and analysis (default: plots)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create output directory
    args.output_dir.mkdir(exist_ok=True)

    try:
        # Load benchmark results
        results = load_benchmark_results(args.input_dir)
        if not results:
            logger.error("No benchmark results found")
            return 1

        # Organize data by runtime
        organized_data = organize_data(results)

        logger.info("Creating visualizations...")

        # Generate focused throughput plots
        plot_single_thread_scaling(organized_data["single_thread"], args.output_dir)
        plot_parallel_scaling(organized_data["parallel"], args.output_dir)
        plot_parallel_heatmap(organized_data["parallel"], args.output_dir)
        plot_distributed_scaling(organized_data["distributed"], args.output_dir)
        plot_distributed_heatmap(organized_data["distributed"], args.output_dir)

        # Create summary
        create_summary_table(organized_data, args.output_dir)

        logger.info("Analysis complete! Results saved to %s", args.output_dir)
        logger.info("Generated files:")
        for file in sorted(args.output_dir.glob("*")):
            logger.info("  - %s", file.name)
    except Exception:
        logger.exception("Analysis failed")
        if args.verbose:
            traceback.print_exc()
        return 1
    else:
        return 0


if __name__ == "__main__":
    sys.exit(main())
