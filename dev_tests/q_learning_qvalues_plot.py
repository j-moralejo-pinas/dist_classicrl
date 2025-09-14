"""
Q-Learning Q-values Evolution Plotter.

This script trains a Q-learning agent on the TicTacToe environment and tracks
the evolution of Q-values for each state-action pair. It creates plots showing
how Q-values change over time during training.
"""

import logging
from collections.abc import Sequence
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces
from gymnasium.vector import SyncVectorEnv, VectorEnv
from gymnasium.vector.vector_env import AutoresetMode
from mpi4py import MPI

from dist_classicrl.algorithms.base_algorithms.q_learning_optimal import OptimalQLearningBase
from dist_classicrl.algorithms.runtime.base_runtime import BaseRuntime
from dist_classicrl.algorithms.runtime.parallel_runtime import ParallelQLearning
from dist_classicrl.algorithms.runtime.q_learning_async_dist import DistAsyncQLearning
from dist_classicrl.algorithms.runtime.single_thread_runtime import (
    SingleThreadQLearning,
)
from dist_classicrl.environments.tiktaktoe_mod import TicTacToeEnv
from dist_classicrl.schedules.exponential_schedule import ExponentialSchedule
from dist_classicrl.wrappers.flatten_multidiscrete_wrapper import (
    FlattenMultiDiscreteObservationsWrapper,
)

comm = MPI.COMM_WORLD
RANK = comm.Get_rank()
NUM_NODES = comm.Get_size()
MASTER_RANK = 0

# Constants
VARIANCE_THRESHOLD = 1e-20
MAX_LEGEND_LINES = 10
PRINT_STATS_INTERVAL = 10

logger = logging.getLogger(__name__)


def check_print(runtime: str) -> bool:
    """
    Check if printing is allowed based on the runtime environment.

    Parameters
    ----------
    runtime : str
        The runtime environment string.

    Returns
    -------
    bool
        True if printing is allowed, False otherwise.
    """
    return runtime != "distributed" or RANK == MASTER_RANK


def make_env() -> FlattenMultiDiscreteObservationsWrapper:
    """Create a single TicTacToe environment with wrapper."""
    env = TicTacToeEnv()
    return FlattenMultiDiscreteObservationsWrapper(env)


class QValueTracker:
    """Tracks Q-values evolution during training."""

    def __init__(self, state_size: int, action_size: int) -> None:
        self.state_size = state_size
        self.action_size = action_size
        self.q_values_history: dict[int, list[np.ndarray]] = {
            action: [] for action in range(action_size)
        }
        self.step_numbers: list[int] = []
        self.learning_rates: list[float] = []
        self.exploration_rates: list[float] = []

    def record_q_values(
        self,
        agent: BaseRuntime,
        step: int,
    ) -> None:
        """Record Q-values, learning rate, and exploration rate at a specific step."""
        self.step_numbers.append(step)
        self.learning_rates.append(agent.lr_schedule.get_value())
        self.exploration_rates.append(agent.exploration_rate_schedule.get_value())

        # Record Q-values for each action
        for action in range(self.action_size):
            action_q_values = agent.algorithm.get_action_q_values(action).copy()
            self.q_values_history[action].append(action_q_values)

    def create_plots(self, output_dir: str = "q_values_plots") -> None:
        """Create plots for Q-values evolution."""
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)

        # Create a plot for each action
        for action in range(self.action_size):
            self._plot_action_qvalues(action, output_dir)

        # Create separate plots for learning rate and exploration rate
        self._plot_learning_rate(output_dir)
        self._plot_exploration_rate(output_dir)

    def _plot_action_qvalues(self, action: int, output_dir: str) -> None:
        """Create a plot for a specific action showing Q-values evolution for all states."""
        fig, ax = plt.subplots(figsize=(12, 8))

        # Get Q-values history for this action
        action_history = np.array(self.q_values_history[action])  # shape: (n_steps, n_states)

        # Filter out states that have flat lines (no evolution)
        states_to_plot = []
        final_values = []

        for state in range(self.state_size):
            state_values = action_history[:, state]
            # Check if the line is not flat (has some variation)
            if np.std(state_values) >= VARIANCE_THRESHOLD:
                states_to_plot.append(state)
                final_values.append(state_values[-1])

        if not states_to_plot:
            logger.info("No evolving Q-values found for action %d", action)
            plt.close(fig)
            return

        # Normalize final values to [0, 1] for color mapping
        if len(final_values) > 1:
            min_val, max_val = min(final_values), max(final_values)
            if max_val > min_val:
                normalized_final_values = [
                    (val - min_val) / (max_val - min_val) for val in final_values
                ]
            else:
                normalized_final_values = [0.5] * len(final_values)
        else:
            normalized_final_values = [0.5]

        # Create color spectrum between two colors (e.g., blue to red)
        color1 = np.array([0.0, 0.0, 1.0])  # Blue
        color2 = np.array([1.0, 0.0, 0.0])  # Red

        # Plot lines for each state
        for i, state in enumerate(states_to_plot):
            state_values = action_history[:, state]

            # Interpolate color based on final value
            t = normalized_final_values[i]
            color = (1 - t) * color1 + t * color2

            # Plot with low alpha and thin line
            ax.plot(
                self.step_numbers,
                state_values,
                color=color,
                alpha=0.3,
                linewidth=0.5,
                label=f"State {state}" if len(states_to_plot) <= MAX_LEGEND_LINES else None,
            )

        # Customize plot
        ax.set_xlabel("Training Steps")
        ax.set_ylabel("Q-Value")
        ax.set_title(
            f"Q-Values Evolution for Action {action}\n"
            f"({len(states_to_plot)} states with evolving Q-values)"
        )
        ax.grid(visible=True, alpha=0.3)

        # Add legend only if we have few lines to avoid clutter
        if len(states_to_plot) <= MAX_LEGEND_LINES:
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        # Add colorbar to show the meaning of colors
        sm = plt.cm.ScalarMappable(
            cmap="coolwarm",
            norm=mcolors.Normalize(
                vmin=min(final_values) if final_values else 0,
                vmax=max(final_values) if final_values else 1,
            ),
        )
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label("Final Q-Value")

        plt.tight_layout()

        # Save plot
        plot_path = Path(output_dir) / f"action_{action}_qvalues_evolution.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        logger.info("Saved plot for action %d: %s", action, plot_path)
        plt.close(fig)

    def _plot_learning_rate(self, output_dir: str) -> None:
        """Create a plot showing the evolution of learning rate."""
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot learning rate
        ax.plot(self.step_numbers, self.learning_rates, "b-", linewidth=2, label="Learning Rate")
        ax.set_xlabel("Training Steps")
        ax.set_ylabel("Learning Rate")
        ax.set_title("Learning Rate Evolution During Training")
        ax.grid(visible=True, alpha=0.3)
        ax.legend()

        plt.tight_layout()

        # Save plot
        plot_path = Path(output_dir) / "learning_rate_evolution.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        logger.info("Saved learning rate plot: %s", plot_path)
        plt.close(fig)

    def _plot_exploration_rate(self, output_dir: str) -> None:
        """Create a plot showing the evolution of exploration rate."""
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot exploration rate
        ax.plot(
            self.step_numbers, self.exploration_rates, "r-", linewidth=2, label="Exploration Rate"
        )
        ax.set_xlabel("Training Steps")
        ax.set_ylabel("Exploration Rate")
        ax.set_title("Exploration Rate Evolution During Training")
        ax.grid(visible=True, alpha=0.3)
        ax.legend()

        plt.tight_layout()

        # Save plot
        plot_path = Path(output_dir) / "exploration_rate_evolution.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        logger.info("Saved exploration rate plot: %s", plot_path)
        plt.close(fig)


def create_environments(
    num_agents: int, num_cores: int = 1, runtime: str = "single_thread"
) -> tuple[VectorEnv | Sequence[VectorEnv], VectorEnv]:
    """Create training and validation environments based on runtime type."""
    val_env = SyncVectorEnv([make_env for _ in range(1)], autoreset_mode=AutoresetMode.SAME_STEP)

    if runtime == "parallel":
        # For parallel runtime, create list of environments for multiple cores
        env = [
            SyncVectorEnv(
                [make_env for _ in range(num_agents)], autoreset_mode=AutoresetMode.SAME_STEP
            )
            for _ in range(num_cores)
        ]
    else:
        # For single-thread runtime, create single vectorized environment
        env = SyncVectorEnv(
            [make_env for _ in range(num_agents)], autoreset_mode=AutoresetMode.SAME_STEP
        )

    return env, val_env


def initialize_agent(
    env: VectorEnv | Sequence[VectorEnv], runtime: str = "single_thread"
) -> tuple[BaseRuntime, int, int]:
    """Initialize the Q-learning agent based on environment spaces and runtime type."""
    # For parallel runtime, use the first environment in the list
    ref_env = env[0] if isinstance(env, Sequence) else env

    # Verify environment spaces
    assert isinstance(ref_env.single_action_space, spaces.Discrete)
    assert isinstance(ref_env.single_observation_space, (spaces.Discrete, spaces.Dict))

    # Initialize agent based on observation space type
    if isinstance(ref_env.single_observation_space, spaces.Dict):
        assert "observation" in ref_env.single_observation_space.spaces
        assert isinstance(ref_env.single_observation_space.spaces["observation"], spaces.Discrete)
        state_size = ref_env.single_observation_space.spaces["observation"].n
    else:
        state_size = ref_env.single_observation_space.n

    action_size = ref_env.single_action_space.n
    algo = OptimalQLearningBase(
        state_size=state_size,
        action_size=action_size,
        discount_factor=1.0,
    )
    lr_schedule = ExponentialSchedule(value=0.5, min_value=1e-5, decay_rate=0.99999)
    exploration_rate_schedule = ExponentialSchedule(value=1.0, min_value=0.1, decay_rate=0.9999999)

    # Create agent based on runtime type

    if runtime == "single_thread":
        agent = SingleThreadQLearning(algo, lr_schedule, exploration_rate_schedule)
    elif runtime == "parallel":
        agent = ParallelQLearning(
            algo,
            lr_schedule,
            exploration_rate_schedule,
        )
    else:
        agent = DistAsyncQLearning(
            algo,
            lr_schedule,
            exploration_rate_schedule,
        )

    return agent, int(state_size), int(action_size)


def run_training(
    agent: BaseRuntime,
    env: VectorEnv | Sequence[VectorEnv],
    val_env: VectorEnv,
    tracker: QValueTracker | None,
    total_steps: int,
    val_every_n_steps: int,
    runtime: str = "single_thread",
) -> None:
    """Run training using the built-in train method and periodically save Q-values."""
    # Calculate effective step multiplier

    step_multiplier = env[0].num_envs if isinstance(env, Sequence) else env.num_envs

    curr_state = None

    # Record initial Q-values
    if check_print(runtime):
        assert tracker is not None, "Tracker must be initialized for Q-value recording"
        tracker.record_q_values(agent, 0)
        logger.info("Recorded initial Q-values")

    steps_completed = 0

    while steps_completed < total_steps:
        # Train for this batch - handle different method signatures

        _, validation_history, env, curr_state = agent.train(
            env=env,
            steps=val_every_n_steps,
            val_env=val_env,
            val_every_n_steps=val_every_n_steps,
            val_steps=None,
            val_episodes=10,
            curr_state_dict=curr_state,
        )  # type: ignore[]

        steps_completed += val_every_n_steps
        effective_steps = steps_completed * step_multiplier

        # Record Q-values after this batch
        if check_print(runtime):
            assert tracker is not None, "Tracker must be initialized for Q-value recording"
            tracker.record_q_values(agent, effective_steps)
            logger.info(
                "Step %d/%d (effective: %d) - Validation Rewards: %s",
                steps_completed,
                total_steps,
                effective_steps,
                validation_history[0],
            )

    if check_print(runtime):
        logger.info(
            "Training completed! Total steps: %d (effective: %d)",
            total_steps,
            total_steps * step_multiplier,
        )


def main() -> None:
    """Run Q-learning training and create Q-value plots."""
    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Configuration - change this to switch between runtimes
    runtime = "distributed"  # or "single_thread" "parallel"

    steps = 1000000
    val_every_n_steps = 10000

    # Training parameters
    num_cores = 8 if runtime == "parallel" else 1
    num_agents = 10
    steps = int(steps / num_agents)
    val_every_n_steps = int(val_every_n_steps / num_agents)

    # Create environments
    env, val_env = create_environments(num_agents, num_cores, runtime)

    # Initialize agent
    agent, state_size, action_size = initialize_agent(env, runtime)

    tracker = None

    if check_print(runtime):
        # Initialize Q-value tracker
        tracker = QValueTracker(state_size, action_size)

        logger.info("Starting Q-learning training...")
        logger.info("Runtime: %s", runtime)
        logger.info("Number of agents: %d", num_agents)
        if runtime == "parallel":
            logger.info("Number of cores: %d", num_cores)
        logger.info("State space size: %d", state_size)
        logger.info("Action space size: %d", action_size)
        logger.info("Training steps per process: %d", steps)

        if runtime == "parallel":
            total_effective_steps = steps * num_agents * num_cores
        else:
            total_effective_steps = steps * num_agents

        logger.info("Total effective steps: %d", total_effective_steps)

    # Run training
    run_training(
        agent,
        env,
        val_env,
        tracker,
        steps,
        val_every_n_steps,
        runtime,
    )

    # Training is complete, final Q-values already recorded by run_training

    if check_print(runtime):
        # Create plots
        assert tracker is not None, "Tracker must be initialized for Q-value plotting"
        logger.info("Creating Q-value evolution plots...")
        script_dir = Path(__file__).parent  # Get the dev_tests directory
        output_dir = script_dir / "q_values_plots"
        tracker.create_plots(str(output_dir))
        logger.info("All plots saved successfully!")


if __name__ == "__main__":
    main()
