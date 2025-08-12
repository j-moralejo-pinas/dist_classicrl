"""Multi-agent Q-learning trainer implementation using multiprocessing."""

# TODO(Javier): Fix shared memory staying open after Ctrl+C
from __future__ import annotations

import logging
import multiprocessing as mp
from multiprocessing import connection, shared_memory
from signal import SIGTERM, signal
from typing import TYPE_CHECKING, Any

import numpy as np

from dist_classicrl.algorithms.runtime.base_runtime import BaseRuntime

if TYPE_CHECKING:
    from multiprocessing.sharedctypes import Synchronized
    from multiprocessing.synchronize import Lock

    from gymnasium.vector import VectorEnv

    from dist_classicrl.environments.custom_env import DistClassicRLEnv

logger = logging.getLogger(__name__)


class ParallelQLearning(BaseRuntime):
    """
    Single environment Q-learning agent.

    Parameters
    ----------
    *args : Any
        Variable length argument list for base class initialization.
    **kwargs : Any
        Arbitrary keyword arguments for base class initialization.

    Attributes
    ----------
    num_agents : int
        Number of agents in the environment.
    learning_rate : Synchronized
        Shared learning rate value.
    exploration_rate : Synchronized
        Shared exploration rate value.
    sm : shared_memory.SharedMemory
        Shared memory object for the Q-table.
    sm_lock : Lock
        Lock for synchronizing access to shared memory.
    sm_name : str
        Name of the shared memory object.
    """

    num_agents: int
    learning_rate: Synchronized
    exploration_rate: Synchronized
    sm: shared_memory.SharedMemory
    sm_lock: Lock

    sm_name: str = "q_table"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.sm_lock = mp.Lock()
        self.lr_schedule.set_mp()
        self.exploration_rate_schedule.set_mp()

    def init_training(self) -> None:
        """Initialize the training environment."""
        self.sm = shared_memory.SharedMemory(
            name=self.sm_name, create=True, size=self.algorithm.q_table.nbytes
        )
        new_q_table = np.ndarray(
            self.algorithm.q_table.shape, dtype=self.algorithm.q_table.dtype, buffer=self.sm.buf
        )
        new_q_table[:] = self.algorithm.q_table[:]
        self.algorithm.q_table = new_q_table
        signal(SIGTERM, self.handle_sigterm)

    def run_steps(
        self,
        steps: int,
        env: list[DistClassicRLEnv] | list[VectorEnv],
        curr_state_dict: list[dict] | None,
    ) -> tuple[float, list[float], list[DistClassicRLEnv | VectorEnv], list[dict[str, Any]]]:
        """
        Run the training steps.

        Parameters
        ----------
        steps : int
            The number of steps to run.
        env : DistClassicRLEnv | VectorEnv
            The environment to train.
        states : Any
            The current states of the environments.
        infos : Any
            The current infos of the environments.
        agent_rewards : np.ndarray
            The rewards received by the agent.

        Returns
        -------
        float
            The average training reward.
        list[float]
            The list of episode rewards.
        DistClassicRLEnv | VectorEnv
            The current environments.
        dict[str, Any]
            The current state of the environments, including states, infos and episode rewards.
        """
        if curr_state_dict is None:
            curr_states = [(env, None) for env in env]
        else:
            curr_states = [(env, state) for (env, state) in zip(env, curr_state_dict, strict=True)]

        reward_history = []

        curr_states_pipe_list = []
        process_list = []
        for curr_state in curr_states:
            parent_conn, child_conn = mp.Pipe()
            curr_states_pipe_list.append(parent_conn)
            p = mp.Process(
                target=self.run_process,
                args=(
                    curr_state,
                    int(steps / len(env)),
                    self.sm_lock,
                    self.exploration_rate_schedule.value,
                    self.lr_schedule.value,
                    child_conn,
                ),
                daemon=True,
            )
            p.start()
            process_list.append(p)
            child_conn.close()

        curr_states = []

        reward_histories: list[list[float]] = []

        for p, curr_states_pipe in zip(process_list, curr_states_pipe_list, strict=False):
            curr_state = curr_states_pipe.recv()
            assert curr_state is not None, "Current state cannot be None"

            # Extract episode rewards from pipe communication
            if "episode_rewards" in curr_state[1]:
                reward_histories.append(curr_state[1]["episode_rewards"])
                del curr_state[1]["episode_rewards"]
            curr_states.append(curr_state)
            curr_states_pipe.close()
            p.join()

        for reward_history_step in zip(*reward_histories, strict=False):
            reward_history.extend([r for r in reward_history_step if r is not None])

        return (
            sum(reward_history) / len(reward_history) if reward_history else 0.0,
            reward_history,
            [curr_state[0] for curr_state in curr_states],
            [curr_state[1] for curr_state in curr_states if curr_state[1] is not None],
        )

    def handle_sigterm(self, signum: int, frame: signal.FrameType | None) -> None:  # noqa: ARG002
        """Handle SIGTERM signal."""
        self.close_training()

    def close_training(self) -> None:
        """Close training."""
        q_table_copy = self.algorithm.q_table.copy()
        self.sm.close()
        self.sm.unlink()
        self.algorithm.q_table = q_table_copy

    def run_process(
        self,
        curr_state: tuple[DistClassicRLEnv | VectorEnv, dict | None],
        num_steps: int,
        sm_lock: Lock,
        exploration_rate: Synchronized,
        learning_rate: Synchronized,
        curr_state_pipe: connection.Connection | None,
    ) -> None:
        """
        Run a single environment with multiple agents for a given number of steps.

        Parameters
        ----------
        curr_state : tuple[DistClassicRLEnv | VectorEnv, tuple[Any, Any, Any] | None]
            The current state of the environment. It contains the environment instance
            and, optionally, another dict with the states, infos and episode rewards.
        num_steps : int
            Number of steps to run.
        sm_lock : Lock
            Lock for synchronizing access to shared memory.
        exploration_rate : Synchronized
            Shared exploration rate value.
        learning_rate : Synchronized
            Shared learning rate value.
        curr_state_pipe : connection.Connection | None
            Pipe for communicating current state.
        curr_state : dict | None
            Current state dictionary, by default None.
        """
        try:
            self.sm_lock = sm_lock
            self.lr_schedule.set_value(learning_rate)
            self.exploration_rate_schedule.set_value(exploration_rate)
            self.sm = shared_memory.SharedMemory(name=self.sm_name)
            self.algorithm.q_table = np.ndarray(
                self.algorithm.q_table.shape, dtype=self.algorithm.q_table.dtype, buffer=self.sm.buf
            )

            env = curr_state[0]

            if curr_state[1] is None:
                states, infos = env.reset()
                n_agents = len(states["observation"]) if isinstance(states, dict) else len(states)
                agent_rewards = np.zeros(n_agents, dtype=np.float32)
            else:
                states = curr_state[1]["states"]
                infos = curr_state[1]["infos"]
                agent_rewards = curr_state[1]["rewards"]

            # Collect episode rewards locally to avoid queue deadlock
            episode_rewards = []

            for _ in range(num_steps):
                states, infos = self.run_single_step(env, states, agent_rewards, episode_rewards)

            if curr_state_pipe is not None:
                curr_state_pipe.send(
                    (
                        env,
                        {
                            "states": states,
                            "infos": infos,
                            "rewards": agent_rewards,
                            "episode_rewards": episode_rewards,
                        },
                    )
                )
        finally:
            self.sm.close()
            curr_state_pipe.close() if curr_state_pipe is not None else None

    def _learn(
        self, states: Any, actions: Any, rewards: Any, next_states: Any, terminateds: Any
    ) -> None:
        with self.sm_lock:
            super()._learn(states, actions, rewards, next_states, terminateds)

    def _choose_actions(self, states: Any) -> Any:
        with self.sm_lock:
            return super()._choose_actions(states)
