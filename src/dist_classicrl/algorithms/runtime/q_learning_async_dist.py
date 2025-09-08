"""Multi-agent Q-learning trainer implementation using MPI."""

from __future__ import annotations

import logging
import queue
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

import numpy as np
from mpi4py import MPI

from dist_classicrl.algorithms.runtime.base_runtime import BaseRuntime
from dist_classicrl.environments.custom_env import DistClassicRLEnv

if TYPE_CHECKING:
    from gymnasium.vector import VectorEnv
    from numpy.typing import NDArray

comm = MPI.COMM_WORLD
RANK = comm.Get_rank()
NUM_NODES = comm.Get_size()
MASTER_RANK = 0

logger = logging.getLogger(__name__)


class DistAsyncQLearning(BaseRuntime):
    """
    Distributed asynchronous Q-learning implementation using MPI.

    This class implements a distributed Q-learning algorithm where multiple worker
    nodes run environments in parallel and a master node coordinates training and
    evaluation. The implementation uses MPI for communication between nodes.

    Attributes
    ----------
    num_agents : int
        Number of agents in the environment.
    experience_queue : queue.Queue
        Queue for storing experience tuples.
    batch_size : int
        Batch size for learning updates.
    """

    num_agents: int
    experience_queue: queue.Queue
    batch_size: int

    def init_training(self) -> None:
        """Initialize the training environment."""

    def run_steps(self) -> None:
        """Run training steps."""

    def close_training(self) -> None:
        """Close the training environment."""

    def update_q_table(
        self,
        val_env: DistClassicRLEnv | VectorEnv,
        val_every_n_steps: int,
        val_steps: int | None,
        val_episodes: int | None,
    ) -> list[float]:
        """
        Update Q-table using experiences from the experience queue.

        This method runs in a separate thread and continuously processes experiences
        from the queue to update the Q-table. It also handles validation at
        specified intervals.

        Parameters
        ----------
        val_env : DistClassicRLEnv | VectorEnv
            Environment for validation.
        val_every_n_steps : int
            Number of steps between validation runs.
        val_steps : int | None
            Number of steps for validation (mutually exclusive with val_episodes).
        val_episodes : int | None
            Number of episodes for validation (mutually exclusive with val_steps).

        Returns
        -------
        list[float]
            The validation reward history.
        """
        running = True
        val_reward_history = []
        val_agent_reward_history = []
        step = 0
        steps_since_val = 0
        while running:
            state_batch = []
            next_state_batch = []
            action_batch = []
            reward_batch = []
            terminated_batch = []
            next_action_masks_batch = []
            while len(state_batch) < self.batch_size and steps_since_val < val_every_n_steps:
                try:
                    element = self.experience_queue.get(timeout=0.1)
                except queue.Empty:
                    break
                if element is None:
                    running = False
                    break
                state, action, reward, next_state, terminated = element
                state_batch.append(state)
                action_batch.append(action)
                reward_batch.append(reward)
                if isinstance(next_state, dict):
                    next_state_batch.append(next_state["observation"])
                    next_action_masks_batch.append(next_state["action_mask"])
                else:
                    next_state_batch.append(next_state)
                terminated_batch.append(terminated)
                steps_since_val += 1
                step += 1

            if state_batch:
                np_state_batch = (
                    {"observation": np.fromiter(state_batch, dtype=np.int32)}
                    if next_action_masks_batch
                    else np.fromiter(state_batch, dtype=np.int32)
                )

                np_action_batch = np.fromiter(action_batch, dtype=np.int32)
                np_reward_batch = np.fromiter(reward_batch, dtype=np.float32)
                np_next_state_batch = (
                    {
                        "observation": np.fromiter(next_state_batch, dtype=np.int32),
                        "action_mask": np.array(next_action_masks_batch, dtype=np.int32),
                    }
                    if next_action_masks_batch
                    else np.fromiter(next_state_batch, dtype=np.int32)
                )
                np_terminated_batch = np.fromiter(terminated_batch, dtype=np.bool)

                self._learn(
                    np_state_batch,
                    np_action_batch,
                    np_reward_batch,
                    np_next_state_batch,
                    np_terminated_batch,
                )

            if steps_since_val >= val_every_n_steps:
                val_total_rewards, val_agent_rewards = 0.0, {}
                if val_steps is not None:
                    val_total_rewards, val_agent_rewards = self.evaluate_steps(val_env, val_steps)
                elif val_episodes is not None:
                    val_total_rewards, val_agent_rewards = self.evaluate_episodes(
                        val_env, val_episodes
                    )

                val_reward_history.append(val_total_rewards)
                val_agent_reward_history.append(val_agent_rewards)
                steps_since_val = 0
                logger.debug("Step %d, Eval total rewards: %s", step, val_total_rewards)
        return val_reward_history

    def communicate_master(self, steps: int) -> list[float]:  # noqa: C901 PLR0912
        # TODO(Javier): Try to fix this
        """
        Handle communication between master and worker nodes.

        This method runs on the master node and coordinates with worker nodes
        to collect experiences and send actions. It manages the training loop
        and collects reward history.

        Parameters
        ----------
        steps : int
            Total number of training steps to run.

        Returns
        -------
        list[float]
            History of episode rewards collected during training.
        """
        num_workers = NUM_NODES - 1
        reward_history = []
        worker_rewards = [np.array(()) for _ in range(num_workers)]
        worker_prev_states: list[NDArray[np.int32] | dict[str, NDArray[np.int32]] | None] = [
            None for _ in range(num_workers)
        ]
        worker_prev_actions: list[NDArray[np.int32] | None] = [None for _ in range(num_workers)]
        requests = [comm.irecv(source=worker_id) for worker_id in range(1, num_workers + 1)]
        step = 0
        while step < steps:
            for worker_id, request in enumerate(requests):
                test_flag, data = request.test()
                if not test_flag:
                    continue
                assert data is not None, "Received None from worker"
                (
                    next_states,
                    rewards,
                    terminateds,
                    truncateds,
                    infos,
                    # firsts,
                ) = data
                actions = self._choose_actions(next_states)
                comm.isend(actions, dest=worker_id + 1, tag=1)
                requests[worker_id] = comm.irecv(source=worker_id + 1)

                # if firsts[0]:
                #     worker_rewards[worker_id] = np.zeros(len(firsts), dtype=np.float32)
                if worker_prev_states[worker_id] is None:
                    worker_rewards[worker_id] = np.zeros(len(rewards), dtype=np.float32)
                else:
                    step += 1
                    worker_rewards[worker_id] += rewards
                    if isinstance(next_states, dict):
                        prev_states = worker_prev_states[worker_id]
                        assert isinstance(prev_states, dict)
                        prev_actions = worker_prev_actions[worker_id]
                        assert prev_actions is not None
                        for idx, (
                            next_state,
                            next_action_mask,
                            reward,
                            terminated,
                        ) in enumerate(
                            zip(
                                next_states["observation"],
                                next_states["action_mask"],
                                rewards,
                                terminateds,
                                strict=True,
                            )
                        ):
                            self.experience_queue.put(
                                (
                                    prev_states["observation"][idx],
                                    prev_actions[idx],
                                    reward,
                                    {
                                        "observation": next_state,
                                        "action_mask": next_action_mask,
                                    },
                                    terminated,
                                )
                            )
                    else:
                        prev_states = worker_prev_states[worker_id]
                        assert prev_states is not None
                        assert not isinstance(prev_states, dict)
                        prev_actions = worker_prev_actions[worker_id]
                        assert prev_actions is not None
                        for idx, (
                            next_state,
                            reward,
                            terminated,
                        ) in enumerate(zip(next_states, rewards, terminateds, strict=True)):
                            self.experience_queue.put(
                                (
                                    prev_states[idx],
                                    prev_actions[idx],
                                    reward,
                                    next_state,
                                    terminated,
                                )
                            )

                # Track previous state/action for the next transition
                worker_prev_states[worker_id] = next_states
                worker_prev_actions[worker_id] = actions

                for idx, (terminated, truncated) in enumerate(
                    zip(terminateds, truncateds, strict=True)
                ):
                    if terminated or truncated:
                        reward_history.append(worker_rewards[worker_id][idx])
                        worker_rewards[worker_id][idx] = 0

                if step >= steps:
                    self.experience_queue.put(None)
                    break

        # Wait for all pending requests to complete before sending termination signals
        for request in requests:
            if not request.test()[0]:
                request.wait()
        for worker_id in range(1, num_workers + 1):
            comm.isend(None, dest=worker_id, tag=0)
        return reward_history

    def run_environment(
        self,
        env: DistClassicRLEnv | VectorEnv,
        curr_state_dict: dict | None = None,
    ) -> tuple[DistClassicRLEnv | VectorEnv, dict[str, Any]]:
        """
        Run environment on worker nodes.

        This method runs on worker nodes and handles environment execution.
        It sends environment states to the master node and receives actions
        to execute, creating a continuous loop of environment interaction.

        Parameters
        ----------
        env : DistClassicRLEnv | VectorEnv
            Environment instance to run on this worker node.
        curr_state_dict : dict | None
            Current state dictionary for the environment.

        Returns
        -------
        DistClassicRLEnv | VectorEnv
            The environment instance after running on the worker node.
        dict[str, Any]
            The info dictionary containing additional information.
        """
        status = MPI.Status()

        while True:
            flag = comm.Iprobe(source=MASTER_RANK, tag=MPI.ANY_TAG, status=status)
            if not flag:
                break
            # Consume the pending message
            comm.recv(source=MASTER_RANK)
        comm.Barrier()

        num_agents_or_envs = env.num_agents if isinstance(env, DistClassicRLEnv) else env.num_envs

        if curr_state_dict is None:
            states, infos = env.reset()
        else:
            states = curr_state_dict["states"]
            infos = curr_state_dict["infos"]

        data_sent = (
            states,
            np.fromiter((0.0 for _ in range(num_agents_or_envs)), dtype=np.float32),
            np.fromiter((False for _ in range(num_agents_or_envs)), dtype=np.bool),
            np.fromiter((False for _ in range(num_agents_or_envs)), dtype=np.bool),
            infos,
            # np.fromiter((True for _ in range(num_agents_or_envs)), dtype=np.bool),
        )
        comm.send(
            data_sent,
            dest=MASTER_RANK,
        )
        # agent_first = np.fromiter((False for _ in range(num_agents_or_envs)), dtype=np.bool)

        while True:
            comm.Probe(source=MASTER_RANK, tag=MPI.ANY_TAG, status=status)
            if status.tag == 0:
                break
            actions = comm.recv(source=MASTER_RANK)
            next_states, rewards, terminated, truncated, infos = env.step(actions)

            data_sent = (
                next_states,
                rewards,
                terminated,
                truncated,
                infos,
                # agent_first,
            )

            comm.send(data_sent, dest=MASTER_RANK)

        comm.Barrier()
        return env, {"states": next_states, "infos": infos, "rewards": rewards}

    def train(
        self,
        env: DistClassicRLEnv | VectorEnv,
        steps: int,
        val_env: DistClassicRLEnv | VectorEnv,
        val_every_n_steps: int,
        val_steps: int | None,
        val_episodes: int | None,
        curr_state_dict: dict[str, Any] = {},  # noqa: B006
        *,
        batch_size: int = 32,
    ) -> tuple[
        list[float],
        list[float],
        DistClassicRLEnv | VectorEnv | list[DistClassicRLEnv] | list[VectorEnv] | None,
        dict[str, Any] | None,
    ]:
        """
        Train the agent in the environment for a given number of steps.

        For the master node:
        First, launch 2 threads: one for updating the Q-table and one for communication from master.
        For the worker nodes:
        Run the environment.

        Parameters
        ----------
        env : DistClassicRLEnv | VectorEnv
            The environment to train.
        steps : int
            Number of steps to train.
        val_env : DistClassicRLEnv | VectorEnv
            The validation environment.
        val_every_n_steps : int
            Validate the agent every n steps.
        val_steps : int | None
            Number of steps to validate.
        val_episodes : int | None
            Number of episodes to validate.
        batch_size : int
            Batch size for training.
        auto_cleanup : bool
            Whether to enable automatic cleanup after training ends.

        Return
        ------
        List[float]
            The reward history during training.
        List[float]
            The validation reward history.
        DistClassicRLEnv | VectorEnv | list[DistClassicRLEnv] | list[VectorEnv]
            The current environments.
        dict[str, Any]
            The current state of the environments, including states, infos and episode rewards.
        """
        assert (val_steps is None) ^ (val_episodes is None), (
            "Either val_steps or val_episodes should be provided."
        )

        # Master Node
        if RANK == MASTER_RANK:
            for i in range(1, NUM_NODES):
                while True:
                    flag = comm.Iprobe(source=i, tag=MPI.ANY_TAG, status=MPI.Status())
                    if not flag:
                        break
                    # Consume the pending message
                    comm.recv(source=i)

            # Run the Q-learning update in a separate thread
            comm.Barrier()

            self.experience_queue = queue.Queue(maxsize=-1)
            self.batch_size = batch_size

            with ThreadPoolExecutor() as executor:
                future = executor.submit(
                    self.update_q_table, val_env, val_every_n_steps, val_steps, val_episodes
                )

                # Run the communication, queuing and metric logging in the main thread
                reward_history = self.communicate_master(steps)

                val_reward_history = future.result()
            comm.Barrier()
            return reward_history, val_reward_history, None, None
        # Worker Nodes
        env, curr_state_dict = self.run_environment(env)
        return [], [], env, curr_state_dict
