"""Multi-agent Q-learning trainer implementation using MPI."""

from __future__ import annotations

import logging
import queue
import threading
from typing import TYPE_CHECKING

import numpy as np
from mpi4py import MPI

from dist_classicrl.algorithms.runtime.single_thread_runtime import (
    OptimalQLearningBase,
)
from dist_classicrl.environments.custom_env import DistClassicRLEnv

if TYPE_CHECKING:
    from gymnasium.vector import VectorEnv
    from numpy.typing import NDArray

comm = MPI.COMM_WORLD
RANK = comm.Get_rank()
NUM_NODES = comm.Get_size()
MASTER_RANK = 0

logger = logging.getLogger(__name__)


class DistAsyncQLearning(OptimalQLearningBase):
    """
    Distributed asynchronous Q-learning implementation using MPI.

    This class implements a distributed Q-learning algorithm where multiple worker
    nodes run environments in parallel and a master node coordinates training and
    evaluation. The implementation uses MPI for communication between nodes.

    Attributes
    ----------
    num_agents : int
        Number of agents in the environment.
    state_size : int
        Size of the state space.
    action_size : int
        Size of the action space.
    learning_rate : float
        Learning rate for Q-learning.
    discount_factor : float
        Discount factor for future rewards.
    exploration_rate : float
        Initial exploration rate for epsilon-greedy policy.
    exploration_decay : float
        Decay rate for exploration rate.
    min_exploration_rate : float
        Minimum exploration rate.
    q_table : NDArray[np.float32]
        Q-table for storing state-action values.
    stable_q_table : NDArray[np.float32]
        Stable copy of Q-table for evaluation.
    experience_queue : queue.Queue
        Queue for storing experience tuples.
    batch_size : int
        Batch size for learning updates.
    """

    num_agents: int
    state_size: int
    action_size: int
    learning_rate: float
    discount_factor: float
    exploration_rate: float
    exploration_decay: float
    min_exploration_rate: float
    q_table: NDArray[np.float32]
    stable_q_table: NDArray[np.float32]

    experience_queue: queue.Queue
    batch_size: int

    def update_q_table(
        self,
        val_env: DistClassicRLEnv | VectorEnv,
        val_every_n_steps: int,
        val_steps: int | None,
        val_episodes: int | None,
    ) -> None:
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

            if not state_batch:
                continue

            np_state_batch = np.fromiter(state_batch, dtype=np.int32)
            np_action_batch = np.fromiter(action_batch, dtype=np.int32)
            np_reward_batch = np.fromiter(reward_batch, dtype=np.float32)
            np_next_state_batch = np.fromiter(next_state_batch, dtype=np.int32)
            np_terminated_batch = np.fromiter(terminated_batch, dtype=np.bool)
            np_next_action_masks_batch = (
                np.array(next_action_masks_batch, dtype=np.int32)
                if next_action_masks_batch
                else None
            )
            self.learn(
                np_state_batch,
                np_action_batch,
                np_reward_batch,
                np_next_state_batch,
                np_terminated_batch,
                np_next_action_masks_batch,
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
        worker_prev_states: list[NDArray[np.int32] | dict[str, NDArray[np.int32]]] = [
            np.array(()) for _ in range(num_workers)
        ]
        requests = [comm.irecv(source=worker_id) for worker_id in range(1, num_workers + 1)]
        step = 0
        while step < steps:
            for worker_id, request in enumerate(requests):
                test_flag, data = request.test()
                if not test_flag:
                    continue
                assert data is not None, "Received None from worker"
                step += 1
                next_states, rewards, terminateds, truncateds, infos, firsts = data
                if isinstance(next_states, dict):
                    actions = self.choose_actions(
                        next_states["observation"], action_masks=next_states["action_mask"]
                    )
                else:
                    actions = self.choose_actions(next_states)
                comm.isend(actions, dest=worker_id + 1, tag=1)
                requests[worker_id] = comm.irecv(source=worker_id + 1)

                if firsts[0]:
                    worker_rewards[worker_id] = np.zeros(len(firsts), dtype=np.float32)
                else:
                    worker_rewards[worker_id] += rewards
                    if isinstance(next_states, dict):
                        for idx, (
                            next_state,
                            next_action_mask,
                            reward,
                            terminated,
                            action,
                        ) in enumerate(
                            zip(
                                next_states["observation"],
                                next_states["action_mask"],
                                rewards,
                                terminateds,
                                actions,
                                strict=True,
                            )
                        ):
                            prev_states = worker_prev_states[worker_id]
                            assert isinstance(prev_states, dict)
                            self.experience_queue.put(
                                (
                                    prev_states["observation"][idx],
                                    action,
                                    reward,
                                    {
                                        "observation": next_state,
                                        "action_mask": next_action_mask,
                                    },
                                    terminated,
                                )
                            )
                    else:
                        for idx, (
                            next_state,
                            reward,
                            terminated,
                            action,
                        ) in enumerate(
                            zip(next_states, rewards, terminateds, actions, strict=True)
                        ):
                            prev_states = worker_prev_states[worker_id]
                            assert not isinstance(prev_states, dict)
                            self.experience_queue.put(
                                (
                                    prev_states[idx],
                                    action,
                                    reward,
                                    next_state,
                                    terminated,
                                )
                            )

                for idx, (terminated, truncated) in enumerate(
                    zip(terminateds, truncateds, strict=True)
                ):
                    if terminated or truncated:
                        reward_history.append(worker_rewards[worker_id][idx])
                        worker_rewards[worker_id][idx] = 0

                worker_prev_states[worker_id] = next_states
                if step >= steps:
                    self.experience_queue.put(None)
                    break
        for worker_id in range(1, num_workers + 1):
            comm.isend(None, dest=worker_id, tag=0)
        return reward_history

    def run_environment(self, env: DistClassicRLEnv | VectorEnv) -> None:
        """
        Run environment on worker nodes.

        This method runs on worker nodes and handles environment execution.
        It sends environment states to the master node and receives actions
        to execute, creating a continuous loop of environment interaction.

        Parameters
        ----------
        env : DistClassicRLEnv | VectorEnv
            Environment instance to run on this worker node.
        """
        status = MPI.Status()

        num_agents_or_envs = env.num_agents if isinstance(env, DistClassicRLEnv) else env.num_envs

        states, infos = env.reset()
        rewards = 0
        data_sent = (
            states,
            np.fromiter((0.0 for _ in range(num_agents_or_envs)), dtype=np.float32),
            np.fromiter((False for _ in range(num_agents_or_envs)), dtype=np.bool),
            np.fromiter((False for _ in range(num_agents_or_envs)), dtype=np.bool),
            infos,
            np.fromiter((True for _ in range(num_agents_or_envs)), dtype=np.bool),
        )
        comm.send(
            data_sent,
            dest=MASTER_RANK,
        )
        agent_first = np.fromiter((False for _ in range(num_agents_or_envs)), dtype=np.bool)

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
                agent_first,
            )

            comm.send(data_sent, dest=MASTER_RANK)

    def train(
        self,
        env: DistClassicRLEnv | VectorEnv,
        steps: int,
        val_env: DistClassicRLEnv | VectorEnv,
        val_every_n_steps: int,
        val_steps: int | None,
        val_episodes: int | None,
        batch_size: int = 32,
    ) -> None:
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
        """
        assert (val_steps is None) ^ (val_episodes is None), (
            "Either val_steps or val_episodes should be provided."
        )

        # Master Node
        if RANK == MASTER_RANK:
            # Run the Q-learning update in a separate thread
            self.experience_queue = queue.Queue(maxsize=-1)
            self.batch_size = batch_size
            update_thread = threading.Thread(
                target=self.update_q_table,
                args=(val_env, val_every_n_steps, val_steps, val_episodes),
                daemon=True,
            )
            update_thread.start()
            # Run the communication, queuing and metric logging in the main thread
            self.communicate_master(steps)
            update_thread.join()
        # Worker Nodes
        else:
            self.run_environment(env)

    def evaluate_steps(
        self,
        env: DistClassicRLEnv | VectorEnv,
        steps: int,
    ) -> tuple[float, list[float]]:
        """
        Evaluate the agent in the environment for a given number of steps.

        Parameters
        ----------
        env : DistClassicRLEnv | VectorEnv
            The environment to evaluate.
        steps : int
            Number of steps to evaluate.

        Returns
        -------
        tuple[float, list[float]]
            Total rewards obtained by the agent and rewards for each agent.
        """
        states, infos = env.reset(seed=42)
        n_agents = len(states["observation"]) if isinstance(states, dict) else len(states)
        agent_rewards = np.zeros(n_agents, dtype=np.float32)
        reward_history = []
        for _ in range(steps):
            if isinstance(states, dict):
                actions = self.choose_actions(
                    states=states["observation"],
                    action_masks=states["action_mask"],
                    deterministic=True,
                )
            else:
                actions = self.choose_actions(states, deterministic=True)
            next_states, rewards, terminateds, truncateds, infos = env.step(actions)
            agent_rewards += rewards
            states = next_states
            for i, (terminated, truncated) in enumerate(zip(terminateds, truncateds, strict=True)):
                if terminated or truncated:
                    reward_history.append(agent_rewards[i])
                    agent_rewards[i] = 0
        return sum(reward_history), reward_history

    def evaluate_episodes(
        self,
        env: DistClassicRLEnv | VectorEnv,
        episodes: int,
    ) -> tuple[float, list[float]]:
        """
        Evaluate the agent in the environment for a given number of episodes.

        Parameters
        ----------
        env : DistClassicRLEnv | VectorEnv
            The environment to evaluate.
        episodes : int
            Number of episodes to evaluate.

        Returns
        -------
        tuple[float, list[float]]
            Total rewards obtained by the agent and rewards for each agent.
        """
        states, infos = env.reset(seed=42)
        n_agents = len(states["observation"]) if isinstance(states, dict) else len(states)
        agent_rewards = np.zeros(n_agents, dtype=np.float32)
        reward_history = []
        episode = 0
        while episode < episodes:
            if isinstance(states, dict):
                actions = self.choose_actions(
                    states=states["observation"],
                    action_masks=states["action_mask"],
                    deterministic=True,
                )
            else:
                actions = self.choose_actions(states, deterministic=True)
            next_states, rewards, terminateds, truncateds, infos = env.step(actions)
            agent_rewards += rewards
            states = next_states
            for i, (terminated, truncated) in enumerate(zip(terminateds, truncateds, strict=True)):
                if terminated or truncated:
                    episode += 1
                    reward_history.append(agent_rewards[i])
                    agent_rewards[i] = 0

        return sum(reward_history), reward_history
