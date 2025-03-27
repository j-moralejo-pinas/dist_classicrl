"""This module contains the implementation of multi-agent Q-learning for the Repsol project."""

import queue
import threading
import gymnasium
from mpi4py import MPI
from typing import Dict, List, Optional, Union

import numpy as np
from numpy.typing import NDArray
from dist_classicrl.custom_env import DistClassicRLEnv
from dist_classicrl.q_learning import SingleEnvQLearning
from gymnasium.vector import SyncVectorEnv

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
MASTER_RANK = 0


class DistAsyncQLearning(SingleEnvQLearning):
    """
    Single environment Q-learning agent.

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
    q_table : mp.Array
        Shared memory array for the Q-table.
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

    def update_q_table(self):
        running = True
        while running:
            with self.experience_queue.not_empty:
                while self.experience_queue.empty():
                    self.experience_queue.not_empty.wait()

            state_batch = []
            action_batch = []
            reward_batch = []
            next_state_batch = []
            terminated_batch = []
            action_masks_batch = []
            while not self.experience_queue.empty() and len(state_batch) < self.batch_size:
                element = self.experience_queue.get()
                if element is None:
                    running = False
                    break
                state, action, reward, next_state, terminated = element
                if isinstance(state, dict):
                    state_batch.append(state["observation"])
                    action_masks_batch.append(state["action_mask"])
                else:
                    state_batch.append(state)
                action_batch.append(action)
                reward_batch.append(reward)
                next_state_batch.append(next_state)
                terminated_batch.append(terminated)
            np_state_batch = np.fromiter(state_batch, dtype=np.int32)
            np_action_batch = np.fromiter(action_batch, dtype=np.int32)
            np_reward_batch = np.fromiter(reward_batch, dtype=np.float32)
            np_next_state_batch = np.fromiter(next_state_batch, dtype=np.int32)
            np_terminated_batch = np.fromiter(terminated_batch, dtype=np.bool)
            np_action_masks_batch = (
                np.fromiter(action_masks_batch, dtype=np.int32)
                if len(action_masks_batch) > 0
                else None
            )
            self._learn_vec(
                np_state_batch,
                np_action_batch,
                np_reward_batch,
                np_next_state_batch,
                np_terminated_batch,
                np_action_masks_batch,
            )

    def communicate_master(self, steps: int) -> List[float]:
        num_workers = 10

        reward_history = []
        worker_rewards = [np.array(()) for _ in range(num_workers)]
        worker_prev_states: List[Union[NDArray[np.int32], Dict[str, NDArray[np.int32]]]] = [
            np.array(()) for _ in range(num_workers)
        ]
        requests = [
            comm.irecv(source=worker_id, tag=worker_id) for worker_id in range(1, num_workers)
        ]
        step = 0
        while step >= steps:
            for worker_id, request in enumerate(requests):
                worker_id += 1
                if request.Test():
                    step += 1
                    next_states, rewards, terminateds, truncateds, infos, firsts = request.wait()
                    if isinstance(next_states, dict):
                        actions = self.choose_actions(
                            next_states["observation"], action_masks=next_states["action_mask"]
                        )
                    else:
                        actions = self.choose_actions(next_states)
                    comm.isend(actions, dest=worker_id, tag=1)

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
                            ) in enumerate(
                                zip(
                                    next_states["observation"],
                                    next_states["action_mask"],
                                    rewards,
                                    terminateds,
                                )
                            ):
                                prev_states = worker_prev_states[worker_id]
                                assert isinstance(prev_states, dict)
                                self.experience_queue.put(
                                    (
                                        {
                                            "observation": prev_states["observation"][idx],
                                            "action_mask": prev_states["action_mask"][idx],
                                        },
                                        next_action_mask,
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
                            ) in enumerate(zip(next_states, rewards, terminateds, actions)):
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

                    for idx, (terminated, truncated) in enumerate(zip(terminateds, truncateds)):
                        if terminated or truncated:
                            reward_history.append(worker_rewards[worker_id][idx])
                            worker_rewards[worker_id][idx] = 0

                    worker_prev_states[worker_id] = next_states
                if step >= steps:
                    self.experience_queue.put(None)
                    break
        for worker_id in range(1, num_workers):
            comm.isend(None, dest=worker_id, tag=0)
        return reward_history

    def run_environment(self, env: DistClassicRLEnv):
        status = MPI.Status()
        states, infos = env.reset()
        rewards = 0
        comm.send(
            (
                states,
                np.fromiter((0.0 for _ in range(len(infos))), dtype=np.float32),
                np.fromiter((False for _ in range(len(infos))), dtype=np.bool),
                np.fromiter((False for _ in range(len(infos))), dtype=np.bool),
                infos,
                np.fromiter((True for _ in range(len(infos))), dtype=np.bool),
            ),
            dest=MASTER_RANK,
            tag=rank,
        )

        agent_first = np.fromiter((False for _ in range(len(infos))), dtype=np.bool)

        while True:
            comm.Probe(source=MASTER_RANK, tag=MPI.ANY_TAG, status=status)
            if status.tag == 0:
                break
            actions = comm.recv(source=MASTER_RANK, tag=rank)
            next_states, rewards, terminated, truncated, infos = env.step(actions)
            comm.send(
                (next_states, rewards, terminated, truncated, infos, agent_first),
                dest=MASTER_RANK,
                tag=rank,
            )

    def train(
        self,
        env: DistClassicRLEnv,
        steps: int,
        val_env: DistClassicRLEnv,
        val_every_n_steps: int,
        val_steps: Optional[int],
        val_episodes: Optional[int],
    ) -> None:
        """
        Train the agent in the environment for a given number of steps.
        For the master node:
        First, launch 2 threads: one for updating the Q-table and one for communication from master.
        For the worker nodes:
        Run the environment

        Parameters
        ----------
        env : Env
            The environment to train.
        steps : int
            Number of steps to train.
        eval_env : Env
            The evaluation environment.
        eval_steps : int
            Number of steps to evaluate.
        eval_every_n_steps : int
            Evaluate the agent every n steps.
        """
        assert (val_steps is None) ^ (
            val_episodes is None
        ), "Either val_steps or val_episodes should be provided."

        # Master Node
        if rank == MASTER_RANK:
            # Run the Q-learning update in a separate thread
            self.experience_queue = queue.Queue()
            self.batch_size = 32
            update_thread = threading.Thread(target=self.update_q_table)

            # Run the communication, queuing and metric logging in the main thread
            reward_history = self.communicate_master(steps)
            update_thread.join()
            return

        # Worker Nodes
        else:
            self.run_environment(env)
            return
