"""This module contains the implementation of multi-agent Q-learning for the Repsol project."""

import queue
from mpi4py import MPI
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from pettingzoo import ParallelEnv
from q_learning import SingleEnvQLearning

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
MASTER_RANK = 0


class ParallelQLearning(SingleEnvQLearning):
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
            while not self.experience_queue.empty() and len(state_batch) < self.batch_size:
                element = self.experience_queue.get()
                if element is None:
                    running = False
                    break
                state, action, reward, next_state, terminated = self.experience_queue.get()
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
            self._learn_vec(
                np_state_batch,
                np_action_batch,
                np_reward_batch,
                np_next_state_batch,
                np_terminated_batch,
            )

    def communicate_master(self, steps: int):
        num_workers = 10

        worker_rewards = [[] for _ in range(num_workers)]
        worker_prev_states = [{} for _ in range(num_workers)]
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
                    actions = self.choose_actions(next_states)
                    comm.isend(actions, dest=worker_id, tag=1)
                    for agent, first in firsts.items():
                        if not first:
                            self.experience_queue.put(
                                (
                                    worker_prev_states[worker_id][agent],
                                    actions[agent],
                                    rewards[agent],
                                    next_states[agent],
                                    terminateds[agent],
                                )
                            )
                        worker_prev_states[worker_id][agent] = next_states[agent]
                        worker_prev_states[worker_id] = next_states
                        worker_rewards[worker_id].append(rewards)
                if step >= steps:
                    self.experience_queue.put(None)
                    break
        for worker_id in range(1, num_workers):
            comm.isend(None, dest=worker_id, tag=0)

    def run_environment(self, env: ParallelEnv):
        status = MPI.Status()
        states, infos = env.reset()
        rewards = 0
        terminated = False
        truncated = False
        comm.send(
            (
                states,
                {agent: 0.0 for agent in states.keys()},
                {agent: False for agent in states.keys()},
                {agent: False for agent in states.keys()},
                infos,
                {agent: True for agent in states.keys()},
            ),
            dest=MASTER_RANK,
            tag=rank,
        )

        agent_first = {agent: False for agent in states.keys()}

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
            agent_first = {agent: truncated[agent] and terminated[agent] for agent in states.keys()}

    def train(
        self,
        env: ParallelEnv,
        steps: int,
        val_env: ParallelEnv,
        val_every_n_steps: int,
        val_steps: Optional[int],
        val_episodes: Optional[int],
    ) -> None:
        """
        Train the agent in the environment for a given number of steps.

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
        reward_history = [0.0]
        agent_reward_history = {}
        val_reward_history = []
        val_agent_reward_history = {}
        states, infos = env.reset()
        for step in range(steps):

            actions = self.choose_actions(states)
            next_states, rewards, terminated, truncated, infos = env.step(actions)

            for agent, reward in rewards.items():
                if agent not in agent_reward_history:
                    agent_reward_history[agent] = [0]
                agent_reward_history[agent][-1] += reward

            self.learn(states, actions, rewards, next_states, terminated)
            states = next_states

            if not states:
                states, infos = env.reset()
                for agent, reward in agent_reward_history.items():
                    reward_history[-1] += reward[-1]
                    agent_reward_history[agent].append(0)

            if (step + 1) % val_every_n_steps == 0:
                val_total_rewards, val_agent_rewards = 0.0, {}
                if val_steps is not None:
                    val_total_rewards, val_agent_rewards = self.evaluate_steps(val_env, val_steps)
                elif val_episodes is not None:
                    val_total_rewards, val_agent_rewards = self.evaluate_episodes(
                        val_env, val_episodes
                    )

                val_reward_history.append(val_total_rewards)
                for agent, reward in val_agent_rewards.items():
                    if agent not in val_agent_reward_history:
                        val_agent_reward_history[agent] = []
                    val_agent_reward_history[agent].append(reward)
                print(f"Step {step + 1}, Eval total rewards: {val_total_rewards}")
