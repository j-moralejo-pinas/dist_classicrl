"""This module contains the implementation of multi-agent Q-learning for the Repsol project."""

import multiprocessing as mp
from multiprocessing import connection, shared_memory
from multiprocessing.synchronize import Lock
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from pettingzoo import ParallelEnv
from q_learning import SingleEnvQLearning


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
    sm: shared_memory.SharedMemory
    sm_lock: Lock

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.sm = shared_memory.SharedMemory(create=True, size=self.q_table.nbytes)
        self.q_table = np.ndarray(self.q_table.shape, dtype=self.q_table.dtype, buffer=self.sm.buf)
        self.sm_lock = mp.Lock()

    def train(
        self,
        envs: List[ParallelEnv],
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

        reward_history = []
        val_reward_history = []
        val_agent_reward_history = {}
        reward_queue = mp.Queue()
        curr_states = [None] * len(envs)

        for step in range(0, steps, len(envs) * val_every_n_steps):
            curr_states_pipe_list = []
            process_list = []
            for env, curr_state in zip(envs, curr_states):
                parent_conn, child_conn = mp.Pipe()
                curr_states_pipe_list.append(parent_conn)
                p = mp.Process(
                    target=self.run_steps,
                    args=(env, val_every_n_steps, reward_queue, curr_state, child_conn),
                )
                p.start()
                process_list.append(p)
                child_conn.close()

            curr_states = []

            for p, curr_states_pipe in zip(process_list, curr_states_pipe_list):
                curr_states.append(curr_states_pipe.recv())
                curr_states_pipe.close()
                p.join()

            while not reward_queue.empty():
                reward_history.append(reward_queue.get())

            val_total_rewards, val_agent_rewards = 0.0, {}
            if val_steps is not None:
                val_total_rewards, val_agent_rewards = self.evaluate_steps(val_env, val_steps)
            elif val_episodes is not None:
                val_total_rewards, val_agent_rewards = self.evaluate_episodes(val_env, val_episodes)

            val_reward_history.append(val_total_rewards)
            for agent, reward in val_agent_rewards.items():
                if agent not in val_agent_reward_history:
                    val_agent_reward_history[agent] = []
                val_agent_reward_history[agent].append(reward)
            print(f"Step {step + 1}, Eval total rewards: {val_total_rewards}")

    def run_steps(
        self,
        env: ParallelEnv,
        num_steps: int,
        rewards_queue: mp.Queue,
        sm_name: str,
        sm_lock: Lock,
        curr_state: Optional[Dict] = None,
        curr_state_pipe: Optional[connection.Connection] = None,
    ) -> None:
        """
        Run a single environment with multiple agents for a given number of episodes.

        Parameters
        ----------
        env : Env
            The environment to run.
        agent : MultiAgentQLearning
            The multi-agent Q-learning agent.
        num_agents : int
            Number of agents in the environment.
        episodes : int
            Number of episodes to run.
        max_steps : int
            Maximum number of steps per episode.
        return_queue : mp.Queue
            Queue to collect total rewards from the environment.
        """
        self.sm = shared_memory.SharedMemory(name=sm_name)
        self.q_table = np.ndarray(self.q_table.shape, dtype=self.q_table.dtype, buffer=self.sm.buf)

        if curr_state is None:
            states, infos = env.reset()
            agent_rewards = {}
        else:
            states = curr_state["states"]
            infos = curr_state["infos"]
            agent_rewards = curr_state["agent_rewards"]

        for _ in range(num_steps):
            with sm_lock:
                actions = self.choose_actions(states)
            next_states, rewards, terminated, truncated, infos = env.step(actions)
            with sm_lock:
                self.learn(states, actions, rewards, next_states, terminated)

            states = next_states
            if not states:
                states, infos = env.reset()
                rewards_queue.put(sum(agent_rewards.values()))
                for agent, reward in agent_rewards.items():
                    if agent not in agent_rewards:
                        agent_rewards[agent] = 0
                    agent_rewards[agent] += reward
        if curr_state_pipe is not None:
            curr_state_pipe.send({"states": states, "infos": infos, "agent_rewards": agent_rewards})
            curr_state_pipe.close()
