"""This module contains the implementation of multi-agent Q-learning for the Repsol project."""

import multiprocessing as mp
from multiprocessing import connection, shared_memory
from multiprocessing.synchronize import Lock
from typing import Dict, List, Optional, Tuple, Union

from gymnasium.vector import SyncVectorEnv

import numpy as np
from numpy.typing import NDArray
from dist_classicrl.algorithms.base_algorithms.q_learning_optimal import OptimalQLearningBase
from dist_classicrl.environments.custom_env import DistClassicRLEnv


class ParallelQLearning(OptimalQLearningBase):
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
        self.sm = shared_memory.SharedMemory(name="q_table", create=True, size=self.q_table.nbytes)
        self.q_table = np.ndarray(self.q_table.shape, dtype=self.q_table.dtype, buffer=self.sm.buf)
        self.sm_lock = mp.Lock()

    def train(
        self,
        envs: Union[List[DistClassicRLEnv], List[SyncVectorEnv]],
        steps: int,
        val_env: Union[DistClassicRLEnv, SyncVectorEnv],
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
        try:
            assert (val_steps is None) ^ (
                val_episodes is None
            ), "Either val_steps or val_episodes should be provided."

            reward_history = []
            val_reward_history = []
            val_agent_reward_history = []
            reward_queue = mp.Queue()
            curr_states = [None] * len(envs)

            for step in range(0, steps, val_every_n_steps):
                curr_states_pipe_list = []
                process_list = []
                for env, curr_state in zip(envs, curr_states):
                    parent_conn, child_conn = mp.Pipe()
                    curr_states_pipe_list.append(parent_conn)
                    p = mp.Process(
                        target=self.run_steps,
                        args=(
                            env,
                            int(val_every_n_steps / len(envs)),
                            reward_queue,
                            self.sm.name,
                            self.sm_lock,
                            child_conn,
                            curr_state,
                        ),
                        daemon=True,
                    )
                    p.start()
                    process_list.append(p)
                    child_conn.close()

                curr_states = []
                envs = []

                for p, curr_states_pipe in zip(process_list, curr_states_pipe_list):
                    curr_state = curr_states_pipe.recv()
                    envs.append(curr_state["env"])
                    curr_state.pop("env")
                    curr_states.append(curr_state)
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
                val_agent_reward_history.append(val_agent_rewards)
                print(f"Step {step * len(envs)}, Eval total rewards: {val_total_rewards}")
        finally:
            self.sm.close()
            self.sm.unlink()


    def run_steps(
        self,
        env: Union[DistClassicRLEnv, SyncVectorEnv],
        num_steps: int,
        rewards_queue: mp.Queue,
        sm_name: str,
        sm_lock: Lock,
        curr_state_pipe: Optional[connection.Connection],
        curr_state: Optional[Dict] = None,
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
            if isinstance(states, dict):
                n_agents = len(states["observation"])
            else:
                n_agents = len(states)
            agent_rewards = np.zeros(n_agents, dtype=np.float32)
        else:
            states = curr_state["states"]
            infos = curr_state["infos"]
            agent_rewards = curr_state["agent_rewards"]

        for _ in range(num_steps):
            if isinstance(states, dict):
                with sm_lock:
                    actions = self.choose_actions(
                        states=states["observation"], action_masks=states["action_mask"]
                    )
            else:
                with sm_lock:
                    actions = self.choose_actions(states)

            next_states, rewards, terminateds, truncateds, infos = env.step(actions)

            agent_rewards += rewards

            if isinstance(next_states, dict):
                assert isinstance(states, dict)
                with sm_lock:
                    self.learn(
                        states["observation"],
                        actions,
                        rewards,
                        next_states["observation"],
                        terminateds,
                        next_states["action_mask"],
                    )
            else:
                assert not isinstance(states, dict)
                with sm_lock:
                    self.learn(states, actions, rewards, next_states, terminateds)

            states = next_states

            for i, (terminated, truncated) in enumerate(zip(terminateds, truncateds)):
                if terminated or truncated:
                    rewards_queue.put(agent_rewards[i])
                    agent_rewards[i] = 0

        if curr_state_pipe is not None:
            curr_state_pipe.send(
                {"env": env, "states": states, "infos": infos, "agent_rewards": agent_rewards}
            )
            curr_state_pipe.close()

    def evaluate_steps(
        self,
        env: Union[DistClassicRLEnv, SyncVectorEnv],
        steps: int,
    ) -> Tuple[float, List[float]]:
        """
        Evaluate the agent in the environment for a given number of steps.

        Parameters
        ----------
        env : Env
            The environment to evaluate.
        steps : int
            Number of steps to evaluate.

        Returns
        -------
        Tuple[float, Dict[Any, float]]
            Total rewards obtained by the agent and rewards for each agent.
        """

        states, infos = env.reset(seed=42)
        if isinstance(states, dict):
            n_agents = len(states["observation"])
        else:
            n_agents = len(states)
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
            for i, (terminated, truncated) in enumerate(zip(terminateds, truncateds)):
                if terminated or truncated:
                    reward_history.append(agent_rewards[i])
                    agent_rewards[i] = 0
        return sum(reward_history), reward_history

    def evaluate_episodes(
        self,
        env: Union[DistClassicRLEnv, SyncVectorEnv],
        episodes: int,
    ) -> Tuple[float, List[float]]:
        """
        Evaluate the agent in the environment for a given number of episodes.

        Parameters
        ----------
        env : Env
            The environment to evaluate.
        episodes : int
            Number of episodes to evaluate.

        Returns
        -------
        Tuple[float, Dict[Any, float]]
            Total rewards obtained by the agent and rewards for each agent.
        """

        states, infos = env.reset(seed=42)
        if isinstance(states, dict):
            n_agents = len(states["observation"])
        else:
            n_agents = len(states)
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
            for i, (terminated, truncated) in enumerate(zip(terminateds, truncateds)):
                if terminated or truncated:
                    episode += 1
                    reward_history.append(agent_rewards[i])
                    agent_rewards[i] = 0

        return sum(reward_history), reward_history
