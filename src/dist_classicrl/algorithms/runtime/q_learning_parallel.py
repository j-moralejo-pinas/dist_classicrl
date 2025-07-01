"""Multi-agent Q-learning trainer implementation using multiprocessing."""

# TODO(Javier): Fix shared memory staying open after Ctrl+C
from __future__ import annotations

import logging
import multiprocessing as mp
from multiprocessing import Value, connection, shared_memory
from typing import TYPE_CHECKING, Any

import numpy as np

from dist_classicrl.algorithms.base_algorithms.q_learning_optimal import (
    OptimalQLearningBase,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from multiprocessing.sharedctypes import Synchronized
    from multiprocessing.synchronize import Lock

    from gymnasium.vector import VectorEnv
    from numpy.typing import NDArray

    from dist_classicrl.environments.custom_env import DistClassicRLEnv

logger = logging.getLogger(__name__)


class ParallelQLearning(OptimalQLearningBase):
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
        Shared memory array for the Q-table.
    sm : shared_memory.SharedMemory
        Shared memory object for the Q-table.
    sm_lock : Lock
        Lock for synchronizing access to shared memory.
    sm_name : str
        Name of the shared memory object.
    """

    num_agents: int
    state_size: int
    action_size: int
    learning_rate: float
    discount_factor: float
    exploration_rate: float
    _exploration_rate: Synchronized
    exploration_decay: float
    min_exploration_rate: float
    q_table: NDArray[np.float32]
    sm: shared_memory.SharedMemory
    sm_lock: Lock

    sm_name: str = "q_table"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.sm = shared_memory.SharedMemory(
            name=self.sm_name, create=True, size=self.q_table.nbytes
        )
        self.q_table = np.ndarray(self.q_table.shape, dtype=self.q_table.dtype, buffer=self.sm.buf)
        self.sm_lock = mp.Lock()
        self._exploration_rate = Value("f", self.exploration_rate)

    def update_explore_rate(self) -> None:
        """
        Update the exploration rate.

        This method overrides the base class method to ensure thread safety when updating the
        exploration rate in a multi-agent setting.
        """
        self._exploration_rate.value = max(
            self._exploration_rate.value * self.exploration_decay,
            self.min_exploration_rate,
        )
        self.exploration_rate = self._exploration_rate.value

    def train(
        self,
        envs: Sequence[DistClassicRLEnv] | Sequence[VectorEnv],
        steps: int,
        val_env: DistClassicRLEnv | VectorEnv,
        val_every_n_steps: int,
        val_steps: int | None,
        val_episodes: int | None,
    ) -> None:
        """
        Train the agent in the environment for a given number of steps.

        Parameters
        ----------
        envs : Sequence[DistClassicRLEnv] | Sequence[VectorEnv]
            The environments to train.
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
        """
        try:
            assert (val_steps is None) ^ (val_episodes is None), (
                "Either val_steps or val_episodes should be provided."
            )

            reward_history = []
            val_reward_history = []
            val_agent_reward_history = []
            reward_queue = mp.Queue()
            curr_states = [None] * len(envs)

            for step in range(0, steps, val_every_n_steps):
                curr_states_pipe_list = []
                process_list = []
                for env, curr_state in zip(envs, curr_states, strict=False):
                    parent_conn, child_conn = mp.Pipe()
                    curr_states_pipe_list.append(parent_conn)
                    p = mp.Process(
                        target=self.run_steps,
                        args=(
                            env,
                            int(val_every_n_steps / len(envs)),
                            reward_queue,
                            self.sm_lock,
                            self._exploration_rate,
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

                for p, curr_states_pipe in zip(process_list, curr_states_pipe_list, strict=False):
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
                    val_total_rewards, val_agent_rewards = self.evaluate_episodes(
                        val_env, val_episodes
                    )

                val_reward_history.append(val_total_rewards)
                val_agent_reward_history.append(val_agent_rewards)
                logger.debug("Step %d, Eval Total Rewards: %s", step * len(envs), val_total_rewards)
        finally:
            self.sm.close()
            self.sm.unlink()

    def run_steps(
        self,
        env: DistClassicRLEnv | VectorEnv,
        num_steps: int,
        rewards_queue: mp.Queue,
        sm_lock: Lock,
        exploration_rate_value: Synchronized,
        curr_state_pipe: connection.Connection | None,
        curr_state: dict | None = None,
    ) -> None:
        """
        Run a single environment with multiple agents for a given number of steps.

        Parameters
        ----------
        env : DistClassicRLEnv | VectorEnv
            The environment to run.
        num_steps : int
            Number of steps to run.
        rewards_queue : mp.Queue
            Queue to collect rewards from the environment.
        sm_lock : Lock
            Lock for synchronizing access to shared memory.
        exploration_rate_value : Synchronized
            Shared exploration rate value.
        curr_state_pipe : connection.Connection | None
            Pipe for communicating current state.
        curr_state : dict | None
            Current state dictionary, by default None.
        """
        self.sm_lock = sm_lock
        self._exploration_rate = exploration_rate_value
        self.sm = shared_memory.SharedMemory(name=self.sm_name)
        self.q_table = np.ndarray(self.q_table.shape, dtype=self.q_table.dtype, buffer=self.sm.buf)

        if curr_state is None:
            states, infos = env.reset()
            n_agents = len(states["observation"]) if isinstance(states, dict) else len(states)
            agent_rewards = np.zeros(n_agents, dtype=np.float32)
        else:
            states = curr_state["states"]
            infos = curr_state["infos"]
            agent_rewards = curr_state["agent_rewards"]

        for _ in range(num_steps):
            if isinstance(states, dict):
                with self.sm_lock:
                    actions = self.choose_actions(
                        states=states["observation"], action_masks=states["action_mask"]
                    )
            else:
                with self.sm_lock:
                    actions = self.choose_actions(states)

            next_states, rewards, terminateds, truncateds, infos = env.step(actions)

            agent_rewards += rewards

            if isinstance(next_states, dict):
                assert isinstance(states, dict)
                with self.sm_lock:
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
                with self.sm_lock:
                    self.learn(states, actions, rewards, next_states, terminateds)
            states = next_states

            for i, (terminated, truncated) in enumerate(zip(terminateds, truncateds, strict=False)):
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
            for i, (terminated, truncated) in enumerate(zip(terminateds, truncateds, strict=False)):
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
            for i, (terminated, truncated) in enumerate(zip(terminateds, truncateds, strict=False)):
                if terminated or truncated:
                    episode += 1
                    reward_history.append(agent_rewards[i])
                    agent_rewards[i] = 0

        return sum(reward_history), reward_history
