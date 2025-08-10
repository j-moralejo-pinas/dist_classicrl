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
    learning_rate_decay : float
        Decay rate for learning rate.
    min_learning_rate : float
        Minimum learning rate.
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
    learning_rate: Synchronized
    learning_rate_decay: float
    min_learning_rate: float
    _learning_rate: Synchronized
    discount_factor: float
    exploration_rate: Synchronized
    _exploration_rate: Synchronized
    exploration_decay: float
    min_exploration_rate: float
    q_table: NDArray[np.float32]
    sm: shared_memory.SharedMemory
    sm_lock: Lock

    sm_name: str = "q_table"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.exploration_rate = Value("f", 0.0)
        self.learning_rate = Value("f", 0.0)
        super().__init__(*args, **kwargs)
        self.sm_lock = mp.Lock()

    def get_exploration_rate(self) -> float:
        """
        Get the current exploration rate.

        This method retrieves the current exploration rate from the synchronized value.
        """
        return self.exploration_rate.value

    def set_exploration_rate(self, rate: float) -> None:
        """
        Set the exploration rate.

        This method sets the exploration rate in a thread-safe manner.
        """
        self.exploration_rate.value = rate

    def get_learning_rate(self) -> float:
        """
        Get the current learning rate.

        Returns
        -------
        float
            Current learning rate.
        """
        return self.learning_rate.value

    def set_learning_rate(self, learning_rate: float) -> None:
        """
        Set the learning rate.

        This method sets the learning rate in a thread-safe manner.
        """
        self.learning_rate.value = learning_rate

    def train(
        self,
        envs: Sequence[DistClassicRLEnv] | Sequence[VectorEnv],
        steps: int,
        val_env: DistClassicRLEnv | VectorEnv,
        val_every_n_steps: int,
        val_steps: int | None,
        val_episodes: int | None,
        curr_state_dicts: list[dict[str, Any]] | None = None,
    ) -> tuple[
        list[float], list[float], Sequence[DistClassicRLEnv | VectorEnv], list[dict[str, Any]]
    ]:
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
        curr_states : list[dict[str, Any]] | None
            The current state of the environments.

        Returns
        -------
        List[float]
            The reward history during training.
        List[float]
            The validation reward history.
        Sequence[DistClassicRLEnv | VectorEnv]
            The current environments
        List[dict[str, Any]]
            The current states of the environments, including states, infos and episode rewards.
        """
        try:
            self.sm = shared_memory.SharedMemory(
                name=self.sm_name, create=True, size=self.q_table.nbytes
            )
            new_q_table = np.ndarray(
                self.q_table.shape, dtype=self.q_table.dtype, buffer=self.sm.buf
            )
            new_q_table[:] = self.q_table[:]
            self.q_table = new_q_table

            assert (val_steps is None) ^ (val_episodes is None), (
                "Either val_steps or val_episodes should be provided."
            )

            reward_history = []
            val_reward_history = []
            val_agent_reward_history = []
            if curr_state_dicts is None:
                curr_states = [(env, None) for env in envs]
            else:
                curr_states = [
                    (env, state) for (env, state) in zip(envs, curr_state_dicts, strict=True)
                ]

            for step in range(0, steps, val_every_n_steps):
                curr_states_pipe_list = []
                process_list = []
                for curr_state in curr_states:
                    parent_conn, child_conn = mp.Pipe()
                    curr_states_pipe_list.append(parent_conn)
                    p = mp.Process(
                        target=self.run_steps,
                        args=(
                            curr_state,
                            int(val_every_n_steps / len(envs)),
                            self.sm_lock,
                            self.exploration_rate,
                            self.learning_rate,
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
            q_table_copy = self.q_table.copy()
            self.sm.close()
            self.sm.unlink()
            self.q_table = q_table_copy

        ret1 = [curr_state[0] for curr_state in curr_states]
        ret2 = [curr_state[1] for curr_state in curr_states if curr_state[1] is not None]
        return (reward_history, val_reward_history, ret1, ret2)

    def run_steps(
        self,
        curr_state: tuple[DistClassicRLEnv | VectorEnv, dict | None],
        num_steps: int,
        sm_lock: Lock,
        exploration_rate_value: Synchronized,
        learning_rate_value: Synchronized,
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
        exploration_rate_value : Synchronized
            Shared exploration rate value.
        learning_rate_value : Synchronized
            Shared learning rate value.
        curr_state_pipe : connection.Connection | None
            Pipe for communicating current state.
        curr_state : dict | None
            Current state dictionary, by default None.
        """
        try:
            self.sm_lock = sm_lock
            self.exploration_rate = exploration_rate_value
            self.learning_rate = learning_rate_value
            self.sm = shared_memory.SharedMemory(name=self.sm_name)
            self.q_table = np.ndarray(
                self.q_table.shape, dtype=self.q_table.dtype, buffer=self.sm.buf
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
                # Use minimal locking - only lock when accessing shared Q-table
                if isinstance(states, dict):
                    with self.sm_lock:
                        actions = self.choose_actions(
                            states=states["observation"], action_masks=states["action_mask"]
                        )
                else:
                    with self.sm_lock:
                        actions = self.choose_actions(states)

                # Environment step doesn't need locking
                next_states, rewards, terminateds, truncateds, infos = env.step(actions)
                agent_rewards += rewards

                # Learning step needs locking for Q-table update
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

                for i, (terminated, truncated) in enumerate(
                    zip(terminateds, truncateds, strict=False)
                ):
                    if terminated or truncated:
                        episode_rewards.append(agent_rewards[i])
                        agent_rewards[i] = 0

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
