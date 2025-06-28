"""This module provides a wrapper for PettingZoo environments."""

import numpy as np


# Dummy PettingZoo environment for testing.
class DummyPettingZooEnv:
    """
    Dummy PettingZoo environment for testing the PettingZooVectorWrapper.

    This class simulates a simple multi-agent environment with two agents
    and structured observations/actions containing 'observation' and 'action_mask' keys.
    """

    def __init__(self):
        """Initialize the dummy environment with two agents."""
        # Fixed agent order.
        self.agents = ["agent_0", "agent_1"]

        # Dummy spaces (only used to set the wrapper attributes).
        # They are not used in the dummy implementation.
        self.observation_space = {
            "agent_0": {"observation": None, "action_mask": None},
            "agent_1": {"observation": None, "action_mask": None},
        }
        self.action_space = {
            "agent_0": {"action": None, "action_mask": None},
            "agent_1": {"action": None, "action_mask": None},
        }

    def reset(self, **kwargs):
        """
        Reset the environment and return initial observations.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments (unused in this dummy implementation).

        Returns
        -------
        dict
            Dictionary mapping agent names to structured observations.
        """
        # Return a dict keyed by agent names with composite observations.
        return {
            "agent_0": {"observation": np.array([0]), "action_mask": np.array([True, False])},
            "agent_1": {"observation": np.array([10]), "action_mask": np.array([False, True])},
        }

    def step(self, actions):
        """
        Execute actions for all agents and return results.

        Expects actions as a dict keyed by agent names, each with a structured action.
        For this dummy env, we simply:
         - For agent_0: add 0 to the provided "action" value.
         - For agent_1: add 10 to the provided "action" value.
        The action_mask is passed through unchanged.

        Parameters
        ----------
        actions : dict
            Dictionary mapping agent names to structured actions.

        Returns
        -------
        tuple
            Tuple containing observations, rewards, terminated flags,
            truncated flags, and info dictionaries for all agents.
        """
        new_obs = {}
        rewards = {}
        terminated = {}
        truncated = {}
        infos = {}
        for agent in self.agents:
            action = actions[agent]
            # Compute a new observation based on the action value.
            if agent == "agent_0":
                new_obs_val = action["action"] + 0
            else:
                new_obs_val = action["action"] + 10
            new_obs[agent] = {
                "observation": np.array([new_obs_val]),
                "action_mask": action["action_mask"],
            }
            # Dummy reward: simply use the new observation value as reward.
            rewards[agent] = float(new_obs_val)
            terminated[agent] = False
            truncated[agent] = False
            infos[agent] = {"dummy_info": True}
        return new_obs, rewards, terminated, truncated, infos

    def render(self, mode="human"):
        """
        Render the environment.

        Parameters
        ----------
        mode : str, optional
            Rendering mode (default: "human").
        """
        # Dummy render simply prints a message.
        print("Rendering DummyPettingZooEnv.")


# The wrapper preserving structured observations/actions.
class PettingZooVectorWrapper:
    """
    A wrapper that converts a PettingZoo parallel environment's dict-based API
    into an interface similar to Gymnasium's sync vector environments while preserving
    the composite structure of observations and actions.

    - Observations (and actions) remain structured. For example, if an observation is
      a dict with keys "observation" and "action_mask", then reset/step will return a dict
      where each key maps to an array with shape (num_agents, ...).
    - The step method returns separate terminated and truncated arrays.
    """

    def __init__(self, env):
        """
        Initialize the PettingZooVectorWrapper.

        Parameters
        ----------
        env : PettingZoo parallel environment
            The PettingZoo environment to wrap.
        """
        self.env = env
        self.agents = env.agents

        # Assume all agents share the same spaces.
        self.observation_space = env.observation_space[self.agents[0]]
        self.action_space = env.action_space[self.agents[0]]

    def reset(self, **kwargs):
        """
        Reset the environment and return vectorized observations.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments to pass to the environment reset.

        Returns
        -------
        dict or array
            Structured observations with shape (num_agents, ...) for each key.
        """
        obs_dict = self.env.reset(**kwargs)
        return self._dict_to_structured(obs_dict)

    def step(self, actions):
        """
        Execute actions and return vectorized results.

        Parameters
        ----------
        actions : dict or array
            Structured actions with shape (num_agents, ...) for each key.

        Returns
        -------
        tuple
            Tuple containing vectorized observations, rewards, terminated flags,
            truncated flags, and info lists.
        """
        # Convert the structured (vectorized) actions into a per-agent dict.
        actions_dict = self._structured_to_dict(actions)

        # Underlying env returns: obs_dict, rewards_dict, terminated_dict, truncated_dict, infos_dict.
        obs_dict, rewards_dict, terminated_dict, truncated_dict, infos_dict = self.env.step(
            actions_dict
        )

        obs_stacked = self._dict_to_structured(obs_dict)
        rewards_array = np.array([rewards_dict[agent] for agent in self.agents])
        terminated_array = np.array([terminated_dict[agent] for agent in self.agents])
        truncated_array = np.array([truncated_dict[agent] for agent in self.agents])
        infos_list = [infos_dict[agent] for agent in self.agents]

        return obs_stacked, rewards_array, terminated_array, truncated_array, infos_list

    def render(self, mode="human"):
        """
        Render the environment.

        Parameters
        ----------
        mode : str, optional
            Rendering mode (default: "human").

        Returns
        -------
        Any
            Rendering output from the underlying environment.
        """
        return self.env.render(mode=mode)

    def _dict_to_structured(self, data_dict):
        """
        Convert a dict (keyed by agent names) into a structured object.

        If the per-agent data is composite (e.g. a dict), then each key is stacked
        separately.

        Parameters
        ----------
        data_dict : dict
            Dictionary mapping agent names to their data.

        Returns
        -------
        dict or array
            Structured data with agent dimension stacked.
        """
        # Build a list of per-agent observations in fixed order.
        obs_list = [data_dict[agent] for agent in self.agents]
        return self._stack_structure(obs_list)

    def _structured_to_dict(self, structured_data):
        """
        Convert structured (vectorized) action input into a dict keyed by agent names.

        It "un-stacks" the leading agent dimension while preserving the overall structure.

        Parameters
        ----------
        structured_data : dict or array
            Structured data with leading agent dimension.

        Returns
        -------
        dict
            Dictionary mapping agent names to per-agent data.
        """
        actions_list = self._unstack_structure(structured_data, len(self.agents))
        return {agent: actions_list[i] for i, agent in enumerate(self.agents)}

    def _stack_structure(self, data_list):
        """
        Recursively stack a list of data items while preserving their structure.

        If each item is a dict (with the same keys), return a dict where each key maps
        to a NumPy array stacking the corresponding values over the agent dimension.

        Parameters
        ----------
        data_list : list
            List of data items to stack.

        Returns
        -------
        dict or array
            Stacked structure with agent dimension as the first axis.
        """
        if isinstance(data_list[0], dict):
            return {k: self._stack_structure([d[k] for d in data_list]) for k in data_list[0]}
        else:
            return np.array(data_list)

    def _unstack_structure(self, data, num_agents):
        """
        Inverse of _stack_structure: given structured data with a leading agent dimension.

        Return a list of per-agent data items.

        Parameters
        ----------
        data : dict or array
            Structured data with leading agent dimension.
        num_agents : int
            Number of agents.

        Returns
        -------
        list
            List of per-agent data items.
        """
        if isinstance(data, dict):
            # Recursively unstack each key.
            unstacked = {k: self._unstack_structure(v, num_agents) for k, v in data.items()}
            return [{k: unstacked[k][i] for k in unstacked} for i in range(num_agents)]
        else:
            # Assume data is a NumPy array with the first dimension corresponding to agents.
            return [data[i] for i in range(num_agents)]


# Test function for the wrapper.
def test_pettingzoo_vector_wrapper():
    """
    Test function for the PettingZooVectorWrapper.

    Creates a dummy environment, wraps it, and tests the reset and step functionality
    to ensure the wrapper correctly converts between dict-based and vectorized APIs.
    """
    # Create the dummy environment and wrap it.
    dummy_env = DummyPettingZooEnv()
    wrapped_env = PettingZooVectorWrapper(dummy_env)

    # Test reset.
    print("=== Reset ===")
    reset_obs = wrapped_env.reset()
    print("Reset observation (structured):")
    print(reset_obs)

    # Check structure:
    # It should be a dict with keys "observation" and "action_mask",
    # and each value should be a NumPy array with shape (num_agents, ...).
    for key, value in reset_obs.items():
        print(f"Key: {key}, shape: {np.array(value).shape}")

    # Create a vectorized (structured) action.
    # For example, we assume the action structure is a dict with:
    #   - "action": scalar actions (one per agent)
    #   - "action_mask": an array for each agent (here, shape (2,))
    vectorized_actions = {
        "action": np.array([5, 15]),  # action for agent_0 and agent_1 respectively.
        "action_mask": np.array([[True, False], [False, True]]),
    }

    # Test step.
    print("\n=== Step ===")
    obs, rewards, terminated, truncated, infos = wrapped_env.step(vectorized_actions)
    print("Step observation (structured):")
    print(obs)
    print("Rewards:", rewards)
    print("Terminated:", terminated)
    print("Truncated:", truncated)
    print("Infos:", infos)

    # The dummy env's step adds 0 for agent_0 and 10 for agent_1 to the action value.
    # Therefore, we expect:
    #  - For agent_0: observation becomes [5]
    #  - For agent_1: observation becomes [15 + 10] = [25]
    print("\n=== Expected Results ===")
    print("Expected agent_0 observation: [5]")
    print("Expected agent_1 observation: [25]")


if __name__ == "__main__":
    test_pettingzoo_vector_wrapper()
