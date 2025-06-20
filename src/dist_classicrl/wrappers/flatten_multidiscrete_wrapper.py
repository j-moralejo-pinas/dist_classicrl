from typing import Any, Dict, Union

import gymnasium
import gymnasium.spaces.dict
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from dist_classicrl.utils import compute_radix, decode_to_multi_discrete, encode_multi_discrete


class FlattenMultiDiscreteActionsWrapper(gymnasium.ActionWrapper):

    action_radix: NDArray[np.int32]
    action_space: spaces.Discrete
    action_nvec: NDArray[np.int32]

    def __init__(self, env):
        super().__init__(env)
        action_space = env.action_space
        assert isinstance(action_space, spaces.MultiDiscrete) or isinstance(
            action_space, spaces.Discrete
        ), f"Expected MultiDiscrete or Discrete action space, got {type(env.action_space)}."

        assert isinstance(
            action_space, spaces.MultiDiscrete
        ), "Expected MultiDiscrete action space."
        self.action_radix = compute_radix(action_space.nvec)
        self.action_nvec = action_space.nvec
        self.action_space = spaces.Discrete(np.prod(action_space.nvec))

    def action(self, action: int) -> NDArray[np.int32]:
        return decode_to_multi_discrete(self.action_nvec, action, self.action_radix)


class FlattenMultiDiscreteObservationsWrapper(gymnasium.ObservationWrapper):

    observation_radix: NDArray[np.int32]
    observation_space: Union[spaces.Discrete, spaces.Dict]
    observation_nvec: NDArray[np.int32]

    def __init__(self, env) -> None:
        super().__init__(env)

        observation_space = env.observation_space
        if isinstance(observation_space, spaces.Dict):
            assert (
                "observation" in observation_space.spaces
            ), "Expected 'observation' key in observation space."
            observation_subspace = observation_space.spaces["observation"]
            assert isinstance(
                observation_subspace, spaces.MultiDiscrete
            ), "Expected MultiDiscrete observation space."
            self.observation_radix = compute_radix(observation_subspace.nvec)
            self.observation_nvec = observation_subspace.nvec
            self.observation_space = (
                observation_space  # TODO: I should probably make a deep copy here
            )
            self.observation_space.spaces["observation"] = spaces.Discrete(
                np.prod(observation_subspace.nvec)
            )
        else:
            assert isinstance(observation_space, spaces.MultiDiscrete) or isinstance(
                observation_space, spaces.Discrete
            ), f"Expected MultiDiscrete or Discrete observation space, got {type(env.observation_space)}."

            assert isinstance(
                observation_space, spaces.MultiDiscrete
            ), "Expected MultiDiscrete observation space."
            self.observation_radix = compute_radix(observation_space.nvec)
            self.observation_nvec = observation_space.nvec
            self.observation_space = spaces.Discrete(np.prod(observation_space.nvec))

    def observation(
        self, observation: Union[NDArray[np.int32], Dict[str, Union[NDArray[np.int32], Any]]]
    ) -> Union[int, Dict[str, Union[int, Any]]]:
        if isinstance(observation, dict):
            observation["observation"] = encode_multi_discrete(
                observation["observation"], self.observation_radix
            )
            return observation
        else:
            return encode_multi_discrete(observation, self.observation_radix)
