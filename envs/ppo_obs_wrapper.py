import gymnasium as gym
from gymnasium import spaces
import numpy as np


class FlattenMiniGridObsWrapper(gym.ObservationWrapper):
    """
    Convert MiniGrid dict obs:
      {"image": 7x7x3, "direction": int, "mission": str}
    into a flat numeric vector:
      [flattened_image, direction]
    """

    def __init__(self, env):
        super().__init__(env)

        image_space = env.observation_space["image"]
        flat_dim = int(np.prod(image_space.shape)) + 1  # +1 for direction

        self.observation_space = spaces.Box(
            low=0.0,
            high=255.0,
            shape=(flat_dim,),
            dtype=np.float32,
        )

    def observation(self, obs):
        image = np.array(obs["image"], dtype=np.float32).flatten()
        direction = np.array([obs["direction"]], dtype=np.float32)
        return np.concatenate([image, direction], axis=0)