import gymnasium as gym

class ThreeActionWrapper(gym.ActionWrapper): # This wrapper reduces the minigrid action space from 7 to 3 actions
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(3)
        self.action_map = {
            0: 0, # turn left
            1: 1, # turn right
            2: 2  # move forward
        }

    def action(self, action):
        return self.action_map[int(action)]