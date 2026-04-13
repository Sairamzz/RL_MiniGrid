import random
import gymnasium as gym


class ActionSlipWrapper(gym.Wrapper):
    def __init__(self, env, slip_prob = 0.10):
        super().__init__(env)
        self.slip_prob = slip_prob

    def step(self, action):
        if random.random() < self.slip_prob:
            action = self.action_space.sample()
        return self.env.step(action)
