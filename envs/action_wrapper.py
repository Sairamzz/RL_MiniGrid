import gymnasium as gym
from gymnasium import spaces


class ActionSubsetWrapper(gym.ActionWrapper):
    def __init__(self, env, allowed_actions):
        super().__init__(env)
        self.allowed_actions = allowed_actions
        self.action_space = spaces.Discrete(len(allowed_actions))

    def action(self, act):
        return self.allowed_actions[act]