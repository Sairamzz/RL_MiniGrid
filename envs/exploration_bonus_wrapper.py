import gymnasium as gym

class ExplorationBonusWrapper(gym.Wrapper):
    def __init__(self, env, bonus=0.01):
        super().__init__(env)
        self.bonus = bonus
        self.visited = set()

    def reset(self, **kwargs):
        self.visited = set()
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        pos = tuple(self.env.unwrapped.agent_pos)
        if pos not in self.visited:
            reward += self.bonus
            self.visited.add(pos)
        return obs, reward, done, truncated, info