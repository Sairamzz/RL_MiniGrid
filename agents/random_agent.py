from agents.base_agent import BaseAgent


class RandomAgent(BaseAgent):
    def begin_episode(self):
        pass

    def act(self, obs):
        return self.action_space.sample()

    def observe(self, obs, action, reward, next_obs, done):
        pass

    def end_episode(self):
        pass

    def name(self):
        return "random"