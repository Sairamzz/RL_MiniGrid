from abc import ABC, abstractmethod

class BaseAgent(ABC): # Base class for pomcp and prioritized sweeping
    def __init__(self, action_space, config):
        self.action_space = action_space
        self.config = config

    @abstractmethod
    def begin_episode(self):
        pass

    @abstractmethod
    def act(self, obs):
        pass

    @abstractmethod
    def observe(self, obs, action, reward, next_obs, done, info=None):
        pass

    @abstractmethod
    def end_episode(self):
        pass

    @abstractmethod
    def name(self):
        pass
