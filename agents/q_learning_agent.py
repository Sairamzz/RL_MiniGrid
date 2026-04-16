from collections import defaultdict
import numpy as np

from agents.base_agent import BaseAgent


class QLearningAgent(BaseAgent):
    """
    Tabular Q-learning baseline intended for the FULL-STATE wrapper.
    This is a sanity-check agent, not the main POMDP method.
    """

    def __init__(self, action_space, config):
        super().__init__(action_space, config)
        self.n_actions = action_space.n

        self.Q = defaultdict(
            lambda: np.ones(self.n_actions, dtype=float) * self.config.optimistic_initial_q
        )

        self.epsilon = self.config.epsilon_start

    def _state_key(self, obs):
        arr = np.asarray(obs, dtype=np.int32).reshape(-1)
        return tuple(int(x) for x in arr)

    def begin_episode(self):
        pass

    def act(self, obs):
        state = self._state_key(obs)

        if np.random.rand() < self.epsilon:
            return self.action_space.sample()

        return int(np.argmax(self.Q[state]))

    def act_greedy(self, obs):
        state = self._state_key(obs)
        return int(np.argmax(self.Q[state]))

    def observe(self, obs, action, reward, next_obs, done):
        state = self._state_key(obs)
        next_state = self._state_key(next_obs)

        best_next = 0.0 if done else np.max(self.Q[next_state])
        td_target = reward + self.config.gamma * best_next
        td_error = td_target - self.Q[state][action]

        self.Q[state][action] += self.config.alpha * td_error

    def end_episode(self):
        self.epsilon = max(
            self.config.epsilon_end,
            self.epsilon * self.config.epsilon_decay
        )

    def name(self):
        return "ql_full"

    def state_dict(self):
        return {
            "Q": {k: v.tolist() for k, v in self.Q.items()},
            "epsilon": float(self.epsilon),
        }

    def load_state_dict(self, data):
        self.Q = defaultdict(
            lambda: np.ones(self.n_actions, dtype=float) * self.config.optimistic_initial_q
        )
        for k, v in data["Q"].items():
            self.Q[k] = np.array(v, dtype=float)

        self.epsilon = float(data.get("epsilon", self.config.epsilon_end))