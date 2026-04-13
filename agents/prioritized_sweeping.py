from collections import defaultdict, Counter
import numpy as np

from agents.base_agent import BaseAgent
from envs.observation import encode_obs_tabular
from models.tabular_model import TabularModel
from models.predecessor_graph import PredecessorGraph
from planning.priority_queue import MaxPriorityQueue


class PrioritizedSweepingAgent(BaseAgent):
    def __init__(self, action_space, config):
        super().__init__(action_space, config)
        self.n_actions = action_space.n
        self.Q = defaultdict(
            lambda: np.ones(self.n_actions, dtype=float) * self.config.optimistic_initial_q
        )
        self.model = TabularModel()
        self.predecessors = PredecessorGraph()
        self.queue = MaxPriorityQueue()

    def begin_episode(self):
        pass

    def act(self, obs):
        state = encode_obs_tabular(obs)
        if np.random.rand() < self.config.epsilon:
            return self.action_space.sample()
        return int(np.argmax(self.Q[state]))

    def act_greedy(self, obs):
        state = encode_obs_tabular(obs)
        return int(np.argmax(self.Q[state]))

    def observe(self, obs, action, reward, next_obs, done):
        state = encode_obs_tabular(obs)
        next_state = encode_obs_tabular(next_obs)

        self.model.update(state, action, next_state, reward)
        self.predecessors.add_transition(state, action, next_state)

        td_target = reward if done else reward + self.config.gamma * np.max(self.Q[next_state])
        priority = abs(td_target - self.Q[state][action])

        if priority > self.config.priority_threshold:
            self.queue.push(priority, (state, action))

        self.planning()

    def planning(self):
        for _ in range(self.config.planning_steps):
            if self.queue.empty():
                break

            _, (state, action) = self.queue.pop()

            dist = self.model.get_transition_probabilities(state, action)
            reward = self.model.get_expected_reward(state, action)

            if not dist:
                continue

            backed_up = 0.0
            for next_state, prob in dist.items():
                backed_up += prob * (reward + self.config.gamma * np.max(self.Q[next_state]))

            self.Q[state][action] = backed_up

            for pred_state, pred_action in self.predecessors.get_predecessors(state):
                pred_reward = self.model.get_expected_reward(pred_state, pred_action)
                pred_dist = self.model.get_transition_probabilities(pred_state, pred_action)

                if not pred_dist:
                    continue

                target = 0.0
                for s_next, prob in pred_dist.items():
                    target += prob * (pred_reward + self.config.gamma * np.max(self.Q[s_next]))

                priority = abs(target - self.Q[pred_state][pred_action])
                if priority > self.config.priority_threshold:
                    self.queue.push(priority, (pred_state, pred_action))

    def end_episode(self):
        pass

    def name(self):
        return "ps"

    def state_dict(self):
        transition_counts = {
            k: dict(v) for k, v in self.model.transition_counts.items()
        }
        reward_sums = dict(self.model.reward_sums)
        reward_counts = dict(self.model.reward_counts)
        predecessors = {
            k: list(v) for k, v in self.predecessors.predecessors.items()
        }

        return {
            "Q": {k: v.tolist() for k, v in self.Q.items()},
            "transition_counts": transition_counts,
            "reward_sums": reward_sums,
            "reward_counts": reward_counts,
            "predecessors": predecessors,
        }

    def load_state_dict(self, data):
        self.Q = defaultdict(
            lambda: np.ones(self.n_actions, dtype=float) * self.config.optimistic_initial_q
        )
        for k, v in data["Q"].items():
            self.Q[k] = np.array(v, dtype=float)

        self.model.transition_counts = defaultdict(Counter)
        for k, v in data["transition_counts"].items():
            self.model.transition_counts[k] = Counter(v)

        self.model.reward_sums = defaultdict(float)
        for k, v in data["reward_sums"].items():
            self.model.reward_sums[k] = float(v)

        self.model.reward_counts = defaultdict(int)
        for k, v in data["reward_counts"].items():
            self.model.reward_counts[k] = int(v)

        self.predecessors.predecessors = defaultdict(set)
        for k, v in data["predecessors"].items():
            self.predecessors.predecessors[k] = set(tuple(x) for x in v)