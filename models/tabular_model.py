from collections import defaultdict, Counter

class TabularModel:
    def __init__(self):
        self.transition_counts = defaultdict(Counter)
        self.reward_sums = defaultdict(float)
        self.reward_counts = defaultdict(int)

    # Update the model with a new transition and reward
    def update(self, state, action, next_state, reward):
        key = (state, action)
        self.transition_counts[key][next_state] += 1
        self.reward_sums[key] += float(reward)
        self.reward_counts[key] += 1

    # Get the transition probabilities for a given state and action
    def get_transition_probabilities(self, state, action):
        key = (state, action)
        total = sum(self.transition_counts[key].values())
        if total == 0:
            return {}
        return {s_next: count / total for s_next, count in self.transition_counts[key].items()}
    
    # Get the average reward
    def get_expected_reward(self, state, action):
        key = (state, action)
        if self.reward_counts[key] == 0:
            return 0.0
        return self.reward_sums[key] / self.reward_counts[key]



