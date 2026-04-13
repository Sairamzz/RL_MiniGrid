from collections import defaultdict

class PredecessorGraph: # For backward propagation of value updates in prioritized sweeping
    def __init__(self):
        self.predecessors = defaultdict(set)

    # next_state has a predecessor (state, action) pair
    def add_transition(self, state, action, next_state):
        self.predecessors[next_state].add((state, action))

    # Get the predecessors of a state that has been observed
    def get_predecessors(self, state):
        return self.predecessors[state]
