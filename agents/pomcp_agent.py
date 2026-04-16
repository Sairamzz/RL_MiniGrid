import math
import random
from typing import FrozenSet

import numpy as np

from agents.base_agent import BaseAgent
from envs.observation import obs_to_history_key

_IMMUTABLE_TYPES: FrozenSet[int] = frozenset([2, 3, 8, 9])


class BeliefNode:
    __slots__ = [
        "n_visits",
        "action_visits",
        "action_values",
        "children",
        "particles",
        "_initialized",
    ]

    def __init__(self) -> None:
        self.n_visits = 0
        self.action_visits = {}
        self.action_values = {}
        self.children = {}
        self.particles = []
        self._initialized = False


class POMCPAgent(BaseAgent):
    def __init__(self, action_space, config):
        super().__init__(action_space, config)

        self.n_actions = action_space.n
        self.gamma = config.gamma
        self.num_sim = config.num_sim
        self.c = config.c
        self.rollout_depth = config.rollout_depth
        self.discount_rollout = config.discount_rollout
        self.step_penalty = config.step_penalty

        self.num_particles = max(config.num_sim, 150)

        self._env = None
        self._root = None
        self._particles = []

    def set_env_reference(self, env):
        self._env = env

    def set_initial_particle(self, env):
        s0 = self._save_state(env)
        self._particles = [s0.copy() for _ in range(self.num_particles)]

    def begin_episode(self):
        self._root = BeliefNode()
        self._particles = []

    def act(self, obs):
        if self._env is None or not self._particles:
            return self.action_space.sample()

        real_state = self._save_state(self._env)

        if not self._root._initialized:
            self._init_node(self._root)

        self._root.particles = list(self._particles)

        for _ in range(self.num_sim):
            particle = random.choice(self._particles)
            self._restore_state(self._env, particle)
            self._simulate(self._root, self.rollout_depth)

        self._restore_state(self._env, real_state)
        return self._best_action(self._root)

    def observe(self, obs, action, reward, next_obs, done, info=None):
        if self._env is None:
            return

        obs_key = obs_to_history_key(next_obs)
        real_state = self._save_state(self._env)

        consistent = []

        for particle in self._particles:
            self._restore_state(self._env, particle)

            _, _, d, t, _ = self._env.step(action)

            # terminal consistency
            if (d or t) != bool(done):
                continue

            if not (d or t):
                sim_obs = self._env.unwrapped.gen_obs()
                if obs_to_history_key(sim_obs) == obs_key:
                    consistent.append(self._save_state(self._env))
            else:
                # if both terminal, keep it
                consistent.append(self._save_state(self._env))

        # aggressive recovery if belief collapses
        if len(consistent) < 20:
            consistent += [real_state.copy() for _ in range(self.num_particles)]

        if consistent:
            self._particles = [
                random.choice(consistent).copy() for _ in range(self.num_particles)
            ]
        else:
            self._particles = [real_state.copy() for _ in range(self.num_particles)]

        self._restore_state(self._env, real_state)

        key = (action, obs_key)
        if self._root is not None and key in self._root.children:
            self._root = self._root.children[key]
        else:
            self._root = BeliefNode()

        self._root.particles = list(self._particles)

    def end_episode(self):
        self._root = None
        self._particles = []

    def name(self):
        return "pomcp"

    def _simulate(self, node, depth):
        if depth == 0:
            return 0.0

        if not node._initialized:
            self._init_node(node)
            return self._rollout(depth)

        action = self._ucb_select(node)

        uw = self._env.unwrapped
        fwd_pos = uw.front_pos
        fwd_cell = uw.grid.get(*fwd_pos) if fwd_pos is not None else None

        if fwd_cell and fwd_cell.type == "door":
            if not fwd_cell.is_open and self.n_actions >= 4:
                _, r1, d1, t1, _ = self._env.step(3)  # toggle
                next_obs, r2, d2, t2, _ = self._env.step(2)  # forward
                reward = float(r1 + r2) - 2 * self.step_penalty
                done = d1 or d2
                truncated = t1 or t2
            else:
                next_obs, reward, done, truncated, _ = self._env.step(2)
                reward = float(reward) - self.step_penalty
        else:
            next_obs, reward, done, truncated, _ = self._env.step(action)
            reward = float(reward) - self.step_penalty

        obs_key = obs_to_history_key(next_obs)
        key = (action, obs_key)

        if key not in node.children:
            node.children[key] = BeliefNode()

        child = node.children[key]

        if done or truncated:
            q = reward
        else:
            q = reward + self.gamma * self._simulate(child, depth - 1)

        node.n_visits += 1
        node.action_visits[action] += 1

        n = node.action_visits[action]
        node.action_values[action] += (q - node.action_values[action]) / n

        return q

    def _rollout(self, depth):
        total = 0.0
        discount = 1.0

        def manhattan(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        uw = self._env.unwrapped

        goal_pos = None
        for i in range(uw.grid.width):
            for j in range(uw.grid.height):
                cell = uw.grid.get(i, j)
                if cell and cell.type == "goal":
                    goal_pos = (i, j)
                    break
            if goal_pos:
                break

        visited = set()
        visited.add(tuple(uw.agent_pos))

        for _ in range(depth):
            base_state = self._save_state(self._env)
            base_pos = tuple(base_state["agent_pos"])

            action_scores = []

            for a in range(self.n_actions):
                self._restore_state(self._env, base_state)

                _, reward, done, truncated, _ = self._env.step(a)
                reward = float(reward) - self.step_penalty
                new_pos = tuple(self._env.unwrapped.agent_pos)

                score = reward

                # penalize turning a bit
                if a in [0, 1]:
                    score -= 0.003

                # penalize forward that doesn't move
                if a == 2 and new_pos == base_pos:
                    score -= 0.05

                # reward novelty, penalize revisits
                if new_pos in visited:
                    score -= 0.05
                else:
                    score += 0.03

                # prefer actions that reduce distance to goal
                if goal_pos is not None:
                    old_d = manhattan(base_pos, goal_pos)
                    new_d = manhattan(new_pos, goal_pos)
                    if new_d < old_d:
                        score += 0.08
                    elif new_d > old_d:
                        score -= 0.03

                action_scores.append(score)

            # epsilon-greedy rollout
            if random.random() < 0.10:
                action = random.randrange(self.n_actions)
            else:
                best_score = max(action_scores)
                best_actions = [a for a, s in enumerate(action_scores) if abs(s - best_score) < 1e-12]
                action = random.choice(best_actions)

            self._restore_state(self._env, base_state)
            _, reward, done, truncated, _ = self._env.step(action)
            reward = float(reward) - self.step_penalty

            new_pos = tuple(self._env.unwrapped.agent_pos)

            if new_pos in visited:
                reward -= 0.05
            else:
                reward += 0.03
                visited.add(new_pos)

            if goal_pos is not None:
                old_d = manhattan(base_pos, goal_pos)
                new_d = manhattan(new_pos, goal_pos)
                if new_d < old_d:
                    reward += 0.08
                elif new_d > old_d:
                    reward -= 0.03

            total += discount * reward
            discount *= self.discount_rollout

            if done or truncated:
                break

        return total

    def _ucb_select(self, node):
        unvisited = [a for a in range(self.n_actions) if node.action_visits[a] == 0]
        if unvisited:
            return random.choice(unvisited)

        log_n = math.log(node.n_visits + 1)

        scores = []
        for a in range(self.n_actions):
            s = node.action_values[a] + self.c * math.sqrt(
                log_n / node.action_visits[a]
            )
            scores.append(s)

        best = max(scores)
        best_actions = [a for a, s in enumerate(scores) if abs(s - best) < 1e-12]
        return random.choice(best_actions)

    def _best_action(self, node):
        if not node._initialized:
            return random.randrange(self.n_actions)

        visited_actions = [a for a in range(self.n_actions) if node.action_visits[a] > 0]
        if not visited_actions:
            return random.randrange(self.n_actions)

        best_q = max(node.action_values[a] for a in visited_actions)
        best = [a for a in visited_actions if abs(node.action_values[a] - best_q) < 1e-12]
        return random.choice(best)

    def _init_node(self, node):
        for a in range(self.n_actions):
            node.action_visits[a] = 0
            node.action_values[a] = 0.0
        node._initialized = True
        node.n_visits = 0

    def _save_state(self, env):
        uw = env.unwrapped
        return {
            "grid": uw.grid.encode().copy(),
            "agent_pos": tuple(uw.agent_pos),
            "agent_dir": uw.agent_dir,
            "step_count": uw.step_count,
        }

    def _restore_state(self, env, state):
        from minigrid.core.world_object import WorldObj

        uw = env.unwrapped
        grid = state["grid"]
        W, H, _ = grid.shape

        for i in range(W):
            for j in range(H):
                t, c, s = grid[i, j]
                if t <= 1:
                    uw.grid.set(i, j, None)
                else:
                    uw.grid.set(i, j, WorldObj.decode(int(t), int(c), int(s)))

        uw.agent_pos = np.array(state["agent_pos"])
        uw.agent_dir = state["agent_dir"]
        uw.step_count = state["step_count"]
