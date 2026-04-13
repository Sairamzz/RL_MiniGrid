import gymnasium as gym
import minigrid

from envs.action_wrapper import ThreeActionWrapper
from envs.slip_wrapper import ActionSlipWrapper


def minigrid_env(env_id: str, slip_prob = 0.0, render_mode=None, seed = 0, max_steps = 250):
    env = gym.make(env_id, render_mode=render_mode, max_steps=max_steps)
    # 3 actions: turn left, turn right, move forward
    env = ThreeActionWrapper(env)
    # Adding a stochastic slip
    env = ActionSlipWrapper(env, slip_prob=slip_prob)
    env.reset(seed=seed)
    return env
