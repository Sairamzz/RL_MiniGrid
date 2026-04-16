import gymnasium as gym
import minigrid

from envs.action_wrapper import ActionSubsetWrapper
from envs.slip_wrapper import ActionSlipWrapper
from envs.exploration_bonus_wrapper import ExplorationBonusWrapper


def minigrid_env(env_id: str, slip_prob=0.0, render_mode=None, seed=0, max_steps=250, exploration_bonus=0.0):
    env = gym.make(env_id, render_mode=render_mode, max_steps=max_steps)

    # MultiRoom needs toggle
    if "MultiRoom" in env_id:
        allowed_actions = [0, 1, 2, 5]   # left, right, forward, toggle
    else:
        allowed_actions = [0, 1, 2]      # left, right, forward

    env = ActionSubsetWrapper(env, allowed_actions)
    env = ActionSlipWrapper(env, slip_prob=slip_prob)

    if exploration_bonus > 0:
        print(f"ExplorationBonusWrapper applied with bonus={exploration_bonus}")
        env = ExplorationBonusWrapper(env, bonus=exploration_bonus)

    env.reset(seed=seed)
    return env