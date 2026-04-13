import argparse
import os

from config import EnvConfig, TrainingConfig, PSConfig, POMCPConfig, PPOConfig
from envs.mingrid_env import minigrid_env
from agents.random_agent import RandomAgent
from agents.prioritized_sweeping import PrioritizedSweepingAgent
from agents.pomcp_agent import POMCPAgent
from agents.ppo_agent import PPOAgent
from trainers.train_tabular import train_tabular_agent
from utils.checkpoints import load_pickle
from utils.visualization import (
    rollout_greedy,
    rollout_ppo,
    save_trajectory_json,
    plot_trajectory_on_grid,
)


def make_tabular_agent(agent_name, env):
    if agent_name == "random":
        return RandomAgent(env.action_space, config=None)
    if agent_name == "ps":
        return PrioritizedSweepingAgent(env.action_space, PSConfig())
    if agent_name == "pomcp":
        return POMCPAgent(env.action_space, POMCPConfig())
    raise ValueError(f"Unknown agent: {agent_name}")


def train_tabular(agent_name, env_id, episodes, max_steps, seed=0):
    env_cfg = EnvConfig(env=env_id, seed=seed, max_steps=max_steps)
    exp_cfg = TrainingConfig()

    env = minigrid_env(env_cfg.env, slip_prob=env_cfg.slip_prob, seed=env_cfg.seed)
    agent = make_tabular_agent(agent_name, env)

    output_csv = f"results/csv/{agent_name}_{env_id.replace('-', '_')}.csv"
    checkpoint_dir = "results/checkpoints"

    rows = train_tabular_agent(
        env=env,
        agent=agent,
        episodes=episodes,
        max_steps=max_steps,
        output_csv=output_csv,
        checkpoint=exp_cfg.checkpoint,
        checkpoint_dir=checkpoint_dir,
    )

    env.close()
    print(f"Saved CSV to {output_csv}")
    print(f"Final checkpoint in {checkpoint_dir}")
    return rows


def train_ppo(env_id, total_timesteps, seed=0):
    env_cfg = EnvConfig(env=env_id, seed=seed)
    ppo_cfg = PPOConfig()

    env = minigrid_env(env_cfg.env_id, slip_prob=env_cfg.slip_prob, seed=env_cfg.seed)
    ppo_cfg.total_timesteps = total_timesteps

    agent = PPOAgent(env, ppo_cfg)
    agent.train()

    rewards = agent.evaluate(episodes=10)
    env.close()

    print("PPO eval rewards:", rewards)


def visualize_ps(env_id, checkpoint_path, max_steps=150, seed=0):
    env = minigrid_env(env_id, slip_prob=0.10, seed=seed)
    agent = PrioritizedSweepingAgent(env.action_space, PSConfig())

    data = load_pickle(checkpoint_path)
    agent.load_state_dict(data)

    rollout = rollout_greedy(env, agent, max_steps=max_steps)
    save_trajectory_json(rollout, "results/trajectories/ps_rollout.json")
    plot_trajectory_on_grid(
        env,
        rollout,
        save_path="results/figures/ps_rollout.png",
        title="PS Greedy Policy"
    )
    env.close()


def visualize_ppo(env_id, checkpoint_path, max_steps=150, seed=0):
    env = minigrid_env(env_id, slip_prob=0.10, seed=seed)
    agent = PPOAgent(env, PPOConfig())
    agent.load(checkpoint_path)

    rollout = rollout_ppo(env, agent.model, max_steps=max_steps)
    save_trajectory_json(rollout, "results/trajectories/ppo_rollout.json")
    plot_trajectory_on_grid(
        env,
        rollout,
        save_path="results/figures/ppo_rollout.png",
        title="PPO Deterministic Policy"
    )
    env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["train", "visualize"])
    parser.add_argument("--agent", type=str, required=True, choices=["random", "ps", "pomcp", "ppo"])
    parser.add_argument("--env-id", type=str, default="MiniGrid-MultiRoom-N4-S5-v0")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--total-timesteps", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", type=str, default="")
    args = parser.parse_args()

    if args.mode == "train":
        if args.agent in ["random", "ps", "pomcp"]:
            train_tabular(
                agent_name=args.agent,
                env_id=args.env_id,
                episodes=args.episodes,
                max_steps=args.max_steps,
                seed=args.seed,
            )
        elif args.agent == "ppo":
            train_ppo(
                env_id=args.env_id,
                total_timesteps=args.total_timesteps,
                seed=args.seed,
            )

    elif args.mode == "visualize":
        if args.agent == "ps":
            ckpt = args.checkpoint or "results/checkpoints/ps_final.pkl"
            visualize_ps(args.env_id, ckpt, max_steps=args.max_steps, seed=args.seed)
        elif args.agent == "ppo":
            ckpt = args.checkpoint or "results/checkpoints/ppo_minigrid.zip"
            visualize_ppo(args.env_id, ckpt, max_steps=args.max_steps, seed=args.seed)
        else:
            raise ValueError("Visualization mode currently supported for ps and ppo only.")


if __name__ == "__main__":
    main()