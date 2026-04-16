import argparse
import os

from config import EnvConfig, TrainingConfig, PSConfig, POMCPConfig, PPOConfig, A2CConfig, QLearningConfig, SACConfig
from envs.mingrid_env import minigrid_env
from agents.random_agent import RandomAgent
from agents.prioritized_sweeping import PrioritizedSweepingAgent
from agents.pomcp_agent import POMCPAgent
from agents.ppo_agent import PPOAgent
from agents.a2c_agent import A2CAgent
from agents.sac_agent import SACAgent
from agents.q_learning_agent import QLearningAgent
from trainers.train_tabular import train_tabular_agent
from utils.checkpoints import load_pickle
from utils.visualization import rollout_greedy, rollout_ppo, rollout_pomcp, save_trajectory_json, plot_trajectory_on_grid
from utils.plotting import plot_rewards


def make_tabular_agent(agent_name, env):
    if agent_name == "random":
        return RandomAgent(env.action_space, config=None)
    if agent_name == "ps":
        return PrioritizedSweepingAgent(env.action_space, PSConfig())
    if agent_name == "pomcp":
        agent = POMCPAgent(env.action_space, POMCPConfig())
        agent.set_env_reference(env)
        return agent
    if agent_name == "qlearning":
        return QLearningAgent(env.action_space, QLearningConfig())
    raise ValueError(f"Unknown agent: {agent_name}")


def train_tabular(agent_name, env_id, episodes, max_steps, seed=0):
    env_cfg = EnvConfig(env=env_id, seed=seed, max_steps=max_steps)
    exp_cfg = TrainingConfig()

    env = minigrid_env(env_cfg.env, slip_prob=env_cfg.slip_prob, seed=env_cfg.seed, max_steps=env_cfg.max_steps)
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


def train_ppo(env_id, total_timesteps, seed = 0, max_steps = 250, load_checkpoint=None, exploration_bonus = 0.05):
    env_cfg = EnvConfig(env=env_id, seed=seed, max_steps=max_steps)
    train_env = minigrid_env(
        env_cfg.env,
        slip_prob=env_cfg.slip_prob,
        seed=env_cfg.seed,
        max_steps=env_cfg.max_steps,
        exploration_bonus=exploration_bonus
    )

    eval_env = minigrid_env(
        env_cfg.env,
        slip_prob=env_cfg.slip_prob,
        seed=env_cfg.seed,
        max_steps=env_cfg.max_steps,
        exploration_bonus=0.0 # No exploration bonus during evaluation
    )

    ppo_cfg = PPOConfig(total_timesteps=total_timesteps)
    ppo_cfg.checkpoint_path = f"results/checkpoints/ppo_{env_id.replace('-', '_')}.pt"
    ppo_cfg.csv_path = f"results/csv/ppo_{env_id.replace('-', '_')}_history.csv"
    agent = PPOAgent(train_env, ppo_cfg)

    if load_checkpoint:                 
        agent.load(load_checkpoint)
        print(f"Loaded checkpoint from {load_checkpoint}")

    best_ckpt = agent.train(eval_env=eval_env, eval_episodes = 10, eval_interval_updates = 10)

    agent.load(best_ckpt) # Load best checkpoint for evaluation
    rewards, successes = agent.evaluate(num_episodes = 10, env=eval_env)

    print("\nBest checkpoint:", best_ckpt)
    print("PPO evaluation rewards:", rewards)
    print("PPO evaluation successes:", sum(successes) / len(successes))

    train_env.close()
    eval_env.close()

def train_a2c(env_id, total_timesteps, seed=0, max_steps=250, load_checkpoint=None, exploration_bonus=0.02):
    env_cfg = EnvConfig(env=env_id, seed=seed, max_steps=max_steps)

    train_env = minigrid_env(
        env_cfg.env,
        slip_prob=env_cfg.slip_prob,
        seed=env_cfg.seed,
        max_steps=env_cfg.max_steps,
        exploration_bonus=exploration_bonus
    )

    eval_env = minigrid_env(
        env_cfg.env,
        slip_prob=env_cfg.slip_prob,
        seed=env_cfg.seed,
        max_steps=env_cfg.max_steps,
        exploration_bonus=0.0
    )

    a2c_cfg = A2CConfig(total_timesteps=total_timesteps)
    a2c_cfg.checkpoint_path = f"results/checkpoints/a2c_{env_id.replace('-', '_')}.pt"
    a2c_cfg.csv_path = f"results/csv/a2c_{env_id.replace('-', '_')}_history.csv"

    agent = A2CAgent(train_env, a2c_cfg)

    if load_checkpoint:
        agent.load(load_checkpoint)
        print(f"Loaded checkpoint from {load_checkpoint}")

    eval_seeds = list(range(30))
    best_ckpt = agent.train(eval_env=eval_env, eval_episodes=30, eval_interval_updates=10, eval_seeds=eval_seeds)

    agent.load(best_ckpt)
    rewards = agent.evaluate(num_episodes=30, env=eval_env, seeds=eval_seeds)

    print("\nBest checkpoint:", best_ckpt)
    print("A2C evaluation rewards:", rewards)
    print("A2C evaluation mean reward:", sum(rewards) / len(rewards))

    train_env.close()
    eval_env.close()

def train_sac(env_id, total_timesteps, seed=0, max_steps=250, load_checkpoint=None, exploration_bonus=0.02):
    env_cfg = EnvConfig(env=env_id, seed=seed, max_steps=max_steps)

    train_env = minigrid_env(
        env_cfg.env,
        slip_prob=env_cfg.slip_prob,
        seed=env_cfg.seed,
        max_steps=env_cfg.max_steps,
        exploration_bonus=exploration_bonus
    )

    eval_env = minigrid_env(
        env_cfg.env,
        slip_prob=env_cfg.slip_prob,
        seed=env_cfg.seed,
        max_steps=env_cfg.max_steps,
        exploration_bonus=0.0
    )

    sac_cfg = SACConfig(total_timesteps=total_timesteps)
    sac_cfg.checkpoint_path = f"results/checkpoints/sac_{env_id.replace('-', '_')}.pt"

    agent = SACAgent(train_env, sac_cfg)

    if load_checkpoint:
        agent.load(load_checkpoint)
        print(f"Loaded checkpoint from {load_checkpoint}")

    best_ckpt = agent.train(eval_env=eval_env, eval_episodes=10, eval_interval_steps=5000)

    agent.load(best_ckpt)
    rewards = agent.evaluate(num_episodes=10, env=eval_env)

    print("\nBest checkpoint:", best_ckpt)
    print("SAC evaluation rewards:", rewards)
    print("SAC evaluation mean reward:", sum(rewards) / len(rewards))

    train_env.close()
    eval_env.close()

def visualize_ps(env_id, checkpoint_path, max_steps=150, seed=0):
    env = minigrid_env(env_id, slip_prob=0.10, seed=seed, max_steps=max_steps)
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

def visualize_pomcp(env_id, max_steps=250, seed=0):
    env_cfg = EnvConfig(env=env_id, seed=seed, max_steps=max_steps)

    env = minigrid_env(
        env_cfg.env,
        slip_prob=env_cfg.slip_prob,
        seed=env_cfg.seed,
        max_steps=env_cfg.max_steps,
        exploration_bonus=0.0
    )

    agent = POMCPAgent(env.action_space, POMCPConfig())
    agent.set_env_reference(env)

    rollout = rollout_pomcp(env, agent, max_steps=max_steps, seed=seed)

    env_tag = env_id.replace("-", "_")
    save_trajectory_json(rollout, f"results/trajectories/pomcp_{env_tag}_rollout.json")
    plot_trajectory_on_grid(
        env,
        rollout,
        save_path=f"results/figures/pomcp_{env_tag}_rollout.png",
        title=f"POMCP Policy - {env_id}"
    )

    env.close()


def visualize_ppo(env_id, checkpoint_path, max_steps=250, seed=0):
    env_cfg = EnvConfig(env=env_id, seed=seed, max_steps=max_steps)
    env = minigrid_env(
        env_cfg.env,
        slip_prob=env_cfg.slip_prob,
        seed=env_cfg.seed,
        max_steps=env_cfg.max_steps,
        exploration_bonus=0.0
    )

    agent = PPOAgent(env, PPOConfig())
    agent.load(checkpoint_path)

    rollout = agent.rollout(max_steps=max_steps, deterministic=True)

    env_tag = env_id.replace('-', '_')
    traj_path = f"results/trajectories/ppo_{env_tag}_rollout.json"
    fig_path = f"results/figures/ppo_{env_tag}_rollout.png"
    
    save_trajectory_json(rollout, traj_path)
    plot_trajectory_on_grid(
        env,
        rollout,
        save_path=fig_path,
        title=f"PPO - {env_id}"
    )

    env.close()

def visualize_a2c(env_id, checkpoint_path, max_steps=250, seed=0):
    env_cfg = EnvConfig(env=env_id, seed=seed, max_steps=max_steps)
    env = minigrid_env(
        env_cfg.env,
        slip_prob=env_cfg.slip_prob,
        seed=env_cfg.seed,
        max_steps=env_cfg.max_steps,
        exploration_bonus=0.0
    )

    agent = A2CAgent(env, A2CConfig())
    agent.load(checkpoint_path)

    rollout = agent.rollout(max_steps=max_steps, deterministic=True)

    env_tag = env_id.replace("-", "_")
    traj_path = f"results/trajectories/a2c_{env_tag}_rollout.json"
    fig_path = f"results/figures/a2c_{env_tag}_rollout.png"

    save_trajectory_json(rollout, traj_path)
    plot_trajectory_on_grid(
        env,
        rollout,
        save_path=fig_path,
        title=f"A2C Deterministic Policy - {env_id}"
    )

    env.close()

def visualize_sac(env_id, checkpoint_path, max_steps=250, seed=0):
    env_cfg = EnvConfig(env=env_id, seed=seed, max_steps=max_steps)
    env = minigrid_env(
        env_cfg.env,
        slip_prob=env_cfg.slip_prob,
        seed=env_cfg.seed,
        max_steps=env_cfg.max_steps,
        exploration_bonus=0.0
    )

    agent = SACAgent(env, SACConfig())
    agent.load(checkpoint_path)

    rollout = agent.rollout(max_steps=max_steps, deterministic=True)

    env_tag = env_id.replace("-", "_")
    traj_path = f"results/trajectories/sac_{env_tag}_rollout.json"
    fig_path = f"results/figures/sac_{env_tag}_rollout.png"

    save_trajectory_json(rollout, traj_path)
    plot_trajectory_on_grid(
        env,
        rollout,
        save_path=fig_path,
        title=f"SAC Deterministic Policy - {env_id}"
    )

    env.close()

def visualize_random(env_id, max_steps=250, seed=0):
    env_cfg = EnvConfig(env=env_id, seed=seed, max_steps=max_steps)

    env = minigrid_env(
        env_cfg.env,
        slip_prob=env_cfg.slip_prob,
        seed=env_cfg.seed,
        max_steps=env_cfg.max_steps,
    )

    agent = RandomAgent(env.action_space, config=None)

    rollout = rollout_greedy(env, agent, max_steps=max_steps)
    save_trajectory_json(rollout, "results/trajectories/random_rollout.json")
    plot_trajectory_on_grid(
        env,
        rollout,
        save_path="results/figures/random_rollout.png",
        title="Random Policy"
    )

    env.close()

def visualize_qlearning(env_id, checkpoint_path, max_steps=250, seed=0, slip_prob=0.1):
    env_cfg = EnvConfig(env=env_id, seed=seed, max_steps=max_steps)
    env_cfg.slip_prob = slip_prob

    env = minigrid_env(
        env_cfg.env,
        slip_prob=env_cfg.slip_prob,
        seed=env_cfg.seed,
        max_steps=env_cfg.max_steps,
    )

    agent = QLearningAgent(env.action_space, QLearningConfig())
    data = load_pickle(checkpoint_path)
    agent.load_state_dict(data)

    rollout = rollout_greedy(env, agent, max_steps=max_steps)
    save_trajectory_json(rollout, "results/trajectories/qlearning_rollout.json")
    plot_trajectory_on_grid(
        env,
        rollout,
        save_path="results/figures/qlearning_rollout.png",
        title="Q-Learning Greedy Policy"
    )

    env.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["train", "visualize", "plot"])
    parser.add_argument("--agent", type=str, required=False, choices=["random", "ps", "pomcp", "ppo", "a2c", "sac"])
    parser.add_argument("--env-id", type=str, default="MiniGrid-MultiRoom-N4-S5-v0")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--total-timesteps", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--load-checkpoint", type=str, default=None)
    parser.add_argument("--csv-path", type=str, default="")
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
                max_steps=args.max_steps,
                load_checkpoint=args.load_checkpoint
            )

        elif args.agent == "a2c":
            train_a2c(
                env_id=args.env_id,
                total_timesteps=args.total_timesteps,
                seed=args.seed,
                max_steps=args.max_steps,
                load_checkpoint=args.load_checkpoint
            )
        
        elif args.agent == "sac":
            train_sac(
                env_id=args.env_id,
                total_timesteps=args.total_timesteps,
                seed=args.seed,
                max_steps=args.max_steps,
                load_checkpoint=args.load_checkpoint
            )

    elif args.mode == "visualize":
        if args.agent == "random":
            visualize_random(args.env_id, max_steps=args.max_steps, seed=args.seed)
        elif args.agent == "ps":
            ckpt = args.checkpoint or "results/checkpoints/ps_final.pkl"
            visualize_ps(args.env_id, ckpt, max_steps=args.max_steps, seed=args.seed)
        elif args.agent == "qlearning":
            ckpt = args.checkpoint or "results/checkpoints/qlearning_final.pkl"
            visualize_qlearning(args.env_id, ckpt, max_steps=args.max_steps, seed=args.seed)
        elif args.agent == "pomcp":
            visualize_pomcp(args.env_id, max_steps=args.max_steps, seed=args.seed)
        elif args.agent == "ppo":
            ckpt = args.checkpoint or "results/checkpoints/ppo_minigrid.pt"
            visualize_ppo(args.env_id, ckpt, max_steps=args.max_steps, seed=args.seed)
        elif args.agent == "a2c":
            ckpt = args.checkpoint or f"results/checkpoints/a2c_{args.env_id.replace('-', '_')}_best.pt"
            visualize_a2c(args.env_id, ckpt, max_steps=args.max_steps, seed=args.seed)
        elif args.agent == "sac":
            ckpt = args.checkpoint or f"results/checkpoints/sac_{args.env_id.replace('-', '_')}_best.pt"
            visualize_sac(args.env_id, ckpt, max_steps=args.max_steps, seed=args.seed)
        else:
            raise ValueError("Visualization mode currently supported for random, ps and ppo only.")


    elif args.mode == "plot":
        if not args.csv_path:
            raise ValueError("Please provide --csv-path for plot mode.")
        plot_rewards(args.csv_path)

if __name__ == "__main__":
    main()