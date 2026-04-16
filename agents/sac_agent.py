import os
import csv
import random
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from envs.observation import encode_obs_ppo
from models.sac_networks import DiscretePolicyNet, DiscreteQNet


class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def add(self, obs, action, reward, next_obs, done):
        self.buffer.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)
        return (
            np.array(obs, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_obs, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class SACAgent:
    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.device = torch.device(config.device)

        obs, _ = self.env.reset()
        obs_vec = encode_obs_ppo(obs)

        self.obs_dim = obs_vec.shape[0]
        self.action_dim = self.env.action_space.n

        self.actor = DiscretePolicyNet(
            obs_dim=self.obs_dim,
            hidden_dim=config.hidden_dim,
            action_dim=self.action_dim,
        ).to(self.device)

        self.q1 = DiscreteQNet(
            obs_dim=self.obs_dim,
            hidden_dim=config.hidden_dim,
            action_dim=self.action_dim,
        ).to(self.device)

        self.q2 = DiscreteQNet(
            obs_dim=self.obs_dim,
            hidden_dim=config.hidden_dim,
            action_dim=self.action_dim,
        ).to(self.device)

        self.q1_target = DiscreteQNet(
            obs_dim=self.obs_dim,
            hidden_dim=config.hidden_dim,
            action_dim=self.action_dim,
        ).to(self.device)

        self.q2_target = DiscreteQNet(
            obs_dim=self.obs_dim,
            hidden_dim=config.hidden_dim,
            action_dim=self.action_dim,
        ).to(self.device)

        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.learning_rate)
        self.q1_optimizer = torch.optim.Adam(self.q1.parameters(), lr=config.learning_rate)
        self.q2_optimizer = torch.optim.Adam(self.q2.parameters(), lr=config.learning_rate)

        self.replay_buffer = ReplayBuffer(capacity=config.buffer_size)

        os.makedirs(os.path.dirname(config.checkpoint_path), exist_ok=True)

    def name(self):
        return "sac"

    def obs_tensor(self, obs):
        obs_vec = encode_obs_ppo(obs)
        obs_tensor = torch.tensor(obs_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        return obs_tensor, obs_vec

    def act(self, obs):
        obs_tensor, _ = self.obs_tensor(obs)
        with torch.no_grad():
            logits = self.actor(obs_tensor)
            probs = F.softmax(logits, dim=-1)
            action = torch.multinomial(probs, num_samples=1)
        return int(action.item())

    def act_greedy(self, obs):
        obs_tensor, _ = self.obs_tensor(obs)
        with torch.no_grad():
            logits = self.actor(obs_tensor)
            action = torch.argmax(logits, dim=-1)
        return int(action.item())

    def soft_update(self, net, target_net):
        tau = self.config.tau
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def update(self):
        if len(self.replay_buffer) < max(self.config.batch_size, self.config.learning_starts):
            return

        obs, actions, rewards, next_obs, dones = self.replay_buffer.sample(self.config.batch_size)

        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_obs = torch.tensor(next_obs, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        alpha = self.config.alpha

        with torch.no_grad():
            next_logits = self.actor(next_obs)
            next_log_probs = F.log_softmax(next_logits, dim=-1)
            next_probs = F.softmax(next_logits, dim=-1)

            q1_next = self.q1_target(next_obs)
            q2_next = self.q2_target(next_obs)
            min_q_next = torch.min(q1_next, q2_next)

            v_next = (next_probs * (min_q_next - alpha * next_log_probs)).sum(dim=1)
            target_q = rewards + self.config.gamma * (1.0 - dones) * v_next

        q1_values = self.q1(obs).gather(1, actions).squeeze(1)
        q2_values = self.q2(obs).gather(1, actions).squeeze(1)

        q1_loss = F.mse_loss(q1_values, target_q)
        q2_loss = F.mse_loss(q2_values, target_q)

        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        logits = self.actor(obs)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = F.softmax(logits, dim=-1)

        q1_pi = self.q1(obs)
        q2_pi = self.q2(obs)
        min_q_pi = torch.min(q1_pi, q2_pi)

        actor_loss = (probs * (alpha * log_probs - min_q_pi)).sum(dim=1).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(self.q1, self.q1_target)
        self.soft_update(self.q2, self.q2_target)

    def train(self, eval_env=None, eval_episodes=10, eval_interval_steps=5000):
        total_steps = 0
        best_mean_reward = float("-inf")

        best_path = self.config.checkpoint_path.replace(".pt", "_best.pt")
        history_path = self.config.checkpoint_path.replace(".pt", "_history.csv")

        obs, _ = self.env.reset()
        episode_steps = 0

        with open(history_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["total_steps", "eval_mean_reward"])

            pbar = tqdm(total=self.config.total_timesteps, desc="Training SAC")

            while total_steps < self.config.total_timesteps:
                action = self.act(obs)
                next_obs, reward, done, truncated, _ = self.env.step(action)
                reward = float(reward) - self.config.step_penalty

                obs_vec = encode_obs_ppo(obs)
                next_obs_vec = encode_obs_ppo(next_obs)
                terminal = bool(done or truncated)

                self.replay_buffer.add(obs_vec, action, reward, next_obs_vec, terminal)

                for _ in range(self.config.updates_per_step):
                    self.update()

                obs = next_obs
                total_steps += 1
                episode_steps += 1

                if terminal:
                    obs, _ = self.env.reset()
                    episode_steps = 0

                if eval_env is not None and total_steps % eval_interval_steps == 0:
                    eval_rewards = self.evaluate(num_episodes=eval_episodes, env=eval_env)
                    mean_reward = float(np.mean(eval_rewards))

                    if mean_reward > best_mean_reward:
                        best_mean_reward = mean_reward
                        self.save(best_path)

                    writer.writerow([total_steps, mean_reward])
                    f.flush()

                    pbar.set_postfix({"best_rew": f"{best_mean_reward:.3f}"})

                pbar.update(1)

            pbar.close()

        self.save(self.config.checkpoint_path)
        print(f"\nSaved SAC history to {history_path}")
        print(f"Best checkpoint: {best_path}")

        return best_path

    def evaluate(self, num_episodes=10, env=None):
        env = self.env if env is None else env
        rewards = []

        for _ in range(num_episodes):
            obs, _ = env.reset()
            done = False
            truncated = False
            total_reward = 0.0

            while not (done or truncated):
                action = self.act_greedy(obs)
                obs, reward, done, truncated, _ = env.step(action)
                reward = float(reward) - self.config.step_penalty
                total_reward += reward

            rewards.append(float(total_reward))

        return rewards

    def rollout(self, max_steps=250, deterministic=True):
        obs, _ = self.env.reset()
        done = False
        truncated = False
        steps = 0
        total_reward = 0.0
        trajectory = []

        pos = tuple(int(x) for x in self.env.unwrapped.agent_pos) if hasattr(self.env.unwrapped, "agent_pos") else None
        direction = int(self.env.unwrapped.agent_dir) if hasattr(self.env.unwrapped, "agent_dir") else None
        trajectory.append({
            "step": 0,
            "pos": pos,
            "dir": direction,
            "action": None,
            "reward": 0.0,
        })

        while not (done or truncated) and steps < max_steps:
            if deterministic:
                action = self.act_greedy(obs)
            else:
                action = self.act(obs)

            next_obs, reward, done, truncated, _ = self.env.step(action)
            reward = float(reward) - self.config.step_penalty

            pos = tuple(int(x) for x in self.env.unwrapped.agent_pos) if hasattr(self.env.unwrapped, "agent_pos") else None
            direction = int(self.env.unwrapped.agent_dir) if hasattr(self.env.unwrapped, "agent_dir") else None

            trajectory.append({
                "step": int(steps + 1),
                "pos": pos,
                "dir": direction,
                "action": int(action),
                "reward": float(reward),
            })

            total_reward += reward
            obs = next_obs
            steps += 1

        return {
            "success": int(done),
            "steps": int(steps),
            "total_reward": float(total_reward),
            "trajectory": trajectory,
        }

    def save(self, path=None):
        ckpt = path if path is not None else self.config.checkpoint_path
        torch.save({
            "actor_state_dict": self.actor.state_dict(),
            "q1_state_dict": self.q1.state_dict(),
            "q2_state_dict": self.q2.state_dict(),
            "q1_target_state_dict": self.q1_target.state_dict(),
            "q2_target_state_dict": self.q2_target.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "q1_optimizer_state_dict": self.q1_optimizer.state_dict(),
            "q2_optimizer_state_dict": self.q2_optimizer.state_dict(),
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
        }, ckpt)

    def load(self, path=None):
        ckpt = path if path is not None else self.config.checkpoint_path
        data = torch.load(ckpt, map_location=self.device, weights_only=False)

        self.actor.load_state_dict(data["actor_state_dict"])
        self.q1.load_state_dict(data["q1_state_dict"])
        self.q2.load_state_dict(data["q2_state_dict"])
        self.q1_target.load_state_dict(data["q1_target_state_dict"])
        self.q2_target.load_state_dict(data["q2_target_state_dict"])
        self.actor_optimizer.load_state_dict(data["actor_optimizer_state_dict"])
        self.q1_optimizer.load_state_dict(data["q1_optimizer_state_dict"])
        self.q2_optimizer.load_state_dict(data["q2_optimizer_state_dict"])