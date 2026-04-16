import os
import csv
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from envs.observation import encode_obs_ppo
from models.actor_critic import ActorCriticLSTM


class A2CAgent:
    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.device = torch.device(config.device)

        obs, _ = self.env.reset()
        obs_vec = encode_obs_ppo(obs)

        self.obs_dim = obs_vec.shape[0]
        self.action_dim = self.env.action_space.n

        self.model = ActorCriticLSTM(
            obs_dim=self.obs_dim,
            hidden_dim=config.hidden_dim,
            action_dim=self.action_dim,
            lstm_hidden=config.lstm_hidden,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        os.makedirs(os.path.dirname(config.checkpoint_path), exist_ok=True)
        os.makedirs(os.path.dirname(config.csv_path), exist_ok=True)

    def name(self):
        return "a2c"

    def _obs_tensor(self, obs):
        obs_vec = encode_obs_ppo(obs)
        obs_tensor = torch.tensor(obs_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        return obs_tensor, obs_vec

    def _init_hidden(self):
        return self.model.init_hidden(batch_size=1, device=self.device)

    def act(self, obs, hx=None, cx=None):
        if hx is None:
            hx, cx = self._init_hidden()

        obs_tensor, _ = self._obs_tensor(obs)
        with torch.no_grad():
            logits, value, hx, cx = self.model(obs_tensor, hx, cx)
            probs = F.softmax(logits, dim=-1)
            action = torch.multinomial(probs, num_samples=1)
            log_prob = -F.cross_entropy(logits, action.squeeze(-1), reduction="none")

        return int(action.item()), float(log_prob.item()), float(value.item()), hx, cx

    def act_greedy(self, obs, hx=None, cx=None):
        if hx is None:
            hx, cx = self._init_hidden()

        obs_tensor, _ = self._obs_tensor(obs)
        with torch.no_grad():
            logits, _, hx, cx = self.model(obs_tensor, hx, cx)
            action = torch.argmax(logits, dim=-1)

        return int(action.item()), hx, cx

    def collect_rollout(self):
        obs, _ = self.env.reset()
        hx, cx = self._init_hidden()

        storage = {
            "obs": [],
            "actions": [],
            "rewards": [],
            "dones": [],
            "values": [],
            "hx": [],
            "cx": [],
        }

        for _ in range(self.config.n_steps):
            obs_tensor, obs_vec = self._obs_tensor(obs)

            storage["hx"].append(hx.squeeze(0).detach().cpu().numpy())
            storage["cx"].append(cx.squeeze(0).detach().cpu().numpy())

            with torch.no_grad():
                logits, value, hx, cx = self.model(obs_tensor, hx, cx)
                probs = F.softmax(logits, dim=-1)
                action = torch.multinomial(probs, num_samples=1).squeeze(dim=1)

            action_item = int(action.item())
            next_obs, reward, done, truncated, _ = self.env.step(action_item)
            reward = float(reward) - self.config.step_penalty
            terminal = bool(done or truncated)

            storage["obs"].append(obs_vec)
            storage["actions"].append(action_item)
            storage["rewards"].append(float(reward))
            storage["dones"].append(float(terminal))
            storage["values"].append(float(value.item()))

            obs = next_obs

            if terminal:
                obs, _ = self.env.reset()
                hx, cx = self._init_hidden()

        obs_tensor, _ = self._obs_tensor(obs)
        with torch.no_grad():
            _, last_value, _, _ = self.model(obs_tensor, hx, cx)

        return storage, float(last_value.item())

    def compute_returns_and_advantages(self, rewards, dones, values, last_value):
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)
        values = np.array(values, dtype=np.float32)

        advantages = np.zeros_like(rewards, dtype=np.float32)
        gae = 0.0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = last_value
            else:
                next_value = values[t + 1]

            mask = 1.0 - dones[t]
            delta = rewards[t] + self.config.gamma * next_value * mask - values[t]
            gae = delta + self.config.gamma * self.config.gae_lambda * mask * gae
            advantages[t] = gae

        returns = advantages + values
        return returns, advantages

    def update(self, storage, last_value):
        returns, advantages = self.compute_returns_and_advantages(
            rewards=storage["rewards"],
            dones=storage["dones"],
            values=storage["values"],
            last_value=last_value,
        )

        obs_arr = np.array(storage["obs"], dtype=np.float32)
        act_arr = np.array(storage["actions"], dtype=np.int64)
        ret_arr = np.array(returns, dtype=np.float32)
        adv_arr = np.array(advantages, dtype=np.float32)
        hx_arr = np.array(storage["hx"], dtype=np.float32)
        cx_arr = np.array(storage["cx"], dtype=np.float32)
        done_arr = np.array(storage["dones"], dtype=np.float32)

        adv_arr = (adv_arr - adv_arr.mean()) / (adv_arr.std() + 1e-8)

        hx = torch.tensor(hx_arr[0], device=self.device).unsqueeze(0)
        cx = torch.tensor(cx_arr[0], device=self.device).unsqueeze(0)

        logits_all, values_all = [], []

        T = len(obs_arr)
        for t in range(T):
            obs_t = torch.tensor(obs_arr[t:t+1], dtype=torch.float32, device=self.device)
            logits, value, hx, cx = self.model(obs_t, hx, cx)
            logits_all.append(logits)
            values_all.append(value)

            if done_arr[t]:
                hx, cx = self._init_hidden()

        logits = torch.cat(logits_all, dim=0)
        values = torch.cat(values_all, dim=0).squeeze(-1)

        actions = torch.tensor(act_arr, dtype=torch.long, device=self.device)
        returns_t = torch.tensor(ret_arr, dtype=torch.float32, device=self.device)
        advantages_t = torch.tensor(adv_arr, dtype=torch.float32, device=self.device)

        log_probs = -F.cross_entropy(logits, actions, reduction="none")
        probs = F.softmax(logits, dim=-1)
        log_probs_all = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs_all).sum(dim=-1).mean()

        actor_loss = -(log_probs * advantages_t.detach()).mean()
        critic_loss = F.mse_loss(values, returns_t)

        total_loss = actor_loss + self.config.vf_coeff * critic_loss - self.config.entropy_coeff * entropy

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        self.optimizer.step()

    def train(self, eval_env=None, eval_episodes=10, eval_interval_updates=10, eval_seeds=None):
        total_steps = 0
        update_idx = 0

        best_mean_reward = float("-inf")
        best_path = self.config.checkpoint_path.replace(".pt", "_best.pt")
        history_path = self.config.csv_path

        with open(history_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["update", "total_steps", "eval_mean_reward"])

            pbar = tqdm(total=self.config.total_timesteps, desc="Training A2C")

            while total_steps < self.config.total_timesteps:
                storage, last_value = self.collect_rollout()
                self.update(storage, last_value)

                total_steps += len(storage["rewards"])
                update_idx += 1

                if eval_env is not None and (update_idx % eval_interval_updates == 0):
                    eval_rewards = self.evaluate(num_episodes=eval_episodes, env=eval_env, seeds=eval_seeds)
                    mean_reward = float(np.mean(eval_rewards))

                    if mean_reward > best_mean_reward:
                        best_mean_reward = mean_reward
                        self.save(best_path)

                    writer.writerow([update_idx, total_steps, mean_reward])
                    f.flush()

                    pbar.set_postfix({"best_rew": f"{best_mean_reward:.3f}"})

                pbar.update(len(storage["rewards"]))

            pbar.close()

        self.save(self.config.checkpoint_path)
        print(f"\nSaved A2C history to {history_path}")
        print(f"Best checkpoint: {best_path}")

        return best_path

    def evaluate(self, num_episodes=10, env=None, seeds=None):
        env = self.env if env is None else env
        rewards = []

        for ep in range(num_episodes):
            if seeds is not None:
                obs, _ = env.reset(seed=seeds[ep])
            else:
                obs, _ = env.reset()

            hx, cx = self._init_hidden()
            done = False
            truncated = False
            total_reward = 0.0

            while not (done or truncated):
                action, hx, cx = self.act_greedy(obs, hx, cx)
                obs, reward, done, truncated, _ = env.step(action)
                reward = float(reward) - self.config.step_penalty
                total_reward += reward

            rewards.append(float(total_reward))

        return rewards

    def rollout(self, max_steps=250, deterministic=True):
        obs, _ = self.env.reset()
        hx, cx = self._init_hidden()
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
                action, hx, cx = self.act_greedy(obs, hx, cx)
            else:
                action, _, _, hx, cx = self.act(obs, hx, cx)

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
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
        }, ckpt)

    def load(self, path=None):
        ckpt = path if path is not None else self.config.checkpoint_path
        data = torch.load(ckpt, map_location=self.device, weights_only=False)

        self.model.load_state_dict(data["model_state_dict"])
        self.optimizer.load_state_dict(data["optimizer_state_dict"])