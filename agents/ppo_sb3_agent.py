import os
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor


class PPOAgent:
    def __init__(self, env, config):
        self.env = Monitor(env)
        self.config = config

        os.makedirs(os.path.dirname(config.checkpoint_path), exist_ok=True)

        self.model = PPO(
            policy="MlpPolicy",
            env=self.env,
            learning_rate=config.learning_rate,
            n_steps=config.n_steps,
            batch_size=config.batch_size,
            gamma=config.gamma,
            verbose=1,
        )

    def train(self):
        self.model.learn(
            total_timesteps=self.config.total_timesteps,
            progress_bar=True
        )
        self.model.save(self.config.checkpoint_path)

    def load(self, path=None):
        ckpt = path if path is not None else self.config.checkpoint_path
        self.model = PPO.load(ckpt, env=self.env)

    def evaluate(self, episodes=10):
        rewards = []

        for _ in range(episodes):
            obs, info = self.env.reset()
            done = False
            truncated = False
            total_reward = 0.0

            while not (done or truncated):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = self.env.step(action)
                reward = float(reward) - 0.001
                total_reward += reward

            rewards.append(total_reward)

        return rewards