from dataclasses import dataclass

@dataclass
class EnvConfig:
    env: str = "MiniGrid-MultiRoom-N4-S5-v0"
    slip_prob: float = 0.0 # Probability of slipping (taking a random action instead of the intended one)
    max_steps: int = 250
    seed: int = 0

@dataclass
class TrainingConfig:
    num_episodes: int = 200
    output_csv: str = "results/csv/results.csv"
    checkpoint: int = 50
    checkpoint_dir: str = "results/checkpoints/"
    
@dataclass
class PSConfig:
    gamma: float = 0.99 # Discount factor
    epsilon: float = 0.40 # Probability of exploration
    planning_steps: int = 50
    priority_threshold: float = 1e-5 # Minimum priority for a transition to be included in the planning queue
    optimistic_initial_q: float = 0.5 # Initial Q-value for optimistic planning

@dataclass
class POMCPConfig:
    gamma: float = 0.99 # Discount factor
    num_sim: int = 100
    c: float = 1.4 # UCB exploration constant
    rollout_depth: int = 30 # Maximum depth of the search tree during rollouts
    discount_rollout: float = 0.99 # Discount factor for rewards during rollouts

@dataclass
class PPOConfig:
    total_timesteps: int = 100000 # Total number of timesteps to train the agent
    learning_rate: float = 3e-4
    n_steps: int = 512 # max rollout length before the update
    batch_size: int = 64
    gamma: float = 0.99 # Discount factor
    checkpoint_path: str = "results/checkpoints/ppo_minigrid"
