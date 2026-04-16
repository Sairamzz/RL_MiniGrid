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
    rollout_depth: int = 40 # Maximum depth of the search tree during rollouts
    discount_rollout: float = 0.99 # Discount factor for rewards during rollouts
    step_penalty: float = 0.001 # Penalty for taking a step to encourage shorter solutions

@dataclass
class PPOConfig:
    total_timesteps: int = 2000000 # Total number of timesteps to train the agent
    learning_rate: float = 3e-4
    n_steps: int = 512 # max rollout length before the update
    seq_len: int = 16
    batch_size: int = 64

    gamma: float = 0.99 # Discount factor
    gae_lambda: float = 0.95 # GAE lambda parameter
    update_epochs: int = 4 # Number of epochs to update the policy 

    clip_eps: float = 0.2 # PPO clipping epsilon
    vf_coeff: float = 0.5 # Coefficient for value function loss
    entropy_coeff: float = 0.1 # Coefficient for entropy bonus

    hidden_dim: int = 128 # Hidden layer size for the actor and critic networks
    lstm_hidden: int = 128 # Hidden size for the LSTM layer

    max_grad_norm: float = 0.5 # Maximum gradient norm for clipping
    step_penalty: float = 0.001 # Penalty for taking a step to encourage shorter solutions

    checkpoint_path: str = "results/checkpoints/ppo_minigrid.pt"
    csv_path: str = "results/csv/ppo_minigrid_history.csv"
    device: str = "cpu"

@dataclass
class A2CConfig:
    total_timesteps: int = 100000
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95

    n_steps: int = 256
    entropy_coeff: float = 0.01
    vf_coeff: float = 0.5
    max_grad_norm: float = 0.5

    hidden_dim: int = 128
    lstm_hidden: int = 128
    step_penalty: float = 0.001

    checkpoint_path: str = "results/checkpoints/a2c_minigrid.pt"
    csv_path: str = "results/csv/a2c_minigrid_history.csv"
    device: str = "cpu"

@dataclass
class QLearningConfig:
    alpha: float = 0.2
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995
    optimistic_initial_q: float = 0.0

@dataclass
class SACConfig:
    total_timesteps: int = 100000 # Total number of timesteps to train the agent
    learning_rate: float = 3e-4 
    gamma: float = 0.99 # Discount factor
    tau: float = 0.005 # Soft update coefficient for target networks

    batch_size: int = 128 # Batch size for training
    buffer_size: int = 100000 # Size of the replay buffer
    learning_starts: int = 1000 # Number of steps before starting to learn
    updates_per_step: int = 1 # Number of updates per training step

    hidden_dim: int = 256 # Hidden layer size for the actor and critic networks
    alpha: float = 0.2
    step_penalty: float = 0.001 # Penalty for taking a step to encourage shorter solutions

    checkpoint_path: str = "results/checkpoints/sac_minigrid.pt"
    device: str = "cpu"
