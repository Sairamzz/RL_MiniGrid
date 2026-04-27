# POMDP Navigation - Reinforcement Learning

## Project Overview:
This project studies **navigation under partial observability** in the MiniGrid benchmark using three different approaches: **POMCP**, **recurrent PPO**, and **recurrent A2C**. The goal is to compare **planning-based** and **learning-based** methods on environments of increasing difficulty, from simple room layouts to longer-horizon MultiRoom tasks that require **memory**, **exploration**, and **door interaction**. PPO and A2C use recurrent actor-critic policies with an LSTM to handle observation aliasing, while POMCP performs online planning from a particle-based belief state.

## Notes

- MiniGrid observations are dictionaries containing `image`, `direction`, and `mission`.  
  For the main benchmark environments used in this project, PPO and A2C encode the local `7x7x3` symbolic observation plus direction into a flattened feature vector.

- The standard PPO/A2C observation encoding has input dimension **984**:
  - `7 x 7 = 49` cells
  - each cell is one-hot encoded as `11` object classes + `6` colors + `3` states = `20`
  - `49 x 20 = 980`
  - plus `4` dimensions for direction
  - total = `984`

- Action subsets are environment-dependent:
  - `Empty` / `FourRooms`: `left`, `right`, `forward`
  - `MultiRoom`: `left`, `right`, `forward`, `toggle`
  Including `toggle` is essential for MultiRoom; without it, the agent cannot open doors and the task becomes unsolvable.

- PPO and A2C use **recurrent actor-critic networks with an LSTM** to handle partial observability.

- Reward shaping is applied during training:
  - step penalty: `-0.001` per step
  - exploration bonus for first-time visits to new cells
  The exploration bonus is disabled during evaluation so checkpoints are selected using clean task performance.

- PPO and A2C save:
  - final checkpoints in `results/checkpoints/`
  - best checkpoints in `results/checkpoints/*_best.pt`
  - reward history CSVs in `results/csv/`

- POMCP is an **online planner**, not a learned policy:
  - it does not save/load learned weights like PPO or A2C
  - visualization runs the planner directly at test time
  - it is much slower than PPO/A2C on larger environments

- For plotting learning curves, use the CSV files in `results/csv/`.  
  PPO/A2C history files log evaluation reward versus total timesteps, while POMCP logs episode-based results.

- The main benchmark environments used in this project are:
  - `MiniGrid-FourRooms-v0`
  - `MiniGrid-MultiRoom-N2-S4-v0`
  - `MiniGrid-MultiRoom-N4-S5-v0`
  - `MiniGrid-MultiRoom-N6-v0`
 
## View Results

Training and visualization outputs are saved under `results/`:

- `results/checkpoints/` — saved model checkpoints (`.pt`)
- `results/csv/` — learning curve data (`.csv`)
- `results/figures/` — generated plots and rollout images (`.png`)
- `results/trajectories/` — saved rollout trajectories (`.json`, if enabled)

To view the generated figures, open the files inside:

```bash
results/figures/
```

## How to Run

Starter code for a MiniGrid MultiRoom project comparing:
- **POMCP** (Partially Observable Monte Carlo Planning)
- **PPO** (Proximal Policy Optimization)
- **A2C** (Advantage Actor-Critic)

### Install

```bash
pip install -r requirements.txt
```

### Run

Random:
```bash
python main.py --agent random --env-id MiniGrid-MultiRoom-N4-S5-v0 --episodes 50
```

POMCP:
```bash
python main.py --mode visualize --agent pomcp --env-id MiniGrid-MultiRoom-N2-S4-v0 --max-steps 250
```

PPO:
- Train
```bash
python main.py --mode train --agent ppo --env-id MiniGrid-MultiRoom-N6-v0 --total-timesteps 200000 --max-steps 400
```

- Visualize
```bash
python main.py --mode visualize --agent ppo --env-id MiniGrid-MultiRoom-N6-v0 --checkpoint results/checkpoints/ppo_MiniGrid_MultiRoom_N6_v0_best.pt --max-steps 400
```

A2C:
- Train
```bash
python main.py --mode train --agent a2c --env-id MiniGrid-MultiRoom-N6-v0 --total-timesteps 200000 --max-steps 400
```

- Visualize
```bash
python main.py --mode visualize --agent a2c --env-id MiniGrid-MultiRoom-N6-v0 --checkpoint results/checkpoints/a2c_MiniGrid_MultiRoom_N6_v0_best.pt --max-steps 400
```

