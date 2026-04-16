# RL Navigation Project

Starter code for a MiniGrid MultiRoom project comparing:
- Random baseline
- Prioritized Sweeping baseline
- POMCP-lite (history tree + particle filter over latent full states)
- PPO via Stable-Baselines3

## Install

```bash
pip install -r requirements.txt
```

## Run

Random:
```bash
python main.py --agent random --env-id MiniGrid-MultiRoom-N4-S5-v0 --episodes 50
```

Prioritized Sweeping baseline:
```bash
python main.py --agent ps --env-id MiniGrid-MultiRoom-N4-S5-v0 --episodes 200
```

POMCP-lite:
```bash
python main.py --agent pomcp --env-id MiniGrid-MultiRoom-N4-S5-v0 --episodes 100 --num-sims 100
```

PPO:
```bash
python main.py --agent ppo --env-id MiniGrid-MultiRoom-N4-S5-v0 --total-timesteps 50000
```

## Notes

- MiniGrid observations are dictionaries with `image`, `direction`, and `mission`; the default partial observation is a compact `7x7x3` symbolic encoding rather than raw pixels. Use wrappers if you want flattened or image-only observations. See Farama MiniGrid docs. citeturn0search2turn0search0
- PPO in Stable-Baselines3 supports dictionary observations via `MultiInputPolicy`, and SB3 supports saving/loading checkpoints directly. citeturn1search1turn1search0
- The POMCP implementation here is intentionally simplified so you can submit and iterate quickly; it uses cloned env states for simulation.
