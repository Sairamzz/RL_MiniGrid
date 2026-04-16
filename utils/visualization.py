import os
import json
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

ACTION_NAMES = {
    0: "left",
    1: "right",
    2: "forward",
    3: "toggle",
}


def rollout_greedy(env, agent, max_steps=150):
    obs, info = env.reset()

    if hasattr(agent, "begin_episode"):
        agent.begin_episode()
    
    if hasattr(agent, "set_env_reference"):
        agent.set_env_reference(env)

    if hasattr(agent, "set_initial_particle"):
        agent.set_initial_particle(env)

    done = False
    truncated = False
    steps = 0
    total_reward = 0.0
    trajectory = []

    # Store initial state
    pos = tuple(int(x) for x in env.unwrapped.agent_pos) if hasattr(env.unwrapped, "agent_pos") else None
    direction = int(env.unwrapped.agent_dir) if hasattr(env.unwrapped, "agent_dir") else None
    trajectory.append({
        "step": 0,
        "pos": pos,
        "dir": direction,
        "action": None,
        "action_name": "start",
        "reward": 0.0,
    })

    pbar = tqdm(total=max_steps, desc=f"Visualizing {agent.name()}", leave=True)

    while not (done or truncated) and steps < max_steps:
        if hasattr(agent, "act_greedy"):
            action = agent.act_greedy(obs)
            # handle agents that return tuples like (action, hx, cx)
            if isinstance(action, tuple):
                action = action[0]
        else:
            action = agent.act(obs)
            if isinstance(action, tuple):
                action = action[0]

        next_obs, reward, done, truncated, info = env.step(action)

        pos = tuple(int(x) for x in env.unwrapped.agent_pos) if hasattr(env.unwrapped, "agent_pos") else None
        direction = int(env.unwrapped.agent_dir) if hasattr(env.unwrapped, "agent_dir") else None

        trajectory.append({
            "step": steps,
            "pos": pos,
            "dir": direction,
            "action": int(action),
            "action_name": ACTION_NAMES.get(int(action), str(action)),
            "reward": float(reward),
        })

        total_reward += reward
        obs = next_obs
        steps += 1

        pbar.update(1)
        pbar.set_postfix({
            "reward": f"{total_reward:.3f}",
            "success": int(done),
        })

    pbar.close()

    return {
        "success": int(done),
        "steps": steps,
        "total_reward": total_reward,
        "trajectory": trajectory,
    }

def rollout_pomcp(env, agent, max_steps=150, seed=None):
    if seed is None:
        obs, info = env.reset()
    else:
        obs, info = env.reset(seed=seed)

    if hasattr(agent, "begin_episode"):
        agent.begin_episode()

    if hasattr(agent, "set_env_reference"):
        agent.set_env_reference(env)

    if hasattr(agent, "set_initial_particle"):
        agent.set_initial_particle(env)

    done = False
    truncated = False
    steps = 0
    total_reward = 0.0
    trajectory = []

    pos = tuple(int(x) for x in env.unwrapped.agent_pos) if hasattr(env.unwrapped, "agent_pos") else None
    direction = int(env.unwrapped.agent_dir) if hasattr(env.unwrapped, "agent_dir") else None
    trajectory.append({
        "step": 0,
        "pos": pos,
        "dir": direction,
        "action": None,
        "action_name": "start",
        "reward": 0.0,
    })

    from tqdm import tqdm
    pbar = tqdm(total=max_steps, desc="Visualizing pomcp", leave=True)

    while not (done or truncated) and steps < max_steps:
        action = agent.act(obs)

        next_obs, reward, done, truncated, info = env.step(action)

        if hasattr(agent, "step_penalty"):
            reward = float(reward) - agent.step_penalty

        try:
            agent.observe(obs, action, reward, next_obs, done or truncated, info)
        except TypeError:
            agent.observe(obs, action, reward, next_obs, done or truncated)

        pos = tuple(int(x) for x in env.unwrapped.agent_pos) if hasattr(env.unwrapped, "agent_pos") else None
        direction = int(env.unwrapped.agent_dir) if hasattr(env.unwrapped, "agent_dir") else None

        trajectory.append({
            "step": steps + 1,
            "pos": pos,
            "dir": direction,
            "action": int(action),
            "action_name": ACTION_NAMES.get(int(action), str(action)),
            "reward": float(reward),
        })

        total_reward += float(reward)
        obs = next_obs
        steps += 1

        pbar.update(1)
        pbar.set_postfix({
            "reward": f"{total_reward:.3f}",
            "done": int(done),
            "trunc": int(truncated),
        })

    pbar.close()

    return {
        "success": int(done),
        "steps": steps,
        "total_reward": total_reward,
        "trajectory": trajectory,
    }


def rollout_ppo(env, model, max_steps=150):
    obs, info = env.reset()
    done = False
    truncated = False
    steps = 0
    total_reward = 0.0
    trajectory = []

    while not (done or truncated) and steps < max_steps:
        pos = tuple(int(x) for x in env.unwrapped.agent_pos) if hasattr(env.unwrapped, "agent_pos") else None
        direction = int(env.unwrapped.agent_dir) if hasattr(env.unwrapped, "agent_dir") else None

        action, _ = model.predict(obs, deterministic=True)
        next_obs, reward, done, truncated, info = env.step(action)

        # Robust conversion for scalar / 0-d / 1-d numpy outputs
        action_int = int(np.asarray(action).item())

        trajectory.append({
            "step": int(steps),
            "pos": pos,
            "dir": direction,
            "action": action_int,
            "action_name": ACTION_NAMES.get(action_int, str(action_int)),
            "reward": float(reward),
        })

        total_reward += float(reward)
        obs = next_obs
        steps += 1

    return {
        "success": int(done),
        "steps": int(steps),
        "total_reward": float(total_reward),
        "trajectory": trajectory,
    }


def _to_jsonable(obj):
    """
    Recursively convert numpy types into plain Python types
    so json.dump() can serialize them.
    """
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def save_trajectory_json(rollout_data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    clean_data = _to_jsonable(rollout_data)
    with open(path, "w") as f:
        json.dump(clean_data, f, indent=2)


def _build_grid_image(env):
    width = env.unwrapped.width
    height = env.unwrapped.height
    img = np.zeros((height, width), dtype=int)

    for x in range(width):
        for y in range(height):
            cell = env.unwrapped.grid.get(x, y)
            if cell is None:
                img[y, x] = 0
            elif getattr(cell, "type", None) == "wall":
                img[y, x] = 1
            elif getattr(cell, "type", None) == "goal":
                img[y, x] = 2
            else:
                img[y, x] = 0

    return img


def plot_trajectory_on_grid(env, rollout_data, save_path=None, title="Learned Trajectory"):
    grid_img = _build_grid_image(env)
    traj = rollout_data["trajectory"]
    positions = [item["pos"] for item in traj if item["pos"] is not None]

    plt.figure(figsize=(6, 6))
    plt.imshow(grid_img, origin="lower")

    if positions:
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        plt.plot(xs, ys, marker="o", linewidth=2)

        plt.scatter(xs[0], ys[0], marker="s", s=100, label="start")
        plt.scatter(xs[-1], ys[-1], marker="*", s=150, label="end")

    plt.title(
        f"{title}\nSuccess={rollout_data['success']} | Steps={rollout_data['steps']} | Reward={rollout_data['total_reward']:.3f}"
    )
    plt.legend()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()