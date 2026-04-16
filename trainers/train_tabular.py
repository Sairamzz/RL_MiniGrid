import os
import time
import numpy as np

from tqdm import tqdm
from utils.io import save_csv
from utils.checkpoints import save_pickle


def run_episode(env, agent, max_steps=150, training=True, seed=None):
    if seed is None:
        obs, info = env.reset()
    else:
        obs, info = env.reset(seed=seed)
    
    agent.begin_episode()

    if hasattr(agent, "set_env_reference"):
        agent.set_env_reference(env)

    if hasattr(agent, "set_initial_particle"):
        agent.set_initial_particle(env)

    total_reward = 0.0
    steps = 0
    done = False
    truncated = False
    decision_times = []

    while not (done or truncated) and steps < max_steps:
        t0 = time.perf_counter()
        action = agent.act(obs)
        t1 = time.perf_counter()
        decision_times.append(t1 - t0)

        next_obs, reward, done, truncated, info = env.step(action)

        if hasattr(agent, "step_penalty"):
            reward = float(reward) - agent.step_penalty
        else:
            reward = float(reward)

        # Error Debug
        # print("action:", action, type(action))
        # print("reward:", reward, type(reward))
        # print("done:", done, type(done))
        # print("truncated:", truncated, type(truncated))
        # print("info:", info, type(info))

        # if isinstance(next_obs, dict):
        #     print("next_obs keys:", next_obs.keys())
        #     if "image" in next_obs:
        #         print("next_obs['image'] shape:", np.array(next_obs["image"]).shape)
        #         print("next_obs['image'] dtype:", np.array(next_obs["image"]).dtype)
        #     if "direction" in next_obs:
        #         print("next_obs['direction']:", next_obs["direction"], type(next_obs["direction"]))
        # else:
        #     print("next_obs type:", type(next_obs))

        # Normalize reward to a plain float
        if isinstance(reward, tuple):
            reward = reward[0]
        reward = float(np.asarray(reward).reshape(-1)[0])

        if training:
            agent.observe(obs, action, reward, next_obs, done or truncated)

        total_reward += reward
        obs = next_obs
        steps += 1

    agent.end_episode()

    return {
        "reward": total_reward,
        "steps": steps,
        "success": int(done),
        "avg_decision_time": sum(decision_times) / max(1, len(decision_times)),
    }


def train_tabular_agent(
    env,
    agent,
    episodes,
    max_steps,
    output_csv,
    checkpoint_dir=None,
    checkpoint=50,
):
    rows = []

    for ep in tqdm(range(episodes), desc=f"Training {agent.name()}"):
        result = run_episode(env, agent, max_steps=max_steps, training=True)
        result["episode"] = ep
        result["agent"] = agent.name()
        rows.append(result)

        if checkpoint_dir and hasattr(agent, "state_dict"):
            if (ep + 1) % checkpoint == 0:
                ckpt_path = os.path.join(
                    checkpoint_dir,
                    f"{agent.name()}_episode_{ep+1}.pkl"
                )
                save_pickle(agent.state_dict(), ckpt_path)

    save_csv(rows, output_csv)

    if checkpoint_dir and hasattr(agent, "state_dict"):
        final_ckpt = os.path.join(checkpoint_dir, f"{agent.name()}_final.pkl")
        save_pickle(agent.state_dict(), final_ckpt)

    return rows