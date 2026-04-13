from __future__ import annotations

import time


def run_episode(env, agent, training: bool = True):
    obs, info = env.reset()
    if hasattr(agent, "begin_episode"):
        agent.begin_episode()
    if hasattr(agent, "set_env_reference"):
        agent.set_env_reference(env)
        agent.set_initial_particle(env)

    done = False
    truncated = False
    ep_reward = 0.0
    steps = 0
    times = []

    while not (done or truncated):
        t0 = time.perf_counter()
        action = agent.act(obs)
        t1 = time.perf_counter()
        next_obs, reward, done, truncated, info = env.step(action)
        if hasattr(agent, "set_env_reference"):
            agent.set_env_reference(env)
        if training:
            agent.observe(obs, action, reward, next_obs, done or truncated, info)
        obs = next_obs
        ep_reward += reward
        steps += 1
        times.append(t1 - t0)

    if hasattr(agent, "end_episode"):
        agent.end_episode()

    return {
        "reward": ep_reward,
        "steps": steps,
        "success": int(done),
        "avg_decision_time": sum(times) / max(1, len(times)),
    }
