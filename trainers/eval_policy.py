def evaluate_greedy_policy(env, agent, max_steps=150):
    obs, info = env.reset()
    done = False
    truncated = False
    steps = 0
    total_reward = 0.0
    trajectory = []

    while not (done or truncated) and steps < max_steps:
        pos = tuple(env.unwrapped.agent_pos) if hasattr(env.unwrapped, "agent_pos") else None
        direction = int(env.unwrapped.agent_dir) if hasattr(env.unwrapped, "agent_dir") else None

        if hasattr(agent, "act_greedy"):
            action = agent.act_greedy(obs)
        else:
            action = agent.act(obs)

        next_obs, reward, done, truncated, info = env.step(action)

        trajectory.append({
            "step": steps,
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
        "steps": steps,
        "total_reward": total_reward,
        "trajectory": trajectory,
    }