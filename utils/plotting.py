from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt


def plot_rewards(csv_path: str, out_path: str | None = None):
    df = pd.read_csv(csv_path)
    g = df.groupby("episode")["reward"].mean().reset_index()
    plt.figure(figsize=(8, 4))
    plt.plot(g["episode"], g["reward"])
    plt.xlabel("Episode")
    plt.ylabel("Mean reward")
    plt.title("Learning curve")
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path)
    plt.show()
