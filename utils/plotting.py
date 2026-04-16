import pandas as pd
import matplotlib.pyplot as plt


def plot_multi_env_learning_curves(
    csv_paths: list[str],
    labels: list[str],
    title: str,
    out_path: str | None = None,
    max_timesteps: int | None = None,
):
    if len(csv_paths) != len(labels):
        raise ValueError("csv_paths and labels must have the same length.")

    plt.figure(figsize=(9, 5))

    for csv_path, label in zip(csv_paths, labels):
        df = pd.read_csv(csv_path)

        if "total_steps" not in df.columns or "eval_mean_reward" not in df.columns:
            raise ValueError(
                f"{csv_path} does not have the required columns "
                "['total_steps', 'eval_mean_reward']."
            )

        if max_timesteps is not None:
            df = df[df["total_steps"] <= max_timesteps]

        plt.plot(df["total_steps"], df["eval_mean_reward"], label=label)

    plt.xlabel("Timesteps")
    plt.ylabel("Mean reward")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path)

    plt.show()