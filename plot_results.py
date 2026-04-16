import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def infer_label(path: str) -> str:
    name = os.path.basename(path).replace("_history.csv", "").replace(".csv", "")

    if "FourRooms" in name:
        return "FourRooms"
    if "N2_S4" in name:
        return "MultiRoom-N2-S4"
    if "N4_S5" in name:
        return "MultiRoom-N4-S5"
    if "N6" in name:
        return "MultiRoom-N6"
    if "Empty_8x8" in name:
        return "Empty-8x8"

    return name


def plot_single_csv(csv_path: str, out_path: str | None = None, title: str | None = None):
    df = pd.read_csv(csv_path)

    plt.figure(figsize=(8, 4))

    if "episode" in df.columns and "reward" in df.columns:
        g = df.groupby("episode")["reward"].mean().reset_index()
        plt.plot(g["episode"], g["reward"], label=infer_label(csv_path))
        plt.xlabel("Episode")
        plt.ylabel("Mean reward")
        plt.title(title or "Learning curve")

    elif "total_steps" in df.columns and "eval_mean_reward" in df.columns:
        plt.plot(df["total_steps"], df["eval_mean_reward"], label=infer_label(csv_path))
        plt.xlabel("Timesteps")
        plt.ylabel("Mean reward")
        plt.title(title or "Learning curve")

    else:
        raise ValueError(
            f"Unsupported CSV format in {csv_path}. "
            "Expected either ['episode', 'reward'] or ['total_steps', 'eval_mean_reward']."
        )

    plt.legend()
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, bbox_inches="tight")

    plt.show()


def plot_multi_csv(
    csv_paths: list[str],
    out_path: str | None = None,
    title: str = "Learning curves",
    max_timesteps: int | None = None,
):
    plt.figure(figsize=(9, 5))

    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        label = infer_label(csv_path)

        if "total_steps" in df.columns and "eval_mean_reward" in df.columns:
            if max_timesteps is not None:
                df = df[df["total_steps"] <= max_timesteps]

            plt.plot(df["total_steps"], df["eval_mean_reward"], label=label)

        elif "episode" in df.columns and "reward" in df.columns:
            g = df.groupby("episode")["reward"].mean().reset_index()
            plt.plot(g["episode"], g["reward"], label=label)

        else:
            raise ValueError(
                f"Unsupported CSV format in {csv_path}. "
                "Expected either ['episode', 'reward'] or ['total_steps', 'eval_mean_reward']."
            )

    # axis label based on first csv
    first_df = pd.read_csv(csv_paths[0])
    if "total_steps" in first_df.columns:
        plt.xlabel("Timesteps")
    else:
        plt.xlabel("Episode")

    plt.ylabel("Mean reward")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, bbox_inches="tight")

    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", type=str, nargs="+", required=True)
    parser.add_argument("--title", type=str, default="Learning curves")
    parser.add_argument("--out-path", type=str, default=None)
    parser.add_argument("--max-timesteps", type=int, default=None)
    args = parser.parse_args()

    if len(args.csv_path) == 1:
        plot_single_csv(
            csv_path=args.csv_path[0],
            out_path=args.out_path,
            title=args.title,
        )
    else:
        plot_multi_csv(
            csv_paths=args.csv_path,
            out_path=args.out_path,
            title=args.title,
            max_timesteps=args.max_timesteps,
        )


if __name__ == "__main__":
    main()