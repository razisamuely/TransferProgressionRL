import matplotlib.pyplot as plt
import pandas as pd


def print_parameters(title, my_dict):
    print(title)
    for k, v in vars(my_dict).items():
        print(f"{k}={v}")


def plot_metrics(episodes_rewards, losses, paths):
    plt.figure(figsize=(12, 6))
    plt.plot(episodes_rewards, label="Rewards", color="blue", linewidth=0.5)
    plt.plot(
        pd.Series(episodes_rewards).rolling(100).mean(),
        label="Rewards MA@50",
        color="green",
        linewidth=3,
    )
    plt.title("Rewards")
    plt.legend()
    plt.grid(True)
    plt.savefig(paths[0])

    plt.figure(figsize=(12, 6))
    plt.plot(losses, label="Losses", color="red", linewidth=0.5)
    plt.plot(
        pd.Series(losses).rolling(100).mean(),
        label="Losses MA@50",
        color="orange",
        linewidth=3,
    )

    plt.title("Losses")
    plt.legend()
    plt.grid(True)
    plt.savefig(paths[1])
