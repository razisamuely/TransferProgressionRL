import gymnasium as gym
import numpy as np
import optuna
import argparse
from src.section1.utils.trainer import train
from src.section1.utils.ac_with_optuna import optimize_hyperparameters
from pathlib import Path
from src.section1.utils.constants import (
    DEFAULT_GAMMA,
    DEFAULT_EPISODES,
    DEFAULT_LOG_INTERVAL,
    DEFAULT_ACTOR_LEARNING_RATE,
    DEFAULT_CRITIC_LEARNING_RATE,
    DEFAULT_VISUAL_MODE,
    DEFAULT_OPTIMIZE_HYPER_PARAMETERS_MODE,
    MAX_STEPS,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--optimize_hyper_parameters",
        type=bool,
        default=DEFAULT_OPTIMIZE_HYPER_PARAMETERS_MODE,
    )
    parser.add_argument(
        "--env",
        type=str,
        choices=["Acrobot-v1", "MountainCarContinuous-v0", "CartPole-v1"],
    )
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument(
        "--visual_mode",
        type=str,
        choices=["pyplot", "tensorboard"],
        default=DEFAULT_VISUAL_MODE,
    )
    parser.add_argument("--log_interval", type=int, default=DEFAULT_LOG_INTERVAL)
    parser.add_argument(
        "--learning_rate_actor", type=float, default=DEFAULT_ACTOR_LEARNING_RATE
    )
    parser.add_argument(
        "--learning_rate_critic", type=float, default=DEFAULT_CRITIC_LEARNING_RATE
    )
    parser.add_argument(
        "--target_env_name",
        type=str,
        choices=["Acrobot-v1", "MountainCarContinuous-v0", "CartPole-v1"],
    )
    parser.add_argument("--models_dir", type=str)
    parser.add_argument(
        "--max_steps", type=int, default=MAX_STEPS, help="Maximum steps per episode"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.optimize_hyper_parameters:
        optimize_hyperparameters(args)
    else:
        train(args)

# Train without hyperparameter optimization CartPole
# python src/section1/train_actor_critic_agent.py \
# --env CartPole-v1 \
# --models_dir assets/section_1/actor_critic/cart_pole/models

# Train without hyperparameter optimization MountainCarContinuous
# python src/section1/train_actor_critic_agent.py \
# --env MountainCarContinuous-v0 \
# --models_dir assets/section_1/actor_critic/mountain_car/models

# Train without hyperparameter optimization Acrobot
# python src/section1/train_actor_critic_agent.py \
# --env Acrobot-v1 \
# --models_dir assets/section_1/actor_critic/acrobot/models

# optimize_hyperparameters
# python src/section1/train_actor_critic_agent.py \
# --optimize_hyper_parameters True \
# --env CartPole-v1 \
# --models_dir assets/section_1/actor_critic/cart_pole/models

# python src/section1/train_actor_critic_agent.py \
# --optimize_hyper_parameters True \
# --env MountainCarContinuous-v0 \
# --models_dir assets/section_1/actor_critic/mountain_car/models
