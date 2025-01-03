import gymnasium as gym
import numpy as np
import optuna
import argparse
from src.section1.utils.trainer import train
from src.section1.utils.ac_with_optuna import optimize_hyperparameters
from src.section1.agents.actor_critic_agent import ActorCriticAgent
from src.section1.utils.constants import (
    DEFAULT_ENV,
    DEFAULT_GAMMA,
    DEFAULT_EPISODES,
    DEFAULT_LOG_INTERVAL,
    DEFAULT_ACTOR_LEARNING_RATE,
    DEFAULT_CRITIC_LEARNING_RATE,
    DEFAULT_VISUAL_MODE,
    DEFAULT_OPTIMIZE_HYPER_PARAMETERS_MODE,
    DEFAULT_TASK_DATA_DIR
)

from pathlib import Path

def continue_training(args, model_path):
    """Continue training from a saved model"""
    saved_model_path = Path(DEFAULT_TASK_DATA_DIR) / "section_1" / "actor_critic" / "models" / f"episode_{model_path}"
    if not saved_model_path.exists():
        raise ValueError(f"Model path not found: {saved_model_path}")
    ac_agent = ActorCriticAgent.load_models(saved_model_path)
    train(args, ac_agent)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--optimize_hyper_parameters", 
        type=bool, 
        default=DEFAULT_OPTIMIZE_HYPER_PARAMETERS_MODE
    )
    parser.add_argument(
        "--continue_training",
        type=str,
        choices=["best", "final"],
        help="Continue training from 'best' or 'final' saved model"
    )
    parser.add_argument(
        "--env",
        type=str,
        default=DEFAULT_ENV
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=DEFAULT_GAMMA
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=DEFAULT_EPISODES
    )
    parser.add_argument(
        "--visual_mode",
        type=str,
        choices=["pyplot", "tensorboard"],
        default=DEFAULT_VISUAL_MODE
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=DEFAULT_LOG_INTERVAL
    )
    parser.add_argument(
        "--learning_rate_actor",
        type=float,
        default=DEFAULT_ACTOR_LEARNING_RATE
    )
    parser.add_argument(
        "--learning_rate_critic",
        type=float,
        default=DEFAULT_CRITIC_LEARNING_RATE
    )
    parser.add_argument(
        "--transfer_learning",
        type=str,
        choices=["Acrobot-v1", "MountainCarContinuous-v0"],
        help="Target environment for transfer learning"
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.optimize_hyper_parameters:
        optimize_hyperparameters(args)
    elif args.continue_training:
        continue_training(args, args.continue_training)
    elif args.transfer_learning:
        transfer_learning(args, "best", args.transfer_learning)
    else:
        train(args)