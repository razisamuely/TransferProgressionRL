import gymnasium as gym
import numpy as np
import optuna
import argparse
from src.section1.utils.trainer import train
from src.section1.agents.actor_critic_agent import ActorCriticAgent, MAX_INPUT_DIM
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
import os 


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
        default="CartPole-v1",
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
    parser.add_argument("--exp_name", type=str, default="exp1")
    parser.add_argument("--hidden1", type=int, default=256)
    parser.add_argument("--hidden2", type=int, default=256)
    parser.add_argument("--models_dir", type=str, default="assets/section_1/actor_critic/models")
    parser.add_argument(
        "--max_steps", type=int, default=MAX_STEPS, help="Maximum steps per episode"
    )

    args = parser.parse_args()
    
    # Create models directory if it doesn't exist
    os.makedirs(args.models_dir, exist_ok=True)
    
    return args


def train_with_args(args):
    env = gym.make(args.env)
    env = env.unwrapped

    agent = ActorCriticAgent(
        env=env,
        batch_size=128,
        hidden1=args.hidden1,
        hidden2=args.hidden2,
        gamma=args.gamma,
        lr_actor=args.learning_rate_actor,
        lr_critic=args.learning_rate_critic,
        exp_name=args.exp_name,
        models_dir=args.models_dir,
    )

    state_padding_size = MAX_INPUT_DIM - env.observation_space.shape[0]
    turns, scores, avg_scores, goals = train(
        env=env,
        agent=agent,
        episodes=args.episodes,
        state_padding_size=state_padding_size,
        max_steps_per_episode=args.max_steps
    )

    agent.save_model(args.episodes, avg_scores[-1])
    return turns, scores, avg_scores, goals


if __name__ == "__main__":
    args = parse_args()
    if args.optimize_hyper_parameters:
        optimize_hyperparameters(args)
    else:
        train_with_args(args)



# For MountainCarContinuous-v0
# python src/section1/train_actor_critic_agent.py \
#     --env MountainCarContinuous-v0 \
#     --exp_name mountaincar_exp \
#     --gamma 0.99 \
#     --learning_rate_actor 0.00001 \
#     --learning_rate_critic 0.0001 \
#     --episodes 2000 \
#     --max_steps 2000 \
#     --models_dir assets/section_1/actor_critic/mountain_car/models

# # For CartPole-v1
# python src/section1/train_actor_critic_agent.py \
#     --env CartPole-v1 \
#     --exp_name cartpole_exp \
#     --gamma 0.99 \
#     --learning_rate_actor 0.00001 \
#     --learning_rate_critic  0.0001 \
#     --episodes 3000 \
#     --max_steps 1000 \
#     --models_dir assets/section_1/actor_critic/cart_pole/models

# # For Acrobot-v1
# python src/section1/train_actor_critic_agent.py \
#     --env Acrobot-v1 \
#     --exp_name acrobot_exp \
#     --gamma 0.99 \
#     --learning_rate_actor 0.00001 \
#     --learning_rate_critic 0.0001 \
#     --episodes 1000 \
#     --max_steps 500 \
#     --models_dir assets/section_1/actor_critic/acrobot/models