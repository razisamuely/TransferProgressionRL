from pathlib import Path
from src.section1.agents.actor_critic_agent import ActorCriticAgent
import argparse
from src.section1.utils.ac_with_optuna import optimize_hyperparameters
from src.section1.utils.trainer import train
import gymnasium as gym
from src.section1.utils.constants import (
    DEFAULT_GAMMA,
    DEFAULT_EPISODES,
    DEFAULT_LOG_INTERVAL,
    DEFAULT_ACTOR_LEARNING_RATE,
    DEFAULT_CRITIC_LEARNING_RATE,
    DEFAULT_VISUAL_MODE,
    DEFAULT_OPTIMIZE_HYPER_PARAMETERS_MODE,
    MAX_INPUT_DIM,
    MAX_STEPS,
)
from src.section3.progressive_networks_actor_critic_agent import ProgressiveActorCriticAgent
from visualize_agent import create_agent
import os 

def train_with_args(args):
    # load acrobot
    acrobot_env = gym.make("Acrobot-v1")
    acrobot_env = acrobot_env.unwrapped
    acrobot_model =  create_agent(acrobot_env, Path(args.pretrained_acrobot_model_path))

    # 
    # load cartpole
    cartpole_env = gym.make("CartPole-v1")
    cartpole_env = cartpole_env.unwrapped
    cartpole_model = create_agent(cartpole_env,Path(args.pretrained_cart_pole_model_path))
    
    
    # load the env we train 
    env = gym.make(args.env)
    env = env.unwrapped

    agent = ProgressiveActorCriticAgent(
        env=env,
        gamma=args.gamma,
        h11_w1=args.h11_w1,
        h11_w2=args.h11_w2,
        h12_w1=args.h12_w1,
        h12_w2=args.h12_w2,
        lr_actor=args.learning_rate_actor,
        lr_critic=args.learning_rate_critic,
        hidden1=args.hidden1,
        hidden2=args.hidden2,
        agent_1_env_name="Acrobot-v1",
        agent_1_weights_path=args.pretrained_acrobot_model_path,
        agent_2_env_name="CartPole-v1",
        agent_2_weights_path=args.pretrained_cart_pole_model_path,
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_acrobot_model_path",
        type=str,
        default="assets/section_1/actor_critic/acrobot/models/acrobot_exp/episode_best",
    )
    parser.add_argument(
        "--pretrained_cart_pole_model_path",
        type=str,
        default="assets/section_1/actor_critic/cart_pole/models/cartpole_exp/episode_best",
    )
    parser.add_argument(
        "--optimize_hyper_parameters",
        type=bool,
        default=DEFAULT_OPTIMIZE_HYPER_PARAMETERS_MODE,
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
    
    parser.add_argument("--h11_w1", type=float, default=0.0)
    parser.add_argument("--h12_w1", type=float, default=0.0)
    parser.add_argument("--h11_w2", type=float, default=0.0)
    parser.add_argument("--h12_w2", type=float, default=0.0)
    parser.add_argument("--exp_name", type=str, default="mountaincar_exp")
    parser.add_argument("--hidden1", type=int, default=256)
    parser.add_argument("--hidden2", type=int, default=256)
    parser.add_argument("--models_dir", type=str, default="assets/section_3/actor_critic/mountain_car/models")
    parser.add_argument(
        "--max_steps", type=int, default=MAX_STEPS, help="Maximum steps per episode"
    )
    parser.add_argument(
        "--early_exit", type=int, default=None, help="Criteria for early stop"
    )
    parser.add_argument(
        "--env",
        type=str,
        choices=["Acrobot-v1", "MountainCarContinuous-v0", "CartPole-v1"],
        default="MountainCarContinuous-v0",
    )
    
    return parser.parse_args()



if __name__ == "__main__":
    args = parse_args()
    if args.optimize_hyper_parameters:
        optimize_hyperparameters(args)
    else:
        train_with_args(args)

