from src.section1.agents.actor_critic_agent import ActorCriticAgent
import argparse
from src.section1.utils.trainer import train
import gymnasium as gym
from src.section1.utils.constants import (
    DEFAULT_GAMMA,
    DEFAULT_EPISODES,
    DEFAULT_LOG_INTERVAL,
    DEFAULT_ACTOR_LEARNING_RATE,
    DEFAULT_CRITIC_LEARNING_RATE,
    DEFAULT_VISUAL_MODE,
    MAX_STEPS,
)
from src.section3.progressive_networks_actor_critic_agent import ProgressiveActorCriticAgent


def load_acrobot(args):
    return ActorCriticAgent.load_models(
        args.pretrained_acrobot_model_path
    )


def load_cartpole(args):
    return ActorCriticAgent.load_models(
        args.pretrained_cart_pole_model_path
    )


def main(args):

    # Load pre-trained Acrobot and CartPole models
    acrobot_model = load_acrobot(args)
    cartpole_model = load_cartpole(args)

    save_dir = args.models_dir
    env = gym.make(args.env)

    # Determine input and output dimensions
    input_dim = env.observation_space.shape[0]
    if isinstance(env.action_space, gym.spaces.Discrete):
        output_dim = env.action_space.n
    elif isinstance(env.action_space, gym.spaces.Box):
        output_dim = env.action_space.shape[0]
    else:
        raise ValueError(f"Unsupported action space type: {type(env.action_space)}")

    # Initialize the ProgressiveActorCriticAgent
    model = ProgressiveActorCriticAgent(
        input_dim=input_dim,
        output_dim=output_dim,
        gamma=args.gamma,
        learning_rate_actor=args.learning_rate_actor,
        learning_rate_critic=args.learning_rate_critic,
        agent_1=acrobot_model,
        agent_2=cartpole_model,
        models_dir=args.models_dir,
    )

    # Set up model directory
    model.models_dir = args.models_dir

    # Override original input and output dimensions for masking
    model.orig_input_dim = input_dim
    model.orig_output_dim = output_dim

    # Train the model
    train(args, ac_agent=model, save_models=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_acrobot_model_path",
        type=str,
        default="assets/section_1/actor_critic/acrobot/models/episode_best",
    )
    parser.add_argument(
        "--pretrained_cart_pole_model_path",
        type=str,
        default="assets/section_1/actor_critic/cart_pole/models/episode_best",
    )
    parser.add_argument(
        "--env",
        type=str,
        choices=["CartPole-v1", "Acrobot-v1", "MountainCarContinuous-v0"],
        default="MountainCarContinuous-v0",
    )
    parser.add_argument(
        "--early_exit", type=int, default=None, help="Criteria for early stop"
    )
    parser.add_argument(
        "--models_dir",
        type=str,
        default="assets/section_3/progressive_network_cartpole_acrobot_to_mountaincar_3_1/models",
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
        "--max_steps", type=int, default=MAX_STEPS, help="Maximum steps per episode"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
