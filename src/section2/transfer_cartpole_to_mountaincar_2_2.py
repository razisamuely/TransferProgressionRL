
from src.section1.agents.actor_critic_agent import ActorCriticAgent
import argparse
from src.section1.utils.script.trainer import train
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


def main(args):
    pretrained_acrobot_model_path = args.pretrained_acrobot_model_path
    pretrained_acrobot_model = ActorCriticAgent.load_models(
        pretrained_acrobot_model_path
    )

    # Override model dir path
    models_dir = args.models_dir
    pretrained_acrobot_model.models_dir = models_dir

    # Override original input and output dimensions - Note: That is not effect the Nets input and output dimensions,
    # Its serve only for masking the input and output dimensions in the agent

    env = gym.make(args.env)
    input_dim = env.observation_space.shape[0]
    if isinstance(env.action_space, gym.spaces.Discrete):
        output_dim = env.action_space.n
    elif isinstance(env.action_space, gym.spaces.Box):
        output_dim = env.action_space.shape[0]
    else:
        raise ValueError(f"Unsupported action space type: {type(env.action_space)}")

    pretrained_acrobot_model.orig_input_dim = input_dim
    pretrained_acrobot_model.orig_output_dim = output_dim

    pretrained_acrobot_model.reinit_final_layers()
    train(args, ac_agent=pretrained_acrobot_model, save_models=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_acrobot_model_path",
        type=str,
        default="assets/section_1/actor_critic/cart_pole/models/episode_best",
    )
    parser.add_argument(
        "--models_dir",
        type=str,
        default="assets/section_2/transfer_cartpole_to_mountaincar_2_2/models",
    )

    parser.add_argument(
        "--env",
        type=str,
        choices=["CartPole-v1", "Acrobot-v1", "MountainCarContinuous-v0"],
        default="MountainCarContinuous-v0",
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
