import argparse
from src.utils.constants import (
    DEFAULT_GAMMA,
    DEFAULT_EPISODES,
    DEFAULT_LOG_INTERVAL,
    DEFAULT_ACTOR_LEARNING_RATE,
    DEFAULT_CRITIC_LEARNING_RATE,
    DEFAULT_VISUAL_MODE,
    DEFAULT_OPTIMIZE_HYPER_PARAMETERS_MODE,
    MAX_STEPS,
)

def default_parse_args(env_env=None,early_exit=None,models_dir=None):
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--models_dir",default=f"assets/section_1/actor_critic/{env_env}/models", type=str)
    parser.add_argument(
        "--max_steps", type=int, default=MAX_STEPS, help="Maximum steps per episode"
    )

    if models_dir is None:
        parser.add_argument("--models_dir", type=str)

    if env_env is None:
        parser.add_argument(
            "--env",
            type=str,
            choices=["CartPole-v1", "Acrobot-v1", "MountainCarContinuous-v0"],
            default="MountainCarContinuous-v0",
        )
        
    if early_exit is None:
        parser.add_argument(
            "--early_exit", type=int, default=None, help="Criteria for early stop"
        )



    parsed_args= parser.parse_args()
    
    #
    if early_exit is not  None:
        parsed_args.early_exit=early_exit
    if env_env is not None:
        parsed_args.env=env_env
    if models_dir is not None:
        parsed_args.models_dir=models_dir
        
    return parsed_args
