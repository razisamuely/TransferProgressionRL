import argparse
import os
import gymnasium as gym
from src.section1.agents.actor_critic_agent import ActorCriticAgent
from src.section1.utils.trainer import train
from src.section1.utils.constants import (
    MAX_INPUT_DIM,
    MAX_OUTPUT_DIM,
    DEFAULT_GAMMA,
    DEFAULT_EPISODES,
    DEFAULT_LOG_INTERVAL,
    DEFAULT_ACTOR_LEARNING_RATE,
    DEFAULT_CRITIC_LEARNING_RATE,
    DEFAULT_VISUAL_MODE,
    MAX_STEPS,
    BATCH_SIZE,
    BUFFER_SIZE
)

def load_source_model(agent,model_path, target_env_name, target_models_dir):
    agent.load_model(model_path)
    agent.models_dir = target_models_dir    
    return agent

def adapt_agent_to_env(agent, env):
    input_dim = env.observation_space.shape[0]
    if isinstance(env.action_space, gym.spaces.Box):
        output_dim = env.action_space.shape[0]
    else:
        output_dim = env.action_space.n

    agent.n_state = MAX_INPUT_DIM
    agent.n_action = MAX_OUTPUT_DIM
    agent.orig_input_dim = input_dim
    agent.orig_output_dim = output_dim

def main():
    args = parse_arguments()
    env = gym.make(args.target_env)
    agent = ActorCriticAgent(env, hidden1=args.hidden1, hidden2=args.hidden2, gamma=args.gamma, 
                             lr_actor=args.learning_rate_actor, lr_critic=args.learning_rate_critic, buf_size=args.buf_size, 
                             batch_size=args.batch_size, exp_name=args.exp_name, 
                             models_dir=args.target_models_dir)

    # Loading weights only
    agent = load_source_model(agent, args.source_model_path, args.target_env, args.target_models_dir)
    state_padding_size = MAX_INPUT_DIM - env.observation_space.shape[0]
    
    train(
        env=env,
        agent=agent,
        episodes=args.num_episodes,
        state_padding_size=state_padding_size,
        max_steps_per_episode=args.max_steps
    )

def parse_arguments():
    parser = argparse.ArgumentParser(description='Transfer learning from Acrobot to CartPole')
    
    parser.add_argument(
        '--target_env',
        type=str,
        choices=['Acrobot-v1', 'MountainCarContinuous-v0', 'CartPole-v1'],
        default='CartPole-v1'
    )
    
    parser.add_argument('--gamma', type=float, default=DEFAULT_GAMMA)
    parser.add_argument('--num_episodes', type=int, default=DEFAULT_EPISODES)
    parser.add_argument(
        '--visual_mode',
        type=str,
        choices=['pyplot', 'tensorboard'],
        default=DEFAULT_VISUAL_MODE
    )
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--buf_size', type=int, default=BUFFER_SIZE)
    parser.add_argument('--log_interval', type=int, default=DEFAULT_LOG_INTERVAL)
    parser.add_argument('--learning_rate_actor', type=float, default=DEFAULT_ACTOR_LEARNING_RATE)
    parser.add_argument('--learning_rate_critic', type=float, default=DEFAULT_CRITIC_LEARNING_RATE)
    parser.add_argument('--exp_name', type=str, default='exp1')
    parser.add_argument('--hidden1', type=int, default=256)
    parser.add_argument('--hidden2', type=int, default=256)
    
    parser.add_argument(
        '--source_model_path',
        type=str,
        default='assets/section_1/actor_critic/acrobot/models/episode_best'
    )
    
    parser.add_argument(
        '--target_models_dir',
        type=str,
        default='assets/section_2/transfer_acrobot_to_cartpole/models'
    )
    
    parser.add_argument(
        '--max_steps',
        type=int,
        default=MAX_STEPS,
        help='Maximum steps per episode'
    )
    
    args = parser.parse_args()
    os.makedirs(args.target_models_dir, exist_ok=True)
    return args

if __name__ == '__main__':
    main()


# python src/section2/transfer_acrobot_to_cartpole_2_1.py \
# --source_model_path assets/section_1/actor_critic/acrobot/models/acrobot_exp/episode_best \
# --target_env CartPole-v1 \
# --target_models_dir assets/section_2/transfer_acrobot_to_cartpole/models \
# --num_episodes 1000
# --models_dir "assets/section_2/transfer_acrobot_to_cartpole/models"