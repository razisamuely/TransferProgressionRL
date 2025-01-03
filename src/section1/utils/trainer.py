import torch
import gymnasium as gym
import numpy as np
import loguru
import argparse
from torch.utils.tensorboard import SummaryWriter
import os 
import optuna
from src.section1.utils import common
from src.section1.agents.actor_critic_agent import ActorCriticAgent
from src.section1.utils.constants import (
    DEFAULT_EARLY_EXIT_CRITERIA,
    DEFAULT_TASK_DATA_DIR
)

def episode_step(env, ac_agent):
    state = env.reset()[0]
    episode_reward = []
    done = False
    policy_losses = []
    
    while not done:
        action = ac_agent.get_action(state)
        next_state, reward, done, _, _ = env.step(action)
        policy_loss, _ = ac_agent.train_step(state, action, reward, next_state, done)

        episode_reward.append(reward)
        policy_losses.append(policy_loss)
        state = next_state
        
    return episode_reward, policy_losses
       
def train(args,ac_agent = None):
    common.print_parameters("Question 2", args)
    env = gym.make(args.env)
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    
    if ac_agent is None:
        ac_agent = ActorCriticAgent(
            input_dim=input_dim, 
            output_dim=output_dim,
            gamma=args.gamma,
            learning_rate_actor=args.learning_rate_actor,
            learning_rate_critic=args.learning_rate_critic
        )
    
    os.makedirs(DEFAULT_TASK_DATA_DIR, exist_ok=True)
    paths = [
        f"{DEFAULT_TASK_DATA_DIR}/task_2_value_estimation_rewards.png", 
        f"{DEFAULT_TASK_DATA_DIR}/task_2_value_estimation_losses.png"
    ]
        
    tensorboard_logs_dir = f"{DEFAULT_TASK_DATA_DIR}/task_2"
    view_tensorboard = args.visual_mode == 'tensorboard'
    os.makedirs(tensorboard_logs_dir, exist_ok=True)
    writer = SummaryWriter(tensorboard_logs_dir) if view_tensorboard else None

    episode_rewards = []
    episode_policy_losses = []
    best_mean_reward = float('-inf')
    
    loguru.logger.info("Training actor critic started")
    
    for curr_episode in range(args.episodes):
        curr_episode_rewards, curr_policy_losses = episode_step(env, ac_agent)
        total_episode_reward = sum(curr_episode_rewards)
        total_episode_loss = sum(curr_policy_losses)
        
        episode_rewards.append(total_episode_reward)
        episode_policy_losses.append(total_episode_loss)
        mean_reward_last_100_episodes = np.mean(episode_rewards[-100:])

        if mean_reward_last_100_episodes > best_mean_reward:
            best_mean_reward = mean_reward_last_100_episodes
            ac_agent.save_models(episode="best")

        if view_tensorboard:
            writer.add_scalar("Reward/Episode", total_episode_reward, curr_episode)
            writer.add_scalar("Reward/Mean_Last_100", mean_reward_last_100_episodes, curr_episode)
            writer.add_scalar("Loss/Policy_Loss", np.mean(episode_policy_losses), curr_episode)

        if curr_episode % args.log_interval == 0:
            loguru.logger.info(f"Episode {curr_episode} reward: {mean_reward_last_100_episodes}")
            if not view_tensorboard:
                common.plot_metrics(episode_rewards, episode_policy_losses, paths)
        
        if mean_reward_last_100_episodes >= DEFAULT_EARLY_EXIT_CRITERIA:
            if not view_tensorboard:
                common.plot_metrics(episode_rewards, episode_policy_losses, paths)
            break
    
    if writer:
        writer.flush()
        writer.close()
    
    env.close()
    loguru.logger.info("Training actor critic done")