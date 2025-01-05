import torch
import gymnasium as gym
import numpy as np
import loguru
import argparse
from torch.utils.tensorboard import SummaryWriter
import os
import tqdm
from src.section1.utils import common
from src.section1.agents.actor_critic_agent import ActorCriticAgent
from src.section1.utils.constants import DEFAULT_EARLY_EXIT_CRITERIA


def episode_step(env, ac_agent, max_steps: int = 1000):

    state = env.reset()[0]
    episode_reward = []
    policy_losses = []
    done = False
    step_count = 0  # Track the number of steps in the episode

    while not done and step_count < max_steps:
        action = ac_agent.get_action(state)

        next_state, reward, done, _, _ = env.step(action)
        policy_loss, _ = ac_agent.train_step(state, action, reward, next_state, done)

        episode_reward.append(reward)
        policy_losses.append(policy_loss)
        state = next_state

        step_count += 1  # Increment the step count

    if step_count >= max_steps:
        loguru.logger.warning(
            f"Episode terminated due to reaching the maximum step limit ({max_steps})."
        )

    return episode_reward, policy_losses


def train(args, ac_agent=None, save_models=True):
    save_dir = args.models_dir
    env = gym.make(args.env)
    input_dim = env.observation_space.shape[0]
    if isinstance(env.action_space, gym.spaces.Discrete):
        output_dim = env.action_space.n
    elif isinstance(env.action_space, gym.spaces.Box):
        output_dim = env.action_space.shape[0]
    else:
        raise ValueError(f"Unsupported action space type: {type(env.action_space)}")

    if ac_agent is None:
        ac_agent = ActorCriticAgent(
            input_dim=input_dim,
            output_dim=output_dim,
            gamma=args.gamma,
            learning_rate_actor=args.learning_rate_actor,
            learning_rate_critic=args.learning_rate_critic,
            models_dir=args.models_dir,
        )

    env_name = args.env.split("-")[0].lower()
    os.makedirs(save_dir, exist_ok=True)
    paths = [f"{save_dir}/{env_name}_rewards.png", f"{save_dir}/{env_name}_losses.png"]

    tensorboard_logs_dir = f"{save_dir}/tensorboard_{env_name}"
    view_tensorboard = args.visual_mode == "tensorboard"
    os.makedirs(tensorboard_logs_dir, exist_ok=True)
    writer = SummaryWriter(tensorboard_logs_dir) if view_tensorboard else None

    episode_rewards = []
    episode_policy_losses = []
    best_mean_reward = float("-inf")

    for curr_episode in tqdm.tqdm(range(args.episodes)):
        curr_episode_rewards, curr_policy_losses = episode_step(
            env, ac_agent, max_steps=args.max_steps
        )
        total_episode_reward = sum(curr_episode_rewards)
        total_episode_loss = sum(curr_policy_losses)

        episode_rewards.append(total_episode_reward)
        episode_policy_losses.append(total_episode_loss)
        mean_reward_last_100_episodes = np.mean(episode_rewards[-100:])

        if mean_reward_last_100_episodes > best_mean_reward:
            best_mean_reward = mean_reward_last_100_episodes
            if save_models:
                ac_agent.save_models(episode=f"best")

        if view_tensorboard:
            writer.add_scalar(
                f"{env_name}/Reward/Episode", total_episode_reward, curr_episode
            )
            writer.add_scalar(
                f"{env_name}/Reward/Mean_Last_100",
                mean_reward_last_100_episodes,
                curr_episode,
            )
            writer.add_scalar(
                f"{env_name}/Loss/Policy_Loss",
                np.mean(episode_policy_losses),
                curr_episode,
            )

        if curr_episode % args.log_interval == 0:
            loguru.logger.info(
                f"Episode {curr_episode} reward for {args.env}: {mean_reward_last_100_episodes}"
            )
            if not view_tensorboard:
                common.plot_metrics(episode_rewards, episode_policy_losses, paths)


    if writer:
        writer.flush()
        writer.close()

    env.close()
    loguru.logger.info(f"Training actor critic done for {args.env}")

    training_stats = {
        "episode_rewards": episode_rewards,
        "episode_policy_losses": episode_policy_losses,
        "best_mean_reward": best_mean_reward,
        "episodes_trained": curr_episode + 1,
        "env_name": args.env,
    }

    return training_stats, ac_agent
