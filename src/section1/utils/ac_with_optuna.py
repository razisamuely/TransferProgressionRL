import gymnasium as gym
import numpy as np
import optuna
import argparse
from src.section1.agents.actor_critic_agent import ActorCriticAgent
from src.section1.utils.trainer import train, episode_step
from src.section1.utils.constants import (
    GAMMA_SEARCH_CONFIG,
    ACTOR_LR_SEARCH_CONFIG,
    CRITIC_LR_SEARCH_CONFIG,
    OPTUNA_NUM_TRIALS
)

def optimize_hyperparameters(args):
    def objective(trial):
        gamma = trial.suggest_float(
            GAMMA_SEARCH_CONFIG["name"],
            GAMMA_SEARCH_CONFIG["min"],
            GAMMA_SEARCH_CONFIG["max"],
            step=GAMMA_SEARCH_CONFIG["step"]
        )
        
        learning_rate_actor = trial.suggest_loguniform(
            ACTOR_LR_SEARCH_CONFIG["name"],
            ACTOR_LR_SEARCH_CONFIG["min"],
            ACTOR_LR_SEARCH_CONFIG["max"]
        )
        
        learning_rate_critic = trial.suggest_loguniform(
            CRITIC_LR_SEARCH_CONFIG["name"],
            CRITIC_LR_SEARCH_CONFIG["min"],
            CRITIC_LR_SEARCH_CONFIG["max"]
        )
        
        env = gym.make(args.env)
        input_dim = env.observation_space.shape[0]
        output_dim = env.action_space.n

        ac_agent = ActorCriticAgent(
            input_dim=input_dim,
            output_dim=output_dim,
            gamma=gamma,
            learning_rate_actor=learning_rate_actor,
            learning_rate_critic=learning_rate_critic
        )
        
        episode_rewards = []
        for _ in range(args.episodes): 
            curr_episode_rewards, _ = episode_step(env, ac_agent)
            episode_rewards.append(sum(curr_episode_rewards))

        mean_reward = np.mean(episode_rewards[-100:])
        env.close()
        return mean_reward

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=OPTUNA_NUM_TRIALS)
    return study.best_params
