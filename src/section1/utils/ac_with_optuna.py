import json
import loguru
from pathlib import Path
from src.section1.utils.constants import (
    GAMMA_SEARCH_CONFIG,
    ACTOR_LR_SEARCH_CONFIG,
    CRITIC_LR_SEARCH_CONFIG,
    OPTUNA_NUM_TRIALS
)
from src.section1.agents.actor_critic_agent import ActorCriticAgent
from src.section1.utils.trainer import episode_step
import gymnasium as gym
import optuna
import numpy as np

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
        
        # Save model if this trial has the best performance so far
        if mean_reward > trial.study.best_value:
            ac_agent.save_models(episode="optuna_best")
            
        env.close()
        return mean_reward

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=OPTUNA_NUM_TRIALS)
    
    # Save best hyperparameters
    env_name = args.env.split('-')[0].lower()
    save_dir = Path("assets/task_2/section_1/actor_critic/optuna")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    best_params = study.best_params
    best_params['best_value'] = study.best_value
    
    with open(save_dir / f"best_params_{env_name}.json", 'w') as f:
        json.dump(best_params, f, indent=4)
        
    loguru.logger.info(f"Best hyperparameters for {args.env}:")
    loguru.logger.info(f"Best value: {study.best_value}")
    loguru.logger.info(f"Best parameters: {study.best_params}")
    loguru.logger.info(f"Saved to: {save_dir}/best_params_{env_name}.json")
    
    return study.best_params