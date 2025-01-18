import json
import gymnasium as gym
from pathlib import Path
import optuna
import shutil
from src.section1.utils.constants import (
    GAMMA_SEARCH_CONFIG,
    ACTOR_LR_SEARCH_CONFIG,
    CRITIC_LR_SEARCH_CONFIG,
    OPTUNA_NUM_TRIALS,
    MAX_INPUT_DIM
)
from src.section1.utils.trainer import train  # Added the missing import
from src.section1.agents.actor_critic_agent import ActorCriticAgent

def optimize_hyperparameters(args):
    def objective(trial):
        # Suggest hyperparameters
        gamma = trial.suggest_float(
            GAMMA_SEARCH_CONFIG["name"],
            GAMMA_SEARCH_CONFIG["min"],
            GAMMA_SEARCH_CONFIG["max"],
            step=GAMMA_SEARCH_CONFIG["step"],
        )

        learning_rate_actor = trial.suggest_float(
            ACTOR_LR_SEARCH_CONFIG["name"],
            ACTOR_LR_SEARCH_CONFIG["min"],
            ACTOR_LR_SEARCH_CONFIG["max"],
            log=True,
        )

        learning_rate_critic = trial.suggest_float(
            CRITIC_LR_SEARCH_CONFIG["name"],
            CRITIC_LR_SEARCH_CONFIG["min"],
            CRITIC_LR_SEARCH_CONFIG["max"],
            log=True,
        )

        # Create environment
        env = gym.make(args.env)
        env = env.unwrapped

        # Create trial directory
        trial_dir = Path(args.models_dir) / "optuna" / f"trial_{trial.number}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        # Create agent with trial hyperparameters
        agent = ActorCriticAgent(
            env=env,
            batch_size=128,
            hidden1=args.hidden1,
            hidden2=args.hidden2,
            gamma=gamma,
            lr_actor=learning_rate_actor,
            lr_critic=learning_rate_critic,
            exp_name=f"trial_{trial.number}",
            models_dir=str(trial_dir)
        )

        # Train the agent
        state_padding_size = MAX_INPUT_DIM - env.observation_space.shape[0]
        turns, scores, avg_scores, goals = train(
            env=env,
            agent=agent,
            episodes=args.episodes,
            state_padding_size=state_padding_size,
            max_steps_per_episode=args.max_steps
        )

        mean_reward = avg_scores[-1] if avg_scores else 0

        # Save if best model
        if trial.number == 0 or mean_reward > trial.study.best_value:
            best_model_dir = Path(args.models_dir) / "optuna" / "best_model"
            best_model_dir.mkdir(parents=True, exist_ok=True)

            # Save the model
            agent.save_model("best", mean_reward)

            # Save optimization results
            best_params = {
                "parameters": {
                    "gamma": gamma,
                    "learning_rate_actor": learning_rate_actor,
                    "learning_rate_critic": learning_rate_critic,
                },
                "best_value": mean_reward,
                "env": args.env,
                "training_stats": {
                    "turns": turns,
                    "final_score": scores[-1],
                    "best_mean_reward": mean_reward,
                    "goals_achieved": sum(goals)
                },
                "trial_number": trial.number,
            }

            with open(best_model_dir / "optimization_results.json", "w") as f:
                json.dump(best_params, f, indent=4)

            print(f"New best model found in trial {trial.number}")
            print(f"Parameters: {best_params['parameters']}")
            print(f"Value: {mean_reward}")

        return mean_reward

    # Create and run the study
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=OPTUNA_NUM_TRIALS)

    # Clean up trials except the best one
    best_trial_number = study.best_trial.number
    optuna_dir = Path(args.models_dir) / "optuna"
    for trial_dir in optuna_dir.glob("trial_*"):
        if trial_dir.name != f"trial_{best_trial_number}":
            shutil.rmtree(trial_dir)

    return study.best_trial.params