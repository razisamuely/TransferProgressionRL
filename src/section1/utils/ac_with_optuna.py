import json
import loguru
from pathlib import Path
import optuna
import argparse
from src.section1.utils.constants import (
    GAMMA_SEARCH_CONFIG,
    ACTOR_LR_SEARCH_CONFIG,
    CRITIC_LR_SEARCH_CONFIG,
    OPTUNA_NUM_TRIALS,
)
from src.section1.utils.trainer import train


def optimize_hyperparameters(args):
    def objective(trial):
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

        trial_args = argparse.Namespace(**vars(args))
        trial_args.gamma = gamma
        trial_args.learning_rate_actor = learning_rate_actor
        trial_args.learning_rate_critic = learning_rate_critic
        trial_args.models_dir = (
            Path(args.models_dir) / "optuna" / f"trial_{trial.number}"
        )

        training_stats, ac_agent = train(trial_args, save_models=False)
        mean_reward = training_stats["best_mean_reward"]

        if trial.number == 0 or mean_reward > trial.study.best_value:
            best_model_dir = Path(args.models_dir) / "optuna" / "best_model"
            best_model_dir.mkdir(parents=True, exist_ok=True)

            ac_agent.save_models(episode="best")

            best_params = {
                "parameters": {
                    "gamma": gamma,
                    "learning_rate_actor": learning_rate_actor,
                    "learning_rate_critic": learning_rate_critic,
                },
                "best_value": mean_reward,
                "env": args.env,
                "training_stats": training_stats,
                "trial_number": trial.number,
            }

            with open(best_model_dir / "optimization_results.json", "w") as f:
                json.dump(best_params, f, indent=4, default=str)

            loguru.logger.info(f"New best model found in trial {trial.number}")
            loguru.logger.info(f"Parameters: {best_params['parameters']}")
            loguru.logger.info(f"Value: {mean_reward}")

        return mean_reward

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=OPTUNA_NUM_TRIALS)

    # Delete all the trials tensorboard directories except the best trial
    from shutil import rmtree

    best_trial_number = study.best_trial.number
    for trial in study.trials:
        if trial.number != best_trial_number:
            trial_dir = Path(args.models_dir) / "optuna" / f"trial_{trial.number}"
            if trial_dir.exists():
                rmtree(trial_dir)
    return study.best_params
