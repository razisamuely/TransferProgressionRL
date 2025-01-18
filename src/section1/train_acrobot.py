from src.section1.section1_utils import default_parse_args
from src.utils.trainer import train
from src.utils.ac_with_optuna import optimize_hyperparameters
if __name__ == "__main__":

    args = default_parse_args("Acrobot-v1",
                              -90,
                              f"assets/section_1/actor_critic/acrobot/models")
    if args.optimize_hyper_parameters:
        optimize_hyperparameters(args)
    else:
        train(args)

# Train without hyperparameter optimization CartPole
# python src/section1/train_actor_critic_agent.py \
# --env CartPole-v1 \
# --models_dir assets/section_1/actor_critic/cart_pole/models

# Train without hyperparameter optimization MountainCarContinuous
# python src/section1/train_actor_critic_agent.py \
# --env MountainCarContinuous-v0 \
# --models_dir assets/section_1/actor_critic/mountain_car/models

# Train without hyperparameter optimization Acrobot
# python src/section1/train_actor_critic_agent.py \
# --env Acrobot-v1 \
# --models_dir assets/section_1/actor_critic/acrobot/models

# optimize_hyperparameters
# python src/section1/train_actor_critic_agent.py \
# --optimize_hyper_parameters True \
# --env CartPole-v1 \
# --models_dir assets/section_1/actor_critic/cart_pole/models

# python src/section1/train_actor_critic_agent.py \
# --optimize_hyper_parameters True \
# --env MountainCarContinuous-v0 \
# --models_dir assets/section_1/actor_critic/mountain_car/models
