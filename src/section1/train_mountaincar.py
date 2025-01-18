from src.section1.section1_utils import default_parse_args
from src.utils.trainer import train
from src.utils.ac_with_optuna import optimize_hyperparameters
if __name__ == "__main__":
    args = default_parse_args("MountainCarContinuous-v0",
                              90,
                              f"assets/section_1/actor_critic/mountain_car/models")
    if args.optimize_hyper_parameters:
        optimize_hyperparameters(args)
    else:
        train(args)