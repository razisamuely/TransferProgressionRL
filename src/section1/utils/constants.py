# src/section1/constants.py

DEFAULT_ENV = "CartPole-v1"
DEFAULT_GAMMA = 0.99
DEFAULT_EPISODES = 3000
DEFAULT_LOG_INTERVAL = 100
DEFAULT_ACTOR_LEARNING_RATE = 0.0004
DEFAULT_CRITIC_LEARNING_RATE = 0.001
DEFAULT_EARLY_EXIT_CRITERIA = 475
DEFAULT_TASK_DATA_DIR = "assets/task_2"
DEFAULT_VISUAL_MODE = "tensorboard"
DEFAULT_OPTIMIZE_HYPER_PARAMETERS_MODE = False
GAMMA_SEARCH_CONFIG = {
    "min": 0.8,
    "max": 0.999,
    "step": 0.01,
    "name": "gamma"
}

ACTOR_LR_SEARCH_CONFIG = {
    "min": 1e-5,
    "max": 1e-1,
    "name": "learning_rate_actor",
    "distribution": "log-uniform"
}

CRITIC_LR_SEARCH_CONFIG = {
    "min": 1e-5,
    "max": 1e-1,
    "name": "learning_rate_critic",
    "distribution": "log-uniform" 
}

OPTUNA_NUM_TRIALS = 50