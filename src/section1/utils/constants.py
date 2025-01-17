# src/section1/constants.py

DEFAULT_ENV = "CartPole-v1"
DEFAULT_GAMMA = 0.99
DEFAULT_EPISODES = 2000
DEFAULT_LOG_INTERVAL = 5
DEFAULT_ACTOR_LEARNING_RATE = 0.00001
DEFAULT_CRITIC_LEARNING_RATE = 0.0001
DEFAULT_TASK_DATA_DIR = "assets/task_2"
DEFAULT_VISUAL_MODE = "tensorboard"
DEFAULT_OPTIMIZE_HYPER_PARAMETERS_MODE = False
MAX_STEPS = 3000
DEFAULT_MODELS_DIR = "assets/section_1/actor_critic/cart_pole/models"
GAMMA_SEARCH_CONFIG = {"min": 0.8, "max": 0.999, "step": 0.01, "name": "gamma"}
MAX_INPUT_DIM = 6
MAX_OUTPUT_DIM = 3
BATCH_SIZE = 64
BUFFER_SIZE = 100000
ACTOR_LR_SEARCH_CONFIG = {
    "min": 1e-5,
    "max": 1e-2,
    "name": "learning_rate_actor",
    "distribution": "log-uniform",
}

CRITIC_LR_SEARCH_CONFIG = {
    "min": 1e-5,
    "max": 1e-2,
    "name": "learning_rate_critic",
    "distribution": "log-uniform",
}

OPTUNA_NUM_TRIALS = 50
