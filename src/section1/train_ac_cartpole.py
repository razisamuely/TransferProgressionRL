import gymnasium as gym
import numpy as np
import optuna
import argparse
from src.section1.utils.trainer import train
from src.section1.utils.ac_with_optuna import optimize_hyperparameters
from src.section1.agents.actor_critic_agent import ActorCriticAgent
from src.section1.utils.constants import (
   DEFAULT_ENV,
   DEFAULT_GAMMA,
   DEFAULT_EPISODES,
   DEFAULT_LOG_INTERVAL,
   DEFAULT_ACTOR_LEARNING_RATE,
   DEFAULT_CRITIC_LEARNING_RATE,
   DEFAULT_VISUAL_MODE,
   DEFAULT_OPTIMIZE_HYPER_PARAMETERS_MODE,
   DEFAULT_TASK_DATA_DIR
)
from pathlib import Path
import torch
import loguru
import time
import types

def continue_training(args, model_path):
   saved_model_path = Path(DEFAULT_TASK_DATA_DIR) / "section_1" / "actor_critic" / "models" / f"episode_{model_path}"
   if not saved_model_path.exists():
       raise ValueError(f"Model path not found: {saved_model_path}")
   ac_agent = ActorCriticAgent.load_models(saved_model_path)
   train(args, ac_agent)

def transfer_learning(args, model_path, target_env_name):
   saved_model_path = Path(DEFAULT_TASK_DATA_DIR) / "section_1" / "actor_critic" / "models" / f"episode_{model_path}"
   if not saved_model_path.exists():
       raise ValueError(f"Model path not found: {saved_model_path}")

   source_env = gym.make(args.env)
   target_env = gym.make(target_env_name)

   max_input_dim = max(
       source_env.observation_space.shape[0],
       target_env.observation_space.shape[0]
   )
   
   max_output_dim = max(
       source_env.action_space.n if hasattr(source_env.action_space, 'n') else source_env.action_space.shape[0],
       target_env.action_space.n if hasattr(target_env.action_space, 'n') else target_env.action_space.shape[0]
   )

   target_input_dim = target_env.observation_space.shape[0]
   target_output_dim = target_env.action_space.n if hasattr(target_env.action_space, 'n') else target_env.action_space.shape[0]

   loguru.logger.info(f"Using unified input dim: {max_input_dim}")
   loguru.logger.info(f"Using unified output dim: {max_output_dim}")
   loguru.logger.info(f"Target input dim: {target_input_dim}")
   loguru.logger.info(f"Target output dim: {target_output_dim}")
   
   ac_agent = ActorCriticAgent(
       input_dim=max_input_dim,
       output_dim=max_output_dim,
       gamma=args.gamma,
       learning_rate_actor=args.learning_rate_actor,
       learning_rate_critic=args.learning_rate_critic
   )

   pretrained = ActorCriticAgent.load_models(saved_model_path)

   with torch.no_grad():
       for i in range(len(ac_agent.actor_model)):
           if isinstance(ac_agent.actor_model[i], torch.nn.Linear):
               layer = ac_agent.actor_model[i]
               prev_layer = pretrained.actor_model[i]
               layer.weight[:prev_layer.weight.shape[0], :prev_layer.weight.shape[1]].copy_(prev_layer.weight)
               layer.bias[:prev_layer.bias.shape[0]].copy_(prev_layer.bias)

       for i in range(len(ac_agent.critic_model)):
           if isinstance(ac_agent.critic_model[i], torch.nn.Linear):
               layer = ac_agent.critic_model[i]
               prev_layer = pretrained.critic_model[i]
               layer.weight[:prev_layer.weight.shape[0], :prev_layer.weight.shape[1]].copy_(prev_layer.weight)
               layer.bias[:prev_layer.bias.shape[0]].copy_(prev_layer.bias)

   def get_action_with_padding(self, state, max_input_dim=max_input_dim, target_output_dim=target_output_dim):
       if len(state) < max_input_dim:
           padded_state = np.pad(state, (0, max_input_dim - len(state)), 'constant')
       else:
           padded_state = state

       state_tensor = torch.tensor(padded_state, dtype=torch.float32).to(self.device)
       policy_probs = self.actor_model(state_tensor)
       
       valid_probs = policy_probs[:target_output_dim]
       valid_probs = valid_probs / valid_probs.sum()
       
       action = np.random.choice(target_output_dim, p=valid_probs.detach().cpu().numpy())
       return action

   ac_agent.get_action = types.MethodType(get_action_with_padding, ac_agent)
   args.env = target_env_name
   
   start_time = time.time()
   train(args, ac_agent)
   training_time = time.time() - start_time
   
   loguru.logger.info(f"Transfer Learning Statistics:")
   loguru.logger.info(f"Training time: {training_time:.2f} seconds")
   loguru.logger.info(f"Source env: {args.env}")
   loguru.logger.info(f"Target env: {target_env_name}")

def parse_args():
   parser = argparse.ArgumentParser()
   parser.add_argument("--optimize_hyper_parameters", type=bool, default=DEFAULT_OPTIMIZE_HYPER_PARAMETERS_MODE)
   parser.add_argument("--continue_training", type=str, choices=["best", "final"])
   parser.add_argument("--env", type=str, default=DEFAULT_ENV)
   parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
   parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
   parser.add_argument("--visual_mode", type=str, choices=["pyplot", "tensorboard"], default=DEFAULT_VISUAL_MODE)
   parser.add_argument("--log_interval", type=int, default=DEFAULT_LOG_INTERVAL)
   parser.add_argument("--learning_rate_actor", type=float, default=DEFAULT_ACTOR_LEARNING_RATE)
   parser.add_argument("--learning_rate_critic", type=float, default=DEFAULT_CRITIC_LEARNING_RATE)
   parser.add_argument("--transfer_learning", type=str, choices=["Acrobot-v1", "MountainCarContinuous-v0"])
   return parser.parse_args()

if __name__ == "__main__":
   args = parse_args()
   if args.optimize_hyper_parameters:
       optimize_hyperparameters(args)
   elif args.continue_training:
       continue_training(args, args.continue_training)
   elif args.transfer_learning:
       transfer_learning(args, "best", args.transfer_learning)
   else:
       train(args)