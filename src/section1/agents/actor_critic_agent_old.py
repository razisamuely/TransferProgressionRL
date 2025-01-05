import torch
import gymnasium as gym
import numpy as np
import os
import json
from pathlib import Path


class ActorCriticAgent:
    MAX_INPUT_DIM = 6
    MAX_OUTPUT_DIM = 3

    def __init__(
        self,
        input_dim,
        output_dim,
        gamma,
        learning_rate_actor,
        learning_rate_critic,
        models_dir,
        entropy_weight=0.1,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.orig_input_dim = input_dim
        self.orig_output_dim = output_dim
        self.input_dim = self.MAX_INPUT_DIM
        self.output_dim = self.MAX_OUTPUT_DIM
        self.gamma = gamma
        self.entropy_weight = entropy_weight
        self.learning_rate_actor = learning_rate_actor
        self.learning_rate_critic = learning_rate_critic
        self.actor_model = self.create_policy_network(
            self.MAX_INPUT_DIM, self.MAX_OUTPUT_DIM
        ).to(self.device)
        self.critic_model = self.create_critic_network(self.MAX_INPUT_DIM, 1).to(
            self.device
        )
        self.actor_optimizer = torch.optim.Adam(
            self.actor_model.parameters(), lr=self.learning_rate_actor
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic_model.parameters(), lr=self.learning_rate_critic
        )
        self.models_dir = models_dir
        path = Path(self.models_dir)
        path.mkdir(parents=True, exist_ok=True)
        self.standard_normal = torch.distributions.Normal(
            loc=torch.tensor(0.0).to(self.device),
            scale=torch.tensor(1.0).to(self.device)
        )

    def save_models(self, episode=None):
        save_path = self.models_dir
        if episode is not None:
            save_path = os.path.join(save_path, f"episode_{episode}")
            os.makedirs(save_path, exist_ok=True)

        actor_path = os.path.join(save_path, "actor.pth")
        torch.save(
            {
                "model_state_dict": self.actor_model.state_dict(),
                "optimizer_state_dict": self.actor_optimizer.state_dict(),
                "input_dim": int(self.input_dim),
                "output_dim": int(self.output_dim),
                "learning_rate": float(self.learning_rate_actor),
            },
            actor_path,
        )

        critic_path = os.path.join(save_path, "critic.pth")
        torch.save(
            {
                "model_state_dict": self.critic_model.state_dict(),
                "optimizer_state_dict": self.critic_optimizer.state_dict(),
                "input_dim": int(self.input_dim),
                "output_dim": 1,
                "learning_rate": float(self.learning_rate_critic),
            },
            critic_path,
        )

        hyper_params = {
            "gamma": float(self.gamma),
            "learning_rate_actor": float(self.learning_rate_actor),
            "learning_rate_critic": float(self.learning_rate_critic),
            "input_dim": int(self.input_dim),
            "output_dim": int(self.output_dim),
            "orig_input_dim": int(self.orig_input_dim),
            "orig_output_dim": int(self.orig_output_dim),
            "models_dir": str(self.models_dir),
        }
        with open(os.path.join(save_path, "hyperparameters.json"), "w") as f:
            json.dump(hyper_params, f)

    @classmethod
    def load_models(cls, load_path):
        with open(os.path.join(load_path, "hyperparameters.json"), "r") as f:
            hyper_params = json.load(f)

        instance = cls(
            input_dim=hyper_params["orig_input_dim"],
            output_dim=hyper_params["orig_output_dim"],
            gamma=hyper_params["gamma"],
            learning_rate_actor=hyper_params["learning_rate_actor"],
            learning_rate_critic=hyper_params["learning_rate_critic"],
            models_dir=hyper_params["models_dir"],
        )

        actor_checkpoint = torch.load(os.path.join(load_path, "actor.pth"))
        instance.actor_model.load_state_dict(actor_checkpoint["model_state_dict"])
        instance.actor_optimizer.load_state_dict(
            actor_checkpoint["optimizer_state_dict"]
        )

        critic_checkpoint = torch.load(os.path.join(load_path, "critic.pth"))
        instance.critic_model.load_state_dict(critic_checkpoint["model_state_dict"])
        instance.critic_optimizer.load_state_dict(
            critic_checkpoint["optimizer_state_dict"]
        )

        return instance

    def create_policy_network(self, input_dim, output_dim):
        return torch.nn.Sequential(
            *[
                torch.nn.Linear(input_dim, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, output_dim),
                torch.nn.Softmax(dim=-1),
            ]
        )

    def create_critic_network(self, input_dim, output_dim):
        return torch.nn.Sequential(
            *[
                torch.nn.Linear(input_dim, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, output_dim),
            ]
        )

    def get_action(self, state):
        if len(state) < self.input_dim:
            state = np.pad(state, (0, self.input_dim - len(state)), "constant")

        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        action_params = self.actor_model(state)  # Will always output 3 values

        if self.orig_output_dim == 1:  # Continuous action space (MountainCarContinuous)
            # mean = action_params[0]
            # log_std = action_params[1]
            # log_std = torch.clamp(log_std, -20, 2)
            # std = torch.exp(log_std)
            
            # # Use standard normal and reparameterization trick
            # epsilon = self.standard_normal.sample()
            # action = mean + std * epsilon
            
            # # Clip the action to valid range
            # action = torch.clamp(action, -1.0, 1.0)
            
            # action = action.detach().cpu().numpy()
            valid_probs = action_params[: 2] # mean and log_std
            valid_probs = valid_probs / valid_probs.sum()

            mean = valid_probs[0]
            log_std = valid_probs[1]

            log_std = torch.clamp(log_std, -20, 2)
            std = torch.exp(log_std) * 0.1

            epsilon = self.standard_normal.sample()
            action = mean + std * epsilon
            action = torch.clamp(action, -1.0, 1.0)
            action = action.detach().cpu().numpy()

            
            return np.array([action])
        else:  # Discrete action space (CartPole and Acrobot)
            valid_probs = action_params[: self.orig_output_dim]
            valid_probs = valid_probs / valid_probs.sum()

            action = np.random.choice(
                self.orig_output_dim, p=valid_probs.detach().cpu().numpy()
            )
            return action

    def train_step(self, state, action, reward, next_state, done):
        if len(state) < self.input_dim:
            state = np.pad(state, (0, self.input_dim - len(state)), "constant")
        if len(next_state) < self.input_dim:
            next_state = np.pad(next_state, (0, self.input_dim - len(next_state)), "constant")

        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        next_state = torch.tensor(next_state, dtype=torch.float32).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float32).to(self.device)

        value = self.critic_model(state)
        next_value = self.critic_model(next_state) if not done else torch.tensor(0.0).to(self.device)
        td_err = reward + self.gamma * next_value - value

        # Get the action parameters and calculate log probability
        action_params = self.actor_model(state)
        
        if self.orig_output_dim == 1:  # Continuous action space
            mean = action_params[0]
            log_std = action_params[1]
            
            # Clamp log_std for numerical stability
            log_std = torch.clamp(log_std, -20, 2)
            std = torch.exp(log_std)
            
            action = torch.tensor(action, dtype=torch.float32).to(self.device)
            distribution = torch.distributions.Normal(mean, std)
            log_prob = distribution.log_prob(action)
            
        else:  # Discrete action space
            valid_probs = action_params[: self.orig_output_dim]
            valid_probs = valid_probs / valid_probs.sum()
            
            action = torch.tensor(action, dtype=torch.int64).to(self.device)
            distribution = torch.distributions.Categorical(valid_probs)
            log_prob = distribution.log_prob(action)

        # Policy loss
        policy_loss = -log_prob * td_err.detach()
        
        # Entropy loss to encourage exploration
        entropy = distribution.entropy().mean()
        entropy_loss = -self.entropy_weight * entropy

        # Total policy loss
        total_policy_loss = policy_loss + entropy_loss

        # Value loss
        value_loss = td_err.pow(2)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()

        # Optimize the actor (policy)
        self.actor_optimizer.zero_grad()
        total_policy_loss.backward()
        self.actor_optimizer.step()

        return total_policy_loss.item(), value_loss.item()

