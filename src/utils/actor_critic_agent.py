from src.utils.network_uils import create_critic_network, create_policy_network
import torch
import gymnasium as gym
import numpy as np
import os
import json
from pathlib import Path
from torch.nn.init import xavier_uniform_

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
        models_dir=None,
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
        self.actor_model = create_policy_network(
            self.MAX_INPUT_DIM, 12,self.MAX_OUTPUT_DIM
        ).to(self.device)
        self.critic_model = create_critic_network(self.MAX_INPUT_DIM,12, 1).to(
            self.device
        )

        self.actor_optimizer = torch.optim.Adam(
            self.actor_model.parameters(), lr=self.learning_rate_actor
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic_model.parameters(), lr=self.learning_rate_critic
        )

        self.models_dir = models_dir
        Path(self.models_dir).mkdir(parents=True, exist_ok=True)
        self.standard_normal = torch.distributions.Normal(
            loc=torch.tensor(0.0).to(self.device),
            scale=torch.tensor(1.0).to(self.device),
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

        actor_checkpoint = torch.load(os.path.join(load_path, "actor.pth"),
                                              map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")
                                      )
        instance.actor_model.load_state_dict(actor_checkpoint["model_state_dict"])
        instance.actor_optimizer.load_state_dict(
            actor_checkpoint["optimizer_state_dict"]
        )

        critic_checkpoint = torch.load(os.path.join(load_path, "critic.pth"),
                                               map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")
                                       )
        instance.critic_model.load_state_dict(critic_checkpoint["model_state_dict"])
        instance.critic_optimizer.load_state_dict(
            critic_checkpoint["optimizer_state_dict"]
        )

        return instance

    def reinit_final_layers(self):
        xavier_uniform_(self.actor_model[-1].weight)
        if self.actor_model[-1].bias is not None:
            self.actor_model[-1].bias.data.fill_(0.0)
        
        xavier_uniform_(self.critic_model[-1].weight)
        if self.critic_model[-1].bias is not None:
            self.critic_model[-1].bias.data.fill_(0.0)

    def reinint_all_layers(self):
        for layer in self.actor_model:
            if hasattr(layer, "weight"):
                xavier_uniform_(layer.weight)
            if hasattr(layer, "bias") and layer.bias is not None:
                layer.bias.data.fill_(0.0)

        for layer in self.critic_model:
            if hasattr(layer, "weight"):
                xavier_uniform_(layer.weight)
            if hasattr(layer, "bias") and layer.bias is not None:
                layer.bias.data.fill_(0.0)
 

    def get_action(self, state):
        if len(state) < self.input_dim:
            state = np.pad(state, (0, self.input_dim - len(state)), "constant")

        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        action_params = self.actor_model(state)

        if self.orig_output_dim == 1:
            mu = action_params[0]
            sigma = torch.nn.functional.softplus(action_params[1]) + 1e-5
            distribution = torch.distributions.Normal(mu, sigma)
            action = distribution.sample()
            action = torch.clamp(action, -1.0, 1.0)
            
            return np.array([action.detach().cpu().numpy()])

        else:
            action_params = torch.nn.functional.softmax(action_params, dim=-1)
            valid_probs = action_params[: self.orig_output_dim]
            valid_probs = valid_probs / valid_probs.sum()
            return np.random.choice(
                self.orig_output_dim, p=valid_probs.detach().cpu().numpy()
            )

    def train_step(self, state, action, reward, next_state, done):
        if len(state) < self.input_dim:
            state = np.pad(state, (0, self.input_dim - len(state)), "constant")
        if len(next_state) < self.input_dim:
            next_state = np.pad(
                next_state, (0, self.input_dim - len(next_state)), "constant"
            )

        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        next_state = torch.tensor(next_state, dtype=torch.float32).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float32).to(self.device)

        value = self.critic_model(state)
        next_value = (
            self.critic_model(next_state)
            if not done
            else torch.tensor(0.0).to(self.device)
        )
        td_err = reward + self.gamma * next_value - value

        action_params = self.actor_model(state)

        if self.orig_output_dim == 1:
            mu = action_params[0]
            sigma = torch.nn.functional.softplus(action_params[1]) + 1e-5
            action = torch.tensor(action, dtype=torch.float32).to(self.device)
            distribution = torch.distributions.Normal(mu, sigma)
            log_prob = distribution.log_prob(action)
        else:
            action_params = torch.nn.functional.softmax(action_params, dim=-1)
            valid_probs = action_params[: self.orig_output_dim]
            valid_probs = valid_probs / valid_probs.sum()
            action = torch.tensor(action, dtype=torch.int64).to(self.device)
            distribution = torch.distributions.Categorical(valid_probs)
            log_prob = distribution.log_prob(action)

        policy_loss = -log_prob * td_err.detach()
        entropy = distribution.entropy().mean()
        entropy_loss = -self.entropy_weight * entropy
        total_policy_loss = policy_loss + entropy_loss
        value_loss = td_err.pow(2)

        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad()
        total_policy_loss.backward()
        self.actor_optimizer.step()

        return total_policy_loss.item(), value_loss.item()
