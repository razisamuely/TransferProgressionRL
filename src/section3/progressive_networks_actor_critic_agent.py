import os
import json
from pathlib import Path
import numpy as np
from src.utils.actor_critic_agent import ActorCriticAgent
import torch
import torch.nn as nn
torch.autograd.set_detect_anomaly(True)
class ProgressiveActorNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, model_1, model_2):
        super(ProgressiveActorNetwork, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_1 = model_1.to(self.device)
        self.model_2 = model_2.to(self.device)
        self.freeze_model(self.model_1)
        self.freeze_model(self.model_2)

        # Create main network
        self.main_network = self.create_network(input_dim, hidden_dim, output_dim).to(self.device)

        # Dynamically set adapter input dimensions based on model output sizes
        model_1_output_dim = self.get_model_output_dim(self.model_1, input_dim)
        model_2_output_dim = self.get_model_output_dim(self.model_2, input_dim)

        self.adapter_1_layer1 = self._create_adapter(model_1_output_dim, hidden_dim).to(self.device)
        self.adapter_1_layer2 = self._create_adapter(model_1_output_dim, hidden_dim).to(self.device)
        self.adapter_2_layer1 = self._create_adapter(model_2_output_dim, hidden_dim).to(self.device)
        self.adapter_2_layer2 = self._create_adapter(model_2_output_dim, hidden_dim).to(self.device)

    def freeze_model(self, model):
        for param in model.parameters():
            param.requires_grad = False

    def create_network(self, input_dim, hidden_dim, output_dim):
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def _create_adapter(self, input_dim, hidden_dim):
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
        )

    def get_model_output_dim(self, model, input_dim):
        """Infer the output dimension of the given model."""
        dummy_input = torch.zeros(1, input_dim).to(self.device)
        with torch.no_grad():
            output = model(dummy_input)
        return output.shape[1]

    def forward(self, state):
        # Pass state through the base layer of the main network
        x = self.main_network[0](state)
        x = self.main_network[1](x)  # ReLU activation

        # Compute outputs from frozen models
        cp_h11 = self.model_1(state)  # Output from model_1
        cp_adapter_h11 = self.adapter_1_layer1(cp_h11.detach())  # Pass through adapter with detached gradients

        ac_h11 = self.model_2(state)  # Output from model_2
        ac_adapter_h11 = self.adapter_2_layer1(ac_h11.detach())  # Pass through adapter with detached gradients

        # Combine outputs with the base network
        combined = x + 0.01 * cp_adapter_h11 + 0.01 * ac_adapter_h11

        # Pass combined outputs through the second layer of the main network
        base_output = self.main_network[2](combined)
        base_output = self.main_network[3](base_output)  # ReLU activation

        # Apply second set of adapters
        cp_adapter_h12 = self.adapter_1_layer2(cp_h11.detach())
        ac_adapter_h12 = self.adapter_2_layer2(ac_h11.detach())

        # Final combination
        combined_final = base_output + 0.01 * cp_adapter_h12 + 0.01 * ac_adapter_h12

        # Pass through the final layer of the main network
        output = self.main_network[4](combined_final)

        return output

class ProgressiveActorCriticAgent(ActorCriticAgent):
    MAX_INPUT_DIM = 6
    MAX_OUTPUT_DIM = 3

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        gamma: float,
        learning_rate_actor: float,
        learning_rate_critic: float,
        agent_1: ActorCriticAgent,
        agent_2: ActorCriticAgent,
        models_dir: str = None,
        entropy_weight: float = 0.1,
    ):
        super(ProgressiveActorCriticAgent, self).__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            gamma=gamma,
            learning_rate_actor=learning_rate_actor,
            learning_rate_critic=learning_rate_critic,
            models_dir=models_dir,
            entropy_weight=entropy_weight,
        )
        self.agent_1 = agent_1
        self.agent_2 = agent_2
        self.actor_model = ProgressiveActorNetwork(
            input_dim=self.MAX_INPUT_DIM,
            hidden_dim=128,
            output_dim=self.MAX_OUTPUT_DIM,
            model_1=self.agent_1.actor_model,
            model_2=self.agent_2.actor_model,
        ).to(self.device)
        self.critic_model = ProgressiveActorNetwork(
            input_dim=self.MAX_INPUT_DIM,
            hidden_dim=128,
            output_dim=1,
            model_1=self.agent_1.critic_model,
            model_2=self.agent_2.critic_model,
        ).to(self.device)
        self.actor_optimizer = torch.optim.Adam(
            self.actor_model.parameters(), lr=self.learning_rate_actor
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic_model.parameters(), lr=self.learning_rate_critic
        )

        if models_dir:
            Path(models_dir).mkdir(parents=True, exist_ok=True)


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
