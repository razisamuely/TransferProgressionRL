import torch
import gymnasium as gym
import numpy as np
import os 
import json 
from pathlib import Path


class ActorCriticAgent:
    def __init__(self, input_dim, output_dim, gamma, learning_rate_actor,learning_rate_critic):
        self.device = torch.device("cpu")
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.gamma = gamma
        self.learning_rate_actor = learning_rate_actor
        self.learning_rate_critic = learning_rate_critic
        #
        self.actor_model = self.create_policy_network(input_dim, output_dim).to(self.device)
        self.critic_model = self.create_critic_network(input_dim, 1).to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor_model.parameters(), lr=self.learning_rate_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic_model.parameters(), lr=self.learning_rate_critic) 

        self.models_dir = "assets/task_2/section_1/actor_critic/models"
        path = Path(self.models_dir)
        path.mkdir(parents=True, exist_ok=True)
        
        # Creat path if not exist 

    def save_models(self, episode=None):
        """Save both actor and critic models"""
        save_path = self.models_dir
        if episode is not None:
            save_path = os.path.join(save_path, f"episode_{episode}")
            os.makedirs(save_path, exist_ok=True)
            
        # Save actor model
        actor_path = os.path.join(save_path, "actor.pth")
        torch.save({
            'model_state_dict': self.actor_model.state_dict(),
            'optimizer_state_dict': self.actor_optimizer.state_dict(),
            'input_dim': int(self.input_dim),  # Convert to native Python int
            'output_dim': int(self.output_dim),  # Convert to native Python int
            'learning_rate': float(self.learning_rate_actor)  # Convert to native Python float
        }, actor_path)
        
        # Save critic model
        critic_path = os.path.join(save_path, "critic.pth")
        torch.save({
            'model_state_dict': self.critic_model.state_dict(),
            'optimizer_state_dict': self.critic_optimizer.state_dict(),
            'input_dim': int(self.input_dim),  # Convert to native Python int
            'output_dim': 1,
            'learning_rate': float(self.learning_rate_critic)  # Convert to native Python float
        }, critic_path)
        
        # Save hyperparameters with explicit type conversion
        hyper_params = {
            'gamma': float(self.gamma),
            'learning_rate_actor': float(self.learning_rate_actor),
            'learning_rate_critic': float(self.learning_rate_critic),
            'input_dim': int(self.input_dim),
            'output_dim': int(self.output_dim)
        }
        with open(os.path.join(save_path, "hyperparameters.json"), 'w') as f:
            json.dump(hyper_params, f)

    @classmethod
    def load_models(cls, load_path):
        """Load a saved model"""
        # Load hyperparameters
        with open(os.path.join(load_path, "hyperparameters.json"), 'r') as f:
            hyper_params = json.load(f)
            
        # Create new instance
        instance = cls(
            input_dim=hyper_params['input_dim'],
            output_dim=hyper_params['output_dim'],
            gamma=hyper_params['gamma'],
            learning_rate_actor=hyper_params['learning_rate_actor'],
            learning_rate_critic=hyper_params['learning_rate_critic']
        )
        
        # Load actor model
        actor_checkpoint = torch.load(os.path.join(load_path, "actor.pth"))
        instance.actor_model.load_state_dict(actor_checkpoint['model_state_dict'])
        instance.actor_optimizer.load_state_dict(actor_checkpoint['optimizer_state_dict'])
        
        # Load critic model
        critic_checkpoint = torch.load(os.path.join(load_path, "critic.pth"))
        instance.critic_model.load_state_dict(critic_checkpoint['model_state_dict'])
        instance.critic_optimizer.load_state_dict(critic_checkpoint['optimizer_state_dict'])
        
        return instance
    
    def create_policy_network(self, input_dim, output_dim):
        return torch.nn.Sequential(*[
            torch.nn.Linear(input_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, output_dim),
            torch.nn.Softmax(dim=-1),
        ])

    def create_critic_network(self, input_dim, output_dim):
        return torch.nn.Sequential(*[
            torch.nn.Linear(input_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, output_dim),
        ])
    
    def get_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        policy_probs = self.actor_model(state)
        action = np.random.choice(self.output_dim, p=policy_probs.detach().cpu().numpy())
        return action


    def train_step(self, state, action, reward, next_state, done):
        #  
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        next_state = torch.tensor(next_state, dtype=torch.float32).to(self.device)
        action = torch.tensor(action, dtype=torch.int64).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float32).to(self.device)

        # Compute the TD errror
        value = self.critic_model(state)
        next_value = self.critic_model(next_state) if not done else torch.tensor(0.0).to(self.device)
        td_err = reward + self.gamma * next_value - value
        
        # computer actor error       
        policy_probs = self.actor_model(state)
        distribution = torch.distributions.Categorical(policy_probs)
        log_prob = distribution.log_prob(action)
        policy_loss = -log_prob * td_err.detach()

        # compute actor loss
        value_loss = td_err.pow(2)


        # update critic
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()

        # update policy
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        return policy_loss.item(), value_loss.item()