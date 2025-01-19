import os
import json
from pathlib import Path
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym

from src.section1.agents.actor_critic_agent import ActorCriticAgent
from visualize_agent import create_agent

class ProgressiveActorNetwork(nn.Module):
    def __init__(self,
             n_state,
                 max_n_action,
                 n_action,
                 hidden1, hidden2, 
                 model_1,model_2,
                 h11_w1,h11_w2,h12_w1,h12_w2,
                 action_space=None
                ):
        super(ProgressiveActorNetwork, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_1 = model_1.to(self.device)
        self.model_2 = model_2.to(self.device)
        self.freeze_model(self.model_1)
        self.freeze_model(self.model_2)

        # Create main network
        self.base = nn.Sequential(
            nn.Linear(n_state, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, max_n_action),
        )
        self.mu_layer = nn.Linear(1, 1)
        self.sigma_layer = nn.Linear(1, 1)

        self.max_n_action = max_n_action
        self.n_action = n_action
        self.is_discrete = isinstance(action_space, gym.spaces.Discrete)
        self.distribution = torch.distributions.Normal


        model_1_output_dim = self.get_model_output_dim(self.model_1, n_state)
        model_2_output_dim = self.get_model_output_dim(self.model_2, n_state)
        self.adapter_1_layer1 = self._create_adapter(model_1_output_dim, hidden1).to(self.device)
        self.adapter_1_layer2 = self._create_adapter(model_1_output_dim, hidden2).to(self.device)
        self.adapter_2_layer1 = self._create_adapter(model_2_output_dim, hidden1).to(self.device)
        self.adapter_2_layer2 = self._create_adapter(model_2_output_dim, hidden2).to(self.device)
        self.adapter_1_layer1.apply(self.init_weights)
        self.adapter_1_layer2.apply(self.init_weights)
        self.adapter_2_layer1.apply(self.init_weights)
        self.adapter_2_layer2.apply(self.init_weights)
        
        self.h11_w1=h11_w1
        self.h11_w2=h11_w2
        self.h12_w1=h12_w1
        self.h12_w2=h12_w2
        self.epsilon = 1e-5

        self.base.to(self.device)
        self.mu_layer.to(self.device)
        self.sigma_layer.to(self.device)


    def freeze_model(self, model):
        for param in model.parameters():
            param.requires_grad = False
    def _apply_gradient_mask(self, x):
        gradient_mask = torch.zeros_like(x)
        mask_indices = slice(None, self.n_action)
        
        if x.dim() == 1:
            gradient_mask[mask_indices] = 1
        else:
            gradient_mask[:, mask_indices] = 1
            
        return x * gradient_mask.detach()
    def _get_discrete_distribution(self, x):
        action_probs = F.softmax(x, dim=-1)
        valid_indices = slice(None, self.n_action)
        
        if x.dim() == 1:
            valid_probs = action_probs[valid_indices]
        else:
            valid_probs = action_probs[:, valid_indices]
            
        valid_probs = valid_probs / valid_probs.sum(dim=-1, keepdim=True)
        return torch.distributions.Categorical(probs=valid_probs)
    def _get_continuous_distribution(self, x):
        if x.dim() == 1:
            mu = self.mu_layer(x[:1])
            sigma = self.sigma_layer(x[1:2])
        else:
            mu = self.mu_layer(x[:,:1])
            sigma = self.sigma_layer(x[:,1:2])

        mu_final = 2 * torch.tanh(mu)
        sigma_final = F.softplus(sigma) + self.epsilon
        
        return self.distribution(mu_final, sigma_final)

    def _create_adapter(self, input_dim, hidden_dim):
        print(input_dim, hidden_dim)
        
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim  ),
            nn.ReLU(),
            nn.Linear(hidden_dim  , hidden_dim),
        )

    def get_model_output_dim(self, model, input_dim):
        """Infer the output dimension of the given model."""
        dummy_input = torch.zeros(1, input_dim).to(self.device)
        with torch.no_grad():
            output = model(dummy_input)
        try:
            return output.probs.shape[1]
        except Exception as err:
            pass
        return output.shape[1]

    def __joined_forward(self,state):
        x = self.base[0](state)
        x = self.base[1](x) 
        cp_h11 = self.model_1(state)  
        cp_adapter_h11 = self.adapter_1_layer1(cp_h11.detach()) 
        ac_h11 = self.model_2(state) 
        ac_adapter_h11 = self.adapter_2_layer1(ac_h11.detach())  
        combined = x +self.h11_w1 * cp_adapter_h11 + self.h11_w2 * ac_adapter_h11
        base_output = self.base[2](combined)
        base_output = self.base[3](base_output)  
        cp_adapter_h12 = self.adapter_1_layer2(cp_h11.detach())
        ac_adapter_h12 = self.adapter_2_layer2(ac_h11.detach())
        combined_final = base_output + self.h12_w1 * cp_adapter_h12 + self.h12_w2* ac_adapter_h12
        x = self.base[4](combined_final)
        return x    
    
    def forward(self, state):
        x=self.__joined_forward(state)
        
        if self.max_n_action > self.n_action:
            x = self._apply_gradient_mask(x)
            dist = (self._get_discrete_distribution(x) if self.is_discrete 
                   else self._get_continuous_distribution(x))
        else:
            dist = self._get_discrete_distribution(x)
                
        return dist
    def init_weights(self,m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            nn.init.zeros_(m.bias)
            
class ProgressiveCriticNetwork(nn.Module):
    def __init__(self,
             n_state,
                 hidden1, 
                 hidden2,
                model_1,
                model_2,h11_w1,h11_w2,h12_w1,h12_w2):
        
        super(ProgressiveCriticNetwork, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_1 = model_1.to(self.device)
        self.model_2 = model_2.to(self.device)
        self.freeze_model(self.model_1)
        self.freeze_model(self.model_2)

        # Create main network
        self. net = nn.Sequential(
            nn.Linear(n_state, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1)
        )
        
        #
        model_1_output_dim = self.get_model_output_dim(self.model_1, n_state)
        model_2_output_dim = self.get_model_output_dim(self.model_2, n_state)
        self.adapter_1_layer1 = self._create_adapter(model_1_output_dim, hidden1).to(self.device)
        self.adapter_1_layer2 = self._create_adapter(model_1_output_dim, hidden2).to(self.device)
        self.adapter_2_layer1 = self._create_adapter(model_2_output_dim, hidden1).to(self.device)
        self.adapter_2_layer2 = self._create_adapter(model_2_output_dim, hidden2).to(self.device)
        self.h11_w1=h11_w1
        self.h11_w2=h11_w2
        self.h12_w1=h12_w1
        self.h12_w2=h12_w2

    def freeze_model(self, model):
        for param in model.parameters():
            param.requires_grad = False


    def _create_adapter(self, input_dim, hidden_dim):
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim  ),
            nn.ReLU(),
            nn.Linear(hidden_dim  , hidden_dim),
        )

    def get_model_output_dim(self, model, input_dim):
        """Infer the output dimension of the given model."""
        dummy_input = torch.zeros(1, input_dim).to(self.device)
        with torch.no_grad():
            output = model(dummy_input)
        try:
            return output.probs.shape[1]
        except Exception as err:
            pass
        return output.shape[1]
    
    def __join_forward(self,state):
        x = self.net[0](state)
        x = self.net[1](x) 
        cp_h11 = self.model_1(state)  
        cp_adapter_h11 = self.adapter_1_layer1(cp_h11.detach())  
        ac_h11 = self.model_2(state)  
        ac_adapter_h11 = self.adapter_2_layer1(ac_h11.detach()) 
        combined = x + self.h11_w1 * cp_adapter_h11 +   self.h11_w2*ac_adapter_h11
        base_output = self.net[2](combined)
        base_output = self.net[3](base_output) 
        cp_adapter_h12 = self.adapter_1_layer2(cp_h11.detach())
        ac_adapter_h12 = self.adapter_2_layer2(ac_h11.detach())
        combined_final = base_output + self.h12_w1 * cp_adapter_h12 + self.h12_w2* ac_adapter_h12
        output = self.net[4](combined_final)
        return output    

    def forward(self, state):
        return self.__join_forward(state)

class ProgressiveActorCriticAgent(ActorCriticAgent):
    MAX_INPUT_DIM = 6
    MAX_OUTPUT_DIM = 3

    def __init__(self, env,
                 hidden1, hidden2,
        gamma, 
        agent_1_env_name:str,
        agent_1_weights_path:str,
        agent_2_env_name:str,
        agent_2_weights_path:str,

        h11_w1=0.0,
        h11_w2=0.0,
        h12_w1=0.0,
        h12_w2=0.0,
        lr_actor=0.00001,
        lr_critic=0.0001, 
        buf_size=1000000, 
        sync_freq=100, 
        batch_size=64, 
        exp_name='exp1', 
        device='cuda', 
        models_dir='./models'):
        super(ProgressiveActorCriticAgent, self).__init__(
            env,
            hidden1, 
            hidden2, 
            gamma, 
            lr_actor=lr_actor,
            lr_critic=lr_critic,
            buf_size=buf_size, 
            sync_freq=sync_freq,
            batch_size=batch_size, 
            exp_name=exp_name, 
            device=device,
            models_dir=models_dir)
        
        #
        self.agent_1_env_name=agent_1_env_name
        self.agent_1_weights_path=agent_1_weights_path
        self.agent_2_env_name=agent_2_env_name
        self.agent_2_weights_path=agent_2_weights_path
        self.agent_1 = self.load_agent(agent_1_env_name,agent_1_weights_path)
        self.agent_2 = self.load_agent(agent_2_env_name,agent_2_weights_path)
        self.actor = ProgressiveActorNetwork(
                    n_state= self.n_state,
                     max_n_action=self.max_n_action,
                     n_action=self.n_action, 
                     hidden1=hidden1,
                     hidden2=hidden2,
                     model_1=self.agent_1.actor.base,
                     model_2=self.agent_2.actor.base,
                     action_space=self.env.action_space,
                     h11_w1=h11_w1,
                     h11_w2=h11_w2,
                     h12_w1=h12_w1,
                     h12_w2=h12_w2
        ).to(self.device)
        self.critic = ProgressiveCriticNetwork(
            n_state=self.n_state,
            hidden1=hidden1,
            hidden2=hidden2, 
            model_1=self.agent_1.critic,
            model_2=self.agent_2.critic,
            h11_w1=h11_w1,
            h11_w2=h11_w2,
            h12_w1=h12_w1,
            h12_w2=h12_w2
        ).to(self.device)
        self.target_critic = copy.deepcopy(self.critic)
        self.optim_actor = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optim_critic = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        #
        self.h11_w1=h11_w1
        self.h11_w2=h11_w2
        self.h12_w1=h12_w1
        self.h12_w2=h12_w2
    def load_agent(self,env_name,env_weights_path):
        return  create_agent(gym.make(env_name).unwrapped,
                             Path(env_weights_path))

    ########################################################
    def save_model(self, epoch, avg_score):
        save_path = os.path.join(self.models_dir, self.exp_name)
        if epoch is not None:
            save_path = os.path.join(save_path, f"episode_{epoch}")
        os.makedirs(save_path, exist_ok=True)

        actor_path = os.path.join(save_path, "actor.pth")
        torch.save({
            "model_state_dict": self.actor.state_dict(),
            "optimizer_state_dict": self.optim_actor.state_dict(),
            "input_dim": int(self.n_state),
            "output_dim": int(self.n_action),
            "learning_rate": float(self.lr_actor),
            
            
        }, actor_path)

        critic_path = os.path.join(save_path, "critic.pth")
        torch.save({
            "model_state_dict": self.critic.state_dict(),
            "optimizer_state_dict": self.optim_critic.state_dict(),
            "input_dim": int(self.n_state),
            "output_dim": 1,
            "learning_rate": float(self.lr_critic),
        }, critic_path)

        hyper_params = {
            "gamma": float(self.gamma),
            "learning_rate_actor": float(self.lr_actor),
            "learning_rate_critic": float(self.lr_critic),
            "input_dim": int(self.n_state),
            "output_dim": int(self.n_action),
            "orig_input_dim": int(self.n_state),
            "orig_output_dim": int(self.n_action),
            "models_dir": str(save_path),
            "average_score": float(avg_score),
            "h11_w1":self.h11_w1,
            "h11_w2":self.h11_w2,
            "h12_w1":self.h12_w1,
            "h12_w2":self.h12_w2,
            "agent_1_env_name":self.agent_1_env_name,
            "agent_1_weights_path":self.agent_1_weights_path,
            "agent_2_env_name":self.agent_2_env_name,
            "agent_2_weights_path":self.agent_2_weights_path,
        }
        with open(os.path.join(save_path, "hyperparameters.json"), "w") as f:
            json.dump(hyper_params, f)

    def load_model(self, load_path):
        actor_checkpoint = torch.load(os.path.join(load_path, "actor.pth"))
        self.actor.load_state_dict(actor_checkpoint["model_state_dict"])
        self.optim_actor.load_state_dict(actor_checkpoint["optimizer_state_dict"])

        critic_checkpoint = torch.load(os.path.join(load_path, "critic.pth"))
        self.critic.load_state_dict(critic_checkpoint["model_state_dict"])
        self.optim_critic.load_state_dict(critic_checkpoint["optimizer_state_dict"])
        
        