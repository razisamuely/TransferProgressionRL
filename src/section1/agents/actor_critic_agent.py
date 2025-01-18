import copy
import numpy as np
import torch
import torch.nn as nn
import os
import json
from collections import deque
from gymnasium.spaces import Box
from src.section1.utils.ounoise import OUNoise
from src.section1.utils.dataloger import DataLoger
from src.section1.agents.actor_critic import Actor, Critic
from src.section1.utils.constants import MAX_INPUT_DIM, MAX_OUTPUT_DIM

class ActorCriticAgent:
    MAX_INPUT_DIM = MAX_INPUT_DIM
    MAX_OUTPUT_DIM = MAX_OUTPUT_DIM

    def __init__(self, env, 
                 hidden1, hidden2,
                 gamma,
                 lr_actor=0.00001,
                 lr_critic=0.0001, 
                 buf_size=1000000,
                 sync_freq=100,
                 batch_size=64,
                 exp_name='exp1', 
                 device='cuda',
                 models_dir='./models'):
        self.env = env
        self.max_n_action = self.MAX_OUTPUT_DIM
        self.n_action = self._get_action_space_size(env)
        self.n_state = self.MAX_INPUT_DIM
        self.device = device
        self._is_discrete = not isinstance(env.action_space, Box)

        self._init_networks(hidden1, hidden2)
        self._init_memory(buf_size, batch_size)
        self._init_training_params(gamma, lr_actor, lr_critic, batch_size, sync_freq)
        self._init_exploration()
        self._init_logging(exp_name, models_dir)
    
    def _init_networks(self, hidden1, hidden2):
        self.actor = Actor(self.n_state, self.max_n_action, self.n_action, 
                         hidden1, hidden2, self.env.action_space).to(self.device)
        self.critic = Critic(self.n_state, hidden1, hidden2).to(self.device)
        self.target_critic = copy.deepcopy(self.critic)

    def _init_memory(self, buf_size, batch_size):
        self.buf_size = buf_size
        self.batch_size = batch_size
        self.buffer = deque(maxlen=buf_size)
        self.bf_counter = 0

    def _init_training_params(self, gamma, lr_actor, lr_critic, batch_size, sync_freq):
        self.gamma = gamma
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.batch_size = batch_size
        self.sync_freq = sync_freq
        self.learn_counter = 0
        self.optim_actor = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optim_critic = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.mse_loss = nn.MSELoss()

    def _init_exploration(self):
        self.explore_mu = 0.2
        self.explore_theta = 0.15
        self.explore_sigma = 0.2
        self.noise = OUNoise(self.n_action, self.explore_mu, self.explore_theta, self.explore_sigma)

    def _init_logging(self, exp_name, models_dir):
        self.exp_name = exp_name
        self.models_dir = models_dir
        self.loger = DataLoger(models_dir+ '/tensorboard/' + exp_name)

    @staticmethod
    def _get_action_space_size(env):
        return env.action_space.shape[0] if isinstance(env.action_space, Box) else env.action_space.n

    def save_transition(self, state, action, reward, nxt_state, done):
        transition = np.hstack((state, action, reward, nxt_state, done))
        self.buffer.append(transition)
        self.bf_counter += 1

    def predict(self, state):
        state = torch.Tensor(state).to(self.device)
        dist = self.actor(state)
        return dist.sample().cpu().numpy()

    def sample_act(self, state, noise_scale=1.0):
        action = self.predict(state)
        if self._is_discrete:
            return action
        return action + self.noise.sample() * noise_scale

    def sample_batch(self):
        max_buffer = min(self.bf_counter, self.buf_size)
        if max_buffer < self.batch_size:
            return None
            
        # Convert last two transitions from deque to numpy array
        # Note: deque[-n:] returns last n items
        batch = np.array(list(self.buffer)[-100:])
        
        state = batch[:, :self.n_state]
        action = batch[:, self.n_state:self.n_state + 1]
        reward = batch[:, self.n_state + 1:self.n_state + 2]
        nxt_state = batch[:, self.n_state + 2:self.n_state * 2 + 2]
        done = batch[:, self.n_state * 2 + 2:]
        
        return state, action, reward, nxt_state, done

    def learn(self, epoch, step, cur_state):
        max_buffer = min(self.bf_counter, self.buf_size)
        if max_buffer < self.batch_size:
            return

        self.learn_counter += 1
        if self.learn_counter % self.sync_freq == 0:
            self.target_critic.load_state_dict(self.critic.state_dict())

        state, action, reward, nxt_state, done = self.sample_batch()
        state = torch.Tensor(state).to(self.device)
        action = torch.Tensor(action).to(self.device)
        reward = torch.Tensor(reward).to(self.device)
        nxt_state = torch.Tensor(nxt_state).to(self.device)
        done = torch.Tensor(done).to(self.device)

        dist, V = self.actor(state), self.critic(state)
        nxt_V = self.target_critic(nxt_state).detach()
        td_error = reward + self.gamma * nxt_V * (1 - done) - V

        actor_loss = (-dist.log_prob(action) * td_error.detach()).mean()
        critic_loss = self.mse_loss(V, reward + self.gamma * nxt_V * (1 - done))

        self.optim_actor.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 3)
        self.optim_actor.step()

        self.optim_critic.zero_grad()
        critic_loss.backward()
        self.optim_critic.step()

        if step % 100 == 0:
            self.loger.log('actor_loss', actor_loss.item(), step)
            self.loger.log('critic_loss', critic_loss.item(), step)

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
            "learning_rate": float(self.lr_actor)
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
        
        