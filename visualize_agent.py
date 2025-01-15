import gymnasium as gym
import numpy as np
import argparse
from pathlib import Path
import torch
from src.section1.agents.actor_critic_agent import ActorCriticAgent

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        type=str,
        choices=["Acrobot-v1", "MountainCarContinuous-v0", "CartPole-v1"],
        required=True,
        help="Environment to run"
    )
    parser.add_argument(
        "--models_dir",
        type=str,
        required=True,
        help="Directory containing the trained models"
    )
    parser.add_argument(
        "--episode",
        type=str,
        default="best",
        help="Episode to load (default: 'best')"
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=5,
        help="Number of episodes to render"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run the model on (default: cuda if available, else cpu)"
    )
    return parser.parse_args()

def create_agent(env, model_path, device='cuda'):
    """Create and initialize the agent with proper parameters"""
    # First create instance with default parameters
    agent = ActorCriticAgent(
        env=env,
        hidden1=256,
        hidden2=256,
        gamma=0.99,
        device=device,
        models_dir=str(model_path.parent),
        exp_name=model_path.name
    )
    
    # Then load the saved model parameters
    ActorCriticAgent.load_model(agent, str(model_path))
    return agent

def evaluate_agent(env, agent, num_episodes=5):
    total_rewards = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        truncated = False
        total_reward = 0
        steps = 0
        
        while not (done or truncated):
            env.render()
            # Use predict instead of get_action for evaluation
            action = agent.predict(state)
            next_state, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            state = next_state
            steps += 1
            
        print(f"Episode {episode + 1} - Steps: {steps}, Reward: {total_reward}")
        total_rewards.append(total_reward)
    
    print(f"\nAverage Reward over {num_episodes} episodes: {np.mean(total_rewards):.2f}")
    print(f"Standard Deviation: {np.std(total_rewards):.2f}")
    return np.mean(total_rewards)

def main():
    args = parse_args()
    
    # Create environment
    env = gym.make(args.env, render_mode="human")
    
    # Construct the model path
    base_path = Path(args.models_dir)
    model_path = base_path / f"episode_{args.episode}"
    
    if not model_path.exists():
        raise ValueError(f"Model directory not found: {model_path}")
        
    print(f"Loading model from: {model_path}")
    
    # Create and load the agent
    agent = create_agent(env, model_path, args.device)
    
    # Evaluate the agent
    evaluate_agent(env, agent, args.num_episodes)
    
    env.close()

if __name__ == "__main__":
    main()

# python visualize_agent.py \
# --env CartPole-v1 \
# --models_dir assets/section_1/actor_critic/cart_pole/models

# python visualize_agent.py \
# --env Acrobot-v1 \
# --models_dir assets/section_1/actor_critic/acrobot/models/acrobot_exp/ \
# --episode 1000 \
# --num_episodes 5

# python visualize_agent.py \
# --env MountainCarContinuous-v0 \
# --models_dir assets/section_1/actor_critic/mountain_car/models \
# --episode 1000 \
# --num_episodes 5

### Transfer 
# Acrobot to CartPole
# python visualize_agent.py \
# --env CartPole-v1 \
# --models_dir assets/section_2/transfer_acrobot_to_cartpole_2_1/models
