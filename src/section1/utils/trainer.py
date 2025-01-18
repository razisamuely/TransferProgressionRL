import sys
import tqdm
import numpy as np
import loguru

def pad_state(nxt_state, state_padding_size):
    return np.hstack((nxt_state, np.zeros(state_padding_size)))

def check_episode_end(done, step_per_episode, max_steps_per_episode, episode, score):
    if done or (step_per_episode >= max_steps_per_episode):
        loguru.logger.info(f"Episode: {episode}")
        if done:
            loguru.logger.success(
                f"Episode ended successfully after {step_per_episode} steps with final reward: {score:.2f}"
            )
        elif (step_per_episode >= max_steps_per_episode):
            loguru.logger.error(
                f"Episode terminated due to max steps ({max_steps_per_episode}) with final reward: {score:.2f}"
            )
        return True
    return False

def train(env,
          agent,
          episodes,
          state_padding_size, 
          max_steps_per_episode=3000):
    scores, avg_scores, turns, goals = [], [], [], []
    step = 0
    episode_steps = []
    best_avg_score = float('-inf')
    
    for episode in tqdm.tqdm(range(episodes), file=sys.stdout):
        score, current_steps = episode_step(
            env, agent,
            state_padding_size, step,
            episode, max_steps_per_episode)
        step += current_steps
        episode_steps.append(current_steps)
        
        scores.append(score)
        avg_score = np.mean(scores[-100:])
        avg_steps = np.mean(episode_steps[-100:])
        
        if avg_score > best_avg_score:
            best_avg_score = avg_score
            agent.save_model('best', avg_score)
            loguru.logger.success(f"New best model saved with avg_score: {avg_score:.2f}")
        
        agent.loger.log('episode/reward', score, episode)
        agent.loger.log('episode/steps', current_steps, episode)
        agent.loger.log('episode/avg_reward_100', avg_score, episode)
        agent.loger.log('episode/avg_steps_100', avg_steps, episode)
        agent.loger.log('episode/total_avg_reward', np.mean(scores), episode)
        agent.loger.log('episode/total_avg_steps', np.mean(episode_steps), episode)
        
        avg_scores.append(avg_score)
        turns.append(episode)
        goals.append(90)
        
        loguru.logger.info(f"Episode: {episode}, avg_score: {avg_score:.1f}, steps: {current_steps}")
        
    
    return turns, scores, avg_scores, goals

def episode_step(env, agent, state_padding_size, step, episode, max_steps_per_episode):
    state = env.reset()[0]
    state = np.hstack((state, np.zeros(state_padding_size)))
    score = 0
    step_per_episode = 0
    
    while True:
        step += 1
        step_per_episode += 1
        action = agent.sample_act(state)
        nxt_state, reward, done, _, info = env.step(action)
        nxt_state = pad_state(nxt_state, state_padding_size)
        score += reward
        agent.save_transition(state, action, reward, nxt_state, done)
        agent.learn(episode, step, state)
        state = nxt_state
        if check_episode_end(done, step_per_episode, max_steps_per_episode, episode, score):
            break
        
    return score, step_per_episode
