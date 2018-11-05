import gym

import random
import torch
from collections import deque

from unityagents import UnityEnvironment
import numpy as np


from dqn_agent import Agent

def load_checkpoint(model, filename='model/checkpoint.pth'):
    """
    Load checkpoint of model.
    return trained model
    """
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint)
    return model

def dqn(env, agent, n_episodes=100, max_t = 10000, eps_end=0.01):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    brain_name = env.brain_names[0]
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_end                    # initialize epsilon
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        score = 0
        
        agent.qnetwork_local = load_checkpoint(agent.qnetwork_local)
        agent.qnetwork_target = load_checkpoint(agent.qnetwork_target)
        for t in range(max_t):
            action = agent.act(state, eps)
            agent.qnetwork_local.eval()
            env_info = env.step(action)[brain_name]        # send the action to the environment
            next_state = env_info.vector_observations[0]   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]                  # see if episode has finished
            
            state = next_state
            score += reward
            if done:
                break 
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score

        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))

    return scores

def main():
    env = UnityEnvironment(file_name="Banana_Linux/Banana.x86_64")
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]
    # number of actions
    action_size = brain.vector_action_space_size
    # examine the state space 
    state = env_info.vector_observations[0]
    state_size = len(state)
    env_info = env.reset(train_mode=True)[brain_name]
    
    paras = {'hidden_size': 64, 'hidden_n': 2, 'batch_size': 64, 'gamma': 0.99}
    agent = Agent(state_size=state_size, action_size=action_size, seed=0,
                  hidden_size = paras['hidden_size'], hidden_n = paras['hidden_n'],
                 batch_size = paras['batch_size'], gamma = paras['gamma'])
    state = env.reset()
    scores = dqn(env, agent)
    
if __name__ == "__main__":
    main()
