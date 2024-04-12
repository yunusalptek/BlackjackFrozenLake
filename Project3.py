import gym
import numpy as np
from tqdm import tqdm
from collections import defaultdict


env = gym.make('Blackjack-v1', natural=False, sab=False, render_mode='human')

learning_rate = 0.0001 #The lower the better
num_runs = 10_000 #Number of games played
initial_epsilon = 1.0
epsilon_decay = initial_epsilon / (num_runs / 2)
final_epsilon = 0.1

#Initialize Q-table
q_table = defaultdict(lambda: np.zeros(env.action_space.n))

# Initialize epsilon
epsilon = initial_epsilon



# Iterate over episodes
for episode_num in tqdm(range(num_runs)):
    episode_trajectory = []
    observation, info = env.reset()
    is_done = False

    #For 1 game 
    while not is_done:
        #Choose action using epsilon-greedy strategy
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = int(np.argmax(q_table[observation]))

        #Take action
        next_observation, reward, is_done, info, _ = env.step(action)
        
        #Store transition
        episode_trajectory.append((observation, action, reward, next_observation))

        observation = next_observation

    #Reduce epsilon to learn and act optimally eventually
    epsilon = max(final_epsilon, epsilon - epsilon_decay)

    #Update Q-values based on games
    for i in range(len(episode_trajectory)):
        state, action, reward, next_state = episode_trajectory[i]
        current_q = q_table[state][action]
        next_q = np.max(q_table[next_state]) if not is_done else 0  # Terminal state
        temporal_difference = reward + 0.95 * next_q - current_q
        q_table[state][action] += learning_rate * temporal_difference
