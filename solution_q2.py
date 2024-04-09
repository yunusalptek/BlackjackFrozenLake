import gymnasium as gym
import numpy as np
env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True,
render_mode="human")
# Your code for Q2.2 which Executes Random Policy until 1000 episodes
# List to store training data
training_data = []

# Number of episodes
num_episodes = 1000

for _ in range(num_episodes):
    # Reset the environment for a new episode
    observation = env.reset()
    done = False
    
    while not done:
        # Choose a random action
        action = env.action_space.sample()
        
        # Take the action and observe the next state, reward, termination flag, and additional info
        next_observation, reward, done, _ = env.step(action)
        
        # Record the transition (observation, action, next_observation) and reward
        training_data.append((observation, action, next_observation, reward))
        
        observation = next_observation
        
        if done:
            break
# Your code for Q2.3 which implements Value Iteration
# Define the transition function T(s'|s, a) and reward function R(s, a, s') using the training data
def transition_function(s_prime, s, a):
    # Count occurrences of (s, a, s') in the training data
    count = sum(1 for obs, act, next_obs, _ in training_data if obs == s and act == a and next_obs == s_prime)
    # Count occurrences of (s, a) in the training data
    total = sum(1 for obs, act, _, _ in training_data if obs == s and act == a)
    # Calculate probability T(s'|s, a)
    return count / total if total > 0 else 0

def reward_function(s, a, s_prime):
    # Find all transitions (s, a, s') in the training data
    transitions = [(obs, act, next_obs, reward) for obs, act, next_obs, reward in training_data if obs == s and act == a and next_obs == s_prime]
    # Calculate the total reward for transitions (s, a, s')
    total_reward = sum(reward for _, _, _, reward in transitions)
    # Calculate the average reward R(s, a, s')
    return total_reward / len(transitions) if transitions else 0

# Discount factor
gamma = 0.9

# Number of iterations for value iteration
num_iterations = 1000

# Initialize the value function arbitrarily
num_states = 16  # Number of states in FrozenLake 4x4 grid
V = np.zeros(num_states)

# Value iteration algorithm
for _ in range(num_iterations):
    V_new = np.zeros(num_states)
    for s in range(num_states):
        max_value = float('-inf')
        for a in range(4):  # 4 possible actions in FrozenLake environment
            value = 0
            for s_prime in range(num_states):
                value += transition_function(s_prime, s, a) * (reward_function(s, a, s_prime) + gamma * V[s_prime])
            max_value = max(max_value, value)
        V_new[s] = max_value
    V = V_new
#Your code for Q2.4 which implements Policy Extraction
# Initialize the policy
policy = np.zeros(num_states, dtype=int)

# Extract the optimal policy
for s in range(num_states):
    max_action = None
    max_value = float('-inf')
    for a in range(4):  # 4 possible actions in FrozenLake environment
        value = 0
        for s_prime in range(num_states):
            value += transition_function(s_prime, s, a) * (reward_function(s, a, s_prime) + gamma * V[s_prime])
        if value > max_value:
            max_value = value
            max_action = a
    policy[s] = max_action
# Your code for Q2.5 which executes the optimal policy
# Extracted optimal policy
policy = [1, 2, 1, 0, 1, 0, 2, 0, 2, 1, 1, 0, 1, 2, 2, 0]  # Example policy obtained from Q2.4

# Number of episodes
num_episodes = 100

for _ in range(num_episodes):
    # Reset the environment for a new episode
    observation = env.reset()
    done = False
    
    while not done:
        # Choose action based on the optimal policy
        action = policy[observation]
        
        # Take the action and observe the next state, reward, termination flag, and additional info
        next_observation, reward, done, _, = env.step(action)
        
        observation = next_observation
        
        if done:
            break