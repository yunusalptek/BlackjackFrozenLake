import gymnasium as gym
import numpy as np

# Initialize environment
env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True, render_mode="human")

# Function to execute random policy for given episodes
def execute_random_policy(env, episodes=10):
    transitions = []  # List to store transitions (s, a, r, s', done)
    
    for episode in range(episodes):
        state, info = env.reset()
        #done = False
        action = env.action_space.sample()  # Choose a random action
        next_state, reward, terminated, truncated, info = env.step(action)
        transitions.append((state, action, reward, next_state))
        state = next_state

        if terminated or truncated:
            break

    return transitions

# Collect training data
training_data = execute_random_policy(env)

# Define functions to estimate transition and reward functions
def estimate_functions(training_data, n_states, n_actions):
    T_counts = np.zeros((n_states, n_actions, n_states))  # Transition counts
    R_sums = np.zeros((n_states, n_actions, n_states))     # Sum of rewards

    for transition in training_data:
        s, a, r, s_prime = transition
        T_counts[s, a, s_prime] += 1
        R_sums[s, a, s_prime] += r

    # Normalize to get probabilities
    T = T_counts / np.maximum(T_counts.sum(axis=2, keepdims=True), 1)
    R = np.divide(R_sums, T_counts, out=np.zeros_like(R_sums), where=T_counts != 0)

    return T, R

# Get number of states and actions
n_states = env.observation_space.n
n_actions = env.action_space.n

# Estimate transition and reward functions
T_hat, R_hat = estimate_functions(training_data, n_states, n_actions)

# Value Iteration to find optimal value function
def value_iteration(T, R, gamma=0.99, epsilon=1e-6):
    n_states, n_actions, _ = T.shape
    V = np.zeros(n_states)

    while True:
        V_new = np.zeros_like(V)
        for s in range(n_states):
            V_new[s] = np.max(np.sum(T[s] * (R[s] + gamma * V), axis=1))
        if np.max(np.abs(V - V_new)) < epsilon:
            break
        V = V_new

    return V

# Get optimal value function
V_optimal = value_iteration(T_hat, R_hat)

# Extract policy
def extract_policy(V, T, R, gamma=0.99):
    Q = np.zeros((V.shape[0], T.shape[1]))  # Initialize Q with the correct shape

    for s in range(V.shape[0]):
        for a in range(T.shape[1]):  # Use T.shape[1] to ensure correct indexing
            Q[s, a] = np.sum(T[s, a, :] * (R[s, a, :] + gamma * V))

    policy = np.argmax(Q, axis=1)
    return policy

optimal_policy = extract_policy(V_optimal, T_hat, R_hat)

# Execute optimal policy
def execute_optimal_policy(env, policy):
    observation, info = env.reset()
    total_reward = 0
    #done = False
    for _ in range(10):
        if isinstance(observation, tuple):  # Check if observation is a tuple
            observation = observation[0]  # Take the first element if it's a tuple
        action = int(policy[observation])  # Convert observation to int
        observation, reward, truncated, terminated, info = env.step(action)
        total_reward += reward
    
        if terminated or truncated:
            break
        
    return total_reward

total_reward = execute_optimal_policy(env, optimal_policy)
print("Total reward with optimal policy:", total_reward)

# Close environment
env.close()
