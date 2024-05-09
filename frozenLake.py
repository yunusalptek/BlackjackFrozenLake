# Functions for Steps 1 to 5 (as defined above)
import numpy as np
import gymnasium as gym

# Step 1: Collect Training Data
def collect_training_data(env, num_episodes=1000):
    transitions = []
    rewards = []
    for _ in range(num_episodes):
        obs = env.reset()
        episode_transitions = []
        episode_rewards = []
        done = False
        while not done:
            action = env.action_space.sample()  # Random action
            next_obs, reward, done, info, _ = env.step(action)
            # Convert observations to integers using env.unwrapped
            s = env.unwrapped.s  # Current state
            s_next = next_obs  # Next state
            episode_transitions.append((s, action, s_next))
            episode_rewards.append(reward)
            obs = next_obs
        transitions.extend(episode_transitions)
        rewards.extend(episode_rewards)
    return transitions, rewards

# Step 2: Estimate Transition and Reward Functions
def estimate_functions(transitions, rewards, num_states, num_actions):
    # Check if the environment has discrete observation and action spaces
    assert isinstance(env.observation_space, gym.spaces.Discrete), "Observation space must be discrete"
    assert isinstance(env.action_space, gym.spaces.Discrete), "Action space must be discrete"
    
    # Initialize counts for transitions and rewards
    transition_counts = np.zeros((num_states, num_actions, num_states))
    reward_counts = np.zeros((num_states, num_actions, num_states))

    # Count occurrences of transitions and rewards
    for transition, reward in zip(transitions, rewards):
        s, a, s_next = transition
        transition_counts[s, a, s_next] += 1
        reward_counts[s, a, s_next] += reward

    # Estimate transition probabilities and rewards
    T = transition_counts / np.maximum(1, transition_counts.sum(axis=2, keepdims=True))
    R = reward_counts / np.maximum(1, transition_counts.sum(axis=2, keepdims=True))

    return T, R

# Step 3: Value Iteration
def value_iteration(T, R, gamma=0.99, epsilon=1e-6):
    num_states, num_actions, _ = T.shape
    V = np.zeros(num_states)
    
    while True:
        V_new = np.max(np.sum(T * (R + gamma * V), axis=2), axis=1)
        if np.max(np.abs(V_new - V)) < epsilon:
            break
        V = V_new
    return V

# Step 4: Policy Extraction
def extract_policy(T, R, V, gamma=0.99):
    num_states, num_actions, _ = T.shape
    Q = np.sum(T * (R + gamma * V), axis=2)
    policy = np.argmax(Q, axis=1)
    return policy

# Step 5: Execute Optimal Policy
def execute_optimal_policy(env, policy):
    obs = env.reset()
    env.render()
    done = False
    while not done:
        # Convert observation to integer state
        s = env.unwrapped.s
        action = policy[s]
        obs, _, done, _, info = env.step(action)
        env.render()

# Environment Setup
env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True, render_mode="human")

# Step 1: Collect Training Data
transitions, rewards = collect_training_data(env, num_episodes=1000)

# Step 2: Estimate Transition and Reward Functions
num_states = env.observation_space.n
num_actions = env.action_space.n
T, R = estimate_functions(transitions, rewards, num_states, num_actions)

# Step 3: Value Iteration
V = value_iteration(T, R)

# Step 4: Policy Extraction
policy = extract_policy(T, R, V)

# Step 5: Execute Optimal Policy
execute_optimal_policy(env, policy)

env.close()