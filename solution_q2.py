import numpy as np
import gymnasium as gym

# Initialize environment
env = gym.make("FrozenLake-v1", desc=None, map_name="4x4", render_mode="human", is_slippery=True)

# Initialize dictionaries to store counts
transition_counts = {}  # Count of transitions (s, a) -> s'
reward_counts = {}      # Count of rewards (s, a, s') -> r
state_action_counts = {} # Count of state-action pairs

# Execute random policy for 1000 episodes
for _ in range(10):
    state, info = env.reset()
    #done = False
    action = env.action_space.sample()  # Random action
    next_state, reward, terminated, truncated, info = env.step(action)

    # Update state-action counts
    state_action_pair = (state, action)
    if state_action_pair not in state_action_counts:
        state_action_counts[state_action_pair] = 1
    else:
        state_action_counts[state_action_pair] += 1

    # Update transition counts
    if state_action_pair not in transition_counts:
        transition_counts[state_action_pair] = {}
    if next_state not in transition_counts[state_action_pair]:
        transition_counts[state_action_pair][next_state] = 1
    else:
        transition_counts[state_action_pair][next_state] += 1

    # Update reward counts
    if state_action_pair not in reward_counts:
        reward_counts[state_action_pair] = {}
    if next_state not in reward_counts[state_action_pair]:
        reward_counts[state_action_pair][next_state] = reward
    else:
        reward_counts[state_action_pair][next_state] += reward

    state = next_state

    if terminated or truncated:
        break

# Estimate transition function T(s'|s, a)
transition_probabilities = {}
for state_action_pair, transitions in transition_counts.items():
    total_transitions = sum(transitions.values())
    transition_probabilities[state_action_pair] = {next_state: count / total_transitions
                                                    for next_state, count in transitions.items()}

# Estimate reward function R(s, a, s')
reward_function = {}
for state_action_pair, rewards in reward_counts.items():
    total_rewards = sum(rewards.values())
    reward_function[state_action_pair] = {next_state: reward / count
                                           for next_state, reward in rewards.items()
                                           for next_state, count in transition_counts[state_action_pair].items()}

# Print estimated transition function and reward function
print("Estimated Transition Function:")
for state_action_pair, transitions in transition_probabilities.items():
    print(f"State-Action Pair: {state_action_pair}, Transitions: {transitions}")

print("\nEstimated Reward Function:")
for state_action_pair, rewards in reward_function.items():
    print(f"State-Action Pair: {state_action_pair}, Rewards: {rewards}")

def value_iteration(transition_probabilities, num_states, num_actions, gamma=0.99, max_iterations=1000, epsilon=1e-6):
    # Initialize value function
    V = np.zeros(num_states)

    for _ in range(max_iterations):
        prev_V = np.copy(V)
        for state in range(num_states):
            action_values = []
            for action in range(num_actions):
                if (state, action) in transition_probabilities:
                    next_state = list(transition_probabilities[(state, action)].keys())[0]
                    probability = transition_probabilities[(state, action)][next_state]
                    try:
                        action_value = probability * (gamma * prev_V[next_state])
                        action_values.append(action_value)
                    except:
                        pass
            if action_values:
                V[state] = np.max(action_values)

        if np.max(np.abs(V - prev_V)) < epsilon:
            break

    return V

# Number of states and actions
num_states = len(set(s for s, _ in transition_probabilities.keys()))
num_actions = env.action_space.n

# Call the value iteration function
optimal_value_function = value_iteration(transition_probabilities, num_states, num_actions)

# Print the optimal value function
print("Optimal Value Function:")
print(optimal_value_function)

def extract_policy(value_function, transition_probabilities, num_states, num_actions, gamma=0.99):
    policy = np.zeros(num_states, dtype=int)
    for state in range(num_states):
        action_values = np.zeros(num_actions)
        for action in range(num_actions):
            if (state, action) in transition_probabilities:
                for next_state in transition_probabilities[(state, action)]:
                    try:
                        probability = transition_probabilities[(state, action)][next_state]
                        reward = reward_function[(state, action)][next_state]
                        action_values[action] += probability * (reward + gamma * value_function[next_state])
                    except:
                        pass
        # Choose the action that maximizes the expected return
        policy[state] = np.argmax(action_values)
    return policy

# Extract optimal policy
optimal_policy = extract_policy(optimal_value_function, transition_probabilities, num_states, num_actions)

# Print the optimal policy
print("Optimal Policy:")
print(optimal_policy)

# Reset the environment
observation, info = env.reset()

# Act according to the optimal policy
for _ in range(50):
    # Select action according to the optimal policy
    try:
        action = optimal_policy[observation]
    except:
        pass    
    
    # Take action and observe the next state and reward
    observation, reward, terminated, truncated, info = env.step(action)
    
    # Render the environment
    env.render()

    if terminated or truncated:
        print("Episode finished")
        break

env.close()
