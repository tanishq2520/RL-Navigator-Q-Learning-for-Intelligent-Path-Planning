import numpy as np
import matplotlib.pyplot as plt
from grid_env import GridWorldEnv


# Create environment
env = GridWorldEnv()

# State and action sizes
state_size = env.grid_size * env.grid_size
action_size = env.action_space.n

# Initialize Q-table
Q = np.zeros((state_size, action_size))

# Hyperparameters
learning_rate = 0.1
discount_factor = 0.95

epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01

episodes = 2000

# Store rewards for analysis
episode_rewards = []


# Convert state (row,col) → index
def state_to_index(state):
    row, col = state
    return row * env.grid_size + col


# =========================
# TRAINING LOOP
# =========================
for episode in range(episodes):

    state, _ = env.reset()
    state_index = state_to_index(state)

    done = False
    total_reward = 0

    while not done:

        # Exploration vs Exploitation
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state_index])

        # Take action
        next_state, reward, terminated, truncated, _ = env.step(action)

        next_state_index = state_to_index(next_state)

        # Best future Q-value
        best_next_q = np.max(Q[next_state_index])

        # Q-learning update
        Q[state_index, action] = Q[state_index, action] + learning_rate * (
            reward + discount_factor * best_next_q - Q[state_index, action]
        )

        state_index = next_state_index

        total_reward += reward

        done = terminated or truncated

    # Store reward
    episode_rewards.append(total_reward)

    # Decay exploration
    epsilon = max(epsilon_min, epsilon * epsilon_decay)


print("Training Finished")

# Save Q-table
np.save("q_table.npy", Q)


# =========================
# TEST LEARNED POLICY
# =========================

print("\nTesting learned policy\n")

state, _ = env.reset()
state_index = state_to_index(state)

done = False

env.render()

while not done:

    action = np.argmax(Q[state_index])

    next_state, reward, terminated, truncated, _ = env.step(action)

    env.render()

    state_index = state_to_index(next_state)

    done = terminated or truncated

    if terminated:
        if reward > 0:
            print("Goal reached!")
        else:
            print("Hit obstacle!")


# =========================
# PLOT TRAINING PERFORMANCE
# =========================

plt.plot(episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Q-learning Training Convergence")
plt.show()