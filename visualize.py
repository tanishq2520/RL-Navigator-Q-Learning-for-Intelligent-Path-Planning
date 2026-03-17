import numpy as np
import matplotlib.pyplot as plt
from grid_env import GridWorldEnv


env = GridWorldEnv()

# Load trained Q-table
Q = np.load("q_table.npy")


def state_to_index(state):
    row, col = state
    return row * env.grid_size + col


state, _ = env.reset()
state_index = state_to_index(state)

path = []

done = False

while not done:

    path.append(tuple(state))

    action = np.argmax(Q[state_index])

    next_state, reward, terminated, truncated, _ = env.step(action)

    state_index = state_to_index(next_state)

    state = next_state

    done = terminated or truncated

path.append(tuple(state))

grid = np.zeros((env.grid_size, env.grid_size))

for r, c in env.obstacles:
    grid[r, c] = -1

for r, c in path:
    grid[r, c] = 1

gr, gc = env.goal
grid[gr, gc] = 2


plt.imshow(grid)
plt.title("Learned Path")
plt.colorbar()
plt.show()