import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt


class GridWorldEnv(gym.Env):

    def __init__(self):
        super(GridWorldEnv, self).__init__()

        # Grid size
        self.grid_size = 5

        # Define action space (4 possible moves)
        self.action_space = spaces.Discrete(4)

        # Define observation space (agent position)
        self.observation_space = spaces.Box(
            low=0,
            high=self.grid_size - 1,
            shape=(2,),
            dtype=np.int32
        )

        # Define the grid
        self.grid = np.zeros((self.grid_size, self.grid_size))

        # Obstacles
        self.obstacles = [(0,3),(1,1),(1,3),(2,1),(3,2),(3,3)]

        for obs in self.obstacles:
            self.grid[obs] = -1

        # Goal position
        self.goal = (4,4)
        self.grid[self.goal] = 2

        # Agent start
        self.agent_pos = [0,0]


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.agent_pos = [0,0]

        observation = np.array(self.agent_pos, dtype=np.int32)

        info = {}

        return observation, info


    def step(self, action):

        row, col = self.agent_pos

        if action == 0:  # UP
            row -= 1
        elif action == 1:  # DOWN
            row += 1
        elif action == 2:  # LEFT
            col -= 1
        elif action == 3:  # RIGHT
            col += 1

        row = np.clip(row, 0, self.grid_size - 1)
        col = np.clip(col, 0, self.grid_size - 1)

        new_pos = (row, col)

        reward = -1
        terminated = False

        if new_pos in self.obstacles:
            reward = -10
            terminated = True

        elif new_pos == self.goal:
            reward = 10
            terminated = True

        self.agent_pos = [row, col]

        observation = np.array(self.agent_pos, dtype=np.int32)

        truncated = False
        info = {}

        return observation, reward, terminated, truncated, info


    
    def render(self):

        grid = np.zeros((self.grid_size, self.grid_size))

        # obstacles
        for r, c in self.obstacles:
            grid[r, c] = -1

        # goal
        gr, gc = self.goal
        grid[gr, gc] = 2

        # agent
        r, c = self.agent_pos
        grid[r, c] = 1

        plt.clf()
        plt.imshow(grid)
        plt.title("RL Path Planning Agent")

        plt.xticks(np.arange(self.grid_size))
        plt.yticks(np.arange(self.grid_size))
        plt.grid(True)

        plt.pause(0.4)

