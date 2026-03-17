from grid_env import GridWorldEnv

env = GridWorldEnv()

obs, info = env.reset()

for _ in range(10):
    action = env.action_space.sample()

    obs, reward, terminated, truncated, info = env.step(action)

    env.render()

    if terminated:
        print("Episode ended")
        break



# ---------------------------------

# import gymnasium as gym

# # Initialise the environment
# env = gym.make("FrozenLake-v1", render_mode="human")

# # Reset the environment to generate the first observation
# observation, info = env.reset(seed=42)
# for _ in range(1000):
#     # this is where you would insert your policy
#     action = env.action_space.sample()

#     # step (transition) through the environment with the action
#     # receiving the next observation, reward and if the episode has terminated or truncated
#     observation, reward, terminated, truncated, info = env.step(action)

#     # If the episode has ended then we can reset to start a new episode
#     if terminated or truncated:
#         observation, info = env.reset()

# env.close()

# import gymnasium as gym
# import ale_py
# import time

# # 1. Register the Atari environments
# gym.register_envs(ale_py)

# # 2. Create the environment with render_mode="human" 
# # This is the most important part to see the window!
# env = gym.make("ALE/Solaris-v5", render_mode="human")

# # 3. Reset the environment
# obs, info = env.reset()

# # 4. Run a short loop so the window stays open
# for _ in range(500):
#     # Take a random action
#     action = env.action_space.sample()
#     obs, reward, terminated, truncated, info = env.step(action)
    
#     # Slow it down slightly so you can see it
#     time.sleep(0.01)

#     if terminated or truncated:
#         obs, info = env.reset()

# # 5. Close the window properly
# env.close()