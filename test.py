import gym
import gym_carla

import tqdm
import time


env = gym.make('Carla-v0')
obs = env.reset()
start = time.time()

for i in tqdm.tqdm(range(2000)):
    action = [0.01, 0.30, 0.0]
    obs, rewards, done, info = env.step(action)
    if i % 100 == 0:
        print(env._wrapper.ticks / (time.time() - start))
