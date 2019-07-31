import gym
import gym_carla

import tqdm
import time

N_HEROES = 8

env = gym.make('Carla-v0', n_heroes=N_HEROES)
env.reset(n_vehicles=200)

for i in tqdm.tqdm(range(2000)):
    action = list()
    for _ in range(N_HEROES):
        action.append((0.0, 0.25, 0.0))

    obs, rewards, done, info = env.step(action)
    env.render()
