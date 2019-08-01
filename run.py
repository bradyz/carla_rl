import gym
import gym_carla

import tqdm

from replay_buffer import MultiReplayBuffer


N_HEROES = 4
CAPACITY = 10000
PORT = 3000
N_VEHICLES = 50
N_PEDESTRIANS = 25
BATCH_SIZE = 64


class DummyAgent(object):
    def __call__(self, states, infos=None):
        return [(0.15, 0.4, 0.0) for _ in states]


def main():
    env = gym.make('Carla-v0', n_heroes=N_HEROES, port=PORT)
    agent = DummyAgent()
    replay = MultiReplayBuffer(CAPACITY)

    for _ in tqdm.tqdm(range(10)):
        states = env.reset(n_vehicles=N_VEHICLES, n_pedestrians=N_PEDESTRIANS)

        for i in tqdm.tqdm(range(2000)):
            actions = agent(states)
            new_states, rewards, dones, infos = env.step(actions)

            replay.add(states, actions, rewards, new_states, dones)

            states = new_states

            if len(replay) > BATCH_SIZE:
                replay.sample(BATCH_SIZE)


if __name__ == '__main__':
    main()
