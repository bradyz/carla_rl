import argparse

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
OBSERVATION_SHAPE = [192, 192, 7]
N_ACTIONS = 3


class DummyAgent(object):
    def __call__(self, states, infos=None):
        return [(0.15, 0.4, 0.0) for _ in states]


def main(args):
    env = gym.make('Carla-v0', n_heroes=N_HEROES, port=PORT)
    agent = DummyAgent()
    replay = MultiReplayBuffer(CAPACITY)

    from sac import SAC

    updates = 0
    trainer = SAC(OBSERVATION_SHAPE, N_ACTIONS, args)

    for _ in tqdm.tqdm(range(10)):
        states = env.reset(n_vehicles=N_VEHICLES, n_pedestrians=N_PEDESTRIANS)

        for i in tqdm.tqdm(range(2000)):
            actions = agent(states)
            new_states, rewards, dones, infos = env.step(actions)

            replay.add(states, actions, rewards, new_states, dones)

            states = new_states

            if len(replay) > BATCH_SIZE:
                trainer.update_parameters(replay, args.batch_size, updates)
                updates += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
    parser.add_argument('--policy', default="Gaussian")
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.005, help='target smoothing coefficient')
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--alpha', type=float, default=0.2, help='entropy:reward ratio')
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=False)
    parser.add_argument('--seed', type=int, default=456)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_steps', type=int, default=10000)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--target_update_interval', type=int, default=1)

    args = parser.parse_args()

    main(args)
