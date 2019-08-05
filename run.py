import argparse

import gym
import gym_carla

import tqdm
import numpy as np

from replay_buffer import MultiReplayBuffer


N_HEROES = 16
CAPACITY = 100000
PORT = 3000
N_VEHICLES = 50
N_PEDESTRIANS = 25
BATCH_SIZE = 128
OBSERVATION_SHAPE = [84, 84, 7]
N_ACTIONS = 3


def preprocess(states):
    import torch

    x = np.float32(states).transpose(0, 3, 1, 2)
    x = torch.from_numpy(x).cuda()

    return x


def main(args):
    env = gym.make('Carla-v0', n_heroes=N_HEROES, port=PORT)
    replay = MultiReplayBuffer(CAPACITY)

    from sac import SAC
    import torch
    import bz_utils as bzu

    bzu.log.init('log_v1')

    updates = 0
    trainer = SAC(OBSERVATION_SHAPE, N_ACTIONS, args)
    agent = trainer.policy
    # agent.load_state_dict(torch.load('log/latest.t7'))

    for _ in tqdm.tqdm(range(1000)):
        totals = [0 for _ in range(N_HEROES)]
        finished = list()
        states = env.reset(n_vehicles=N_VEHICLES, n_pedestrians=N_PEDESTRIANS)

        for i in tqdm.tqdm(range(1000), desc='Experiences'):
            _, _, actions = agent.sample(preprocess(states))
            actions = actions.detach().cpu().numpy()
            new_states, rewards, dones, infos = env.step(actions)

            for j in range(N_HEROES):
                totals[j] += rewards[j]

                if dones[j]:
                    finished.append(totals[j])
                    totals[j] = 0

            # env.render()
            replay.add(states, actions, rewards, new_states, dones)

            states = new_states

        for j in range(N_HEROES):
            totals[j] += rewards[j]
            finished.append(totals[j])

        bzu.log.scalar(is_train=True, **{'cumulative': np.mean(finished)})

        for i in tqdm.tqdm(range(1000), desc='Batch'):
            loss_q1, loss_q2, p_loss, a_loss, a_tlog = trainer.update_parameters(replay, args.batch_size, updates)
            scalars = {
                    'loss_q1': loss_q1,
                    'loss_q2': loss_q2,
                    'p_loss': p_loss,
                    'a_loss': a_loss,
                    'a_tlog': a_tlog,
                    }
            bzu.log.scalar(is_train=True, **scalars)
            updates += 1

        bzu.log.end_epoch(agent)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
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
