import numpy as np


OBSERVATION_SHAPE = [192, 192, 7]
ACTION_SHAPE = [3]


class MultiReplayBuffer(object):
    def __init__(self, capacity):
        self.states = np.zeros([capacity] + OBSERVATION_SHAPE, dtype=np.float32)
        self.actions = np.zeros([capacity] + ACTION_SHAPE, dtype=np.float32)
        self.rewards = np.zeros([capacity], dtype=np.int32)
        self.new_states = np.zeros([capacity] + OBSERVATION_SHAPE, dtype=np.float32)
        self.dones = np.zeros([capacity] + OBSERVATION_SHAPE, dtype=np.int32)

        self.valid = list()
        self.spot = 0
        self.capacity = capacity

    def add(self, states, actions, rewards, new_states, dones):
        for i in range(len(states)):
            self.states[self.spot] = states[i]
            self.actions[self.spot] = actions[i]
            self.rewards[self.spot] = rewards[i]
            self.new_states[self.spot] = new_states[i]

            self.valid.append(self.spot)
            self.spot = (self.spot + 1) % self.capacity

    def sample(self, batch_size):
        spots = np.random.choice(self.valid, batch_size)

        states = self.states[spots]
        actions = self.actions[spots]
        rewards = self.rewards[spots]
        new_states = self.new_states[spots]
        dones = self.dones[spots]

        return states, actions, rewards, new_states, dones

    def __len__(self):
        return len(self.valid)