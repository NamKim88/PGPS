import os
import torch
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
USE_CUDA = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
Device = torch.device("cuda" if USE_CUDA else "cpu")

class ReplayBuffer(object):
    def __init__(self, args, state_dim, action_dim, max_size=1e6):
        self.max_size = np.int(max_size)
        self.state_dim = np.int(state_dim)
        self.action_dim = np.int(action_dim)
        self.size = 0
        self.ptr = 0

        self.state = np.zeros((self.max_size, self.state_dim))
        self.action = np.zeros((self.max_size, self.action_dim))
        self.next_state = np.zeros((self.max_size, self.state_dim))
        self.reward = np.zeros((self.max_size, 1))
        self.not_done = np.zeros((self.max_size, 1))

    def clear(self):
        self.size = 0
        self.state = np.zeros((self.max_size, self.state_dim))
        self.action = np.zeros((self.max_size, self.action_dim))
        self.next_state = np.zeros((self.max_size, self.state_dim))
        self.reward = np.zeros((self.max_size, 1))
        self.not_done = np.zeros((self.max_size, 1))

    def add(self, state, action, next_state, reward, not_done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = not_done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        index = np.random.randint(0, self.size, size=batch_size)

        return (FloatTensor(self.state[index]).to(Device),
                FloatTensor(self.action[index]).to(Device),
                FloatTensor(self.next_state[index]).to(Device),
                FloatTensor(self.reward[index]).to(Device),
                FloatTensor(self.not_done[index]).to(Device))
