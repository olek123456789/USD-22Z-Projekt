import numpy as np
import torch


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, buffer_size):
        self.buffer_size = buffer_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((buffer_size, state_dim))
        self.action = np.zeros((buffer_size, action_dim))
        self.next_state = np.zeros((buffer_size, state_dim))
        self.reward = np.zeros((buffer_size, 1))
        self.not_done = np.zeros((buffer_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Add specific state to the buffer
    def add(self, state, action, next_state, reward, done):
        if (len(state) == 2):
            state_to_pass = state[0]
        else:
            state_to_pass = state          
        
        self.state[self.ptr] = state_to_pass
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    #Generate a sample to smooth the policy
    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        
        return (
        	torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)