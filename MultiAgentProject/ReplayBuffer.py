import numpy as np
import random
import copy
from collections import namedtuple, deque

import torch
import torch.nn.functional as F
import torch.optim as optim

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed, device):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done", "priority"])
        self.seed = random.seed(seed)
        self.device = device

    def add(self, state, action, reward, next_state, done, priority):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done, priority)
        self.memory.append(e)
        
    def sample(self):
        """Randomly sample a batch of experiences from memory, weighted by priorities."""

        # get priorities
        priorities = [self.memory[i].priority for i in range(len(self))]

        # get sample numbers by priority
        cumsum_priorities = np.cumsum(priorities)
        stopping_values = [random.random()*sum(priorities) for i in range(self.batch_size)]
        stopping_values.sort()  
        # stopping values are where we pick the experience samples, sorting them (of size batch_size) is much faster than sorting the priorities, and having this sorted lets us go through the cumsum_priorities list just once

        experience_idx = []
        experiences = []
        for i in range(len(cumsum_priorities)-1):
            if len(stopping_values) <= 0:
                break
            if stopping_values[0] < cumsum_priorities[i+1]:
                experience_idx.append(i)
                experiences.append(self.memory[i])
                stopping_values.pop(0)

        #         experiences = random.sample(self.memory, k=self.batch_size)      # uniform sampling
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
     
        return (states, actions, rewards, next_states, dones, experience_idx)

    
    
    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
