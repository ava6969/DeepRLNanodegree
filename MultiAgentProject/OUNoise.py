import numpy as np
import random
import copy
from collections import namedtuple, deque
import hyperparameters as h

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, mu=0.0):
        """Initialize parameters and noise process.
        Params
        ======
            mu (float)    : long-running mean
            theta (float) : speed of mean reversion
            sigma (float) : volatility parameter
        """
        self.mu = mu * np.ones(size)
        self.theta = h.OU_THETA
        self.sigma = h.OU_SIGMA
        self.seed = random.seed(h.SEED)
        self.size = size
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
    
        return self.state