LR_ACTOR = 1e-3         # learning rate of the actor (1e-3 was fine at first, but ended up diverging)
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
TAU = 5e-3              # for soft update of target parameters (tried 1e-3)
OU_SIGMA = 0.5          # Ornstein-Uhlenbeck noise parameter, volatility
OU_THETA = 0.15         # Ornstein-Uhlenbeck noise parameter, speed of mean reversion

EPS_START = 10.0         # initial value for epsilon in noise decay process in Agent.act()

# EPS_EP_END = 300        # episode to end the noise decay process
# EPS_FINAL = 0           # final value for epsilon after decay

EPS_DECAY_MULTIPLICATIVE = 0.999

OU_SIGMA = 0.5          # Ornstein-Uhlenbeck noise parameter, volatility
OU_THETA = 0.15         # Ornstein-Uhlenbeck noise parameter, speed of mean reversion

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
LEARN_EVERY = 1         # learning timestep interval
LEARN_NUM = 10           # number of learning passes per call
GAMMA = 0.99            # discount factor

SEED = 301