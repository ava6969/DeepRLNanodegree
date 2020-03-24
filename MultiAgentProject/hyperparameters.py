BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128        # minibatch size

LR_ACTOR = 1e-3         # learning rate of the actor
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay

LEARN_EVERY = 1         # learning timestep interval
LEARN_PASSES = 5        # number of learning passes

GAMMA = 0.99            # discount factor
TAU = 1e-2              # for soft update of target parameters

OU_SIGMA = 0.5          # Ornstein-Uhlenbeck noise parameter, volatility
OU_THETA = 0.15         # Ornstein-Uhlenbeck noise parameter, speed of mean reversion

EPS_START = 6.0         # initial value for epsilon in noise decay process in Agent.act()
EPS_EP_END = 350        # episode to end the noise decay process
EPS_FINAL = 0           # final value for epsilon after decay

PRIORITY_EPS = 0.01     # small factor to ensure that no experience has zero sample probability
PRIORITY_ALPHA = 0.5    # how much to prioritize replay of high-error experiences