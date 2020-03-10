import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic
from OUNoise import OUNoise
import hyperparameters as h

import torch
import torch.nn.functional as F
import torch.optim as optim


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, num_agents, random_seed, replay_buffer, device):
        """Initialize an Agent object.
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            num_agents (int): number of agents
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.seed = random.seed(random_seed)
        self.eps = h.EPS_START
#         self.eps_decay = 1/(h.EPS_EP_END*h.LEARN_NUM)  # set decay rate based on epsilon end target
        self.eps_decay = h.EPS_DECAY_MULTIPLICATIVE
        self.timestep = 0
        self.device = device

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(self.device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=h.LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed).to(self.device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=h.LR_CRITIC, weight_decay=h.WEIGHT_DECAY)

        # Noise process
        self.noise = OUNoise((num_agents, action_size))

        # Replay memory
        self.buffer = replay_buffer

    def step(self, state, action, reward, next_state, done, agent_number):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        self.timestep += 1
        
        # Save experience / reward, initializing the priority using reward as error
        priority = (abs(reward) + h.PRIORITY_EPS)**h.PRIORITY_ALPHA
        
        self.buffer.add(state, action, reward, next_state, done, priority)

    def act(self, states, add_noise):
        """Returns actions for both agents as per current policy, given their respective states."""
        states = torch.from_numpy(states).float().to(self.device)
        actions = np.zeros((self.num_agents, self.action_size))
        self.actor_local.eval()
        with torch.no_grad():
            # get action for each agent and concatenate them
            for agent_num, state in enumerate(states):
                action = self.actor_local(state).cpu().data.numpy()
                actions[agent_num, :] = action
        self.actor_local.train()
        # add noise to actions
        if add_noise:
            actions += self.eps * self.noise.sample()
        actions = np.clip(actions, -1, 1)
        return actions

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, agent_number):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done, experience indices) tuples
        """
        states, actions, rewards, next_states, dones, experience_idx = experiences

        # ---------------------------- update critic ---------------------------- #
        Q_expected, Q_targets = self.get_Q_loss(states, actions, rewards, next_states, dones, agent_number)      
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()
        
        # update priorities based on new Q values
        Q_expected = Q_expected.detach().numpy()
        Q_targets = Q_targets.detach().numpy()
        for i in range(len(experience_idx)):
            self.buffer.memory[experience_idx[i]]._replace(priority = (abs(Q_expected[i]-Q_targets[i])+h.PRIORITY_EPS)**h.PRIORITY_ALPHA)

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        # Construct action prediction vector relative to each agent
        if agent_number == 0:
            actions_pred = torch.cat((actions_pred, actions[:,2:]), dim=1)
        else:
            actions_pred = torch.cat((actions[:,:2], actions_pred), dim=1)
        # Compute actor loss
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, h.TAU)
        self.soft_update(self.actor_local, self.actor_target, h.TAU)

        # update noise decay parameter
#         self.eps -= self.eps_decay
#         self.eps = max(self.eps, h.EPS_FINAL)
        self.eps *= self.eps_decay
    
        self.noise.reset()
    
    def get_Q_loss(self, states, actions, rewards, next_states, dones, agent_number):
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        # Construct next actions vector relative to the agent
        if agent_number == 0:
            actions_next = torch.cat((actions_next, actions[:,2:]), dim=1)
        else:
            actions_next = torch.cat((actions[:,:2], actions_next), dim=1)
        # Compute Q targets for current states (y_i)
        Q_targets_next = self.critic_target(next_states, actions_next)
        Q_targets = rewards + (h.GAMMA * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        return Q_expected, Q_targets
        
    
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
          
                        