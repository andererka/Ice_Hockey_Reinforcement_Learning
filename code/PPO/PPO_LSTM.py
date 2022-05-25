import numpy as np
import torch
import math 

from gym import spaces

import torch
import collections
import random
from torch.distributions import Categorical
import optparse
import torch.nn as nn


import PPO.feedforward_ACTOR_CRITIC_LSTM as AC



class PPO:
    def __init__(self, state_dim, action_dim, action_std, n_latent_var, lr, betas, gamma, K_epochs, eps_clip, device, batch_size=64, t_max=10):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.device = device

        self.policy = AC.ActorCritic(state_dim, action_dim, n_latent_var, action_std).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        self.policy_old = AC.ActorCritic(state_dim, action_dim, n_latent_var, action_std).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()
        self.batch_size= batch_size
        self.t_max = t_max
        
        self.loss = 0


    def update(self, memory):

        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
            
        # Normalizing the rewards:
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        
        # convert list to tensor
        old_states = torch.squeeze(torch.stack(memory.states).to(self.device), 1).detach()
        old_actions = torch.squeeze(torch.stack(memory.actions).to(self.device), 1).detach()
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs), 1).to(self.device).detach()
        
        h_in = memory.h_in[0]
        h_out = memory.h_out[0]

        (h1_in, h2_in) = h_in
        
        (h1_out, h2_out) = h_out

        first_hidden  = (h1_in.detach(), h2_in.detach())
        second_hidden = (h1_out.detach(), h2_out.detach())
        # Calculate advantage:

        _, state_values, _, _ = self.policy.evaluate(old_states, old_actions, second_hidden)
        advantages = rewards - state_values.detach()

        ##normalizing advantages?
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)

        #print('_',i)
        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):

            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy, _ = self.policy.evaluate(old_states, old_actions, first_hidden)

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())


            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) #- 0.01*dist_entropy
            #print('loss', loss)
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward(retain_graph=True)


            self.optimizer.step()
            self.loss =  loss.mean().detach().numpy()
        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        return self.loss
    
    def random_action(self):
        action = np.random.uniform(-1., 1., 8)
        return action
    
    def select_action(self, state, memory, hidden):
        state = torch.FloatTensor(state.reshape(1, -1))
  
        action, hidden = self.policy_old.act(state, memory, hidden)
        return action.cpu().data.numpy().flatten(), hidden
    


   

