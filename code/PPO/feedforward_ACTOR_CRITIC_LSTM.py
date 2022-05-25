#parts adapted from: https://github.com/seungeunrho/minimalRL/blob/8c364c36d0c757e0341284cf6d7f0eb731215c85/ppo-lstm.py#L48

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical
from torch.distributions import MultivariateNormal




class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var, action_std):
        super(ActorCritic, self).__init__()

        # actor
        self.state_dim = state_dim
        self.n_latent_var = n_latent_var
        self.action_dim = action_dim

        self.cov_var = torch.full(size=(self.action_dim,), fill_value=0.1)
        self.cov_mat = torch.diag(self.cov_var)

        self.fc1   = nn.Linear(state_dim,64)
        self.lstm  = nn.LSTM(64,32)
        self.fc_pi = nn.Linear(32,8)
        self.fc_v  = nn.Linear(32,1)
    

        self.device = 'cpu'

        self.action_var = torch.full((action_dim,), action_std*action_std).to(self.device)


    def action_layer(self, x, hidden):
        x = torch.tanh(self.fc1(x))
        x = x.view(-1, 1, 64)
        x, lstm_hidden = self.lstm(x, hidden)
        x = self.fc_pi(x)
        prob = torch.tanh(x)
        return prob, lstm_hidden
    
    def value_layer(self, x, hidden):
        x = torch.tanh(self.fc1(x))
        x = x.view(-1, 1, 64)
        x, lstm_hidden = self.lstm(x, hidden)
        v = self.fc_v(x)
        return v


    def logprobs(self, state, hidden):
        state = torch.from_numpy(state).float().to(self.device)

        action_probs, hidden_new = self.action_layer(state, hidden)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_logprobs = dist.log_prob(action)
        

        return action_logprobs, hidden_new

    def act(self, state, memory, hidden):
        action_mean, hidden_new = self.action_layer(state, hidden)
        cov_mat = torch.diag(self.action_var).to(self.device)
        
        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        
        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)
        
        return action.detach(), hidden_new





    def evaluate(self, state, action, hidden):

        action_probs, hidden_new = self.action_layer(state, hidden)
        dist = MultivariateNormal(action_probs, self.cov_mat)
        log_probs = dist.log_prob(action)
        state_value = self.value_layer(state, hidden)
        dist_entropy = dist.entropy()

        return log_probs, torch.squeeze(state_value), dist_entropy, hidden_new

    def predict(self, x):
        with torch.no_grad():
            return self.forward(x)
