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

        self.action_layer = nn.Sequential(
            nn.Linear(state_dim, n_latent_var),
            nn.ReLU(),
            nn.Linear(n_latent_var, n_latent_var),
            nn.ReLU(),
            nn.Linear(n_latent_var, action_dim),
            nn.ReLU()
            # nn.Softmax(dim=-1)
        )

        # critic
        self.value_layer = nn.Sequential(
            nn.Linear(state_dim, n_latent_var),
            nn.ReLU(),
            nn.Linear(n_latent_var, n_latent_var),
            nn.ReLU(),
            nn.Linear(n_latent_var, 1),
        )

        self.device = "cpu"

        self.action_var = torch.full((action_dim,), action_std * action_std).to(
            self.device
        )

    def logprobs(self, state):
        state = torch.from_numpy(state).float().to(self.device)

        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_logprobs = dist.log_prob(action)

        return action_logprobs

    def act(self, state, memory):
        action_mean = self.action_layer(state)
        cov_mat = torch.diag(self.action_var).to(self.device)

        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)

        return action.detach()

    def evaluate(self, state, action):

        action_probs = self.action_layer(state)
        dist = MultivariateNormal(action_probs, self.cov_mat)
        log_probs = dist.log_prob(action)
        state_value = self.value_layer(state)
        dist_entropy = dist.entropy()

        return log_probs, torch.squeeze(state_value), dist_entropy

    def predict(self, x):
        with torch.no_grad():
            return self.forward(x)
