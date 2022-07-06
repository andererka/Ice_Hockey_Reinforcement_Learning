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

import feedforward_ACTOR_CRITIC as AC

# import DDPG.feedforward_ACTOR_CRITIC_LSTM as AC
# import DDPG.feedforward_ACTOR_CRITIC_ReLU as AC


class PPO:
    def __init__(
        self,
        state_dim,
        action_dim,
        action_std,
        n_latent_var,
        n_latent_var2,
        lr,
        betas,
        gamma,
        K_epochs,
        eps_clip,
        device,
        batch_size=64,
        t_max=10,
    ):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.device = device

        self.policy = AC.ActorCritic(
            state_dim, action_dim, n_latent_var, n_latent_var2, action_std
        ).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        self.policy_old = AC.ActorCritic(
            state_dim, action_dim, n_latent_var, n_latent_var2, action_std
        ).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.policy_for_self_play = AC.ActorCritic(
            state_dim, action_dim, n_latent_var, n_latent_var2, action_std
        ).to(device)

        self.policy_for_self_play.load_state_dict(self.policy_old.state_dict())

        self.MseLoss = nn.MSELoss()
        self.batch_size = batch_size
        self.t_max = t_max

        self.loss = 0

    def update(self, memory):

        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(
            reversed(memory.rewards), reversed(memory.is_terminals)
        ):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # convert list to tensor
        old_states = torch.squeeze(
            torch.stack(memory.states).to(self.device), 1
        ).detach()
        old_actions = torch.squeeze(
            torch.stack(memory.actions).to(self.device), 1
        ).detach()
        old_logprobs = (
            torch.squeeze(torch.stack(memory.logprobs), 1).to(self.device).detach()
        )

        # Calculate advantage:
        _, state_values, _ = self.policy.evaluate(old_states, old_actions)
        advantages = rewards - state_values.detach()

        ##normalizing advantages?
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)

        # print('_',i)
        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):

            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(
                old_states, old_actions
            )

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            surr1 = ratios * advantages
            surr2 = (
                torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            )

            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(
                state_values, rewards
            )  # - 0.01*dist_entropy
            # print('loss', loss)
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward(retain_graph=True)

            self.optimizer.step()
            self.loss = loss.mean().detach().numpy()
        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())

        return self.loss

    def random_action(self, state, memory):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.policy_old.random_act(state, memory).cpu().data.numpy().flatten()

    def select_action(self, state, memory):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.policy_old.act(state, memory).cpu().data.numpy().flatten()

    def select_action_self_play(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return (
            self.policy_for_self_play.act_without_mem(state)
            .cpu()
            .data.numpy()
            .flatten()
        )
