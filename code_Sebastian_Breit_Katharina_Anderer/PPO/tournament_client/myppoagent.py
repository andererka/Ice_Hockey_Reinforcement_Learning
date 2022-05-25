import PPO as PPO
import numpy as np
import laserhockey.hockey_env as h_env
import gym
import torch


class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]



class MyPlayer():
    def __init__(self, ppo, memory):
        self.ppo = ppo
        self.memory = memory

    def act(self, obs, verbose=False):
        action = self.ppo.select_action(obs, self.memory)
        return action[:4]

