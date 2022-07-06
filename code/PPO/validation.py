import os

# print(os.getcwd())
# os.chdir('/home/kathi/Dokumente/ReIn_Course/project/laser-hockey-env-master')
import numpy as np
import laserhockey.hockey_env as h_env
import gym
import torch
import time
import gc
import matplotlib.pyplot as plt

# from pathlib import Path


render = False

max_episodes = 3000  # max training episodes
max_timesteps = 250  # max timesteps in one episode
n_latent_var = 64  # number of variables in hidden layer
n_latent_var2 = 64

lr = 0.0001
betas = (0.9, 0.999)
gamma = 0.99  # discount factor
K_epochs = 80  # update policy for K epochs
eps_clip = 0.2  # clip parameter for PPO


import math

import PPO.PPO as PPO


env = h_env.HockeyEnv(mode=0)
env_name = "HockeyEnv"

o_space = env.observation_space
ac_space = env.action_space


action_std = 0.01

ppo = PPO.PPO(
    o_space.shape[0],
    ac_space.shape[0],
    action_std,
    n_latent_var,
    n_latent_var2,
    lr,
    betas,
    gamma,
    K_epochs,
    eps_clip,
    t_max=5,
    device="cpu",
)

###testing the performance of the 4 best versions of the PPO algorithm


##40-70% win rates
ppo.policy.load_state_dict(
    torch.load(
        "/home/kathi/Dokumente/ReIn_Course/project/laser-hockey-env-master/PPO/results/15-00:29:49/best_weights/PPO_HockeyEnv-eps03-discount099-lr00001.pth"
    )
)
ppo.policy_old.load_state_dict(
    torch.load(
        "/home/kathi/Dokumente/ReIn_Course/project/laser-hockey-env-master/PPO/results/15-00:29:49/best_weights/PPO_HockeyEnv-eps03-discount099-lr00001.pth"
    )
)

##40-50% win rates
# ppo.policy.load_state_dict(torch.load('/home/kathi/Dokumente/ReIn_Course/project/laser-hockey-env-master/PPO/results/15-09:22:16/best_weights/PPO_HockeyEnv-eps02-discount099-lr00001.pth'))
# ppo.policy_old.load_state_dict(torch.load('/home/kathi/Dokumente/ReIn_Course/project/laser-hockey-env-master/PPO/results/15-09:22:16/best_weights/PPO_HockeyEnv-eps02-discount099-lr00001.pth'))


##38-70% win rates
# ppo.policy_old.load_state_dict(torch.load('//home/kathi/Dokumente/ReIn_Course/project/laser-hockey-env-master/PPO/results/09-17_without closeness_to_puck&opponent_goal/PPO_HockeyEnv-eps02-discount099-lr0001-random_choice85.pth'))
# ppo.policy.load_state_dict(torch.load('//home/kathi/Dokumente/ReIn_Course/project/laser-hockey-env-master/PPO/results/09-17_without closeness_to_puck&opponent_goal/PPO_HockeyEnv-eps02-discount099-lr0001-random_choice85.pth'))

## ~50% win rate
# ppo.policy.load_state_dict(torch.load('/home/kathi/Dokumente/ReIn_Course/project/laser-hockey-env-master/PPO/results/12-13:47:39/PPO_HockeyEnv-eps02-discount099-lr00001-action_std04.pth'))
# ppo.policy_old.load_state_dict(torch.load('/home/kathi/Dokumente/ReIn_Course/project/laser-hockey-env-master/PPO/results/12-13:47:39/PPO_HockeyEnv-eps02-discount099-lr00001-action_std04.pth'))


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


memory = Memory()


###validation
if __name__ == "__main__":
    lost = 0
    won = 0
    tie = 0
    opponent = h_env.BasicOpponent(weak=False)
    # env.render()
    winner = 10
    obs_agent2 = env.obs_agent_two()
    for _ in range(1000):
        obs = env.reset()
        obs_agent2 = env.obs_agent_two()
        for _ in range(250):

            # time.sleep(0.05)
            # env.render()
            a1 = ppo.select_action(obs, memory)
            a2 = opponent.act(obs_agent2)
            obs, r, d, info = env.step(np.hstack([a1[:4], a2]))
            winner = info["winner"]
            if winner == 1:
                won += 1
            if winner == -1:
                lost += 1
            if winner == 0:
                tie += 1
            obs_agent2 = env.obs_agent_two()
            if d:
                break

    average_win_rate = won / (won + lost)
    print(average_win_rate)
    print(won)
    print(lost)
    print(tie)
