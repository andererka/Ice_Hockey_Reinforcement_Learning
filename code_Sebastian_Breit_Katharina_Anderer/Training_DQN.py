import numpy as np
import laserhockey.hockey_env as h_env
import gym
import torch
import time
from pathlib import Path

env_list = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def clear_envs():
    for env in env_list:
        env.close()


def add_env(env):
    env_list.append(env)


# env = h_env.HockeyEnv(mode=h_env.HockeyEnv.TRAIN_SHOOTING)
# add_env(env)

# obs = env.reset()
# obs_agent2 = env.obs_agent_two()
# _ = env.render()

# In[4]:


# player1 = h_env.BasicOpponent(weak=False)
# player2 = h_env.HumanOpponent(env=env, player=1)
# clear_envs()

# In[5]:


# How to step & get obs
# obs, r, d, info = env.step(np.hstack([a1,a2]))
# obs_agent2 = env.obs_agent_two()


import DDPG.DDPG as DDPG

#env_name = 'Pendulum-v0'
#env = gym.make(env_name)
env = h_env.HockeyEnv(mode=0)
env_name = 'HockeyEnv'
#ac_space = env.action_space
#o_space = env.observation_space

o_space = env.observation_space
ac_space = env.action_space
print(ac_space)
print(o_space)
print(list(zip(env.observation_space.low, env.observation_space.high)))

use_target = True
ddpg = DDPG.DDPG(o_space.shape[0], ac_space.shape[0])
print("DDPG with state dim: {} and action dim {}".format(o_space.shape[0], ac_space.shape[0]))

import matplotlib.pyplot as plt


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def subplot(fig, R, S, P, Q):
    r = list(zip(*R))
    s = list(zip(*S))
    p = list(zip(*P))
    q = list(zip(*Q))
    if fig != None:
        plt.close(fig)
    # clear_output(wait=True)

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))

    ax[0, 0].plot(list(r[1]), list(r[0]), 'r')  # row=0, col=0
    ax[0, 1].plot(list(s[1]), list(s[0]), 'k')  # row=0, col=1

    ax[1, 0].plot(list(p[1]), list(p[0]), 'b')  # row=1, col=0
    ax[1, 1].plot(list(q[1]), list(q[0]), 'g')  # row=1, col=1
    ax[0, 0].title.set_text('Reward')
    ax[0, 1].title.set_text('Smoothed Reward')
    ax[1, 0].title.set_text('Policy loss')
    ax[1, 1].title.set_text('Q loss')
    plt.pause(0.02)
    return fig


plot_reward = []
plot_policy_loss = []
plot_value_loss = []
plot_smoothed_reward = []

max_episodes = 2000
max_steps = 200
fps = 50
show = False
stats = []
losses = []
random_explore_no = 300
total_steps = 0
current_fig = None
best_reward = -np.inf
best_episode = 0
model_dir = "saved_models"

for episode in range(max_episodes):
    episode_reward = 0
    ob = env.reset()
    value_loss = 0
    policy_loss = 0
    steps_per_episode = 0
    for ep_step in range(max_steps):
        a = None
        d = False
        if total_steps < random_explore_no:
            a = ddpg.random_action()

            ob_new, r, d, info = env.step(a)
            ob_new = np.squeeze(np.transpose(ob_new))  # TODO due to a bad env we need the transpose here
            ddpg.store_transition([ob, a, r, ob_new, d])
            episode_reward += r
        else:
            a = ddpg.select_action(ob)

            ob_new, r, d, info = env.step(a)
            ob_new = np.squeeze(np.transpose(ob_new))  # TODO due to a bad env we need the transpose here
            ddpg.store_transition([ob, a, r, ob_new, d])
            episode_reward += r

            value_loss, policy_loss = ddpg.update_policy()

        ob = ob_new
        steps_per_episode = ep_step
        if show:
            time.sleep(1.0 / fps)
            env.render(mode='human')
        if d: break
        total_steps += 1

    losses.append([value_loss, policy_loss])
    stats.append([episode, episode_reward, steps_per_episode + 1])

    try:
        plot_reward.append([episode_reward, episode + 1])
        plot_policy_loss.append([policy_loss, episode + 1])
        plot_value_loss.append([value_loss, episode + 1])
        smoothed_rewards = running_mean(np.asarray(stats)[:, 1], 20).tolist()
        if smoothed_rewards:
            plot_smoothed_reward.append([smoothed_rewards[-1], episode + 1])
        #
    except:
        continue

    if episode_reward > best_reward:
        model_save_path = model_dir + "/" + env_name
        Path(model_save_path).mkdir(parents=True, exist_ok=True)
        ddpg.save_model(model_save_path)
        best_reward = episode_reward
        best_episode = episode

    if ((episode > 0) and (episode % 25 == 0)):
        print("{}: Done after {} steps. Episode reward: {} \n Value-loss {}, policy-loss {}".format(episode,
                                                                                                    steps_per_episode + 1,
                                                                                                    episode_reward,
                                                                                                    value_loss,
                                                                                                    policy_loss))
        print("Best reward {} at episode {}".format(best_reward, best_episode))
        # print(np.mean(losses, 0)[0], np.mean(losses, 0)[1])
        current_fig = subplot(current_fig, plot_reward, plot_smoothed_reward, plot_policy_loss, plot_value_loss)

