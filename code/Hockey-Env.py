#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic("load_ext", "autoreload")
get_ipython().run_line_magic("autoreload", "")
import numpy as np
import laserhockey.hockey_env as h_env
import gym
from importlib import reload
import time
import torch


# In[2]:


np.set_printoptions(suppress=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# In[3]:


env_list = []


def clear_envs():
    for env in env_list:
        env.close()


def add_env(env):
    env_list.append(env)


# # Normal Game Play

# In[4]:


env = h_env.HockeyEnv()
add_env(env)


# have a look at the initialization condition: alternating who starts and are random in puck position

# In[5]:


obs = env.reset()
obs_agent2 = env.obs_agent_two()
_ = env.render()


# In[6]:


# env.close()
clear_envs()


# one episode with random agents

# In[7]:


obs = env.reset()
obs_agent2 = env.obs_agent_two()

for _ in range(600):
    env.render()
    a1 = np.random.uniform(-1, 1, 4)
    a2 = np.random.uniform(-1, 1, 4)
    obs, r, d, info = env.step(np.hstack([a1, a2]))
    obs_agent2 = env.obs_agent_two()
    if d:
        break


# Without rendering, it runs much faster

# "info" dict contains useful proxy rewards and winning information

# In[8]:


info


# Winner == 0: draw
#
# Winner == 1: you (left player)
#
# Winner == -1: opponent wins (right player)

# In[9]:


env.close()


# # Train Shooting

# In[10]:


env = h_env.HockeyEnv(mode=h_env.HockeyEnv.TRAIN_SHOOTING)
add_env(env)


# In[11]:


o = env.reset()
_ = env.render()

for _ in range(50):
    env.render()
    a1 = [1, 0, 0, 1]  # np.random.uniform(-1,1,4)
    a2 = [0, 0.0, 0, 0]
    obs, r, d, info = env.step(np.hstack([a1, a2]))
    obs_agent2 = env.obs_agent_two()
    if d:
        break


# In[ ]:


# In[12]:


env.close()


# # Train DEFENDING

# In[13]:


env = h_env.HockeyEnv(mode=h_env.HockeyEnv.TRAIN_DEFENSE)
add_env(env)


# In[14]:


o = env.reset()
_ = env.render()

for _ in range(60):
    env.render()
    a1 = [0.1, 0, 0, 1]  # np.random.uniform(-1,1,3)
    a2 = [0, 0.0, 0, 0]
    obs, r, d, info = env.step(np.hstack([a1, a2]))
    # print(r)
    obs_agent2 = env.obs_agent_two()
    if d:
        break


# In[15]:


env.close()


# # Using discrete actions

# In[16]:


import random


# In[17]:


env = h_env.HockeyEnv(mode=h_env.HockeyEnv.TRAIN_SHOOTING)
add_env(env)


# In[18]:


env.reset()
for _ in range(200):
    env.render()
    a1_discrete = random.randint(0, 7)

    a1 = env.discrete_to_continous_action(a1_discrete)
    # print(a1)
    a2 = [0, 0.0, 0, 0]
    obs, r, d, info = env.step(np.hstack([a1, a2]))
    obs_agent2 = env.obs_agent_two()
    if d:
        break


# In[19]:


env.close()


# # Hand-crafted Opponent

# In[20]:


env = h_env.HockeyEnv()
add_env(env)


# In[21]:


o = env.reset()
_ = env.render()
player1 = h_env.BasicOpponent(weak=False)
player2 = h_env.BasicOpponent()


# In[22]:


obs_buffer = []
reward_buffer = []
obs = env.reset()
obs_agent2 = env.obs_agent_two()
for _ in range(300):
    env.render()
    a1 = player1.act(obs)
    a2 = player2.act(obs_agent2)
    obs, r, d, info = env.step(np.hstack([a1, a2]))
    obs_buffer.append(obs)
    reward_buffer.append(r)
    obs_agent2 = env.obs_agent_two()
    if d:
        break
obs_buffer = np.asarray(obs_buffer)
reward_buffer = np.asarray(reward_buffer)


# In[23]:


np.mean(obs_buffer, axis=0)


# In[24]:


env.close()


# In[25]:


np.std(obs_buffer, axis=0)


# If you want to use a fixed observation scaling, this might be a reasonable choice

# In[26]:


scaling = [
    1.0,
    1.0,
    0.5,
    4.0,
    4.0,
    4.0,
    1.0,
    1.0,
    0.5,
    4.0,
    4.0,
    4.0,
    2.0,
    2.0,
    10.0,
    10.0,
    4,
    0,
    4,
    0,
]


# In[27]:


import pylab as plt


# In[28]:


plt.plot(obs_buffer[:, 2])
plt.plot(obs_buffer[:, 8])


# In[29]:


plt.plot(obs_buffer[:, 12])


# In[30]:


plt.plot(reward_buffer[:])


# In[31]:


np.sum(reward_buffer)


# In[32]:


env.close()


# # Human Opponent

# In[33]:


env = h_env.HockeyEnv()
add_env(env)


# In[ ]:


# In[34]:


player1 = h_env.HumanOpponent(env=env, player=1)
player2 = h_env.BasicOpponent()


# In[35]:


o = env.reset()
env.render()
# time.sleep(1)
obs_agent2 = env.obs_agent_two()
for _ in range(300):
    # time.sleep(0.2)
    env.render()
    a1 = player1.act(obs)
    a2 = player2.act(obs_agent2)
    # print(a2)
    obs, r, d, info = env.step(np.hstack([a1, a2]))
    obs_agent2 = env.obs_agent_two()
    if d:
        break


# In[36]:


env.close()


# In[ ]:


# In[ ]:


clear_envs()


# In[ ]:


# Plotting


# In[46]:


import matplotlib.pyplot as plt

plt.plot(losses)


# In[47]:


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


# In[48]:


stats_np = np.asarray(stats)
plt.plot(stats_np[:, 1], label="return")
plt.plot(running_mean(stats_np[:, 1], 20), label="smoothed-return")
plt.legend()


# In[49]:


########################### Initial Training Code


# In[ ]:


env = h_env.HockeyEnv()
add_env(env)
player2 = h_env.BasicOpponent(weak=True)

ob = env.reset()

use_target = True
target_update = 20
ac_space = env.action_space
o_space = env.observation_space

print(ac_space)
print(o_space)
print(o_space.shape[0])
q_agent = DQNAgent(
    o_space,
    ac_space,
    discount=0.95,
    eps=0.2,
    use_target_net=use_target,
    update_target_every=target_update,
    buffer_size=int(1e6),
)


# In[ ]:


q_agent.Q.predict(ob)


# In[ ]:


max_episodes = 600
max_steps = 1000
fps = 50
show = True
stats = []
losses = []

for i in range(max_episodes):
    total_reward = 0
    ob = env.reset()
    for t in range(max_steps):

        done = False
        a = q_agent.act(ob)
        # TODO: change action to continuous
        a1 = env.discrete_to_continous_action(a)

        a2 = player2.act(ob)

        ob_new, r, d, info = env.step(np.hstack([a1, a2]))
        total_reward += r
        q_agent.store_transition((ob, a, r, ob_new, d))
        ob = ob_new
        if show:
            time.sleep(1.0 / fps)
            env.render(mode="human")
        if done:
            break
    losses.extend(q_agent.train(32))
    stats.append([i, total_reward, t + 1])

    if (i - 1) % 20 == 0:
        print("{}: Done after {} steps. Reward: {}".format(i, t + 1, total_reward))


# In[ ]:


env.reset()
env.render()
obs_agent2 = env.obs_agent_two()
for _ in range(300):
    env.render()
    a1 = q_agent.act(obs)
    a1 = env.discrete_to_continous_action(a1)
    a2 = player2.act(obs_agent2)
    # print(a2)
    obs, r, d, info = env.step(np.hstack([a1, a2]))
    obs_agent2 = env.obs_agent_two()
    if d:
        break


# In[ ]:


env.close()
