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

env_list = []

##hyperparameters

render = False
solved_reward = 430  # stop training if avg_reward > solved_reward
log_interval = 20  # print avg reward in the interval

max_timesteps = 250  # max timesteps in one episode
n_latent_var = 64  # number of variables in hidden layer
n_latent_var2 = 64
update_timestep = 4000  # update policy every n timesteps (was set to 2000)
lr = 0.0001
betas = (0.9, 0.999)
gamma = 0.99  # discount factor
K_epochs = 80  # update policy for K epochs
eps_clip = 0.2  # clip parameter for PPO
random_seed = 20


import math

import PPO.PPO as PPO


env = h_env.HockeyEnv(mode=0)
env_name = "HockeyEnv"

o_space = env.observation_space
ac_space = env.action_space


action_std = 0.3


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
# ppo2 = PPO.PPO(o_space.shape[0], ac_space.shape[0] , action_std, n_latent_var, n_latent_var2, lr, betas, gamma, K_epochs, eps_clip, t_max= 5, device='cpu' )

print(
    "PPO with state dim: {} and action dim {}".format(
        o_space.shape[0], ac_space.shape[0]
    )
)


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def subplot(fig, R, S, P, Q, path):
    plt.clf()
    r = list(zip(*R))
    s = list(zip(*S))
    p = list(zip(*P))
    q = list(zip(*Q))
    if fig != None:
        plt.close(fig)
    # clear_output(wait=True)

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))

    ax[0, 0].plot(list(r[1]), list(r[0]), "r")  # row=0, col=0

    ax[0, 1].plot(list(s[1]), list(s[0]), "k")  # row=0, col=1

    ax[1, 0].plot(list(p[1]), list(p[0]), "b")  # row=1, col=0

    ax[1, 1].plot(list(q[1]), list(q[0]), "g")  # row=1, col=1
    ax[0, 0].title.set_text("Reward")
    ax[0, 1].title.set_text("Smoothed Reward")
    ax[1, 0].title.set_text("Loss")
    ax[1, 1].title.set_text("Smoothed Loss")

    # plt.pause(0.02)
    fig.savefig(path)
    plt.close()
    plt.close(fig)
    fig.clear()
    fig.clf()
    gc.collect()
    # return fig


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


def make_dir(path):
    os.mkdir(path)


### loading already trained versions of the agent:

# ppo.policy_for_self_play.load_state_dict(torch.load('/home/kathi/Dokumente/ReIn_Course/project/laser-hockey-env-master/results/12-13:47:39/PPO_HockeyEnv-eps02-discount099-lr00001-action_std04.pth'))

# ppo.policy_for_self_play2.load_state_dict(torch.load('/home/kathi/Dokumente/ReIn_Course/project/laser-hockey-env-master/results/15-00:29:49/best_weights/PPO_HockeyEnv-eps03-discount099-lr00001.pth'))

# ppo.policy.load_state_dict(torch.load('/home/kathi/Dokumente/ReIn_Course/project/laser-hockey-env-master/results/15-00:29:49/best_weights/PPO_HockeyEnv-eps03-discount099-lr00001.pth'))
# ppo.policy_old.load_state_dict(torch.load('/home/kathi/Dokumente/ReIn_Course/project/laser-hockey-env-master/results/15-00:29:49/best_weights/PPO_HockeyEnv-eps03-discount099-lr00001.pth'))

# ppo.optimizer.load_state_dict(torch.load('/home/kathi/Dokumente/ReIn_Course/project/laser-hockey-env-master/results/15-00:29:49/best_optimizer/PPO_HockeyEnv-eps03-discount099-lr00001.pth'))

# ppo.policy.load_state_dict(torch.load('/home/kathi/Dokumente/ReIn_Course/project/laser-hockey-env-master/results/12-13:47:39/PPO_HockeyEnv-eps02-discount099-lr00001-action_std04.pth'))
# ppo.policy_old.load_state_dict(torch.load('/home/kathi/Dokumente/ReIn_Course/project/laser-hockey-env-master/results/12-13:47:39/PPO_HockeyEnv-eps02-discount099-lr00001-action_std04.pth'))

# ppo.optimizer.load_state_dict(torch.load('/home/kathi/Dokumente/ReIn_Course/project/laser-hockey-env-master/results/12-13:47:39/PPO_HockeyEnv-eps02-discount099-lr00001-action_std04optimizer.pth'))

# ppo.policy.load_state_dict(torch.load('/home/kathi/Dokumente/ReIn_Course/project/laser-hockey-env-master/results/15-09:22:16/best_weights/PPO_HockeyEnv-eps02-discount099-lr00001.pth'))
# ppo.policy_old.load_state_dict(torch.load('/home/kathi/Dokumente/ReIn_Course/project/laser-hockey-env-master/results/15-09:22:16/best_weights/PPO_HockeyEnv-eps02-discount099-lr00001.pth'))
# ppo.optimizer.load_state_dict(torch.load('/home/kathi/Dokumente/ReIn_Course/project/laser-hockey-env-master/results/15-09:22:16/best_optimizer/PPO_HockeyEnv-eps02-discount099-lr00001.pth'))

ppo.policy_old.load_state_dict(
    torch.load(
        "//home/kathi/Dokumente/ReIn_Course/project/laser-hockey-env-master/results/09-17_without closeness_to_puck&opponent_goal/PPO_HockeyEnv-eps02-discount099-lr0001-random_choice85.pth"
    )
)

ppo.policy.load_state_dict(
    torch.load(
        "//home/kathi/Dokumente/ReIn_Course/project/laser-hockey-env-master/results/09-17_without closeness_to_puck&opponent_goal/PPO_HockeyEnv-eps02-discount099-lr0001-random_choice85.pth"
    )
)

import datetime

from pathlib import Path

plot_reward = []
plot_loss = []
plot_smoothed_reward = []
plot_smoothed_losses = []

# replay_number = 5

fps = 50
show = False
stats = []
losses = []

total_steps = 0
current_fig = None
best_reward = -np.inf
best_episode = 0
model_dir = "saved_models"


# NORMAL = 0
# TRAIN_SHOOTING = 1
# TRAIN_DEFENSE = 2


env = h_env.HockeyEnv(mode=0)
opponent = h_env.BasicOpponent(weak=True)
opponent2 = h_env.BasicOpponent(weak=False)

time_step = 0

plot_reward = []
plot_smoothed_reward = []
plot_loss = []
plot_smoothed_reward = []

max_episodes = 10000

# random_choice_par = 95

max_timesteps = 250

loss = 0

memory = Memory()

best_smoothed_reward = -np.inf

time1 = datetime.datetime.now().strftime("%d")
time2 = datetime.datetime.now().strftime("%H:%M:%S")

path = str("results/" + time1 + "-" + time2)
os.mkdir(path)
make_dir(str(path + "/best_weights"))
make_dir(str(path + "/best_optimizer/"))

if __name__ == "__main__":
    # training loop
    for episode in range(1, max_episodes + 1):
        running_reward = 0
        running_loss = 0
        state = env.reset()
        obs_agent2 = env.obs_agent_two()
        for t in range(max_timesteps):

            time_step += 1

            # different variants for playing against different opponents and self-play:
            action = ppo.select_action(state, memory)

            ran_num = np.random.choice([1, 2])
            if ran_num == 1:
                action2 = opponent.act(obs_agent2)
            elif ran_num == 2:
                action2 = opponent2.act(obs_agent2)
            # elif ran_num == 3:
            #   action2 = ppo.select_action_self_play(obs_agent2)
            # elif ran_num == 4:
            # action2 = ppo.select_action_self_play(obs_agent2)

            obs_agent2 = env.obs_agent_two()
            state, reward, done, info = env.step(np.hstack([action[:4], action2[:4]]))

            reward_closeness_to_puck = info[
                "reward_closeness_to_puck"
            ]  # it's already included
            reward_puck_direction = info["reward_puck_direction"]
            reward_touch_puck = info["reward_touch_puck"]
            winner = info["winner"]
            # print(winner)
            if winner == -1:
                reward += 10

            reward = (
                reward - reward_closeness_to_puck
            )  # +  reward_puck_direction #+ reward_touch_puck

            # Saving reward and is_terminals:
            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            # update if its time
            if time_step % update_timestep == 0:
                # print('update', t)
                ppo.update(memory)
                loss = ppo.loss
                memory.clear_memory()
                time_step = 0
                running_loss += loss
            running_reward += reward

            if render:
                env.render()
            if done:
                break

        stats.append([episode, running_reward])
        losses.append([episode, running_loss])
        try:
            plot_reward.append([running_reward, episode + 1])
            plot_loss.append([running_loss, episode + 1])

            smoothed_rewards = running_mean(np.asarray(stats)[:, 1], 500).tolist()
            smoothed_losses = running_mean(np.asarray(losses)[:, 1], 200).tolist()
            if smoothed_rewards:
                plot_smoothed_reward.append([smoothed_rewards[-1], episode + 1])
            if smoothed_losses:
                plot_smoothed_losses.append([smoothed_losses[-1], episode + 1])

            if smoothed_rewards[-1] >= best_smoothed_reward:
                best_smoothed_reward = smoothed_rewards[-1]

                torch.save(
                    ppo.policy.state_dict(),
                    f"./{path}/best_weights/PPO_{env_name}-eps{eps_clip}-discount{gamma}-lr{lr}.pth",
                )
                torch.save(
                    ppo.optimizer.state_dict(),
                    f"./{path}/best_optimizer/PPO_{env_name}-eps{eps_clip}-discount{gamma}-lr{lr}.pth",
                )

        except:
            continue

        if running_reward > best_reward:
            best_reward = running_reward
            best_episode = episode

        if (episode > 0) and (episode % 30 == 0):
            # print("{}: Episode reward: {} \n loss {}".format(episode,running_reward, loss))
            # print("Best reward {} at episode {}".format(best_reward, best_episode))

            current_fig = subplot(
                current_fig,
                plot_reward,
                plot_smoothed_reward,
                plot_loss,
                plot_smoothed_losses,
                path=f"./figures/PPO_{env_name}-eps{eps_clip}-discount{gamma}-lr{lr}.jpg",
            )

    current_fig = subplot(
        current_fig,
        plot_reward,
        plot_smoothed_reward,
        plot_loss,
        plot_smoothed_losses,
        path=f"./figures/{time1}-{time2}_PPO_{env_name}-eps{eps_clip}-discount{gamma}-lr{lr}.jpg",
    )

    torch.save(
        ppo.policy.state_dict(),
        f"./{path}/PPO_{env_name}-eps{eps_clip}-discount{gamma}-lr{lr}-action_std{action_std}.pth",
    )
    torch.save(
        ppo.optimizer.state_dict(),
        f"./{path}/PPO_{env_name}-eps{eps_clip}-discount{gamma}-lr{lr}-action_std{action_std}optimizer.pth",
    )

    ###validation
    lost = 0
    won = 0
    tie = 0
    opponent = h_env.BasicOpponent(weak=True)
    # env.render()
    winner = 10
    obs_agent2 = env.obs_agent_two()
    for _ in range(500):
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
            obs_agent2 = env.obs_agent_two()
            if d:
                break

    average_win_rate = won / (won + lost)
    print(average_win_rate)
    print(won)
    print(lost)
