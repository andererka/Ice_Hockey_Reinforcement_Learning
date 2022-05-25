import numpy as np
import datetime
import matplotlib
from pathlib import Path

matplotlib.use('Agg')
import matplotlib.pyplot as plt

plot_reward = []
plot_smoothed_reward = []
plot_policy_loss = []
plot_smoothed_policy_loss = []
plot_value_loss = []
plot_smoothed_value_loss = []
plot_steps = []
plot_smoothed_steps = []
evals = []


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def subplot(R, R_s, S, S_s, P, P_s, Q, Q_s):
    r = list(zip(*R))
    s = list(zip(*S))
    p = list(zip(*P))
    q = list(zip(*Q))

    r_s = list(zip(*R_s))
    s_s = list(zip(*S_s))
    p_s = list(zip(*P_s))
    q_s = list(zip(*Q_s))
    # clear_output(wait=True)

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))

    ax[0, 0].plot(list(r[1]), list(r[0]), 'r')  # row=0, col=0
    ax[0, 0].plot(list(r_s[1]), list(r_s[0]), 'k')  # row=0, col=0
    ax[0, 1].plot(list(s[1]), list(s[0]), 'm')  # row=0, col=1
    ax[0, 1].plot(list(s_s[1]), list(s_s[0]), 'k')  # row=0, col=1

    ax[1, 0].plot(list(p[1]), list(p[0]), 'b')  # row=1, col=0
    ax[1, 0].plot(list(p_s[1]), list(p_s[0]), 'k')  # row=1, col=0
    ax[1, 1].plot(list(q[1]), list(q[0]), 'g')  # row=1, col=1
    ax[1, 1].plot(list(q_s[1]), list(q_s[0]), 'k')  # row=1, col=1
    ax[0, 0].title.set_text('Reward')
    ax[0, 1].title.set_text('Max steps')
    ax[1, 0].title.set_text('Policy loss')
    ax[1, 1].title.set_text('Q loss')
    return fig


def reset():
    plt.close()


def create_and_store_plot(plot_dir):
    curr_fig = subplot(plot_reward, plot_smoothed_reward, plot_steps, plot_smoothed_steps, plot_policy_loss,
                       plot_smoothed_policy_loss, plot_value_loss, plot_smoothed_value_loss)
    Path(plot_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_dir + "/" + datetime.datetime.now().strftime("%H%M%S") + '_plot.png')
    plt.close(curr_fig)

    r_s = list(zip(*plot_smoothed_reward))
    plt.plot(list(r_s[1]), list(r_s[0]), 'k')
    plt.xlabel('Episode No')
    plt.title('Average Reward over 50 eps')
    plt.ylabel('Reward')
    plt.savefig(plot_dir + "/smoothed_reward_" + datetime.datetime.now().strftime("%H%M%S") + '_plot.png')
    plt.close()



def store_data_to_plot(stats, ep):
    plot_reward.append([np.asarray(stats)[ep, 1], ep + 1])
    plot_value_loss.append([np.asarray(stats)[ep, 2], ep + 1])
    plot_policy_loss.append([np.asarray(stats)[ep, 3], ep + 1])
    plot_steps.append([np.asarray(stats)[ep, 4], ep + 1])

    smoothed_rewards = running_mean(np.asarray(stats)[:, 1], 50).tolist()
    if smoothed_rewards:
        plot_smoothed_reward.append([smoothed_rewards[-1], ep + 1])

    smoothed_value_loss = running_mean(np.asarray(stats)[:, 2], 50).tolist()
    if smoothed_value_loss:
        plot_smoothed_value_loss.append([smoothed_value_loss[-1], ep + 1])

    smoothed_policy_loss = running_mean(np.asarray(stats)[:, 3], 50).tolist()
    if smoothed_policy_loss:
        plot_smoothed_policy_loss.append([smoothed_policy_loss[-1], ep + 1])

    smoothed_steps = running_mean(np.asarray(stats)[:, 4], 50).tolist()
    if smoothed_steps:
        plot_smoothed_steps.append([smoothed_steps[-1], ep + 1])


def create_and_store_evaluation(evaluations, dir):
    curr_time = datetime.datetime.now().strftime("%H%M%S")
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))
    eps, avg_weak, avg_strong, avg_win_weak, avg_win_strong = zip(*evaluations)
    ax[0, 0].plot(eps, avg_weak, 'k')
    ax[0, 1].plot(eps, avg_strong, 'k')
    ax[1, 0].plot(eps, avg_win_weak, 'k')
    ax[1, 1].plot(eps, avg_win_strong, 'k')
    ax[0, 0].title.set_text('Average Reward 100 runs - Weak')
    ax[0, 1].title.set_text('Average Reward 100 runs - Strong')
    ax[1, 0].title.set_text('Average Winrate 100 runs - Weak')
    ax[1, 1].title.set_text('Average Winrate 100 runs - Strong')
    plt.xlabel('Episode No')

    Path(dir + "/evals").mkdir(parents=True, exist_ok=True)
    plt.savefig(dir + "/evals" + "/" + curr_time + '_plot.png')
    plt.close(fig)
