import numpy as np
import laserhockey.hockey_env as h_env
from pathlib import Path
import datetime
from . import DDPG, plotting
import random


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def start_training(store_every, env_mode, lr_critic, max_eps, max_steps, weak_opp,
                   self_train, load_path, prio_replay_enabled, memory_max_size,
                   discount, tau, batch_size, opp_score_reward_disabled, random_exploration,
                   policy_noise, closeness_reward_disabled, load_optim_weights, save_optim_weights, random_explore_no,
                   exploration_noise, experiences_per_sampling, random_opponent_sampling=False,
                   save_path="saved_models" + "/" + datetime.datetime.now().strftime(
                       "%Y%m%d") + "/" + datetime.datetime.now().strftime("%H%M%S")):
    env = h_env.HockeyEnv(mode=env_mode)
    ac_space = env.action_space
    o_space = env.observation_space
    plotting.reset()

    ddpg = DDPG.DDPG(o_space.shape[0], int(ac_space.shape[0] / 2), lr_critic=lr_critic,
                     prio_replay_enabled=prio_replay_enabled,
                     memory_size=memory_max_size, discount=discount, tau=tau, batch_size=batch_size,
                     policy_noise=policy_noise, experiences_per_sampling=experiences_per_sampling)
    print("Training with state dim: {} and action dim {}".format(o_space.shape[0], ac_space.shape[0]))
    if load_path != None:
        print("loading pre-trained weights")
        ddpg.load_weights(load_path, load_optim_weights)

    stats = []
    evaluations = []
    total_steps = 0

    opponent_list = ['weak', 'strong', 'play_self']
    opponent_probs = [0.4, 0.4, 0.2]

    opponent = h_env.BasicOpponent(weak=weak_opp)
    best_smoothed_reward = -np.inf

    value_loss = 0
    policy_loss = 0

    for episode in range(max_eps):
        episode_reward = 0
        ob = env.reset()
        steps_per_episode = 0
        obs_agent2 = env.obs_agent_two()
        if random_opponent_sampling:
            # random_opp_name = random.choice(opponent_list)
            random_opp_name = np.random.choice(np.asarray(opponent_list), 1, p=np.asarray(opponent_probs))[-1]
            rand_opp = None
            if random_opp_name == 'weak':
                rand_opp = h_env.BasicOpponent(weak=True)
            elif random_opp_name == 'strong':
                rand_opp = h_env.BasicOpponent(weak=False)

        for ep_step in range(max_steps):
            a = None
            d = False

            if not random_opponent_sampling:
                if self_train:
                    if random_exploration and (total_steps < random_explore_no):
                        a1 = ddpg.random_action()
                        a2 = ddpg.random_action()

                    else:
                        a1 = ddpg.select_action(ob)
                        a1 = a1.detach().cpu().numpy()[:4]
                        a1 = ddpg.add_exploration_noise(a1, exploration_noise)

                        # a2 = ddpg.select_action(obs_agent2)
                        a2 = ddpg.select_old_action_self_play(obs_agent2)
                        a2 = a2.detach().cpu().numpy()[:4]

                    a = np.hstack([a1, a2])

                else:
                    if random_exploration and (total_steps < random_explore_no):
                        a1 = ddpg.random_action()[:4]
                    else:
                        a1 = ddpg.select_action(ob)
                        a1 = a1.detach().cpu().numpy()[:4]
                        a1 = ddpg.add_exploration_noise(a1, exploration_noise)

                    if env_mode == 0:
                        a2 = opponent.act(obs_agent2)
                    else:
                        a2 = [0, 0, 0, 0]
                    a = np.hstack([a1, a2])
            else:
                if random_exploration and (total_steps < random_explore_no):
                    a1 = ddpg.random_action()
                    a2 = ddpg.random_action()

                else:
                    a1 = ddpg.select_action(ob)
                    a1 = a1.detach().cpu().numpy()[:4]
                    a1 = ddpg.add_exploration_noise(a1, exploration_noise)

                    if random_opp_name == 'play_self':
                        a2 = ddpg.select_action(obs_agent2)
                        a2 = a2.detach().cpu().numpy()[:4]
                    else:
                        a2 = rand_opp.act(obs_agent2)

                a = np.hstack([a1, a2])

            obs_agent2 = env.obs_agent_two()
            ob_new, r, d, info = env.step(a)
            r = modify_reward(r, info, opp_score_reward_disabled, closeness_reward_disabled)
            ddpg.store_transition([ob, a1, r, ob_new, d])
            if (total_steps < random_explore_no):
                pass
            else:
                value_loss, policy_loss = ddpg.update_policy(total_steps, policy_loss)
            episode_reward += r

            ob = ob_new
            steps_per_episode = ep_step

            total_steps += 1
            if d:
                break

        stats.append([episode, episode_reward, value_loss, policy_loss, steps_per_episode + 1])
        try:
            plotting.store_data_to_plot(stats, episode)
            smoothed_rewards = running_mean(np.asarray(stats)[:, 1], 50).tolist()
            if smoothed_rewards[-1] >= best_smoothed_reward:
                ddpg.save_model(save_path + "/best_weights")
                best_smoothed_reward = smoothed_rewards[-1]
        except:
            continue

        if ((episode > 0) and (episode % store_every == 0)):
            print(
                "{}: Done after {} steps. Episode reward: {}, best smoothed reward: {} \n Value-loss {}, policy-loss {}".format(
                    episode, steps_per_episode + 1, episode_reward, best_smoothed_reward, value_loss, policy_loss))

            ddpg.save_model(save_path + "/weights/" + str(episode), save_with_optimizer_weights=save_optim_weights)
            plotting.create_and_store_plot(save_path + "/plots")


    final_save_path = save_path + "/final_config"
    Path(final_save_path).mkdir(parents=True, exist_ok=True)
    ddpg.save_model(final_save_path, save_with_optimizer_weights=save_optim_weights)


def modify_reward(r, info, opp_score_reward_disabled, closeness_reward_disabled):
    if (info.get('winner') == -1) and opp_score_reward_disabled:
        r += 10
    if closeness_reward_disabled:
        r -= info.get('reward_closeness_to_puck')
    return r


def eval(weak, weights_path):
    eval_env = h_env.HockeyEnv(mode=h_env.HockeyEnv.NORMAL)

    ac_space = eval_env.action_space
    o_space = eval_env.observation_space
    ddpg = DDPG.DDPG(o_space.shape[0], int(ac_space.shape[0] / 2), lr_critic=0.001, prio_replay_enabled=False,
                     memory_size=int(1e5), discount=0.95, tau=0.005,
                     batch_size=100, policy_noise=0.2)

    ddpg.load_weights(weights_path)

    wincount = 0
    losscount = 0
    drawcount = 0
    for test_ep in range(1000):
        obs = eval_env.reset()
        opponent = h_env.BasicOpponent(weak=weak)
        obs_agent2 = eval_env.obs_agent_two()
        d = False
        for x in range(250):

            a1 = ddpg.select_action(obs).detach().cpu().numpy()
            a1 = a1[:4]
            # print(a1)
            a2 = opponent.act(obs_agent2)
            obs, r, d, info = eval_env.step(np.hstack([a1, a2]))
            # print(info)
            obs_agent2 = eval_env.obs_agent_two()
            if d:
                if info.get('winner') == 1:
                    wincount += 1
                elif info.get('winner') == -1:
                    losscount += 1
                break
        if not d:
            drawcount += 1

    winrate = wincount / (wincount + losscount)
    params = [
        {'wincount': wincount, 'losscount': losscount, 'drawcount': drawcount, 'winrate': winrate}]
    print(params)
    return params
