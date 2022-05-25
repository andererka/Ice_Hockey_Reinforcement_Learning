import json
from TD3.Training import Training_DDPG as training

from pathlib import Path


def main():
    path_with_run_no = 'TD3/saves/test_random/no_loading/'
    custom_notes = 'with prio replay,  basic-strong, no loading, closeness rew enabled, opp score punish disabled'
    print(custom_notes)
    episodes = 1000
    save_each = 100

    lr_critic = 0.001
    max_steps = 250
    mem_max_size = int(1e5)
    discount = 0.95
    tau = 0.05
    batch_size = 100
    policy_noise = 0.2

    experiences_per_sampling = 100
    initial_random_exploration = max(2500, experiences_per_sampling * batch_size)
    exploration_noise = 0.1

    with_defense = False
    with_shooting = False
    with_basic = True
    with_self_train = False
    with_random_opp = True

    weak_opponent = True
    with_prio_replay = True
    rand_explore = True

    opp_score_rew_disabled = False
    closeness_reward_disabled = False

    load_optim_weights = True
    save_optim_weights = True

    load_path = None

    # Save params before training
    params = [
        {'path_with_run_no': path_with_run_no, 'episodes': episodes, 'lr_critic': lr_critic, 'max_steps': max_steps,
         'mem_max_size': mem_max_size, 'discount': discount, 'tau': tau, 'batch_size': batch_size,
         'policy_noise': policy_noise,
         'with_defense': with_defense, 'with_shooting': with_shooting, 'with_basic': with_basic,
         'with_self_train': with_self_train, 'with_prio_replay': with_prio_replay, 'weak_opponent': weak_opponent,
         'opp_score_rew_disabled': opp_score_rew_disabled, 'rand_explore': rand_explore,
         'closeness_reward_disabled': closeness_reward_disabled,
         'initial_random_exploration': initial_random_exploration, 'exploration_noise': exploration_noise,
         'experiences_per_sampling': experiences_per_sampling,
         'custom_notes': custom_notes}]

    store_params(params, path_with_run_no)

    if with_defense:
        # load_path = 'TD3/saves/final_run_01/DEFENSE/final_config'
        env_mode = 2  # DEFENSE
        self_train = False
        save_path = '%sDEFENSE' % path_with_run_no
        store_params(params, save_path)

        training.start_training(store_every=save_each, env_mode=env_mode, lr_critic=lr_critic,
                                max_eps=episodes, max_steps=max_steps, weak_opp=weak_opponent,
                                self_train=self_train, load_path=load_path, save_path=save_path,
                                prio_replay_enabled=with_prio_replay, memory_max_size=mem_max_size, discount=discount,
                                tau=tau,
                                batch_size=batch_size, opp_score_reward_disabled=opp_score_rew_disabled,
                                random_exploration=rand_explore, policy_noise=policy_noise,
                                closeness_reward_disabled=closeness_reward_disabled,
                                load_optim_weights=load_optim_weights, save_optim_weights=save_optim_weights,
                                random_explore_no=initial_random_exploration, exploration_noise=exploration_noise,
                                experiences_per_sampling=experiences_per_sampling,
                                random_opponent_sampling=with_random_opp)

    if with_shooting:
        load_path = 'TD3/saves/final_run_01/DEFENSE/weights'
        env_mode = 1  # SHOOTING
        self_train = False
        save_path = '%sSHOOTING' % path_with_run_no
        store_params(params, save_path)
        training.start_training(store_every=save_each, env_mode=env_mode, lr_critic=lr_critic,
                                max_eps=episodes, max_steps=max_steps, weak_opp=weak_opponent,
                                self_train=self_train, load_path=load_path, save_path=save_path,
                                prio_replay_enabled=with_prio_replay, memory_max_size=mem_max_size, discount=discount,
                                tau=tau,
                                batch_size=batch_size, opp_score_reward_disabled=opp_score_rew_disabled,
                                random_exploration=rand_explore, policy_noise=policy_noise,
                                closeness_reward_disabled=closeness_reward_disabled,
                                load_optim_weights=load_optim_weights, save_optim_weights=save_optim_weights,
                                random_explore_no=initial_random_exploration, exploration_noise=exploration_noise,
                                experiences_per_sampling=experiences_per_sampling,
                                random_opponent_sampling=with_random_opp)

    if with_basic:
        # load_path = 'TD3/saves/test_a_dim_change_01/BASIC_OPP/final_config'
        env_mode = 0  # NORMAL
        self_train = False

        save_path = '%sBASIC_OPP' % path_with_run_no
        store_params(params, save_path)
        training.start_training(store_every=save_each, env_mode=env_mode, lr_critic=lr_critic,
                                max_eps=episodes, max_steps=max_steps, weak_opp=weak_opponent,
                                self_train=self_train, load_path=load_path, save_path=save_path,
                                prio_replay_enabled=with_prio_replay, memory_max_size=mem_max_size, discount=discount,
                                tau=tau,
                                batch_size=batch_size, opp_score_reward_disabled=opp_score_rew_disabled,
                                random_exploration=rand_explore, policy_noise=policy_noise,
                                closeness_reward_disabled=closeness_reward_disabled,
                                load_optim_weights=load_optim_weights, save_optim_weights=save_optim_weights,
                                random_explore_no=initial_random_exploration, exploration_noise=exploration_noise,
                                experiences_per_sampling=experiences_per_sampling,
                                random_opponent_sampling=with_random_opp)

    if with_self_train:
        #load_path = 'TD3/saves/final_run_01/SHOOTING/weights'
        env_mode = 0  # NORMAL/weights
        self_train = True

        save_path = '%sSELF_PLAY' % path_with_run_no
        store_params(params, save_path)
        training.start_training(store_every=save_each, env_mode=env_mode, lr_critic=lr_critic,
                                max_eps=episodes, max_steps=max_steps, weak_opp=weak_opponent,
                                self_train=self_train, load_path=load_path, save_path=save_path,
                                prio_replay_enabled=with_prio_replay, memory_max_size=mem_max_size, discount=discount,
                                tau=tau,
                                batch_size=batch_size, opp_score_reward_disabled=opp_score_rew_disabled,
                                random_exploration=rand_explore, policy_noise=policy_noise,
                                closeness_reward_disabled=closeness_reward_disabled,
                                load_optim_weights=load_optim_weights, save_optim_weights=save_optim_weights,
                                random_explore_no=initial_random_exploration, exploration_noise=exploration_noise,
                                experiences_per_sampling=experiences_per_sampling,
                                random_opponent_sampling=with_random_opp)

    store_eval(training.eval(weak=True, weights_path=save_path + '/final_config'), save_path + '/weak_opp')
    store_eval(training.eval(weak=False, weights_path=save_path + '/final_config'), save_path + '/strong_opp')


def store_params(params, path_with_run_no):
    Path(path_with_run_no).mkdir(parents=True, exist_ok=True)
    with open(path_with_run_no + "/vars.json", "w") as f:
        # Write it to file
        json.dump(params, f)


def store_eval(eval, path_with_run_no):
    Path(path_with_run_no).mkdir(parents=True, exist_ok=True)
    with open(path_with_run_no + "/eval.json", "w") as f:
        # Write it to file
        json.dump(eval, f)


if __name__ == '__main__':
    main()
