from TD3.Training import DDPG, actorCritic
from laserhockey import hockey_env as h_env
import torch

env = h_env.HockeyEnv(mode=h_env.HockeyEnv.NORMAL)


class TD3Opponent():

    def __init__(self, load_path):
        ac_space = env.action_space
        o_space = env.observation_space
        self.actor = actorCritic.Actor(o_space.shape[0], int(ac_space.shape[0] / 2), client_run=True)
        self.actor.load_state_dict(
            torch.load('{}/actor.pkl'.format(load_path))
        )

    def act(self, obs):
        return self.actor(torch.from_numpy(obs)).detach().numpy()[:4]
