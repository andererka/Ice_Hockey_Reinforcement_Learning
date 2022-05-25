import numpy as np
import PPO as PPO
import numpy as np
import laserhockey.hockey_env as h_env
import gym
import torch

#from laserhockey.hockey_env import BasicOpponent

from myppoagent import MyPlayer 
from myppoagent import Memory
from client.remoteControllerInterface import RemoteControllerInterface
from client.backend.client import Client

memory = Memory()

env = h_env.HockeyEnv(mode=0)
env_name = 'HockeyEnv'

o_space = env.observation_space
ac_space = env.action_space

###setting standard deviation low for tournament
action_std = 0.001



###hyperparameters
n_latent_var = 64           # number of variables in hidden layer
n_latent_var2 = 64
lr = 0.0001
betas = (0.9, 0.999)
gamma = 0.99                # discount factor
K_epochs = 80               # update policy for K epochs
eps_clip = 0.2              # clip parameter for PPO


ppo = PPO.PPO(o_space.shape[0], ac_space.shape[0] , action_std, n_latent_var, n_latent_var2, lr, betas, gamma, K_epochs, eps_clip, t_max= 5, device='cpu' )

###loading model:
ppo.policy.load_state_dict(torch.load('/home/kathi/Dokumente/ReIn_Course/project/laser-hockey-env-master/PPO/results/12-13:47:39/PPO_HockeyEnv-eps02-discount099-lr00001-action_std04.pth'))
ppo.policy_old.load_state_dict(torch.load('/home/kathi/Dokumente/ReIn_Course/project/laser-hockey-env-master/PPO/results/12-13:47:39/PPO_HockeyEnv-eps02-discount099-lr00001-action_std04.pth'))


###alternative model that was also perfromaing well:

#ppo.policy.load_state_dict(torch.load('/home/kathi/Dokumente/ReIn_Course/project/laser-hockey-env-master/PPO/results/15-00:29:49/best_weights/PPO_HockeyEnv-eps03-discount099-lr00001.pth'))
#ppo.policy_old.load_state_dict(torch.load('/home/kathi/Dokumente/ReIn_Course/project/laser-hockey-env-master/PPO/results/15-00:29:49/best_weights/PPO_HockeyEnv-eps03-discount099-lr00001.pth'))





class MyOpponent(MyPlayer, RemoteControllerInterface):

    def __init__(self, ppo, memory):
        MyPlayer.__init__(self, ppo, memory)
        RemoteControllerInterface.__init__(self, identifier='Canucks_ppo')

    def remote_act(self, 
            obs : np.ndarray,
           ) -> np.ndarray:

        return self.act(obs)
        

if __name__ == '__main__':
    controller = MyOpponent(ppo, memory)

    # Play n (None for an infinite amount) games and quit
    client = Client(username='Katharina_Anderer_Canucks', # Testuser
                    password='j#?h=PJqfGW+7^a/',
                    controller=controller, 
                    output_path='/tmp/ALRL2020/client/Katharina_Anderer_Canucks', # rollout buffer with finished games will be saved in here
                    interactive=False,
                    op='start_queuing',
                    num_games=1000
                   )

    # Start interactive mode. Start playing by typing start_queuing. Stop playing by pressing escape and typing stop_queuing
    # client = Client(username='user0', 
    #                 password='1234',
    #                 controller=controller, 
    #                 output_path='/tmp/ALRL2020/client/user0',
    #                )
