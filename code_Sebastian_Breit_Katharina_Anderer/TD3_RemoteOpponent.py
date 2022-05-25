import numpy as np
from TD3.client.remoteControllerInterface import RemoteControllerInterface
from TD3.client.backend.client import Client
from TD3_Opponent import TD3Opponent


class RemoteTD3Opponent(TD3Opponent, RemoteControllerInterface):

    def __init__(self, load_path):
        TD3Opponent.__init__(self, load_path)
        RemoteControllerInterface.__init__(self, identifier='Canucks_TD3')

    def remote_act(self,
                   obs: np.ndarray,
                   ) -> np.ndarray:
        return self.act(obs)


if __name__ == '__main__':
    controller = RemoteTD3Opponent('TD3/saves/basic_strong/no_loading/BASIC_OPP/weights/900')

    # Play n (None for an infinite amount) games and quit
    client = Client(username='Sebastian_Breit_Canucks',  # Testuser
                    password='G*Hv^&6Q',
                    controller=controller,
                    output_path='/tmp/ALRL2020/client/Sebastian_Breit_Canucks',
                    # rollout buffer with finished games will be saved in here
                    interactive=False,
                    op='start_queuing',
                    num_games=1000000
                    )
