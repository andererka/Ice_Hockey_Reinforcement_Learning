import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# reference https://github.com/sfujim/TD3
# Github from the original paper
class Actor(torch.nn.Module):
    def __init__(self, state_dim, action_dim, client_run=False):
        super(Actor, self).__init__()

        self.l1 = torch.nn.Linear(state_dim, 400)

        self.l2 = torch.nn.Linear(400, 300)

        self.l3 = torch.nn.Linear(300, action_dim)

        self.relu = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()
        self.client_run = client_run

    def forward(self, states):
        with torch.enable_grad():
            if not self.client_run:
                states = states.float().to(device)
            else:
                states = states.float()
            out = self.relu(self.l1(states))
            out = self.relu(self.l2(out))
            out = self.tanh(self.l3(out))

            return out


class Critic(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # critic 1
        self.l1 = torch.nn.Linear(state_dim, 400)
        self.l2 = torch.nn.Linear(400 + action_dim, 300)
        self.l3 = torch.nn.Linear(300, 1)

        # critic 2
        self.l4 = torch.nn.Linear(state_dim, 400)
        self.l5 = torch.nn.Linear(400 + action_dim, 300)
        self.l6 = torch.nn.Linear(300, 1)

        self.relu = torch.nn.ReLU()

    def forward(self, states, actions):
        with torch.enable_grad():
            states = states.float().to(device)  # float()?
            out1 = self.relu(self.l1(states))
            out1 = self.relu(self.l2(torch.cat([out1, actions], dim=1)))
            out1 = self.l3(out1)

            out2 = self.relu(self.l4(states))
            out2 = self.relu(self.l5(torch.cat([out2, actions], dim=1)))
            out2 = self.l6(out2)

            return out1, out2

    def single_forward(self, states, actions):
        states = states.float().to(device)

        out = self.relu(self.l1(states))
        out = self.relu(self.l2(torch.cat([out, actions], dim=1)))
        out = self.l3(out)
        return out
