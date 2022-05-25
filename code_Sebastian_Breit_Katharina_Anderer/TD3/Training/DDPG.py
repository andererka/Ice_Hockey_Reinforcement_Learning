import numpy as np
import torch
from . import memory as prio_mem, actorCritic as AC
from pathlib import Path
import copy


class DDPG(object):
    def __init__(self, observation_dim, action_dim, lr_critic, prio_replay_enabled, memory_size, discount, tau,
                 batch_size, policy_noise, experiences_per_sampling=100, compute_weight_corr_imp_sampling=False,
                 update_self_play_opp_every=100,
                 device=None):
        if device is not None:
            self.device = device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.observation_dim = observation_dim
        self.action_dim = action_dim

        self.prio_replay_enabled = prio_replay_enabled
        self.compute_weight_corr_imp_sampling = compute_weight_corr_imp_sampling
        self.experiences_per_sampling = experiences_per_sampling
        if self.prio_replay_enabled:
            self.buffer = prio_mem.PrioritizedExperienceReplay(
                memory_size, batch_size, experiences_per_sampling, compute_weight_corr_imp_sampling)
        else:
            self.buffer = prio_mem.ReplayMemory(memory_size)

        self.discount = discount
        self.tau = tau
        self.batch_size = batch_size

        self.actor = AC.Actor(observation_dim, action_dim).to(self.device)
        self.actor_target = AC.Actor(observation_dim, action_dim).to(self.device)
        self.initialize_params(self.actor, self.actor_target)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=lr_critic / 10, eps=0.000001)

        self.critic = AC.Critic(observation_dim, action_dim).to(self.device)
        self.critic_target = AC.Critic(observation_dim, action_dim).to(self.device)
        self.initialize_params(self.critic, self.critic_target)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=lr_critic, eps=0.000001)

        self.critic_loss = torch.nn.MSELoss()  # MSELoss()
        self.actor_loss = torch.nn.L1Loss()

        self.max_action = 1.0
        self.policy_noise = policy_noise
        self.noise_clip = 0.5
        self.policy_freq = 2

        self.actor_old_self_play = AC.Actor(observation_dim, action_dim).to(self.device)
        self.actor_old_self_play_next = AC.Actor(observation_dim, action_dim).to(self.device)
        self.actor_old_self_play = copy.deepcopy(self.actor_target)
        self.actor_old_self_play_next = copy.deepcopy(self.actor_target)
        self.update_self_play_opp_every = update_self_play_opp_every

    @staticmethod
    def initialize_params(source, target):
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(param.data)

    def soft_update_params(self, source, target):
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )

    def store_transition(self, transition):
        s, a, r, s1, d = tuple(transition)
        exp = prio_mem.EXP(s, a, s1, r, d)
        self.buffer.add(exp)

    def eval(self):
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()

    def load_weights(self, path, load_with_optimizer_weights=False):
        if path is None: return

        self.actor.load_state_dict(
            torch.load('{}/actor.pkl'.format(path))
        )

        self.critic.load_state_dict(
            torch.load('{}/critic.pkl'.format(path))
        )

        if load_with_optimizer_weights:
            self.critic_optim.load_state_dict(torch.load('{}/critic_optim.pkl'.format(path)))
            self.critic_target = copy.deepcopy(self.critic)

            self.actor_optim.load_state_dict(torch.load('{}/actor_optim.pkl'.format(path)))
            self.actor_target = copy.deepcopy(self.actor)
            # for self-play vs old weights
            self.actor_old_self_play = copy.deepcopy(self.actor_target)
            self.actor_old_self_play_next = copy.deepcopy(self.actor_target)

    def save_model(self, output, save_with_optimizer_weights=False):
        Path(output).mkdir(parents=True, exist_ok=True)
        torch.save(self.actor.state_dict(), '{}/actor.pkl'.format(output))
        torch.save(self.critic.state_dict(), '{}/critic.pkl'.format(output))
        if save_with_optimizer_weights:
            torch.save(self.critic_optim.state_dict(), '{}/critic_optim.pkl'.format(output))
            torch.save(self.actor_optim.state_dict(), '{}/actor_optim.pkl'.format(output))

    def update_policy(self, total_it, prev_policy_loss):

        if total_it % self.update_self_play_opp_every == 0:
            self.actor_old_self_play = copy.deepcopy(self.actor_old_self_play_next)
            self.actor_old_self_play_next = copy.deepcopy(self.actor_target)

        if self.prio_replay_enabled:
            if total_it % self.experiences_per_sampling == 0:
                self.buffer.update_parameters()
                self.buffer.update_stored_samples()

            named_batch = self.buffer.sample()


        else:
            batch = self.buffer.sample(self.batch_size)
            named_batch = prio_mem.Transition(*zip(*batch))

        # Create Batch with replayMemory :
        next_state_batch = torch.tensor(torch.from_numpy(np.stack(named_batch.next_state)), requires_grad=True,
                                        device=self.device, dtype=torch.float)
        state_batch = torch.tensor(torch.from_numpy(np.stack(named_batch.state)), requires_grad=True,
                                   device=self.device,
                                   dtype=torch.float)
        action_batch = torch.tensor(torch.from_numpy(np.stack(named_batch.action)), requires_grad=True,
                                    device=self.device,
                                    dtype=torch.float)
        reward_batch = torch.unsqueeze(
            torch.tensor(torch.from_numpy(np.stack(named_batch.reward)), requires_grad=True, device=self.device,
                         dtype=torch.float), 1)
        done_batch = torch.unsqueeze(
            torch.tensor(torch.from_numpy(np.stack(named_batch.done)), requires_grad=True, device=self.device,
                         dtype=torch.float), 1)

        td_target = self.calc_value_loss(action_batch, done_batch, next_state_batch, reward_batch)

        q_values_1, q_values_2 = self.critic(state_batch, action_batch)

        self.critic_optim.zero_grad()
        value_loss = self.critic_loss(q_values_1, td_target) + self.critic_loss(q_values_2, td_target)
        if self.compute_weight_corr_imp_sampling:
            with torch.no_grad():
                weight = sum(np.multiply(named_batch.weight, value_loss.data.cpu().numpy()))
            value_loss *= weight
        value_loss.backward()
        self.critic_optim.step()

        ## update prios for batch
        if self.prio_replay_enabled:
            delta = torch.abs(q_values_1 - td_target + q_values_2 - td_target).detach().cpu().numpy()
            self.buffer.update_priorities(delta, named_batch.index)

        if total_it % self.policy_freq == 0:
            # Actor update
            policy_loss = -self.critic.single_forward(state_batch, self.actor(state_batch)).mean()
            self.actor_optim.zero_grad()
            policy_loss.backward()
            self.actor_optim.step()

            # Target update
            self.soft_update_params(self.critic, self.critic_target)
            self.soft_update_params(self.actor, self.actor_target)

            return value_loss.item(), policy_loss.item()
        else:
            return value_loss.item(), prev_policy_loss

    def calc_value_loss(self, action_batch, done_batch, next_state_batch, reward_batch):
        # Critic update
        noisy_action_preds = self.add_target_policy_smoothing(self.actor_target(next_state_batch))
        preds_1, preds_2 = self.critic_target(next_state_batch, noisy_action_preds)
        td_target = reward_batch + self.discount * (1 - done_batch) * torch.min(preds_1, preds_2)
        return td_target

    def random_action(self):
        action = np.random.uniform(-1., 1., self.action_dim)
        return action

    def select_action(self, state):
        action = self.actor(torch.from_numpy(state))
        return action

    def add_target_policy_smoothing(self, action_batch):
        noise = (torch.randn_like(action_batch) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
        noisy_actions = (action_batch + noise).clamp(-self.max_action, self.max_action)
        return noisy_actions

    def add_exploration_noise(self, action_batch, explore_noise):
        noise = np.random.normal(0, self.max_action * explore_noise, size=np.shape(action_batch))
        noisy_actions = (action_batch + noise).clip(-self.max_action, self.max_action)
        return noisy_actions

    def select_old_action_self_play(self, state):
        with torch.no_grad():
            action = self.actor_old_self_play(torch.from_numpy(state))
        return action
