from collections import namedtuple
import random
from collections import deque
import operator


EXP = namedtuple('EXP', ('state', 'action', 'next_state', 'reward', 'done'))
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))
TransitionPR = namedtuple('TransitionPR', ('state', 'action', 'next_state', 'reward', 'done', 'weight', 'index'))
PriorityData = namedtuple("Priority_data", field_names=["priority", "probability", "weight", "index"])




class ReplayMemory(object):
    def __init__(self, max_size):
        self.replay_buffer = deque([], maxlen=max_size)

    def add(self, exp):
        self.replay_buffer.append(exp)

    def sample(self, batch_size=1):
        if batch_size > self.replay_buffer.maxlen:
            batch_size = self.replay_buffer.maxlen
        out = random.sample(self.replay_buffer, batch_size)
        return out


class PrioritizedExperienceReplay:
    # Inspired by
    # https://github.com/Near32/PYTORCH_RL/blob/master/utils/replayBuffer/replayBuffer.py
    # https://towardsdatascience.com/how-to-implement-prioritized-experience-replay-for-a-deep-q-network-a710beecd77b
    def __init__(self, buffer_size, batch_size, batches_per_sampling, compute_weights):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.batches_per_sampling = batches_per_sampling
        self.alpha = 0.5
        self.alpha_decay = 0.99
        self.exp_count = 0
        self.td_error_max = None

        indexes = []
        prios = []
        for i in range(buffer_size):
            indexes.append(i)
            d = PriorityData(0, 0, 0, i)
            prios.append(d)

        self.exp_memory = {key: EXP for key in indexes}
        self.prio_memory = {key: prio for key, prio in zip(indexes, prios)}
        self.sampled_batches = []
        self.current_batch = 0
        self.priorities_sum_alpha = 0
        self.priorities_max = 1
        self.weights_max = 1

    def update_stored_samples(self):
        self.current_batch = 0
        values = list(self.prio_memory.values())
        random_values = random.choices(self.prio_memory,
                                       [data.probability for data in values],
                                       k=self.batches_per_sampling * self.batch_size)
        self.sampled_batches = [random_values[i:i + self.batch_size]
                                for i in range(0, len(random_values), self.batch_size)]

    def update_priorities(self, tds, indices):
        for td, index in zip(tds, indices):

            updated_priority = td[0]
            if updated_priority > self.priorities_max:
                self.priorities_max = updated_priority

            updated_weight = 1

            old_priority = self.prio_memory[index].priority
            self.priorities_sum_alpha += updated_priority ** self.alpha - old_priority ** self.alpha
            updated_probability = td[0] ** self.alpha / self.priorities_sum_alpha
            data = PriorityData(updated_priority, updated_probability, updated_weight, index)
            self.prio_memory[index] = data

    def update_parameters(self):
        self.priorities_sum_alpha = 0
        sum_prob_before = 0
        # self.alpha*=self.alpha_decay
        for element in self.prio_memory.values():
            sum_prob_before += element.probability
            self.priorities_sum_alpha += element.priority ** self.alpha

        sum_prob_after = 0
        for element in self.prio_memory.values():
            probability = element.priority ** self.alpha / self.priorities_sum_alpha
            sum_prob_after += probability
            weight = 1
            prio_data = PriorityData(element.priority, probability, weight, element.index)
            self.prio_memory[element.index] = prio_data

    def add(self, exp):
        self.exp_count += 1
        index = self.exp_count % self.buffer_size

        # update max_prio and max_weight
        if self.exp_count > self.buffer_size:
            temp = self.prio_memory[index]
            self.priorities_sum_alpha -= temp.priority ** self.alpha
            if temp.priority == self.priorities_max:
                self.prio_memory[index].priority = 0
                self.priorities_max = max(self.prio_memory.items(), key=operator.itemgetter(1)).priority

        # set prio of new element to max_prio and weight to max_weight
        priority = self.priorities_max
        weight = self.weights_max
        self.priorities_sum_alpha += priority ** self.alpha
        probability = priority ** self.alpha / self.priorities_sum_alpha

        self.exp_memory[index] = exp

        prio_data = PriorityData(priority, probability, weight, index)
        self.prio_memory[index] = prio_data

    def sample(self):
        sampled_batch = self.sampled_batches[self.current_batch]
        self.current_batch += 1
        experiences = []
        weights = []
        indices = []

        for data in sampled_batch:
            experiences.append(self.exp_memory.get(data.index))
            weights.append(data.weight)
            indices.append(data.index)
        exps = EXP(*zip(*experiences))
        prio_transitions = TransitionPR(exps.state, exps.action, exps.next_state, exps.reward, exps.done, weights,
                                        indices)
        return prio_transitions

    def __len__(self):
        return len(self.exp_memory)
