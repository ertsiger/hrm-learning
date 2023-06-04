import collections
import numpy as np
import random


class ExperienceBuffer:
    """
    Regular experience replay buffer (i.e. store individual experiences).
    """
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = random.sample(range(len(self.buffer)), batch_size)
        return [self.buffer[x] for x in indices]

    def clear(self):
        self.buffer.clear()


class EpisodeExperienceBuffer:
    """
    Experience replay buffer that stores and samples sequences of experiences within episodes.
    """
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, episode):
        self.buffer.append(episode)

    def sample(self, batch_size, seq_length):
        states = []
        observations = []
        actions = []
        next_states = []
        next_observations = []
        rewards = []
        is_terminals = []

        episode_ids = random.sample(range(len(self.buffer)), batch_size)
        for episode_id in episode_ids:
            episode_length = len(self.buffer[episode_id])

            if seq_length >= episode_length:  # use the full episode
                start_ts = 0
                end_ts = episode_length
            else:  # use a subsequence of the episode (but trying to maximize its length for history accuracy)
                start_ts = np.random.randint(0, episode_length - seq_length)
                end_ts = start_ts + seq_length

            sequence = self.buffer[episode_id][start_ts:end_ts]

            state, obs, action, next_state, next_obs, reward, is_terminal = zip(*sequence)
            states.append(state)
            observations.append(obs)
            actions.append(action)
            next_states.append(next_state)
            next_observations.append(next_obs)
            rewards.append(reward)
            is_terminals.append(is_terminal)

        return states, observations, actions, next_states, next_observations, rewards, is_terminals,

    def clear(self):
        self.buffer.clear()
