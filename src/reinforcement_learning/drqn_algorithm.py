from dataclasses import dataclass
import numpy as np
import random
from reinforcement_learning.dqn_base_algorithm import ActionResult, DQNBaseAlgorithm, EpisodeData, Experience
from reinforcement_learning.model import MinigridDRQN, WaterWorldDRQN
from reinforcement_learning.replay import EpisodeExperienceBuffer
import torch
from torch.nn.utils.rnn import pad_sequence
from typing import Any
from utils.container_utils import get_param
from utils.math_utils import randargmax


@dataclass
class DRQNEpisodeData(EpisodeData):
    hidden_state: Any


@dataclass
class DRQNActionResult(ActionResult):
    next_hidden_state: Any


class DRQNAlgorithm(DQNBaseAlgorithm):
    """
    Implementation of a Deep Recurrent Q-Network (DRQN), a DQN that employs an LSTM layer to support environments
    involving partial observability.

    Paper: Hausknecht and Stone. 2015. "Deep Recurrent Q-Learning for Partially Observable MDPs".
    """
    ER_SEQ_LENGTH = "er_seq_length"    # length of the sequences sampled from the ER buffer

    LSTM_METHOD = "lstm_method"              # what is the input to the LSTM used in the DRQN
    LSTM_METHOD_STATE = "state"              # the input to the LSTM is the state observation (the set of observed propositions is not used anywhere)
    LSTM_METHOD_STATE_AND_OBS = "state+obs"  # the input to the LSTM is a transformation of the state obs. concatenated with the set of seen propositions
    LSTM_METHOD_OBS = "obs"                  # the input to the LSTM is the set of observed propositions, which is then concatenated with the processed state observation
    LSTM_HIDDEN_SIZE = "lstm_hidden_size"

    def __init__(self, params):
        super(DRQNAlgorithm, self).__init__(params)
        self._er_seq_length = get_param(params, DRQNAlgorithm.ER_SEQ_LENGTH, 4)

    def _get_init_episode_data(self, task):
        data = super()._get_init_episode_data(task)
        return DRQNEpisodeData(state=data.state, observation=data.observation, hidden_state=self._net.get_zero_hidden_state(1, self.device))

    def _get_action(self, task, episode_data: DRQNEpisodeData) -> DRQNActionResult:
        state_v = torch.tensor(np.array([episode_data.state]), dtype=torch.float32, device=self.device)
        obs_v = torch.tensor(
            np.array([self._get_observation_embedding(task, episode_data.observation)]), dtype=torch.float32,
            device=self.device
        )
        if self.training_enable and np.random.uniform(0, 1) <= self._get_exploration_rate():
            action, new_hidden_state = self._get_random_action_from_history(
                task, state_v, obs_v, episode_data.hidden_state
            )
        else:
            action, new_hidden_state = self._get_greedy_action_from_history(state_v, obs_v, episode_data.hidden_state)
        return DRQNActionResult(action=action, next_hidden_state=new_hidden_state)

    def _on_step(self, experience: Experience):
        pass

    def _get_next_episode_data(self, next_state, next_observation, action_result: DRQNActionResult):
        data = super()._get_next_episode_data(next_state, next_observation, action_result)
        return DRQNEpisodeData(
            state=data.state, observation=data.observation, hidden_state=action_result.next_hidden_state
        )

    def _on_episode_complete(self, episode):
        if self.training_enable:
            self._er_buffer.append(episode)

    def _sample_batch(self):
        return self._er_buffer.sample(self._er_batch_size, self._er_seq_length)

    def _get_random_action_from_history(self, task, state, obs, hidden_state):
        action = random.choice(range(0, task.action_space.n))
        with torch.no_grad():
            # Even though a random action is chosen, we still need the new hidden state.
            _, new_hidden_state = self._net(state, obs, hidden_state)
        return action, new_hidden_state

    def _get_greedy_action_from_history(self, state, obs, hidden_state):
        with torch.no_grad():
            q_values, new_hidden_state = self._net(state, obs, hidden_state)
        return randargmax(q_values.cpu().numpy()), new_hidden_state

    def _create_cw_network(self, params, task):
        return MinigridDRQN(
            task.observation_space.shape,
            len(task.get_observables()),
            task.action_space.n,
            get_param(params, DRQNAlgorithm.LSTM_METHOD, DRQNAlgorithm.LSTM_METHOD_STATE),
            get_param(params, DRQNAlgorithm.LSTM_HIDDEN_SIZE, 256),
            self._use_max_pool
        )

    def _create_ww_network(self, params, task):
        return WaterWorldDRQN(
            task.observation_space.shape,
            task.action_space.n
        )

    def _create_er_buffer(self, capacity):
        return EpisodeExperienceBuffer(capacity)

    def _update_network(self, experience_batch):
        states, observations, actions, next_states, next_observations, rewards, is_terminal,  = experience_batch

        def _make_padded_tensor(l, dtype):
            # Convert each member of the batch into a tensor and pad it
            return pad_sequence(
                list(map(lambda x: torch.tensor(np.array(x), dtype=dtype, device=self.device), l)),
                batch_first=True
            )

        # Convert to arrays of tensors
        states_v = _make_padded_tensor(states, torch.float32)
        observations_v = _make_padded_tensor(observations, torch.float32)
        actions_v = _make_padded_tensor(actions, torch.int64)
        next_states_v = _make_padded_tensor(next_states, torch.float32)
        next_observations_v = _make_padded_tensor(next_observations, torch.float32)
        rewards_v = _make_padded_tensor(rewards, torch.float32)
        is_terminal_v = _make_padded_tensor(is_terminal, torch.bool)

        hidden_state = self._net.get_zero_hidden_state(self._er_batch_size, self.device)
        loss = 0.0

        seq_lengths = torch.tensor(list(map(len, states)), dtype=torch.int)
        num_timesteps = seq_lengths.max()
        for timestep in range(num_timesteps):
            q_values, hidden_state = self._net(states_v[:, timestep, :], observations_v[:, timestep, :], hidden_state)
            q_values = q_values.gather(1, actions_v[:, timestep].unsqueeze(-1)).squeeze(-1)

            with torch.no_grad():
                if self._use_double_dqn:
                    next_actions = self._net(
                        next_states_v[:, timestep, :], next_observations_v[:, timestep, :], hidden_state
                    )[0].max(1)[1]
                    next_q_values = self._tgt_net(
                        next_states_v[:, timestep, :], next_observations_v[:, timestep, :], hidden_state
                    )[0].gather(1, next_actions.unsqueeze(-1)).squeeze(-1)
                else:
                    next_q_values = self._tgt_net(next_states_v, next_observations_v, hidden_state).max(1)[0]
                next_q_values[is_terminal_v[:, timestep]] = 0.0

            expected_q_values = rewards_v[:, timestep] + self._discount_rate * next_q_values
            error = (q_values - expected_q_values) ** 2
            loss += torch.sum(error.masked_fill(timestep >= seq_lengths, 0))

        loss = loss / sum(seq_lengths)

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
