import numpy as np
from reinforcement_learning.dqn_base_algorithm import ActionResult, DQNBaseAlgorithm, EpisodeData, Experience
from reinforcement_learning.model import MinigridDQN, WaterWorldRegularDQN
from reinforcement_learning.replay import ExperienceBuffer
import torch


class DQNAlgorithm(DQNBaseAlgorithm):
    """
    Regular DQN algorithm implementation. For fairness in the comparisons against our approach, the network does not
    only take an observation from the environment but also a 0-1 vector representation of the observables that the agent
    sees in that timestep.

    Paper: Mnih et al. 2015. "Human-level control through deep reinforcement learning".
    """
    def _create_cw_network(self, params, task):
        return MinigridDQN(
            task.observation_space.shape,
            len(task.get_observables()),
            task.action_space.n,
            self._use_max_pool
        )

    def _create_ww_network(self, params, task):
        return WaterWorldRegularDQN(
            task.observation_space.shape,
            len(task.get_observables()),
            task.observation_space.n
        )

    def _create_er_buffer(self, capacity):
        return ExperienceBuffer(capacity)

    def _get_action(self, task, episode_data: EpisodeData) -> ActionResult:
        return ActionResult(
            action=self._choose_egreedy_action(
                task,
                episode_data.state,
                lambda current_state: self._net(
                    current_state,
                    torch.tensor(
                        np.array([self._get_observation_embedding(task, episode_data.observation)]),
                        dtype=torch.float32)
                ),
                self._get_exploration_rate()
            )
        )

    def _on_step(self, experience: Experience):
        if self.training_enable:
            self._er_buffer.append(experience)

    def _sample_batch(self):
        return self._er_buffer.sample(self._er_batch_size)

    def _on_episode_complete(self, episode):
        pass

    def _update_network(self, experience_batch):
        states, observations, actions, next_states, next_observations, rewards, is_terminal = zip(*experience_batch)

        states_v = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        observations_v = torch.tensor(observations, dtype=torch.float32, device=self.device)
        actions_v = torch.tensor(np.array(actions), dtype=torch.long, device=self.device)
        next_states_v = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        next_observations_v = torch.tensor(next_observations, dtype=torch.float32, device=self.device)
        rewards_v = torch.tensor(np.array(rewards), dtype=torch.float32, device=self.device)
        is_terminal_v = torch.tensor(np.array(is_terminal), dtype=torch.bool, device=self.device)

        q_values = self._net(states_v, observations_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            if self._use_double_dqn:
                next_actions = self._net(next_states_v, next_observations_v).max(1)[1]
                next_q_values = self._tgt_net(next_states_v, next_observations_v).gather(1, next_actions.unsqueeze(-1)).squeeze(-1)
            else:
                next_q_values = self._tgt_net(next_states_v, next_observations_v).max(1)[0]
            next_q_values[is_terminal_v] = 0.0

        expected_q_values = rewards_v + self._discount_rate * next_q_values
        loss = torch.nn.MSELoss()(q_values, expected_q_values)

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
