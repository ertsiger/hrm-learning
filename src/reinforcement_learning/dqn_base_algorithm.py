from abc import abstractmethod
import collections
from dataclasses import dataclass
import os
from reinforcement_learning.learning_algorithm import LearningAlgorithm
import torch
from typing import Any, Set
from utils import file_utils
from utils.container_utils import get_param

Experience = collections.namedtuple(
    "Experience",
    field_names=["state", "observations", "action", "next_state", "next_observations", "reward", "is_terminal"]
)


@dataclass
class EpisodeData:
    state: Any
    observation: Set[str]


@dataclass
class ActionResult:
    action: int


class DQNBaseAlgorithm(LearningAlgorithm):
    """
    Base class for algorithms relying on the use of a DQN, either a regular one (DQNAlgorithm) or one based on the use
    of an LSTM to support partial observability (DRQNAlgorithm).
    """
    LEARNING_RATE = "learning_rate"
    DISCOUNT_RATE = "discount_rate"
    USE_MAX_POOL = "use_max_pool"             # whether to use maxpooling
    USE_DOUBLE_DQN = "use_double_dqn"         # whether double DQN is used instead of simple DQN

    ER_BUFFER_SIZE = "er_buffer_size"  # size of the ER buffer
    ER_START_SIZE = "er_start_size"    # size of the ER after which learning starts
    ER_BATCH_SIZE = "er_batch_size"    # size of the batches sampled from the buffer (i.e., number of sampled episodes)

    NET_UPDATE_FREQUENCY = "net_update_freq"  # with what frequency to update the model (see "update_timescale")
    TGT_UPDATE_FREQUENCY = "tgt_update_freq"  # with what frequency to sync the target net with the actual one
    UPDATE_TIMESCALE = "update_timescale"  # with what frequency updates are performed (after each step or episode)
    UPDATE_TIMESCALE_EPISODE = "episode"
    UPDATE_TIMESCALE_STEP = "step"

    EXPLORATION_RATE_ANNEALING_INIT = "exploration_rate_annealing_init"              # the degree of exploration at the beginning of training
    EXPLORATION_RATE_ANNEALING_END = "exploration_rate_annealing_end"                # the minimum degree of exploration
    EXPLORATION_RATE_ANNEALING_DURATION = "exploration_rate_annealing_duration"      # for how long is the exploration rate decreased
    EXPLORATION_RATE_ANNEALING_TIMESCALE = "exploration_rate_annealing_timescale"    # whether the annealing is based on the number of episodes or steps
    EXPLORATION_RATE_ANNEALING_TIMESCALE_STEPS = "steps"
    EXPLORATION_RATE_ANNEALING_TIMESCALE_EPISODES = "episodes"

    def __init__(self, params):
        super(DQNBaseAlgorithm, self).__init__(params)

        self._discount_rate = get_param(params, DQNBaseAlgorithm.DISCOUNT_RATE, 0.9)
        self._use_double_dqn = get_param(params, DQNBaseAlgorithm.USE_DOUBLE_DQN, True)

        if self.env_name == DQNBaseAlgorithm.ENV_NAME_CRAFTWORLD:
            self._use_max_pool = get_param(params, DQNBaseAlgorithm.USE_MAX_POOL, False)

        self._update_timescale = get_param(
            params, DQNBaseAlgorithm.UPDATE_TIMESCALE, DQNBaseAlgorithm.UPDATE_TIMESCALE_STEP
        )

        self._net = None
        self._tgt_net = None
        self._optimizer = None
        self._init_networks(params)

        self._er_buffer = self._create_er_buffer(get_param(params, DQNBaseAlgorithm.ER_BUFFER_SIZE, 500))  # 500000 for DQN
        self._er_start_size = get_param(params, DQNBaseAlgorithm.ER_START_SIZE, 100)  # 100000 for DQN
        self._er_batch_size = get_param(params, DQNBaseAlgorithm.ER_BATCH_SIZE, 8)  # 32 for DQN

        self._net_update_freq = get_param(params, DQNBaseAlgorithm.NET_UPDATE_FREQUENCY, 1)
        self._num_net_updates = 0  # Number of steps since the last net update
        self._tgt_net_update_freq = get_param(params, DQNBaseAlgorithm.TGT_UPDATE_FREQUENCY, 100)  # 1500 for DQN
        self._num_tgt_net_updates = 0  # Number of steps since the last target update

        self._total_num_steps = 0
        self._exploration_rate_init = get_param(params, DQNBaseAlgorithm.EXPLORATION_RATE_ANNEALING_INIT, 1.0)
        self._exploration_rate_final = get_param(params, DQNBaseAlgorithm.EXPLORATION_RATE_ANNEALING_END, 0.1)
        self._exploration_rate_annealing_timescale = get_param(
            params, DQNBaseAlgorithm.EXPLORATION_RATE_ANNEALING_TIMESCALE,
            DQNBaseAlgorithm.EXPLORATION_RATE_ANNEALING_TIMESCALE_EPISODES
        )
        self._exploration_rate_annealing_duration = get_param(
            params, DQNBaseAlgorithm.EXPLORATION_RATE_ANNEALING_DURATION, 5000
        )

    def _init_networks(self, params):
        self._net = self._create_network(params)
        if not self.training_enable:
            self._net.load_state_dict(torch.load(self._get_model_path()))
        self._tgt_net = self._create_network(params)
        self._tgt_net.load_state_dict(self._tgt_net.state_dict())
        self._optimizer = torch.optim.RMSprop(
            self._net.parameters(),
            lr=get_param(params, DQNBaseAlgorithm.LEARNING_RATE, 5e-4)
        )

    def _create_network(self, params):
        task = self._get_task(0, 0)  # Get any task
        if self.env_name == DQNBaseAlgorithm.ENV_NAME_CRAFTWORLD:
            return self._create_cw_network(params, task).to(self.device)
        elif self.env_name == DQNBaseAlgorithm.ENV_NAME_WATERWORLD:
            return self._create_ww_network(params, task).to(self.device)
        raise RuntimeError(f"Error: Environment {self.env_name} currently unsupported.")

    @abstractmethod
    def _create_cw_network(self, params, task):
        pass

    @abstractmethod
    def _create_ww_network(self, params, task):
        pass

    @abstractmethod
    def _create_er_buffer(self, capacity):
        pass

    def _run_episode(self, domain_id, task_id):
        task = self._get_task(domain_id, task_id)  # Get the task to learn

        episode = []
        total_reward, episode_length = 0, 0
        episode_data = self._get_init_episode_data(task)

        while not task.is_terminal() and episode_length < self.max_episode_length:
            action_result = self._get_action(task, episode_data)
            next_state, reward, is_terminal, _ = task.step(action_result.action)
            next_observation = task.get_observation()

            if self.training_enable:
                experience = Experience(
                    episode_data.state, self._get_observation_embedding(task, episode_data.observation),
                    action_result.action, next_state, self._get_observation_embedding(task, next_observation), reward,
                    is_terminal
                )
                episode.append(experience)
                self._on_step(experience)
                if self.training_enable and self._update_timescale == DQNBaseAlgorithm.UPDATE_TIMESCALE_STEP:
                    self._train_and_sync()

            episode_data = self._get_next_episode_data(next_state, next_observation, action_result)
            total_reward += reward
            episode_length += 1
            self._total_num_steps += 1

        self._on_episode_complete(episode)
        if self.training_enable and self._update_timescale == DQNBaseAlgorithm.UPDATE_TIMESCALE_EPISODE:
            self._train_and_sync()

        return True, total_reward, episode_length, task.is_terminal(), []

    def _get_init_episode_data(self, task):
        return EpisodeData(state=task.reset(), observation=task.get_observation())

    @abstractmethod
    def _get_action(self, task, episode_data: EpisodeData) -> ActionResult:
        pass

    @abstractmethod
    def _on_step(self, experience: Experience):
        pass

    def _get_next_episode_data(self, next_state, next_observation, action_result: ActionResult):
        return EpisodeData(state=next_state, observation=next_observation)

    @abstractmethod
    def _on_episode_complete(self, episode):
        pass

    def _export_models(self):
        file_utils.mkdir(self.models_folder)
        torch.save(self._net.state_dict(), self._get_model_path())

    def _get_model_path(self):
        return os.path.join(self.models_folder, "model.pt")

    def _train_and_sync(self):
        if len(self._er_buffer) >= self._er_start_size:
            self._num_net_updates += 1
            if self._num_net_updates % self._net_update_freq == 0:
                self._update_network(self._sample_batch())

            self._num_tgt_net_updates += 1
            if self._num_tgt_net_updates % self._tgt_net_update_freq == 0:
                self._tgt_net.load_state_dict(self._net.state_dict())

    def _get_exploration_rate(self):
        if self._exploration_rate_annealing_timescale == DQNBaseAlgorithm.EXPLORATION_RATE_ANNEALING_TIMESCALE_STEPS:
            return self._get_annealed_exploration_rate(
                self._total_num_steps, self._exploration_rate_init, self._exploration_rate_final,
                self._exploration_rate_annealing_duration
            )
        elif self._exploration_rate_annealing_timescale == DQNBaseAlgorithm.EXPLORATION_RATE_ANNEALING_TIMESCALE_EPISODES:
            return self._get_annealed_exploration_rate(
                self._curriculum.get_current_episode(), self._exploration_rate_init, self._exploration_rate_final,
                self._exploration_rate_annealing_duration
            )
        raise RuntimeError(f"Error: Unknown timescale for exploration '{self._exploration_rate_annealing_timescale}'.")

    @abstractmethod
    def _sample_batch(self):
        pass

    @abstractmethod
    def _update_network(self, experience_batch):
        pass

    """
    Unimportant methods for this algorithm
    """
    def _on_domains_learned(self, domain_id):
        pass

    def _update_stats(self, stats):
        pass

    def _on_incomplete_episode(self, domain_id):
        pass
