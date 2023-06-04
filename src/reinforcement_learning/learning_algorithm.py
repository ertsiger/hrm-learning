from abc import ABC, abstractmethod
from gym_hierarchical_subgoal_automata.envs.base.base_env import BaseEnv
import logging
import numpy as np
import os
import pandas as pd
import pickle
import random
from reinforcement_learning.curriculum import Curriculum
from timeit import default_timer as timer
import torch
from typing import Dict, List
from utils import file_utils
from utils.container_utils import get_param, sort_by_ord
from utils import math_utils
from utils.rl_utils import get_random_tasks


class InterruptLearningException(Exception):
    pass


class LearningAlgorithm(ABC):
    """
    Generic class for the different implemented learning algorithms.
    """
    ENV_NAME = "environment_name"
    ENV_NAME_CRAFTWORLD = "craftworld"
    ENV_NAME_WATERWORLD = "waterworld"

    ENV_CONFIG = "environments"
    ENV_CONFIG_NAME = "name"
    ENV_CONFIG_AUTOMATON_NAME = "automaton_name"
    ENV_CONFIG_HIERARCHY_LEVEL = "hierarchy_level"
    ENV_CONFIG_DEPENDENCIES = "dependencies"
    ENV_CONFIG_STARTING_SEED = "starting_seed"

    USE_ENVIRONMENT_SEED = "use_environment_seed"
    NUM_TASKS_PER_ENVIRONMENT = "num_environment_tasks"
    EXPERIMENT_FOLDER_NAME = "folder_name"

    DEBUG = "debug"                            # whether to print messages for debugging
    TRAINING_ENABLE = "training_enable"        # whether we are training if true, or testing if false
    NUM_EPISODES = "num_episodes"              # number of episodes to execute the agent
    MAX_EPISODE_LENGTH = "max_episode_length"  # maximum number of steps per episode

    USE_GPU = "use_gpu"                        # whether to use the gpu (e.g., for deep rl)

    STATE_FORMAT = "state_format"
    STATE_FORMAT_TABULAR = "tabular"
    STATE_FORMAT_ONE_HOT = "one_hot"
    STATE_FORMAT_FULL_OBS = "full_obs"

    GREEDY_EVALUATION_ENABLE = "greedy_evaluation_enable"        # whether to periodically evaluate the greedy policy
    GREEDY_EVALUATION_FREQUENCY = "greedy_evaluation_frequency"  # how many episodes are executed between evaluations of the greedy policy
    GREEDY_EVALUATION_EPISODES = "greedy_evaluation_episodes"    # how many episodes are used to evaluate the greedy policy

    CURRICULUM_WEIGHT = "curriculum_weight"                  # weight for the weighted average
    CURRICULUM_SOFTMAX_TEMP = "curriculum_temp"              # temperature for the softmax used in the curriculum
    CURRICULUM_LEVEL_THRESHOLD = "curriculum_threshold"      # return threshold for switching to higher hierarchical levels
    CURRICULUM_SCORING_METHOD = "curriculum_scoring_method"  # how to score the returns for computing the probabilities in the curriculum
    CURRICULUM_RETURN_SRC = "curriculum_return_src"          # where do the returns fed to the curriculum come from (greedy or exploratory policy)

    USE_SEED = "use_seed"  # whether to use a seed for Python's random, numpy and torch
    SEED_VALUE = "seed"    # value of the seed

    CHECKPOINT_ENABLE = "checkpoint_enable"        # whether to save progress checkpoints
    CHECKPOINT_FOLDER = "checkpoint_folder"        # where are checkpoints saved
    CHECKPOINT_FILENAME = "checkpoint_%d.pickle"   # checkpoint name pattern
    CHECKPOINT_FREQUENCY = "checkpoint_frequency"  # every how many episodes a checkpoint is produced

    REWARD_STEPS_FOLDER = "reward_steps_logs"                # folder where the reward-steps are saved
    REWARD_STEPS_GREEDY_FOLDER = "reward_steps_greedy_logs"  # folder where the reward-steps for the greedy evaluation are saved
    REWARD_STEPS_FILENAME = "reward_steps-%d.txt"            # reward-steps log file pattern

    STATS_SUMMARY_FILENAME = "stats.json"          # name of the file registering the curriculum history
    MODELS_FOLDER = "models_folder"                # where are the final models saved at the end of the learning

    def __init__(self, params):
        # Seed attributes (needs to be done first in case any other method afterwards uses randomization)
        self.use_seed = get_param(params, LearningAlgorithm.USE_SEED, False)
        self.seed_value = get_param(params, LearningAlgorithm.SEED_VALUE, None)
        self.python_seed_state = None
        self.numpy_seed_state = None
        self.torch_seed_state = None
        if self.use_seed:
            self._set_random_seed()  # need to set these here, especially before creating the model in the subclasses

        # Tasks information
        self.env_name = get_param(params, LearningAlgorithm.ENV_NAME)
        self.num_domains = len(get_param(params, LearningAlgorithm.ENV_CONFIG))
        self.num_tasks = get_param(params, LearningAlgorithm.NUM_TASKS_PER_ENVIRONMENT, 1)
        self.environment_names: List[str] = []
        self.tasks: Dict[str, List[BaseEnv]] = {}
        self._init_tasks(params)

        # Folders where the interaction information is going to be stored
        self.export_folder_names: Dict[str, str] = {}
        self._init_folder_names(params)

        # General reinforcement learning parameters
        use_gpu = get_param(params, LearningAlgorithm.USE_GPU, False)
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        self.debug = get_param(params, LearningAlgorithm.DEBUG, False)
        self.training_enable = get_param(params, LearningAlgorithm.TRAINING_ENABLE, True)
        self.num_episodes = get_param(params, LearningAlgorithm.NUM_EPISODES, 20000)
        self.max_episode_length = get_param(params, LearningAlgorithm.MAX_EPISODE_LENGTH, 100)
        self.observation_format = get_param(params, LearningAlgorithm.STATE_FORMAT, LearningAlgorithm.STATE_FORMAT_TABULAR)

        # Greedy evaluation parameters
        self.greedy_evaluation_enable = get_param(params, LearningAlgorithm.GREEDY_EVALUATION_ENABLE, False)
        self.greedy_evaluation_frequency = get_param(params, LearningAlgorithm.GREEDY_EVALUATION_FREQUENCY, 1)
        self.greedy_evaluation_episodes = get_param(params, LearningAlgorithm.GREEDY_EVALUATION_EPISODES, 1)

        # Trace compression parameters (just copy them from any task)
        self.use_compressed_traces = self._get_task(0, 0).is_compressing_obs()
        self.ignore_empty_observations = self._get_task(0, 0).is_ignoring_empty_obs()

        # Curriculum and where does the undiscounted returns with which it's fed come from (the exploratory policy or
        # the greedy policy)
        self._curriculum = Curriculum(self.num_domains, self.num_tasks, self._on_domains_learned,
                                      get_param(params, LearningAlgorithm.ENV_CONFIG),
                                      get_param(params, LearningAlgorithm.CURRICULUM_WEIGHT, 0.99),
                                      get_param(params, LearningAlgorithm.CURRICULUM_SOFTMAX_TEMP, 2.0),
                                      get_param(params, LearningAlgorithm.CURRICULUM_LEVEL_THRESHOLD, 0.95),
                                      get_param(params, LearningAlgorithm.CURRICULUM_SCORING_METHOD, Curriculum.SCORING_ANDREAS))
        self._curriculum_return_src = get_param(params, LearningAlgorithm.CURRICULUM_RETURN_SRC, "greedy")

        self.running_time = 0.0
        self.last_timestamp = None
        self.interrupted_learning = False  # whether the training loop has been interrupted

        # Checkpoint attributes
        self.checkpoint_enable = get_param(params, LearningAlgorithm.CHECKPOINT_ENABLE, False)
        self.checkpoint_folder = get_param(params, LearningAlgorithm.CHECKPOINT_FOLDER, ".")
        self.checkpoint_frequency = get_param(params, LearningAlgorithm.CHECKPOINT_FREQUENCY, 5)

        # Logs for the different tasks
        self.reward_steps_loggers = []
        self.reward_steps_greedy_loggers = []

        # Folder from where the models are loaded/stored
        self.models_folder = os.path.join(
            get_param(params, LearningAlgorithm.EXPERIMENT_FOLDER_NAME),
            get_param(params, LearningAlgorithm.MODELS_FOLDER, "models")
        )

        self._stats_summary_path = os.path.join(
            get_param(params, LearningAlgorithm.EXPERIMENT_FOLDER_NAME),
            LearningAlgorithm.STATS_SUMMARY_FILENAME
        )

        # Remove old directories to avoid keeping old data
        file_utils.rm_dirs(self.get_reward_episodes_folders())
        file_utils.rm_dirs(self.get_reward_episodes_greedy_folders())
        file_utils.rm_file(self._stats_summary_path)

    def __getstate__(self):
        # the loggers must be removed to produce a checkpoint
        state = self.__dict__.copy()
        del state['reward_steps_loggers']
        if self.greedy_evaluation_enable:
            del state['reward_steps_greedy_loggers']
        return state

    '''
    Initialization helpers
    '''
    def _init_tasks(self, params):
        for env in get_param(params, LearningAlgorithm.ENV_CONFIG):
            env_name = env.get(LearningAlgorithm.ENV_CONFIG_NAME)
            self.environment_names.append(env_name)
            self.tasks[env_name] = get_random_tasks(
                params, env_name, self.num_tasks,
                get_param(params, LearningAlgorithm.USE_ENVIRONMENT_SEED, True),
                get_param(env, LearningAlgorithm.ENV_CONFIG_STARTING_SEED)
            )

    def _init_folder_names(self, params):
        self.export_folder_names = {
            env_name: os.path.join(
                get_param(params, LearningAlgorithm.EXPERIMENT_FOLDER_NAME),
                env_name
            )
            for env_name in self.environment_names
        }

    '''
    Learning Loop (main loop, what happens when an episode ends, changes or was not completed)
    '''
    def run(self, loaded_checkpoint=False):
        if self.checkpoint_enable and loaded_checkpoint:
            self._restore_uncheckpointed_files()
            if self.use_seed:
                self._load_seed_states()

        self._init_reward_steps_loggers()

        self.last_timestamp = timer()

        if self.training_enable:
            self._run_tasks()
        else:
            self._evaluate_greedy_policies()

        self._write_stats_summary()

        if self.training_enable:
            self._export_models()

    def _run_tasks(self):
        while self._curriculum.get_current_episode() < self.num_episodes and not self.interrupted_learning:
            # Select a new domain and task to perform
            self._curriculum.on_episode_start()

            try:
                completed_episode, total_reward, episode_length, ended_terminal, observation_history = \
                    self._run_episode(self._curriculum.get_current_domain(), self._curriculum.get_current_task())
                self._on_episode_end(completed_episode, ended_terminal, total_reward, episode_length, observation_history)
            except InterruptLearningException as e:
                completed_episode = False
                self.interrupted_learning = True
                if self.debug:
                    print(e)

            # Make a checkpoint (the name of the file has the last completed episode)
            current_episode = self._curriculum.get_current_episode()
            if self.checkpoint_enable and (not completed_episode or (current_episode % self.checkpoint_frequency == 0)):
                self._make_checkpoint(current_episode)

    @abstractmethod
    def _run_episode(self, domain_id, task_id):
        pass

    def _on_episode_end(self, completed_episode, ended_terminal, total_reward, episode_length, history):
        # Logging
        self._show_learning_msg(self._curriculum.get_current_domain(), self._curriculum.get_current_task(),
                                self._curriculum.get_current_episode(), ended_terminal, total_reward, episode_length,
                                history)
        self._log_reward_and_steps(self.reward_steps_loggers, self._curriculum.get_current_domain(),
                                   self._curriculum.get_current_task(), total_reward, episode_length)

        # Needed to log when an automaton is learned (see subclasses)
        if not completed_episode:
            self._on_incomplete_episode(self._curriculum.get_current_domain())

        # Perform evaluation of the greedy policies
        if self.training_enable and self.greedy_evaluation_enable and self._curriculum.get_current_episode() % self.greedy_evaluation_frequency == 0:
            self._evaluate_greedy_policies()

        # Update domain, task and episode to work with
        if self._curriculum_return_src == "exploratory":
            self._curriculum.update(self._curriculum.get_current_domain(), self._curriculum.get_current_task(), total_reward)

    @abstractmethod
    def _on_incomplete_episode(self, domain_id):
        pass

    @abstractmethod
    def _on_domains_learned(self, domain_id):
        pass

    '''
    Task Management Methods (tasks from ids, update task to interact with)
    '''
    def _get_task(self, domain_id, task_id):
        env_name = self.environment_names[domain_id]
        return self.tasks[env_name][task_id]

    '''
    Action Selection (epsilon-greedy)
    '''
    def _choose_egreedy_action(self, task, state, q_function, epsilon):
        if self.training_enable:
            prob = np.random.uniform(0, 1)
            if prob <= epsilon:
                return self._get_random_action(task)
        return self._get_greedy_action(state, q_function)

    def _get_greedy_action(self, state, q_function):
        if self._is_tabular_case():
            return math_utils.randargmax(q_function[state, :])
        else:
            state_v = torch.tensor(np.array([state]), dtype=torch.float32, device=self.device)
            with torch.no_grad():
                q_values = q_function(state_v)
            return math_utils.randargmax(q_values.cpu().numpy())

    def _is_tabular_case(self):
        return self.observation_format == LearningAlgorithm.STATE_FORMAT_TABULAR

    def _get_random_action(self, task):
        return random.choice(range(0, task.action_space.n))

    def _get_annealed_exploration_rate(self, num_steps, expl_init, expl_end, anneal_steps):
        return max((expl_end - expl_init) * num_steps / anneal_steps + expl_init, expl_end)

    '''
    History and Observation Management
    '''
    def _get_observation_as_ordered_tuple(self, observation_set):
        observations_list = list(observation_set)
        sort_by_ord(observations_list)
        return tuple(observations_list)

    def _get_observation_embedding(self, task, observation):
        observables = sorted(task.get_observables())
        return np.array([1 if o in observation else 0 for o in observables], dtype=np.float32)

    '''
    Greedy Policy Evaluation
    '''
    def _evaluate_greedy_policies(self):
        # We need this because self.training_enable might be either true or false (it's not always true)
        training_enable = self.training_enable
        self.training_enable = False

        # Precompute which domains are active (domains might become active as we update the curriculum)
        is_domain_active = [self._curriculum.is_active_domain(domain_id) for domain_id in range(self.num_domains)]

        for domain_id in range(self.num_domains):
            for task_id in range(self.num_tasks):
                self._evaluate_greedy_policies_helper(domain_id, task_id, training_enable, is_domain_active[domain_id])
        self.training_enable = training_enable

    def _evaluate_greedy_policies_helper(self, domain_id, task_id, use_curriculum, is_domain_active):
        sum_total_reward, sum_episode_length = 0, 0
        for evaluation_episode in range(self.greedy_evaluation_episodes):
            # If the current domain is available for selection in the curriculum
            if not use_curriculum or is_domain_active:
                _, total_reward, episode_length, _, _ = self._run_episode(domain_id, task_id)

                # Use each single evaluation to update the curriculum probabilities
                if use_curriculum and self._curriculum_return_src == "greedy":
                    self._curriculum.update(domain_id, task_id, total_reward)
            else:
                # The domain is still not being used for training, so assign a default performance value
                total_reward = 0.0
                episode_length = self.max_episode_length

            sum_total_reward += total_reward
            sum_episode_length += episode_length

        avg_total_reward = sum_total_reward / self.greedy_evaluation_episodes
        avg_episode_length = sum_episode_length / self.greedy_evaluation_episodes

        # Use the average evaluation to update the curriculum probabilities
        if use_curriculum and is_domain_active and self._curriculum_return_src == "greedy_avg":
            self._curriculum.update(domain_id, task_id, avg_total_reward)

        # TODO: maybe we could also log the std dev for the rewards and steps
        self._log_reward_and_steps(self.reward_steps_greedy_loggers, domain_id, task_id, avg_total_reward, avg_episode_length)

    '''
    Logging
    '''
    def _show_learning_msg(self, domain_id, task_id, episode, ended_terminal, total_reward, episode_length, history):
        if self.debug:
            print(f"Domain: {domain_id}  - Task: {task_id} - Episode: {episode} - Terminal: {ended_terminal}"
                  f" - Reward: {total_reward} - Steps: {episode_length} - Observations: {history}")

    def _init_reward_steps_loggers(self):
        if self.training_enable:
            self.reward_steps_loggers = self._init_reward_steps_loggers_helper(LearningAlgorithm.REWARD_STEPS_FOLDER,
                                                                               LearningAlgorithm.REWARD_STEPS_FILENAME)
            if self.greedy_evaluation_enable:
                self.reward_steps_greedy_loggers = self._init_reward_steps_loggers_helper(LearningAlgorithm.REWARD_STEPS_GREEDY_FOLDER,
                                                                                          LearningAlgorithm.REWARD_STEPS_FILENAME)
        else:
            self.reward_steps_greedy_loggers = self._init_reward_steps_loggers_helper(LearningAlgorithm.REWARD_STEPS_GREEDY_FOLDER,
                                                                                      LearningAlgorithm.REWARD_STEPS_FILENAME)

    def _init_reward_steps_loggers_helper(self, root_folder_name, filename_pattern):
        reward_steps_loggers = []
        for domain_id in range(self.num_domains):
            folder_name = os.path.join(self.export_folder_names[self.environment_names[domain_id]], root_folder_name)
            file_utils.mkdir(folder_name)

            task_loggers = []
            for task_id in range(self.num_tasks):
                filename = filename_pattern % task_id

                name = os.path.join(folder_name, filename)
                handler = logging.FileHandler(name)

                logger = logging.getLogger(name)
                logger.setLevel(logging.INFO)
                logger.addHandler(handler)

                task_loggers.append(logger)

            reward_steps_loggers.append(task_loggers)
        return reward_steps_loggers

    def _log_reward_and_steps(self, reward_steps_loggers, domain_id, task_id, episode_reward, episode_length):
        reward_steps_loggers[domain_id][task_id].info(";".join([str(self._curriculum.get_current_episode()),
                                                                str(float(episode_reward)), str(episode_length)]))

    '''
    Checkpoint Management
    '''
    def _make_checkpoint(self, episode):
        self._update_running_time()

        if self.use_seed:
            self._save_seed_states()

        filename = LearningAlgorithm.CHECKPOINT_FILENAME % episode
        file_path = os.path.join(self.checkpoint_folder, filename)
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)

    def _restore_uncheckpointed_files(self):  # inherited by subclasses
        self._unlog_uncheckpointed_episodes()

    def _unlog_uncheckpointed_episodes(self):
        """Removes the lines for uncheckpointed episodes."""
        for domain_id in range(self.num_domains):
            self._unlog_uncheckpointed_episodes_helper(self.get_reward_episodes_folder(domain_id),
                                                       LearningAlgorithm.REWARD_STEPS_FILENAME)
            if self.greedy_evaluation_enable:
                self._unlog_uncheckpointed_episodes_helper(self.get_reward_episodes_greedy_folder(domain_id),
                                                           LearningAlgorithm.REWARD_STEPS_FILENAME)

    def _unlog_uncheckpointed_episodes_helper(self, folder_name, filename_pattern):
        if file_utils.path_exists(folder_name):
            for task_id in range(self.num_tasks):
                reward_episodes_file = filename_pattern % task_id
                reward_episodes_file_path = os.path.join(folder_name, reward_episodes_file)
                if file_utils.path_exists(reward_episodes_file_path):
                    try:
                        df = pd.read_csv(reward_episodes_file_path, sep=';', header=None,
                                         names=["episode", "return", "steps"])
                        df = df[df["episode"] <= self._curriculum.get_current_episode()]
                        df.to_csv(reward_episodes_file_path, sep=';', index=False, header=None)
                    except pd.errors.EmptyDataError:
                        pass

    def get_reward_episodes_folders(self):
        return [self.get_reward_episodes_folder(domain_id) for domain_id in range(self.num_domains)]

    def get_reward_episodes_folder(self, domain_id):
        env_name = self.environment_names[domain_id]
        return os.path.join(self.export_folder_names.get(env_name), LearningAlgorithm.REWARD_STEPS_FOLDER)

    def get_reward_episodes_greedy_folders(self):
        return [self.get_reward_episodes_greedy_folder(domain_id) for domain_id in range(self.num_domains)]

    def get_reward_episodes_greedy_folder(self, domain_id):
        env_name = self.environment_names[domain_id]
        return os.path.join(self.export_folder_names.get(env_name), LearningAlgorithm.REWARD_STEPS_GREEDY_FOLDER)

    '''
    Management of the file keeping track of the total running time
    '''
    def _update_running_time(self):
        current_timestamp = timer()
        self.running_time += current_timestamp - self.last_timestamp
        self.last_timestamp = current_timestamp

    def _write_stats_summary(self):
        self._update_running_time()
        stats = {
            "total_running_time": self.running_time,
            "interrupted": self.interrupted_learning
        }

        if self.training_enable:
            stats["curriculum_history"] = self._curriculum.get_history()

        self._update_stats(stats)
        file_utils.write_json_obj(stats, self._stats_summary_path, indent=4)

    @abstractmethod
    def _update_stats(self, stats):
        pass

    '''
    Random Seed Management
    '''
    def _set_random_seed(self):
        if not isinstance(self.seed_value, int):
            raise RuntimeError("Error: the seed must be an integer value.")

        random.seed(self.seed_value)
        np.random.seed(self.seed_value)
        torch.manual_seed(self.seed_value)
        self._set_torch_cudnn()

    def _set_torch_cudnn(self):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # torch.backends.cudnn.enabled = False

    def _load_seed_states(self):
        assert self.python_seed_state is not None
        assert self.numpy_seed_state is not None
        assert self.torch_seed_state is not None

        random.setstate(self.python_seed_state)
        np.random.set_state(self.numpy_seed_state)
        torch.set_rng_state(self.torch_seed_state)

        self._set_torch_cudnn()

    def _save_seed_states(self):
        self.python_seed_state = random.getstate()
        self.numpy_seed_state = np.random.get_state()
        self.torch_seed_state = torch.get_rng_state()

    '''
    Model Management
    '''
    @abstractmethod
    def _export_models(self):
        pass
