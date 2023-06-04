from abc import ABC, abstractmethod
import json
import numpy as np
import os
import pickle
import random


class Baseline(ABC):
    def __init__(self, args):
        random.seed(args.seed)
        np.random.seed(args.seed)

        self.episode = 0
        self.num_episodes = args.num_episodes
        self.max_episode_length = args.episode_length

        self.num_calls = 0
        self.time_budget = args.timeout
        self.time_left = self.time_budget

        self.output_dir = args.output_dir

        self.rnd_seed_state = None
        self.np_seed_state = None

    def run(self, loaded_checkpoint):
        if loaded_checkpoint:
            random.setstate(self.rnd_seed_state)
            np.random.set_state(self.np_seed_state)
        self.run_algorithm()
        self.export_stats()

    @abstractmethod
    def run_algorithm(self):
        pass

    def export_stats(self):
        with open(os.path.join(self.output_dir, "stats.json"), 'w') as f:
            json.dump({
                "num_states": self.get_num_states(),
                "num_edges": self.get_num_edges(),
                "num_traces": self.get_num_traces(),
                "num_calls": self.num_calls,
                "interrupted": self.time_left < 0,
                "learning_time": self.time_budget - self.time_left
            }, f)

    @abstractmethod
    def get_num_states(self):
        pass

    @abstractmethod
    def get_num_edges(self):
        pass

    @abstractmethod
    def get_num_traces(self):
        pass

    def make_checkpoint(self, episode):
        self.rnd_seed_state = random.getstate()
        self.np_seed_state = np.random.get_state()

        with open(os.path.join(self.output_dir, f"checkpoint_{episode}.pickle"), 'wb') as f:
            pickle.dump(self, f)

    @abstractmethod
    def get_hierarchical_automaton(self):
        pass
