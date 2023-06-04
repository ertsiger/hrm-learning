from abc import ABC, abstractmethod
import argparse
import numpy as np
from scipy import stats
from utils.file_utils import read_json_file


class StatsCollector(ABC):
    @classmethod
    def run(cls):
        args = cls._get_argparser().parse_args()
        cls._export_stats(read_json_file(args.config_file), args.output_file)

    @classmethod
    def _get_argparser(cls):
        parser = argparse.ArgumentParser()
        parser.add_argument("config_file", help="config file")
        parser.add_argument("output_file", help="output file")
        return parser

    @classmethod
    def _mean_std(cls, arr):
        if len(arr) == 0:
            return 0.0, 0.0
        return np.round(np.mean(arr), 1), np.round(np.std(arr), 1)

    @classmethod
    def _mean_sem(cls, arr):
        if len(arr) == 0:
            return 0.0, 0.0
        elif len(arr) == 1:
            return np.round(np.mean(arr), 1), 0.0
        return np.round(np.mean(arr), 1), np.round(stats.sem(arr), 1)

    @classmethod
    @abstractmethod
    def _export_stats(cls, stats_files, output_file):
        raise NotImplementedError
