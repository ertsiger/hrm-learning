#!/usr/bin/env python3

import argparse
from ilasp.ilasp_common import set_ilasp_env_variables
import os
import pickle
from reinforcement_learning.dqn_algorithm import DQNAlgorithm
from reinforcement_learning.drqn_algorithm import DRQNAlgorithm
from reinforcement_learning.ihsa_crm_algorithm import IHSAAlgorithmCRM
from reinforcement_learning.ihsa_dqrm_algorithm import IHSAAlgorithmDQRM
from reinforcement_learning.ihsa_hrl_tabular_algorithm import IHSAAlgorithmHRLTabular
from reinforcement_learning.ihsa_hrl_dqn_algorithm import IHSAAlgorithmHRLDQN
from reinforcement_learning.learning_algorithm import LearningAlgorithm
import sys
from utils.container_utils import get_param
from utils.file_utils import is_file_empty, read_json_file, rm_file


def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="path to the input configuration")
    return parser


def get_checkpoint_filenames(checkpoint_folder):
    if os.path.exists(checkpoint_folder):
        return [x for x in os.listdir(checkpoint_folder)
                if x.startswith("checkpoint") and not is_file_empty(os.path.join(checkpoint_folder, x))]
    return []


def checkpoints_exist(checkpoint_folder):
    return len(get_checkpoint_filenames(checkpoint_folder)) > 0


def get_last_checkpoint_filename(checkpoint_folder):
    checkpoint_filenames = get_checkpoint_filenames(checkpoint_folder)
    checkpoint_filenames.sort(key=lambda x: int(x[len("checkpoint-"):-len(".pickle")]))
    return os.path.join(checkpoint_folder, checkpoint_filenames[-1])


def load_last_checkpoint(checkpoint_folder):
    while checkpoints_exist(checkpoint_folder):
        checkpoint_filename = get_last_checkpoint_filename(checkpoint_folder)
        try:
            with open(checkpoint_filename, 'rb') as f:
                return pickle.load(f)
        except (pickle.UnpicklingError, EOFError) as e:
            print(f"Warning: Checkpoint {checkpoint_filename} was corrupted. It has been removed. Original message: {e}.")
            rm_file(checkpoint_filename)
    return None


if __name__ == "__main__":
    args = get_argparser().parse_args()
    config = read_json_file(args.config_file)
    set_ilasp_env_variables(os.path.dirname(sys.argv[0]))

    algorithm = None
    loaded_checkpoint = False
    if get_param(config, LearningAlgorithm.CHECKPOINT_ENABLE):
        algorithm = load_last_checkpoint(get_param(config, LearningAlgorithm.CHECKPOINT_FOLDER))
        loaded_checkpoint = algorithm is not None

    if not loaded_checkpoint:
        algorithm_name = get_param(config, "algorithm")
        if algorithm_name is None or algorithm_name == "ihsa-hrl":
            state_format = get_param(config, LearningAlgorithm.STATE_FORMAT)
            if state_format == LearningAlgorithm.STATE_FORMAT_TABULAR:
                algorithm = IHSAAlgorithmHRLTabular(config)
            elif state_format in [LearningAlgorithm.STATE_FORMAT_ONE_HOT, LearningAlgorithm.STATE_FORMAT_FULL_OBS]:
                algorithm = IHSAAlgorithmHRLDQN(config)
            else:
                raise RuntimeError(f"Error: Unrecognized observation format '{state_format}'.")
        elif algorithm_name == "ihsa-crm":
            algorithm = IHSAAlgorithmCRM(config)
        elif algorithm_name == "ihsa-dqrm":
            algorithm = IHSAAlgorithmDQRM(config)
        elif algorithm_name == "drqn":
            algorithm = DRQNAlgorithm(config)
        elif algorithm_name == "dqn":
            algorithm = DQNAlgorithm(config)
        else:
            raise RuntimeError(f"Error: Unrecognized algorithm name '{algorithm_name}'.")
    algorithm.checkpoint_folder = algorithm.checkpoint_folder.replace("neurips-experiments-thresh", "icml-experiments")
    algorithm.export_folder_names = {
        task: folder.replace("neurips-experiments-thresh", "icml-experiments")
        for task, folder in algorithm.export_folder_names.items()
    }
    algorithm.models_folder = algorithm.models_folder.replace("neurips-experiments-thresh", "icml-experiments")
    algorithm._stats_summary_path = algorithm._stats_summary_path.replace("neurips-experiments-thresh", "icml-experiments")
    algorithm.run(loaded_checkpoint)
