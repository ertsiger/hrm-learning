#!/usr/bin/env python3

import argparse
from baselines.deepsynth.deepsynth import DeepSynth
from baselines.jirp.jirp import JIRP
from baselines.lrm.lrm import LRM
import os
import pickle
from utils.file_utils import is_file_empty, rm_file


def get_argparser():
    parser = argparse.ArgumentParser()

    # General
    group = parser.add_argument_group("general")
    group.add_argument("--algorithm", required=True, help="which baseline algorithm to run (deepsynth, jirp, lrm)")
    group.add_argument("--domain", required=True, help="which domain (craftworld, waterworld)")
    group.add_argument("--task", required=True, help="what task for the specified domain")
    group.add_argument("--num_instances", type=int, default=10, help="number of instances of the specified task (e.g., number of grids)")
    group.add_argument("--seed", required=True, type=int, help="seed use to instantiate the tasks and run the experiments")
    group.add_argument("--output_dir", required=True, help="where to save output files and checkpoints")
    group.add_argument("--num_episodes", type=int, default=150000, help="number of episodes to run")
    group.add_argument("--episode_length", type=int, default=1000, help="maximum number of steps per episode")
    group.add_argument("--timeout", type=int, default=7200, help="timeout for learning the machines")

    # Algorithms
    algorithms_group = parser.add_argument_group("algorithms")

    group = algorithms_group.add_argument_group("deepsynth")
    group.add_argument("--deepsynth_cbmc_path", help="path to cbmc")

    group = algorithms_group.add_argument_group("jirp")
    group.add_argument("--jirp_max_states", type=int, default=10, help="maximum number of states that JIRP will use")

    group = algorithms_group.add_argument_group("lrm")
    group.add_argument("--lrm_max_states", type=int, default=10, help="maximum number of states that LRM will use")

    # Environments
    envs_group = parser.add_argument_group("environments")
    group = envs_group.add_argument_group("craftworld")
    group.add_argument("--use_lava", action="store_true", help="whether to have a place labeled with lava")

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

    algorithm = load_last_checkpoint(args.output_dir)
    if algorithm is None:
        loaded_checkpoint = False
        if args.algorithm == "deepsynth":
            assert args.deepsynth_cbmc_path is not None
            algorithm = DeepSynth(args)
        elif args.algorithm == "jirp":
            algorithm = JIRP(args)
        elif args.algorithm == "lrm":
            algorithm = LRM(args)
    else:
        loaded_checkpoint = True
    algorithm.run(loaded_checkpoint)
