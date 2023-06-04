from abc import ABC, abstractmethod
import argparse
import os
from utils import file_utils


class ConfigGenerator(ABC):
    BASE_SEEDS = [
        25101993, 25041996, 31121960, 20091953, 30422020, 31071993, 11091714, 25071992, 1123581321, 31415926
    ]

    @classmethod
    def run(cls):
        args = cls._get_argparser().parse_args()

        seeds = cls._gen_random_seeds(args.num_runs)
        root_experiments_path = os.path.abspath(args.root_experiments_path)
        folder_name = os.path.join(root_experiments_path, args.experiment_folder_name)

        for i in range(1, args.num_runs + 1):
            experiment_path = os.path.join(folder_name, f"batch_{i}")
            cls._gen_experiments(args, experiment_path, seeds[i - 1])
            cls._gen_run_script(args, root_experiments_path, experiment_path, seeds[i - 1])
            cls._gen_condor_job(experiment_path, args.timed)

    @classmethod
    def _get_argparser(cls):
        parser = argparse.ArgumentParser()
        cls._add_general_args(parser)
        cls._add_new_args(parser)
        return parser

    @classmethod
    def _add_general_args(cls, parser):
        group = parser.add_argument_group("general")
        group.add_argument(
            "--domain",
            required=True,
            choices=["craftworld", "waterworld"],
            help="domain whose tasks will be used"
        )
        group.add_argument(
            "--num_runs",
            type=int,
            required=True,
            help="how many runs to create"
        )
        group.add_argument(
            "--root_experiments_path",
            required=True,
            help="folder where the experiment folders are created"
        )
        group.add_argument(
            "--experiment_folder_name",
            required=True,
            help="name of the experiment folder"
        )
        group.add_argument(
            "--env",
            default=None,
            help="name of a specific environment to generate the configuration file for (default: None)"
        )
        group.add_argument(
            "--timed",
            action="store_true",
            help="whether it is an experiment whose running time should be compared with others"
        )
        group.add_argument(
            "--num_tasks", "-t",
            type=int,
            default=1,
            help="number of tasks/instances used per domain (default: 1)"
        )
        group.add_argument(
            "--num_episodes",
            type=int, default=300000,
            help="number of episodes to run (default: 300000)"
        )
        group.add_argument(
            "--maximum_episode_length", "-m",
            type=int, default=1000,
            help="maximum episode length (default: 1000)"
        )

    @classmethod
    @abstractmethod
    def _add_new_args(cls, parser):
        raise NotImplementedError

    @classmethod
    def _gen_random_seeds(cls, num_runs):
        seeds = []
        seed_sum = 0
        for i in range(num_runs):
            seed = cls.BASE_SEEDS[i % len(cls.BASE_SEEDS)] + seed_sum
            seeds.append(seed)
            if (i + 1) % len(cls.BASE_SEEDS) == 0:
                seed_sum += 1
        return seeds

    @classmethod
    def _gen_experiments(cls, args, experiment_path, seed_value):
        file_utils.mkdir(experiment_path)
        cls._write_json_config(args, experiment_path, seed_value)

    @classmethod
    @abstractmethod
    def _write_json_config(cls, args, experiment_path, seed_value):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def _use_json_config(cls):
        raise NotImplementedError

    @classmethod
    def _gen_run_script(cls, args, root_experiments_path, experiments_folder, seed):
        script_path = os.path.join(experiments_folder, "run.sh")
        venv_path = os.path.join(root_experiments_path, "anaconda3")
        env_name = "py37"

        with open(script_path, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write("ulimit -Sn 10000\n")  # number of files that can be simultaneously opened
            f.write(f"export PATH={os.path.join(venv_path, 'bin')}:$PATH\n")
            f.write(f"export LD_PRELOAD={os.path.join(venv_path, 'envs', env_name, 'lib', 'libjemalloc.so')}:$LD_PRELOAD\n")
            f.write("eval \"$(conda shell.bash hook)\"\n")
            f.write(f"conda activate {env_name}")
            f.write("\n\n")

            cls._write_algorithm_call_cmd(f, args, root_experiments_path, experiments_folder, seed)

        os.chmod(script_path, 0o777)

    @classmethod
    @abstractmethod
    def _write_algorithm_call_cmd(cls, f, args, root_experiments_path, experiments_folder, seed):
        raise NotImplementedError

    @classmethod
    def _gen_condor_job(cls, experiment_path, timed_experiment):
        job_path = os.path.join(experiment_path, "condor.config")
        script_path = os.path.join(experiment_path, "run.sh")

        with open(job_path, 'w') as f:
            f.write("universe = vanilla\n")
            f.write(f"executable = {script_path}\n")
            if cls._use_json_config():
                f.write("arguments = $(group)/config.json\n")
            f.write("output = $(group)/uname.$(Process).out\n")
            f.write("error = $(group)/uname.$(Process).err\n")
            f.write("log = $(group)/uname.$(Process).log\n")

            if timed_experiment:
                # 3.40GHz Intel Core i7-6700 processors
                lab_machines = "arc|edge|line|point|sprite"
                machines = [
                    "cod", "crete", "dongting", "empani", "euboea", "fushui", "halavra", "icaria", "kythros", "mort",
                    "muli", "nantwich", "patmos", "pergoussa", "preston", "taizi", "texel26", "wuding", "zuli"
                ]
            else:
                # 2.90GHz Intel Core i7-10700, 4.20GHz Intel Core i7-7700K, and 3.20GHz Intel Core i7-8700 processors
                lab_machines = "arc|edge|line|point|sprite|vertex|ray|texel|oak|willow"
                machines = [
                    "allier", "blavet", "chorley", "cod", "crete", "deule", "dongting", "eliteone-306", "empani",
                    "euboea", "fushui", "glastonbury", "halavra", "icaria", "icaria", "kythira", "kythros", "louge",
                    "mort", "muli", "nantwich", "padiham", "patmos", "patmos", "pergoussa", "preston", "rere",
                    "rotherham", "ryde", "shilka", "taizi", "vire", "wuding", "zuli"
                ]

            f.write(f"requirements = (regexp(\"^({lab_machines})[0-9][0-9]\", Machine))")
            for m in machines:
                f.write(f" || (Machine == \"{m}.doc.ic.ac.uk\")")
            # f.write("requirements = (OpSysMajorVer >= 20)")
            f.write("\n\n")

            f.write(f"group = {experiment_path}\n")
            f.write("queue 1\n\n")
