from config.generators.config_generator import ConfigGenerator
from gym_hierarchical_subgoal_automata.envs.craftworld.craftworld_env import CraftWorldTasks
from gym_hierarchical_subgoal_automata.envs.waterworld.waterworld_env import WaterWorldTasks
import os


class BaselineConfigGenerator(ConfigGenerator):
    @classmethod
    def _add_new_args(cls, parser):
        group = parser.add_argument_group("baseline")
        group.add_argument(
            "--algorithm",
            required=True,
            choices=["deepsynth", "jirp", "lrm"],
            help="name of the baseline algorithm"
        )

    @classmethod
    def _write_algorithm_call_cmd(cls, f, args, root_experiments_path, experiments_folder, seed):
        arguments = [
            f"--algorithm={args.algorithm}",
            "--domain", args.domain,
            f"--task='{args.env}'",
            "--num_instances", str(args.num_tasks),
            "--seed", str(seed),
            f"--output_dir='{experiments_folder}'"
        ]

        if args.algorithm == "deepsynth":
            arguments.extend([
                "--deepsynth_cbmc_path", os.path.join(root_experiments_path, "cbmc/src/cbmc")
            ])
        elif args.algorithm == "jirp":
            arguments.extend([
                "--jirp_max_states", str(cls._get_max_states_for_task(args.env))
            ])
        elif args.algorithm == "lrm":
            arguments.extend([
                "--lrm_max_states", str(cls._get_max_states_for_task(args.env))
            ])

        baseline_script_path = os.path.join(
            root_experiments_path, "hrm-learning/src/run_baseline.py"
        )
        f.write(f"{baseline_script_path} {' '.join(arguments)}")

    @classmethod
    def _get_max_states_for_task(cls, task_name):
        if task_name == CraftWorldTasks.MILK_BUCKET.value:
            return 4
        elif task_name == CraftWorldTasks.BOOK.value:
            return 8
        elif task_name == CraftWorldTasks.BOOK_AND_QUILL.value:
            return 21
        elif task_name == CraftWorldTasks.CAKE.value:
            return 15
        elif task_name == WaterWorldTasks.RG.value:
            return 3
        elif task_name == WaterWorldTasks.RG_BC.value:
            return 8
        elif task_name == WaterWorldTasks.RGB_CMY.value:
            return 12
        raise RuntimeError(f"Error: Task {task_name} is not associated with a maximum number of states.")

    @classmethod
    def _use_json_config(cls):
        return False

    @classmethod
    def _write_json_config(cls, args, experiment_path, seed_value):
        pass


if __name__ == "__main__":
    BaselineConfigGenerator.run()
