from abc import abstractmethod
from collections import namedtuple
from config.generators.config_generator import ConfigGenerator
from gym_hierarchical_subgoal_automata.envs.base.base_env import BaseEnv
from gym_hierarchical_subgoal_automata.envs.craftworld.craftworld_env import CraftWorldTasks
from gym_hierarchical_subgoal_automata.envs.craftworld.minigrid import CustomMiniGrid
from gym_hierarchical_subgoal_automata.envs.waterworld.waterworld_env import WaterWorldEnv, WaterWorldTasks
import os
from reinforcement_learning.learning_algorithm import LearningAlgorithm
from utils import file_utils

Environment = namedtuple("Environment", ["name", "automaton_name", "hierarchy_level", "dependencies"])


class RLConfigGenerator(ConfigGenerator):
    @classmethod
    def _add_new_args(cls, parser):
        cls._add_general_args_extended(parser)
        cls._add_rl_args(parser)
        cls._add_curriculum_args(parser)
        cls._add_general_env_args(parser)
        cls._add_craftworld_args(parser)
        cls._add_waterworld_args(parser)
        cls._add_extra_args(parser)

    @classmethod
    def _add_general_args_extended(cls, parser):
        group = parser.add_argument_group("general_extended")
        group.add_argument(
            "--use_gpu",
            action="store_true",
            help="whether to use the gpu"
        )
        group.add_argument(
            "--not_share_starting_seed",
            action="store_true",
            help="whether tasks being learned have different starting seed to generate instances"
        )

    @classmethod
    def _add_rl_args(cls, parser):
        group = parser.add_argument_group("rl_general")
        group.add_argument(
            "--state_format",
            default="tabular",
            choices=["tabular", "one_hot", "full_obs"],
            help="format of the states perceived by the agent (default: tabular)"
        )
        group.add_argument(
            "--greedy_eval_freq",
            type=int, default=100,
            help="every how many episodes a greedy policy evaluation is performed across tasks (default: 100)"
        )
        group.add_argument(
            "--greedy_eval_episodes",
            type=int, default=1,
            help="for how many episodes a greedy policy is evaluated for each task (default: 1)"
        )

    @classmethod
    def _add_curriculum_args(cls, parser):
        group = parser.add_argument_group("curriculum_learning")
        group.add_argument(
            "--curriculum_weight",
            type=float, default=0.99,
            help="weight used in the weighted average that estimates how well we are doing in a task (default: 0.99)"
        )
        group.add_argument(
            "--curriculum_softmax_temp",
            type=float, default=2.0,
            help="temperature used in the softmax in Pierrot's scoring method (default: 2.0)"
        )
        group.add_argument(
            "--curriculum_threshold",
            type=float, default=0.9,
            help="average return threshold used to determine whether a task has been mastered (default: 0.9)"
        )
        group.add_argument(
            "--curriculum_scoring",
            default="andreas",
            choices=["andreas", "pierrot"],
            help="method used to determine the score for a given task, which is then used to determine a probability "
                 "(andreas, pierrot) (default: andreas)"
        )
        group.add_argument(
            "--curriculum_src",
            default="greedy",
            choices=["exploratory", "greedy"],
            help="where do the returns used in the curriculum come from (exploratory, greedy) (default: greedy)"
        )

    @classmethod
    def _add_general_env_args(cls, parser):
        """
        Environment arguments - Even though some of them may look specific to IHSA, they are not: these have to do
        with how is the underlying automaton model traversed.
        """
        group = parser.add_argument_group("environment")
        group.add_argument(
            "--use_uncompressed_traces",
            action="store_true",
            help="whether uncompressed traces are used to traverse and/or learn the automata"
        )
        group.add_argument(
            "--use_empty_obs",
            action="store_true",
            help="whether empty observations are used to traverse and/or learn the automata"
        )
        group.add_argument(
            "--use_flat_hierarchy",
            action="store_true",
            help="whether to use the flat hierarchy to model the environment and, in the case the handcrafted machines"
                 " are used, whether to use the flat version"
        )
        group.add_argument(
            "--random_restart",
            action="store_true",
            help="whether each task is randomly reinitialized when completed according to its seed (this only applies"
                 " to cases using function approximation, like full_obs and not tabular)"
        )

    @classmethod
    def _add_craftworld_args(cls, parser):
        group = parser.add_argument_group("craftworld")
        group.add_argument(
            "--grid_type",
            default="open_plan",
            choices=["open_plan", "corridor", "four_rooms"],
            help="type of grid to use (default: open_plan)"
        )
        group.add_argument(
            "--grid_size",
            type=int, default=7,
            help="size of the grid (default: 7)"
        )
        group.add_argument(
            "--grid_right_rooms_even",
            action="store_true",
            help="whether the doors on the right of the four rooms grid have the same size (else, one room has one more"
                 " row than the other)"
        )
        group.add_argument(
            "--use_lava",
            action="store_true",
            help="whether to use lava locations in the grid (for open plan grids, specify number of locations using"
                 " num_lava)"
        )
        group.add_argument(
            "--num_lava",
            type=int, default=1,
            help="number of lava cells in the grid"
        )
        group.add_argument(
            "--use_lava_walls",
            action="store_true",
            help="whether the walls of the grid are lava"
        )
        group.add_argument(
            "--lava_locations",
            default="door_intersections",
            choices=["door_intersections", "all_corners", "some_corners"],
            help="where to place lava in four rooms (default: door_intersections)"
        )
        group.add_argument(
            "--max_objs_per_class",
            type=int, default=1,
            help="maximum number of objects per class of object"
        )
        group.add_argument(
            "--cw_num_conv_channels",
            nargs='+', type=int, default=[16, 32, 32],
            help="how many output channels convolutional layers have for CraftWorld (default: 16, 32, 32)"
        )
        group.add_argument(
            "--cw_linear_out_units",
            nargs='+', type=int, default=[128],
            help="how many outputs the intermediate linear layers have in CraftWorld (default: 128)"
        )
        group.add_argument(
            "--cw_use_max_pool",
            action="store_true",
            help="whether to use a max pooling layer after the first convolutional layer in CraftWorld"
        )

    @classmethod
    def _add_waterworld_args(cls, parser):
        group = parser.add_argument_group("waterworld")
        group.add_argument(
            "--ww_size",
            type=int, default=400,
            help="size of the waterworld (default: 400)"
        )
        group.add_argument(
            "--ww_balls_per_color",
            type=int, default=2,
            help="number of balls of each color (default: 2)"
        )
        group.add_argument(
            "--ww_ball_velocity",
            type=int, default=30,
            help="velocity of the balls in the environment (default: 30)"
        )
        group.add_argument(
            "--ww_linear_out_units",
            nargs='+', type=int, default=[1024, 1024, 1024],
            help="how many outputs the intermediate linear layers have in WaterWorld (default: 1024, 1024, 1024)"
        )
        group.add_argument(
            "--ww_meta_linear_out_units",
            nargs='+', type=int, default=[256, 256],
            help="how many outputs the intermediate linear layers of the metacontroller have in WaterWorld"
                 " (default: 256, 256)"
        )
        group.add_argument(
            "--ww_use_empty",
            action="store_true",
            help="use the empty proposition as an extra observable (enable it for tasks involving it!)"
        )
        group.add_argument(
            "--ww_avoid_black",
            action="store_true",
            help="whether to avoid black colored balls (only some tasks support this!)"
        )
        group.add_argument(
            "--ww_num_black_balls",
            type=int, default=2,
            help="number of black balls (if avoidance for this type is enabled) (default: 2)"
        )

    @classmethod
    @abstractmethod
    def _add_extra_args(cls, parser):
        raise NotImplementedError

    @classmethod
    def _write_json_config(cls, args, experiment_path, seed_value):
        file_utils.write_json_obj(
            cls._get_experiment_config(args, experiment_path, seed_value),
            os.path.join(experiment_path, "config.json")
        )

    @classmethod
    def _use_json_config(cls):
        return True

    @classmethod
    def _get_experiment_config(cls, args, experiment_path, seed_value):
        config = {
            LearningAlgorithm.DEBUG: False,  # Do not show progress by default
            LearningAlgorithm.USE_GPU: args.use_gpu,
            LearningAlgorithm.USE_SEED: seed_value is not None,
            LearningAlgorithm.USE_ENVIRONMENT_SEED: True,
            LearningAlgorithm.EXPERIMENT_FOLDER_NAME: experiment_path,
            LearningAlgorithm.ENV_NAME: args.domain,
            LearningAlgorithm.ENV_CONFIG: cls._get_environments(args, seed_value),
            LearningAlgorithm.NUM_TASKS_PER_ENVIRONMENT: args.num_tasks,
            LearningAlgorithm.TRAINING_ENABLE: True  # Might be overwritten later by children classes
        }

        if seed_value is not None:
            config[LearningAlgorithm.SEED_VALUE] = seed_value

        cls._set_general_rl_config(config, args)
        cls._set_env_config(config, args)
        cls._set_checkpoint_config(config, experiment_path)
        cls._set_algorithm_config(config, args)
        return config

    @classmethod
    def _get_environments(cls, args, seed_value):
        if args.env is not None:  # just for simple debugging
            envs = [Environment(args.env, "m0", 1, [])]
            num_levels = 1
        elif args.domain == LearningAlgorithm.ENV_NAME_CRAFTWORLD:
            envs = [
                Environment(CraftWorldTasks.BUCKET.value, "m0", 1, []),
                Environment(CraftWorldTasks.SUGAR.value, "m1", 1, []),
                Environment(CraftWorldTasks.BATTER.value, "m2", 1, []),
                Environment(CraftWorldTasks.PAPER.value, "m3", 1, []),
                Environment(CraftWorldTasks.COMPASS.value, "m4", 1, []),
                Environment(CraftWorldTasks.LEATHER.value, "m5", 1, []),
                Environment(CraftWorldTasks.QUILL.value, "m6", 1, []),
                Environment(CraftWorldTasks.MILK_BUCKET.value, "m7", 2, ["m0"]),
                Environment(CraftWorldTasks.MAP.value, "m8", 2, ["m3", "m4"]),
                Environment(CraftWorldTasks.BOOK.value, "m9", 2, ["m3", "m5"]),
                Environment(CraftWorldTasks.MILK_BUCKET_AND_SUGAR.value, "m10", 3, ["m1", "m7"]),
                Environment(CraftWorldTasks.BOOK_AND_QUILL.value, "m11", 3, ["m6", "m9"]),
                Environment(CraftWorldTasks.CAKE.value, "m12", 4, ["m2", "m10"])
            ]
            num_levels = 4
        elif args.domain == LearningAlgorithm.ENV_NAME_WATERWORLD:
            envs = [
                Environment(WaterWorldTasks.RG.value, "m0", 1, []),
                Environment(WaterWorldTasks.BC.value, "m1", 1, []),
                Environment(WaterWorldTasks.MY.value, "m2", 1, []),
                Environment(WaterWorldTasks.RG_BC.value, "m3", 2, ["m0", "m1"]),
                Environment(WaterWorldTasks.BC_MY.value, "m4", 2, ["m1", "m2"]),
                Environment(WaterWorldTasks.RG_MY.value, "m5", 2, ["m0", "m2"]),
                Environment(WaterWorldTasks.RGB.value, "m6", 2, ["m0"]),
                Environment(WaterWorldTasks.CMY.value, "m7", 2, ["m2"]),
                Environment(WaterWorldTasks.RGB_CMY.value, "m8", 3, ["m6", "m7"])
            ]
            num_levels = 3
        else:
            raise RuntimeError(f"Error: Unknown domain '{args.domain}'.")

        # if len(args.min_goal_examples) > 1 and len(args.min_goal_examples) != num_levels:
        #     raise RuntimeError(
        #         f"Error: There must be exactly {num_levels} items in the {IHSAAlgorithmBase.MIN_GOAL_EXAMPLES}"
        #         f"argument."
        #     )

        if args.not_share_starting_seed:
            seeds = [seed_value + i for i in range(len(envs))]
        else:
            seeds = [seed_value] * len(envs)

        return [{
            LearningAlgorithm.ENV_CONFIG_NAME: env.name,
            LearningAlgorithm.ENV_CONFIG_AUTOMATON_NAME: env.automaton_name,
            LearningAlgorithm.ENV_CONFIG_HIERARCHY_LEVEL: env.hierarchy_level,
            LearningAlgorithm.ENV_CONFIG_DEPENDENCIES: env.dependencies,
            LearningAlgorithm.ENV_CONFIG_STARTING_SEED: seed
        } for (env, seed) in zip(envs, seeds)]

    @classmethod
    def _set_general_rl_config(cls, config, args):
        # General duration and format parameters
        config[LearningAlgorithm.NUM_EPISODES] = args.num_episodes
        config[LearningAlgorithm.MAX_EPISODE_LENGTH] = args.maximum_episode_length
        config[LearningAlgorithm.STATE_FORMAT] = args.state_format

        # Curriculum learning parameters
        config[LearningAlgorithm.CURRICULUM_WEIGHT] = args.curriculum_weight
        config[LearningAlgorithm.CURRICULUM_SOFTMAX_TEMP] = args.curriculum_softmax_temp
        config[LearningAlgorithm.CURRICULUM_LEVEL_THRESHOLD] = args.curriculum_threshold
        config[LearningAlgorithm.CURRICULUM_SCORING_METHOD] = args.curriculum_scoring
        config[LearningAlgorithm.CURRICULUM_RETURN_SRC] = args.curriculum_src

        # Evaluation of the policies
        config[LearningAlgorithm.GREEDY_EVALUATION_ENABLE] = True
        config[LearningAlgorithm.GREEDY_EVALUATION_FREQUENCY] = args.greedy_eval_freq
        config[LearningAlgorithm.GREEDY_EVALUATION_EPISODES] = args.greedy_eval_episodes

    @classmethod
    def _set_env_config(cls, config, args):
        config[BaseEnv.COMPRESS_OBS] = not args.use_uncompressed_traces
        config[BaseEnv.IGNORE_EMPTY_OBS] = not args.use_empty_obs
        config[BaseEnv.RANDOM_RESTART] = args.random_restart
        config[BaseEnv.USE_FLAT_HIERARCHY] = args.use_flat_hierarchy

        if args.domain == LearningAlgorithm.ENV_NAME_CRAFTWORLD:
            cls._set_craftworld_config(config, args)
        elif args.domain == LearningAlgorithm.ENV_NAME_WATERWORLD:
            cls._set_waterworld_config(config, args)
        else:
            raise RuntimeError(f"Error: Unknown domain {args.domain}.")

    @classmethod
    def _set_craftworld_config(cls, config, args):
        config[CustomMiniGrid.GRID_PARAMS] = {
            CustomMiniGrid.GRID_TYPE: args.grid_type,
            CustomMiniGrid.GRID_WIDTH: args.grid_size,
            CustomMiniGrid.GRID_HEIGHT: args.grid_size,
            CustomMiniGrid.GRID_SIZE: args.grid_size,
            CustomMiniGrid.GRID_RIGHT_ROOMS_EVEN: args.grid_right_rooms_even,
            CustomMiniGrid.GRID_LAVA_LOCS: args.lava_locations,
            CustomMiniGrid.USE_LAVA: args.use_lava,
            CustomMiniGrid.NUM_LAVA: args.num_lava,
            CustomMiniGrid.USE_LAVA_WALLS: args.use_lava_walls,
            CustomMiniGrid.MAX_NUM_OBJS_PER_CLASS: args.max_objs_per_class
        }

    @classmethod
    def _set_waterworld_config(cls, config, args):
        config[WaterWorldEnv.MAX_X] = args.ww_size
        config[WaterWorldEnv.MAX_Y] = args.ww_size
        config[WaterWorldEnv.BALLS_PER_COLOR] = args.ww_balls_per_color
        config[WaterWorldEnv.BALL_VELOCITY] = args.ww_ball_velocity
        config[WaterWorldEnv.USE_EMPTY] = args.ww_use_empty
        config[WaterWorldEnv.AVOID_BLACK] = args.ww_avoid_black
        config[WaterWorldEnv.NUM_BLACK_BALLS] = args.ww_num_black_balls

    @classmethod
    def _set_checkpoint_config(cls, config, experiment_path):
        config[LearningAlgorithm.CHECKPOINT_ENABLE] = True
        config[LearningAlgorithm.CHECKPOINT_FREQUENCY] = 500  # int(config[LearningAlgorithm.NUM_EPISODES] / 10)
        config[LearningAlgorithm.CHECKPOINT_FOLDER] = experiment_path

    @classmethod
    @abstractmethod
    def _set_algorithm_config(cls, config, args):
        raise NotImplementedError

    @classmethod
    def _write_algorithm_call_cmd(cls, f, args, root_experiments_path, experiments_folder, seed):
        f.write("if [ $# -ne 1 ]; then\n")
        f.write("    echo $0: usage: config-file\n")
        f.write("    exit 1\n")
        f.write("fi\n\n")

        algorithm_script_path = os.path.join(
            root_experiments_path,
            "hrm-learning/src/run_algorithm.py"
        )
        f.write(f"{algorithm_script_path} $1")
