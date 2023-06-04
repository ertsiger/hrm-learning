from gym_hierarchical_subgoal_automata.envs.craftworld.craftworld_env import CraftWorldTasks
from gym_hierarchical_subgoal_automata.envs.waterworld.waterworld_env import WaterWorldTasks

# Folders
FOLDER_DEFAULT_SETTINGS = "01-default"
FOLDER_FLAT = "02-flat"
FOLDER_RESTRICTIONS = "03-restricted-hypothesis-space"
FOLDER_EXPLORATION = "04-exploration"
FOLDER_GOAL_COLLECTION = "05-goal-collection"
FOLDER_HANDCRAFTED = "06-handcrafted"

# Colour resources:
# - https://sashamaps.net/docs/resources/20-colors/
# - https://stats.stackexchange.com/questions/118033/best-series-of-colors-to-use-for-differentiating-series-in-publication-quality
# - https://personal.sron.nl/~pault/#fig:scheme_rainbow_discrete
COLOR_PALETTES = {
    1: ["#000000"],
    9: [
        "#882e72", "#1965b0", "#7bafde", "#4eb265", "#cae0ab", "#f7f056", "#ee8026", "#dc050c", "#72190e"
    ],
    13: [
        "#d1bbd7", "#ae76a3", "#882e72", "#1965b0", "#5289c7", "#7bafde", "#4eb265", "#90c987", "#cae0ab", "#f7f056",
        "#f4a736", "#e8601c", "#dc050c"
    ]
}

# Common RL parameters
GREEDY_EVAL_FREQ = 100
MAX_EPISODE_LENGTH = 1000
NUM_RUNS = 5
NUM_TASKS = 10

# Plotting and statistics parameters
NUM_COMP_EPISODES_PLOT = 4000
PLOT_FILENAME = "plot.json"
STATS_FILENAME = "stats.json"
OUT_STATS_FILENAME = "stats.out"

# Reinforcement Learning Tasks
CW_TASKS = [
    CraftWorldTasks.BATTER.value, CraftWorldTasks.BUCKET.value, CraftWorldTasks.COMPASS.value,
    CraftWorldTasks.LEATHER.value, CraftWorldTasks.PAPER.value, CraftWorldTasks.QUILL.value,
    CraftWorldTasks.SUGAR.value, CraftWorldTasks.BOOK.value, CraftWorldTasks.MAP.value,
    CraftWorldTasks.MILK_BUCKET.value, CraftWorldTasks.BOOK_AND_QUILL.value,
    CraftWorldTasks.MILK_BUCKET_AND_SUGAR.value, CraftWorldTasks.CAKE.value
]
CW_TASK_TO_LABEL = {
    task: ("".join([x.title() for x in task.split("-")])).replace("And", "&")
    for task in CW_TASKS
}

WW_TASKS = [
    WaterWorldTasks.RG.value, WaterWorldTasks.BC.value, WaterWorldTasks.MY.value, WaterWorldTasks.RG_BC.value,
    WaterWorldTasks.BC_MY.value, WaterWorldTasks.RG_MY.value, WaterWorldTasks.RGB.value, WaterWorldTasks.CMY.value,
    WaterWorldTasks.RGB_CMY.value
]
WW_TASK_TO_LABEL = {
    task: task
    for task in WW_TASKS
}

# Default arguments used in all experiments
DEFAULT_ARGS = [
    "--timed",

    "--num_tasks", f"{NUM_TASKS}",

    "--state_format", "full_obs",

    "--maximum_episode_length", f"{MAX_EPISODE_LENGTH}",

    "--greedy_eval_freq", f"{GREEDY_EVAL_FREQ}",
    "--greedy_eval_episodes", "1",

    "--curriculum_weight", "0.99",
    "--curriculum_scoring", "andreas",
    "--curriculum_src", "greedy",

    "--use_empty_obs"
]

DEFAULT_IHSA_HRL_ARGS = [
    "--rw_goal", "1.0",
    "--rw_step", "0.0",
    "--rw_deadend", "0.0",

    "--discount", "0.9",
    "--meta_discount", "0.99",

    "--er_size", "500000",
    "--er_start", "100000",
    "--meta_er_size", "10000",
    "--meta_er_start", "1000",
    "--tgt_net_update_freq", "1500",
    "--meta_tgt_net_update_freq", "500",

    "--exp_automaton_annealing_steps", "10000",

    "--opt", "rmsprop",
    "--q_select_num", "4",

    "--use_mutex_saturation",
    "--ilasp_timeout", "7200",
    "--allow_suboptimal",
]

DEFAULT_IHSA_CRM_ARGS = [
    "--er_start", "100000",
    "--batch_size", "32",
    "--tgt_net_update_freq", "1500",
    "--use_flat_hierarchy",
    "--exp_annealing_scale", "steps",
    "--discount", "0.99",
]


DEFAULT_DRQN_ARGS = [
    "--discount", "0.99",
    "--net_update_freq", "16",
    "--tgt_net_update_freq", "1500",
    "--batch_size", "1",
    "--er_size", "1000",
    "--er_start", "100",
    "--er_seq_length", "128",
    "--update_timescale", "step",
    "--lstm_method", "state+obs",
    "--lstm_hidden_size", "256",
    "--exp_annealing_duration", "300000",
    "--exp_annealing_scale", "episodes"
]


DEFAULT_DQN_ARGS = [
    "--discount", "0.9",
    "--net_update_freq", "1",
    "--tgt_net_update_freq", "1500",
    "--update_timescale", "step",
    "--batch_size", "32",
    "--er_size", "500000",
    "--er_start", "100000",
    "--exp_annealing_scale", "episodes",
    "--exp_annealing_duration", "300000"
]


def get_cw_args(grid_type, use_deadends, use_base_goal_examples, algorithm="hrl-nonflat", env=None):
    args = [
        "--curriculum_threshold", "0.85",
        "--grid_type", grid_type,
        "--grid_size", "7" if grid_type == "open_plan" else "13",
        "--num_lava", "1",  # will only be obeyed if use_lava is true,
        "--max_objs_per_class", "1" if grid_type == "open_plan" else "2",
        "--cw_num_conv_channels", "16", "32", "32",
        "--cw_linear_out_units", "256"
    ]

    if algorithm in ["hrl-flat", "hrl-nonflat"]:
        args.extend([
            "--lr", "0.0005",
            "--meta_lr", "0.0005",
            "--exp_subgoal_annealing_steps", "2000000",
        ])
    elif algorithm == "crm":
        args.extend([
            "--er_size", "1000000",
            "--lr", "0.0005",
            "--exp_annealing_duration", "100000000"
            #"--scale_er_num_states"
        ])
    elif algorithm == "dqn":
        args.extend([
            "--lr", "0.0005",
        ])
    elif algorithm == "drqn":
        args.extend([
            "--lr", "0.0001"
        ])
    else:
        raise RuntimeError(f"Error: Unknown or unsupported algorithm '{algorithm}'.")

    if grid_type == "four_rooms":
        args.append("--cw_use_max_pool")

    if use_deadends:
        args.append("--use_lava")

    if env is not None:
        args.extend(["--env", env])

    if use_base_goal_examples:
        if env is None:
            args.extend([
                "--min_goal_examples", "25", "150", "150", "150",
                "--use_top_shortest_goal_examples",
                "--num_top_shortest_goal_examples", "10", "10", "10", "10"
            ])
        else:
            args.extend([
                "--min_goal_examples", "25",
                "--use_top_shortest_goal_examples",
                "--num_top_shortest_goal_examples", "10"
            ])
    elif algorithm not in ["drqn", "dqn"]:
        args.extend([
            "--min_goal_examples", "1"
        ])

    return args


def get_ww_args(use_base_goal_examples, use_deadends, algorithm="hrl-nonflat", env=None):
    args = [
        "--curriculum_threshold", "0.75",
        "--ww_size", "400",
        "--ww_balls_per_color", "2",
        "--ww_ball_velocity", "30",
        "--ww_linear_out_units", "512", "512", "512",
        "--ww_meta_linear_out_units", "512", "512", "512"
    ]

    if algorithm.startswith("hrl"):
        args.extend([
            "--lr", "0.00001",
            "--meta_lr", "0.001",
            "--exp_subgoal_annealing_steps", "5000000",
        ])
    elif algorithm == "crm":
        args.extend([
            "--er_size", "1000000",
            "--lr", "0.00001",
            "--ww_layer_size", "512",
            "--exp_annealing_duration", "2000000"
            #"--scale_er_num_states"
        ])
    else:
        raise RuntimeError(f"Error: Unknown or unsupported algorithm '{algorithm}'.")

    if use_base_goal_examples:
        if env is None:
            args.extend([
                "--min_goal_examples", "25", "150", "150",
                "--use_top_shortest_goal_examples",
                "--num_top_shortest_goal_examples", "10", "10", "10",
            ])
        else:
            args.extend([
                "--min_goal_examples", "25",
                "--use_top_shortest_goal_examples",
                "--num_top_shortest_goal_examples", "10"
            ])
    else:
        args.extend([
            "--min_goal_examples", "1"
        ])

    if env is not None:
        args.extend(["--env", env])

    if use_deadends:
        args.append("--ww_avoid_black")
        args.extend(["--ww_num_black_balls", "2"])

    return args


def get_exploration_args(use_actions, use_automata, use_formulas):
    args = []
    if use_actions:
        args.append("--use_exploratory_action_options")
    if use_automata:
        args.append("--use_exploratory_automata_options")
    if use_formulas:
        args.append("--use_exploratory_formula_options")
    return args


def get_restricted_args(use_res_dependencies, use_res_observables, filter_res_observables):
    args = []
    if use_res_dependencies:
        args.append("--res_dep_enable")
    if use_res_observables:
        args.append("--res_obs_enable")
    if filter_res_observables:
        args.append("--filter_obs_enable")
    return args
