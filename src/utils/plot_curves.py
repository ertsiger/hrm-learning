import argparse
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
import os
import pandas as pd
from reinforcement_learning.ihsa_base_algorithm import IHSAAlgorithmBase
from scipy import stats
from tqdm import tqdm
from utils.file_utils import read_json_file

RW_IF_GOAL = 1.0

FIG_RW_ID = 0
FIG_STEPS_ID = 1


def moving_average(interval, args):
    interval_df = pd.DataFrame(interval)
    if args.avg_method == "window":
        roll = interval_df.rolling(window=args.window_size, min_periods=1)
    elif args.avg_method == "ewa":
        roll = interval_df.ewm(alpha=args.ewa_alpha, adjust=False, min_periods=1)
    else:
        raise RuntimeError(f"Error: Unknown method {args.avg_method}.")
    return roll.mean().squeeze().to_numpy()


def get_compressed_learning_episodes(learning_episodes, max_diff):
    new_episodes = []
    for ep in learning_episodes:
        if len(new_episodes) == 0 or (ep - new_episodes[-1]) >= max_diff:
            new_episodes.append(ep)
    return new_episodes


def read_reward_steps_list(filename, args):
    try:
        reward_steps = pd.read_csv(filename, sep=';', header=None)
    except pd.errors.EmptyDataError:
        return [], []

    rewards, steps = reward_steps[1], reward_steps[2]
    if len(rewards) > 1:
        rewards = rewards.squeeze()
        steps = steps.squeeze()
    rewards_np, steps_np = rewards.to_numpy(), steps.to_numpy()

    # Add rewards and steps for the 0-th timestep
    rewards_np = np.pad(rewards_np, pad_width=(1, 0), constant_values=0)
    steps_np = np.pad(steps_np, pad_width=(1, 0), constant_values=args.max_episode_length)

    # Unsolved tasks are given the maximum episode length as #steps
    steps_np[rewards_np < RW_IF_GOAL] = args.max_episode_length
    return rewards_np, steps_np


def read_automaton_learning_episodes(experiment_folder):
    stats_filename = os.path.join(experiment_folder, "..", "stats.json")
    try:
        stats = read_json_file(stats_filename)
        if "mode" in stats and stats["mode"] == "learn":
            return stats["domains"][os.path.basename(experiment_folder)]["episodes"]["learned_automaton"]
    except IOError:
        print(f"Warning: The file '{stats_filename}' does not exist.")
    return []


def plot_curve(figure_id, avg, std, min_val, max_val, learning_episodes, setting_conf, args):
    plt.figure(figure_id)
    x_axis = range(0, args.num_episodes + 1, args.greedy_evaluation_frequency)
    plt.plot(x_axis, avg, color=setting_conf["colour"], label=setting_conf["label"])

    if args.show_sem:
        plt.fill_between(
            x_axis,
            np.clip(avg - std, min_val, max_val),
            np.clip(avg + std, min_val, max_val),
            alpha=0.1,
            color=setting_conf["colour"]
        )

    for vl in learning_episodes:
        if vl <= args.num_episodes:
            plt.axvline(x=vl, alpha=0.35, color=setting_conf["colour"], dashes=(1, 2), zorder=-1000)


def create_argparser():
    parser = argparse.ArgumentParser(
        description='Plots learning curves. It assumes rewards are on [0, 1] scale. A reward of 1 is received if the '
                    'goal is achieved and 0 otherwise.'
    )
    parser.add_argument("--config_file", "-c", required=True, help="file containing the configuration")
    parser.add_argument("--num_runs", "-r", required=True, type=int, help="number of runs")
    parser.add_argument("--num_instances", "-i", required=True, type=int, help="number of instances for each run")
    parser.add_argument("--num_episodes", "-e", required=True, type=int, help="number of episodes to plot")

    parser.add_argument("--avg_method", default="window", help="method for averaging learning curve (window, ewa)")
    parser.add_argument("--window_size", "-w", type=int, default=10, help="size of the averaging window")
    parser.add_argument("--ewa_alpha", type=float, default=0.01, help="alpha for exponential weighted average")

    parser.add_argument("--use_tex", action="store_true", help="whether to plot the strings using TeX")
    parser.add_argument("--max_episode_length", type=int, default=1000, help="maximum length of an episode")

    parser.add_argument(
        "--use_greedy_traces", "-g", action="store_true", help="whether to use the traces that use the greedy policy"
    )
    parser.add_argument(
        "--greedy_evaluation_frequency", type=int, default=1,
        help="every how many episodes was the greedy policy evaluated"
    )

    parser.add_argument("--plot_title", "-t", default=None, help="the title of the plot")
    parser.add_argument("--no_legend", action="store_true", help="whether not to include legend")
    parser.add_argument("--legend_loc", default="out", help="where to place the legend (out, lower right)")
    parser.add_argument(
        "--learning_ep_compression", type=int, default=0,
        help="number of automaton learning episodes to compress in one"
    )

    parser.add_argument("--show_sem", action="store_true", help="whether to show the standard error of the mean")

    return parser


def get_num_plotted_steps(args):
    return args.num_episodes // args.greedy_evaluation_frequency + 1  # +1 is due to the 0-th step


def get_avg_instance_rw_steps(run_folder, args):
    rw_steps_folder = os.path.join(
        run_folder,
        IHSAAlgorithmBase.REWARD_STEPS_GREEDY_FOLDER if args.use_greedy_traces else IHSAAlgorithmBase.REWARD_STEPS_FOLDER
    )
    rw_steps_files = [os.path.join(rw_steps_folder, x) for x in os.listdir(rw_steps_folder)]

    assert len(rw_steps_files) == args.num_instances

    rw_sum, steps_sum = None, None

    for file in rw_steps_files:
        rw, steps = read_reward_steps_list(file, args)
        if rw_sum is None:
            rw_sum, steps_sum = rw, steps
        else:
            rw_sum += rw
            steps_sum += steps

    # If the run does not have minimum of episodes, then consider it unfinished
    if len(rw_sum) < get_num_plotted_steps(args):
        rw_sum = np.zeros(get_num_plotted_steps(args))
        steps_sum = np.ones(get_num_plotted_steps(args)) * args.num_instances * args.max_episode_length

    return rw_sum / args.num_instances, steps_sum / args.num_instances


def get_avg_run_rw_steps(run_folders, args):
    assert len(run_folders) == args.num_runs

    run_rws, run_steps, learning_episodes = [], [], []
    for run_folder in run_folders:
        avg_run_reward, avg_run_steps = get_avg_instance_rw_steps(run_folder, args)
        run_rws.append(avg_run_reward[:get_num_plotted_steps(args)])
        run_steps.append(avg_run_steps[:get_num_plotted_steps(args)])
        learning_episodes.extend(read_automaton_learning_episodes(run_folder))

    avg_rw, std_rw = np.mean(run_rws, axis=0), stats.sem(run_rws, axis=0)
    avg_steps, std_steps = np.mean(run_steps, axis=0), stats.sem(run_steps, axis=0)

    return (
        moving_average(avg_rw, args), moving_average(std_rw, args),
        moving_average(avg_steps, args), moving_average(std_steps, args),
        get_compressed_learning_episodes(sorted(set(learning_episodes)), args.learning_ep_compression)
    )


def save_plot(figure_id, plot_title, ylabel_str, ylim, use_legend, legend_loc, base_filename, output_path, type_name):
    class OOMFormatter(ScalarFormatter):
        def __init__(self, order=0, fformat="%1.1f", offset=True, mathText=True):
            self.oom = order
            self.fformat = fformat
            ScalarFormatter.__init__(self, useOffset=offset, useMathText=mathText)

        def _set_order_of_magnitude(self):
            self.orderOfMagnitude = self.oom

        def _set_format(self, vmin=None, vmax=None):
            self.format = self.fformat
            if self._useMathText:
                self.format = r'$\mathdefault{%s}$' % self.format

    plt.figure(figure_id)
    if plot_title is not None:
        plt.title(plot_title, fontsize=20)
    plt.xlabel("Number of episodes", fontsize=20)
    plt.ylabel(ylabel_str, fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.ylim((0, ylim))
    plt.locator_params(axis='x', nbins=6)
    plt.locator_params(axis='y', nbins=6)
    plt.gca().xaxis.set_major_formatter(OOMFormatter(4, "%1.1f"))
    plt.gca().xaxis.get_offset_text().set_fontsize(16)
    plt.ticklabel_format(axis='x', style='sci', scilimits=(4, 4))
    if use_legend:
        if legend_loc == "out":
            plt.legend(fontsize=16, ncol=2, loc=(1.04, 0))
        else:
            plt.legend(fontsize=16, ncol=1, loc=legend_loc)
    out_filename = os.path.join(output_path, f"{base_filename}_{type_name}_avg")

    # https://stackoverflow.com/questions/11837979/removing-white-space-around-a-saved-image
    plt.savefig(f"{out_filename}.png", bbox_inches='tight', pad_inches=0.01)
    plt.savefig(f"{out_filename}.pdf", bbox_inches='tight', pad_inches=0.01)


def save_reward_plot(plot_title, use_legend, legend_loc, base_filename, output_path):
    save_plot(
        FIG_RW_ID, plot_title, "Average return", RW_IF_GOAL + 0.05, use_legend, legend_loc, base_filename, output_path,
        "reward"
    )


def save_steps_plot(plot_title, use_legend, legend_loc, max_steps, base_filename, output_path):
    save_plot(
        FIG_STEPS_ID, plot_title, "Average steps", max_steps + 5, use_legend, legend_loc, base_filename, output_path,
        "steps"
    )


def plot_avg_curves(args):
    rw_fig, steps_fig = plt.figure(FIG_RW_ID), plt.figure(FIG_STEPS_ID)

    config_obj = read_json_file(args.config_file)
    for setting_id in tqdm(range(len(config_obj))):
        setting = config_obj[setting_id]
        avg_rw, std_rw, avg_steps, std_steps, learning_episodes = get_avg_run_rw_steps(setting["folders"], args)
        plot_curve(FIG_RW_ID, avg_rw, std_rw, 0, RW_IF_GOAL, learning_episodes, setting, args)
        plot_curve(FIG_STEPS_ID, avg_steps, std_steps, 0, args.max_episode_length, learning_episodes, setting, args)

    output_filename_base = os.path.basename(args.config_file)[:-len(".json")]
    output_path = os.path.abspath(os.path.dirname(args.config_file))

    save_reward_plot(args.plot_title, not args.no_legend, args.legend_loc, output_filename_base, output_path)
    save_steps_plot(
        args.plot_title, not args.no_legend, args.legend_loc, args.max_episode_length, output_filename_base, output_path
    )

    plt.close(rw_fig)
    plt.close(steps_fig)


if __name__ == "__main__":
    args = create_argparser().parse_args()

    plt.rc('text', usetex=args.use_tex)
    plt.rc('font', family='Times')  # Previously 'serif'

    plot_avg_curves(args)
