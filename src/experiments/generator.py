import argparse
from experiments import common
from gym_hierarchical_subgoal_automata.envs.craftworld.craftworld_env import CraftWorldTasks
from gym_hierarchical_subgoal_automata.envs.waterworld.waterworld_env import WaterWorldTasks
import json
import os
from reinforcement_learning.learning_algorithm import LearningAlgorithm
import subprocess
from utils.file_utils import mkdir


class ExperimentsGenerator:
    """
    Generates the configuration files for all the experiments reported in the paper "Hierarchies of Reward Machines"
    (Furelos Blanco et al., ICML 2023).
    """
    def __init__(self, root_folder, experiments_folder, collection_script_path):
        self._root_folder = root_folder
        self._experiments_folder = experiments_folder
        self._collection_script_path = collection_script_path
        self._result_collection_cmd = []

    def run(self):
        self._result_collection_cmd.clear()

        # Generate the configuration files for all experiments
        self._gen_default_experiments()
        self._gen_flat_experiments()
        self._gen_restriction_experiments()
        self._gen_exploration_experiments()
        self._gen_goal_collection()
        self._gen_handcrafted_experiments()

        # Generate a script that generates plots and summaries of statistics once the experiments have been fully run
        with open(self._collection_script_path, 'w') as f:
            f.write("\n".join(self._result_collection_cmd))

    """
    Generation
    """
    def _gen_default_experiments(self):
        """
        Default experiments with craftworld (open plan, open plan w. lava, four rooms) and waterworld. Exploration using
        skills from lower levels and policies is enabled, and the learning tasks are not provided with the specific automata
        that must be called nor the observables involved in the task.
        """
        def _gen_experiment(domain, env_type, domain_args, num_episodes):
            folder = os.path.join(common.FOLDER_DEFAULT_SETTINGS, domain, env_type)
            self._gen_rl_conf(
                domain=domain,
                folder=folder,
                num_episodes=num_episodes,
                domain_args=domain_args,
                exploration_args=common.get_exploration_args(True, True, True),
                restricted_args=common.get_restricted_args(False, False, False),
                learn_enable=True
            )
            if domain == "craftworld":
                self._gen_cw_result_files(folder, num_episodes)
            else:
                self._gen_ww_result_files(folder, num_episodes)

        _gen_experiment("craftworld", "open_plan", common.get_cw_args("open_plan", False, True), 150000)
        _gen_experiment("craftworld", "open_plan_lava", common.get_cw_args("open_plan", True, True), 150000)
        _gen_experiment("craftworld", "four_rooms", common.get_cw_args("four_rooms", False, True), 300000)
        _gen_experiment("craftworld", "four_rooms_lava", common.get_cw_args("four_rooms", True, True), 300000)
        _gen_experiment("waterworld", "wo_black", common.get_ww_args(True, False), 150000)
        _gen_experiment("waterworld", "w_black", common.get_ww_args(True, True), 300000)

    def _gen_flat_experiments(self):
        """
        Experiments where a flat automaton is learned for each of the tasks involved in the Craftworld. We use the open plan
        without lava for simplicity: it's enough to show that higher level automata cannot be learned even without a rejecting
        state. The open plan is better for this comparison since counterexamples should be collectable for the higher levels.
        Note the curriculum still applies for the instances of each task. The lowest level tasks should be naturally learnable
        (we are already doing this for the hierarchies!), so we keep them commented for the sake of time.
        """
        def _gen_rl_experiment(domain, task_name, domain_args, use_base_goal_examples, num_episodes):
            folder = os.path.join(
                common.FOLDER_FLAT, "with_goal_base" if use_base_goal_examples else "without_goal_base", domain, task_name
            )
            self._gen_rl_conf(
                domain=domain,
                folder=folder,
                num_episodes=num_episodes,
                domain_args=domain_args,
                exploration_args=common.get_exploration_args(False, False, False),  # don't actually need them since there are no calls to lower levels,
                restricted_args=common.get_restricted_args(False, False, False),    # don't actually need them (maybe observables yes, but keep it open as in the previous)
                learn_enable=True
            )
            if domain == "craftworld":
                self._gen_cw_result_files(folder, num_episodes, [task_name])
            else:
                self._gen_ww_result_files(folder, num_episodes, [task_name])

        for use_base_goal_examples in [False, True]:
            for task in [
                CraftWorldTasks.MILK_BUCKET.value, CraftWorldTasks.MAP.value, CraftWorldTasks.BOOK.value,
                CraftWorldTasks.MILK_BUCKET_AND_SUGAR.value, CraftWorldTasks.BOOK_AND_QUILL.value,
                CraftWorldTasks.CAKE.value
            ]:
                _gen_rl_experiment(
                    "craftworld", task, common.get_cw_args("open_plan", False, use_base_goal_examples, env=task),
                    use_base_goal_examples, 150000
                )
            for task in [WaterWorldTasks.RG_BC.value, WaterWorldTasks.RGB_CMY.value]:
                _gen_rl_experiment(
                    "waterworld", task, common.get_ww_args(use_base_goal_examples, False, env=task),
                    use_base_goal_examples, 150000
                )

        def _gen_baseline_experiment(algorithm, domain, task_name, num_episodes):
            folder = os.path.join(common.FOLDER_FLAT, algorithm, domain, task_name)
            self._gen_baseline_conf(algorithm, domain, task_name, folder, num_episodes)
            self._gen_baseline_stats_file(folder)

        for baseline_algorithm in ["deepsynth", "jirp", "lrm"]:
            for task in [
                CraftWorldTasks.MILK_BUCKET.value, CraftWorldTasks.BOOK.value, CraftWorldTasks.BOOK_AND_QUILL.value,
                CraftWorldTasks.CAKE.value
            ]:
                _gen_baseline_experiment(baseline_algorithm, "craftworld", task, 150000)
            for task in [WaterWorldTasks.RG.value, WaterWorldTasks.RG_BC.value, WaterWorldTasks.RGB_CMY.value]:
                _gen_baseline_experiment(baseline_algorithm, "waterworld", task, 150000)

    def _gen_restriction_experiments(self):
        """
        Experiments for checking how faster ILASP is by restricting the hypothesis space, which in turn can reduce the
        amount of time to learn the final working automaton, thus increasing speed of convergence (we could compare how
        much faster a policy is learned in terms of curricula).
        """
        def _gen_experiment(domain, env_type, domain_args, num_episodes):
            folder = os.path.join(common.FOLDER_RESTRICTIONS, domain, env_type)
            self._gen_rl_conf(
                domain=domain,
                folder=folder,
                num_episodes=num_episodes,
                domain_args=domain_args,
                exploration_args=common.get_exploration_args(True, True, True),
                restricted_args=common.get_restricted_args(True, False, False),
                learn_enable=True
            )
            if domain == "craftworld":
                self._gen_cw_result_files(folder, num_episodes)
            else:
                self._gen_ww_result_files(folder, num_episodes)

        _gen_experiment("craftworld", "open_plan", common.get_cw_args("open_plan", False, True), 150000)
        _gen_experiment("craftworld", "open_plan_lava", common.get_cw_args("open_plan", True, True), 150000)
        _gen_experiment("craftworld", "four_rooms", common.get_cw_args("four_rooms", False, True), 300000)
        _gen_experiment("craftworld", "four_rooms_lava", common.get_cw_args("four_rooms", True, True), 300000)
        _gen_experiment("waterworld", "wo_black", common.get_ww_args(True, False), 150000)
        _gen_experiment("waterworld", "w_black", common.get_ww_args(True, True), 300000)

    def _gen_exploration_experiments(self):
        """
        Experiments for evaluating the usefulness of exploring using lower level automata and formula options. The default
        experiments use the previous two and actions. Here we test what if we only use actions (to check if hihg-level
        exploration is useful), and if we use high-level exploration without actions (to check if actions still play some
        important role).
        """
        def _gen_experiment(domain, env_type, domain_args, use_actions, use_formulas, use_automata, num_episodes):
            folder = os.path.join(
                common.FOLDER_EXPLORATION, f"ac{use_actions}_f{use_formulas}_aut{use_automata}", domain, env_type
            )
            self._gen_rl_conf(
                domain=domain,
                folder=folder,
                num_episodes=num_episodes,
                domain_args=domain_args,
                exploration_args=common.get_exploration_args(use_actions, use_automata, use_formulas),
                restricted_args=common.get_restricted_args(False, False, False),
                learn_enable=True
            )
            if domain == "craftworld":
                self._gen_cw_result_files(folder, num_episodes)
            else:
                self._gen_ww_result_files(folder, num_episodes)

        for use_actions, use_formulas, use_automata in [(True, False, False), (False, True, True)]:
            _gen_experiment(
                "craftworld", "open_plan", common.get_cw_args("open_plan", False, True), use_actions,
                use_formulas, use_automata, 150000
            )
            _gen_experiment(
                "craftworld", "open_plan_lava", common.get_cw_args("open_plan", True, True), use_actions,
                use_formulas, use_automata, 150000
            )
            _gen_experiment(
                "craftworld", "four_rooms", common.get_cw_args("four_rooms", False, True), use_actions,
                use_formulas, use_automata, 300000
            )
            _gen_experiment(
                "craftworld", "four_rooms_lava", common.get_cw_args("four_rooms", True, True), use_actions,
                use_formulas, use_automata, 300000
            )
            _gen_experiment(
                "waterworld", "wo_black", common.get_ww_args(True, False), use_actions, use_formulas,
                use_automata, 150000
            )
            _gen_experiment(
                "waterworld", "w_black", common.get_ww_args(True, True), use_actions, use_formulas, use_automata,
                300000
            )

    def _gen_goal_collection(self):
        """
        Experiments for checking effectiveness of goal collection. We use four rooms to show why they are good to use in
        combination with exploration: if we use just a single example, it will be unlikely we learn about other trajectories.
        Since four rooms has some more sparsity, we should be able to see their convenience here.
        """
        def _gen_experiment(domain, env_type, domain_args, num_episodes):
            folder = os.path.join(common.FOLDER_GOAL_COLLECTION, domain, env_type)
            self._gen_rl_conf(
                domain=domain,
                folder=folder,
                num_episodes=num_episodes,
                domain_args=domain_args,
                exploration_args=common.get_exploration_args(True, True, True),
                restricted_args=common.get_restricted_args(False, False, False),
                learn_enable=True
            )
            if domain == "craftworld":
                self._gen_cw_result_files(folder, num_episodes)
            else:
                self._gen_ww_result_files(folder, num_episodes)

        _gen_experiment("craftworld", "open_plan", common.get_cw_args("open_plan", False, False), 150000)
        _gen_experiment("craftworld", "open_plan_lava", common.get_cw_args("open_plan", True, False), 150000)
        _gen_experiment("craftworld", "four_rooms", common.get_cw_args("four_rooms", False, False), 300000)
        _gen_experiment("craftworld", "four_rooms_lava", common.get_cw_args("four_rooms", True, False), 300000)

    def _gen_handcrafted_experiments(self):
        """
        Experiments for evaluating policy learning algorithms (ours and CRM) on handcrafted non-flat and flat HRMs (CRM
        is only applicable on the latter).
        """
        def _gen_experiment(domain, env_type, domain_args, algorithm, task_name, num_episodes):
            if algorithm == "hrl-nonflat":
                subname = "fFalse"  # strange naming but used to stick to old experiment set
            elif algorithm == "hrl-flat":
                subname = "fTrue"   # strange naming but used to stick to old experiment set
            else:
                subname = "crm"

            folder = os.path.join(common.FOLDER_HANDCRAFTED, domain, f"{env_type}-{subname}", task_name)
            self._gen_rl_conf(
                domain=domain,
                folder=folder,
                num_episodes=num_episodes,
                domain_args=domain_args,
                exploration_args=common.get_exploration_args(False, False, False),  # don't actually need them since there are no calls to lower levels
                restricted_args=common.get_restricted_args(False, False, False),    # don't actually need them (maybe observables yes, but keep it open as in the previous)
                learn_enable=False,
                algorithm=algorithm
            )
            self._gen_plot_handcrafted(domain, task_name, env_type, num_episodes)

        for algorithm in ["hrl-nonflat", "hrl-flat", "crm"]:
            for task in [
                CraftWorldTasks.MILK_BUCKET.value, CraftWorldTasks.BOOK_AND_QUILL.value, CraftWorldTasks.CAKE.value
            ]:
                _gen_experiment(
                    "craftworld", "open_plan", common.get_cw_args("open_plan", False, False, algorithm, task),
                    algorithm, task, 100000
                )
                _gen_experiment(
                    "craftworld", "open_plan_lava", common.get_cw_args("open_plan", True, False, algorithm, task),
                    algorithm, task, 100000
                )
                _gen_experiment(
                    "craftworld", "four_rooms", common.get_cw_args("four_rooms", False, False, algorithm, task),
                    algorithm, task, 200000
                )
                _gen_experiment(
                    "craftworld", "four_rooms_lava", common.get_cw_args("four_rooms", True, False, algorithm, task),
                    algorithm, task, 200000
                )

            for task in [WaterWorldTasks.RG.value, WaterWorldTasks.RG_BC.value, WaterWorldTasks.RGB_CMY.value]:
                _gen_experiment(
                    "waterworld", "wo_black", common.get_ww_args(True, False, algorithm, task), algorithm, task, 100000
                )
                _gen_experiment(
                    "waterworld", "w_black", common.get_ww_args(True, True, algorithm, task), algorithm, task, 200000
                )

    def _gen_rl_conf(
        self, domain, folder, num_episodes, domain_args, exploration_args, restricted_args, learn_enable,
        algorithm="hrl-nonflat"
    ):
        if algorithm.startswith("hrl"):
            generator = "rl.ihsa.ihsa_hrl_config_generator"
        elif algorithm == "crm":
            generator = "rl.ihsa.ihsa_crm_config_generator"
        else:
            raise RuntimeError(f"Error: Unknown algorithm '{algorithm}'.")

        args = [
            "python3", "-m", f"config.generators.{generator}",
            "--domain", domain,
            "--num_runs", str(common.NUM_RUNS),
            "--num_episodes", str(num_episodes),
            "--root_experiments_path", self._root_folder,
            "--experiment_folder_name", os.path.join(self._experiments_folder, folder)
        ]

        args.extend(common.DEFAULT_ARGS)

        if algorithm in ["hrl-nonflat", "hrl-flat", "crm"]:
            args.extend([
                "--mode", "train-learn" if learn_enable else "train-handcrafted",
            ])

        if algorithm.startswith("hrl"):
            args.extend(common.DEFAULT_IHSA_HRL_ARGS)
            if algorithm == "hrl-flat":
                args.append("--use_flat_hierarchy")
        elif algorithm == "crm":
            args.extend(common.DEFAULT_IHSA_CRM_ARGS)
        elif algorithm == "drqn":
            args.extend(common.DEFAULT_DRQN_ARGS)
        elif algorithm == "dqn":
            args.extend(common.DEFAULT_DQN_ARGS)

        args.extend(domain_args)
        args.extend(exploration_args)
        args.extend(restricted_args)
        subprocess.call(args)

    def _gen_baseline_conf(self, algorithm, domain, task_name, folder, num_episodes):
        args = [
            "python3", "-m", "config.generators.baseline.baseline_config_generator",
            "--timed",
            "--domain", domain,
            "--env", task_name,
            "--algorithm", algorithm,
            "--num_runs", str(common.NUM_RUNS),
            "--num_tasks", str(common.NUM_TASKS),
            "--maximum_episode_length", str(common.MAX_EPISODE_LENGTH),
            "--num_episodes", str(num_episodes),
            "--root_experiments_path", self._root_folder,
            "--experiment_folder_name", os.path.join(self._experiments_folder, folder)
        ]
        subprocess.call(args)

    def _gen_result_files(self, folder, num_plot_episodes, tasks, tasks_to_label):
        self._gen_plot_file(tasks, tasks_to_label, folder, num_plot_episodes)
        self._gen_ihsa_stats_file(folder)

    def _gen_cw_result_files(self, folder, num_plot_episodes, tasks=common.CW_TASKS):
        self._gen_result_files(folder, num_plot_episodes, tasks, common.CW_TASK_TO_LABEL)

    def _gen_ww_result_files(self, folder, num_plot_episodes, tasks=common.WW_TASKS):
        self._gen_result_files(folder, num_plot_episodes, tasks, common.WW_TASK_TO_LABEL)

    def _gen_plot_file(self, tasks, tasks_to_label, folder, num_plot_episodes, plot_filename=None):
        obj = []
        if len(tasks) not in common.COLOR_PALETTES:
            raise RuntimeError(f"Error: A palette with {len(tasks)} colors is missing.")
        palette = common.COLOR_PALETTES[len(tasks)]
        for i in range(len(tasks)):
            obj.append({
                "colour": palette[i],
                "label": "\\textsc{" + tasks_to_label[tasks[i]] + "}",
                "folders": [
                    os.path.join(self._root_folder, self._experiments_folder, folder, f"batch_{batch}", tasks[i])
                    for batch in range(1, common.NUM_RUNS + 1)
                ]
            })

        full_folder_path = os.path.join(self._root_folder, self._experiments_folder, "results", folder)
        mkdir(full_folder_path)
        plot_filename = os.path.join(full_folder_path, common.PLOT_FILENAME) if plot_filename is None else os.path.join(
            full_folder_path, plot_filename)
        with open(plot_filename, 'w') as f:
            json.dump(obj, f, sort_keys=True, indent=4)

        cmd = "python -m utils.plot_curves --use_tex -w 20 --no_legend " \
              f"--learning_ep_compression {common.NUM_COMP_EPISODES_PLOT} " \
              f"--max_episode_length {common.MAX_EPISODE_LENGTH} " \
              f"--use_greedy_traces --greedy_evaluation_frequency {common.GREEDY_EVAL_FREQ} " \
              f"-c {plot_filename} " \
              f"-i {common.NUM_TASKS} -r {common.NUM_RUNS} -e {num_plot_episodes}"
        cmd = cmd.replace("(", "\(")
        cmd = cmd.replace(")", "\)")
        cmd = cmd.replace("&", "\&")
        self._result_collection_cmd.append(cmd)

    def _gen_stats_file(self, folder_path, script_name):
        obj = [
            os.path.join(self._root_folder, self._experiments_folder, folder_path, f"batch_{batch}", LearningAlgorithm.STATS_SUMMARY_FILENAME)
            for batch in range(1, common.NUM_RUNS + 1)
        ]

        full_folder_path = os.path.join(self._root_folder, self._experiments_folder, "results", folder_path)
        mkdir(full_folder_path)
        in_stats_filename = os.path.join(full_folder_path, common.STATS_FILENAME)
        with open(in_stats_filename, 'w') as f:
            json.dump(obj, f, sort_keys=True, indent=4)

        cmd = f"python -m {script_name} {in_stats_filename} {os.path.join(full_folder_path, common.OUT_STATS_FILENAME)}"
        cmd = cmd.replace("(", "\(")
        cmd = cmd.replace(")", "\)")
        cmd = cmd.replace("&", "\&")
        self._result_collection_cmd.append(cmd)

    def _gen_ihsa_stats_file(self, folder_path):
        self._gen_stats_file(folder_path, "utils.stats.ihsa_collect_stats")

    def _gen_baseline_stats_file(self, folder_path):
        self._gen_stats_file(folder_path, "utils.stats.baseline_collect_stats")

    def _gen_plot_handcrafted(self, env, task, grid_type, num_plot_episodes):
        obj = []
        for algorithm, label, color in [
            ("crm", "Flat -- CRM", "#ffa600"), ("fTrue", "Flat -- HRL", "#bc5090"),
            ("fFalse", "Non-Flat -- HRL", "#003f5c")
        ]:
            obj.append({
                "colour": color,
                "label": label,
                "folders": [
                    os.path.join(
                        self._root_folder, self._experiments_folder, common.FOLDER_HANDCRAFTED, env,
                        f"{grid_type}-{algorithm}", task, f"batch_{batch}", task
                    )
                    for batch in range(1, common.NUM_RUNS + 1)
                ]
            })

        full_folder_path = os.path.join(self._root_folder, self._experiments_folder, "results", common.FOLDER_HANDCRAFTED, env, f"{grid_type}")
        mkdir(full_folder_path)
        plot_filename = os.path.join(full_folder_path, f"{task}.json")
        with open(plot_filename, 'w') as f:
            json.dump(obj, f, sort_keys=True, indent=4)

        task_title = "".join([x.replace("(", "").replace(")", "").title() for x in task.split("-")]).replace("And", "")

        cmd = "python -m utils.plot_curves --use_tex -w 20 --plot_title \"\\textsc{" + task_title + "}\" " \
              f"--max_episode_length {common.MAX_EPISODE_LENGTH} --legend_loc \"lower right\" --show_sem " \
              f"--use_greedy_traces --greedy_evaluation_frequency {common.GREEDY_EVAL_FREQ} " \
              f"-c {plot_filename} " \
              f"-i {common.NUM_TASKS} -r {common.NUM_RUNS} -e {num_plot_episodes}"
        cmd = cmd.replace("(", "\(")
        cmd = cmd.replace(")", "\)")
        cmd = cmd.replace("&", "\&")
        self._result_collection_cmd.append(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", required=True
    )
    parser.add_argument(
        "--experiments_folder", required=True
    )
    parser.add_argument(
        "--collection_script_path", required=True
    )
    args = parser.parse_args()
    ExperimentsGenerator(args.root, args.experiments_folder, args.collection_script_path).run()
