import argparse
import collections
from experiments import common
from gym_hierarchical_subgoal_automata.envs.craftworld.craftworld_env import CraftWorldTasks
from gym_hierarchical_subgoal_automata.envs.waterworld.waterworld_env import WaterWorldTasks
import json
import math
import numpy as np
import os
from utils.file_utils import read_json_file
import xlsxwriter

TableField = collections.namedtuple("TableField", field_names=["heading", "name", "size"])


class ExperimentsResultSummarizer:
    """
    Generates tables that summarize the performance metrics of the hierarchy learning algorithm across tasks. It also
    prints some information on the improvements/worsenings caused by the ablations.
    """
    DOMAINS = {
        "craftworld": "cw",
        "waterworld": "ww"
    }

    SETTINGS = {
        "open_plan": "op",
        "open_plan_lava": "opl",
        "four_rooms": "fr",
        "four_rooms_lava": "frl",
        "w_black": "wd",
        "wo_black": "wod"
    }

    EXPLORATION = {
        "acFalse_fTrue_autTrue": "no-act",
        "acTrue_fFalse_autFalse": "act-only"
    }

    IHSA_SPREADSHEET_DOMAIN_FIELDS = [
        TableField("Goal Found", "num_runs_found_goal", 1),
        TableField("Learned Aut.", "num_runs_learned_automata", 1),
        TableField("Time", "ilasp_time", 2),
        TableField("Calls", "ilasp_calls", 2),
        TableField("States", "num_states", 2),
        TableField("Edges", "num_edges", 2),
        TableField("First Aut", "ep_first_automaton", 2),
        TableField("Elaps Aut", "ep_level_to_first_aut", 2),
        TableField("Elaps Ex", "ep_level_to_first_ex", 2),
        TableField("Total Ex", "total_ex_num", 2),
        TableField("Total G", "goal_ex_num", 2),
        TableField("Total D", "dend_ex_num", 2),
        TableField("Total I", "inc_ex_num", 2),
        TableField("Length G", "goal_ex_length", 2),
        TableField("Length D", "dend_ex_length", 2),
        TableField("Length I", "inc_ex_length", 2)
    ]

    IHSA_SPREADSHEET_GENERAL_FIELDS = [
        TableField("TOTAL ILASP TIME", "ilasp_time", 2),
        TableField("TOTAL ILASP CALLS", "ilasp_calls", 2),
        TableField("NUM COMPLETED RUNS", "num_completed_runs", 1)
    ]

    FLAT_COMPARISON_TASKS = {
        "craftworld": [
            CraftWorldTasks.MILK_BUCKET.value, CraftWorldTasks.BOOK.value, CraftWorldTasks.BOOK_AND_QUILL.value,
            CraftWorldTasks.CAKE.value
        ],
        "waterworld": [WaterWorldTasks.RG.value, WaterWorldTasks.RG_BC.value, WaterWorldTasks.RGB_CMY.value]
    }

    FLAT_COMPARISON_IHSA_SPREADSHEET_FIELDS = [
        TableField("C", "num_runs_learned_automata", 1),
        TableField("Time", "ilasp_time", 2),
        TableField("States", "num_states", 2),
        TableField("Edges", "num_edges", 2)
    ]

    FLAT_COMPARISON_BASELINE_SPREADSHEET_FIELDS = [
        TableField("C", "num_completed_runs", 1),
        TableField("Time", "learning_time", 2),
        TableField("States", "num_states", 2),
        TableField("Edges", "num_edges", 2)
    ]

    CW_TABLE_HEADINGS = {
        CraftWorldTasks.BATTER.value: "Batter",
        CraftWorldTasks.BUCKET.value: "Bucket",
        CraftWorldTasks.COMPASS.value: "Compass",
        CraftWorldTasks.LEATHER.value: "Leather",
        CraftWorldTasks.PAPER.value: "Paper",
        CraftWorldTasks.QUILL.value: "Quill",
        CraftWorldTasks.SUGAR.value: "Sugar",
        CraftWorldTasks.BOOK.value: "Book",
        CraftWorldTasks.MAP.value: "Map",
        CraftWorldTasks.MILK_BUCKET.value: "MilkBucket",
        CraftWorldTasks.BOOK_AND_QUILL.value: "BookQuill",
        CraftWorldTasks.MILK_BUCKET_AND_SUGAR.value: "MilkB.Sugar",
        CraftWorldTasks.CAKE.value: "Cake",
    }

    WW_TABLE_HEADINGS = {
        WaterWorldTasks.RG.value: "rg",
        WaterWorldTasks.BC.value: "bc",
        WaterWorldTasks.MY.value: "my",
        WaterWorldTasks.RG_BC.value: "rg\&bc",
        WaterWorldTasks.BC_MY.value: "bc\&my",
        WaterWorldTasks.RG_MY.value: "rg\&my",
        WaterWorldTasks.RGB.value: "rgb",
        WaterWorldTasks.CMY.value: "cmy",
        WaterWorldTasks.RGB_CMY.value: "rgb\&cmy"
    }

    @classmethod
    def run(cls):
        args = cls._get_argparser().parse_args()
        cls._generate_spreadsheets(args.in_results_path, args.out_path)
        cls._print_ablation_improvements(args.in_results_path)
        cls._generate_tex_tables(args.in_results_path, args.out_path)

    @classmethod
    def _get_argparser(cls):
        parser = argparse.ArgumentParser()
        parser.add_argument("in_results_path", help="directory containing the processed results of the experiments")
        parser.add_argument("out_path", help="directory where the summarized results will be stored")
        return parser

    @classmethod
    def _generate_spreadsheets(cls, in_results_path, out_path):
        spreadsheets_path = os.path.join(out_path, "icml_spreadsheets.xlsx")
        print(f"Generating spreadsheets in {spreadsheets_path}...")
        workbook = xlsxwriter.Workbook(spreadsheets_path)
        cls._write_ihsa_spreadsheets(workbook, in_results_path)
        cls._write_flat_comparison_spreadsheets(workbook, in_results_path)
        workbook.close()

    @classmethod
    def _write_ihsa_spreadsheets(cls, workbook, in_results_path):
        # Experiments with IHSA in the default setting and ablations.
        for experiment_set in [
            common.FOLDER_DEFAULT_SETTINGS, common.FOLDER_RESTRICTIONS, common.FOLDER_GOAL_COLLECTION
        ]:
            cls._write_ihsa_spreadsheet(workbook, experiment_set.split('-')[1], os.path.join(in_results_path, experiment_set))

        # Experiments with IHSA with the exploration ablation.
        exp_set_path = os.path.join(in_results_path, common.FOLDER_EXPLORATION)
        for exp_setting in sorted(os.listdir(exp_set_path)):
            cls._write_ihsa_spreadsheet(
                workbook,
                f"{common.FOLDER_EXPLORATION.split('-')[1]}_{cls.EXPLORATION[exp_setting]})",
                os.path.join(exp_set_path, exp_setting)
            )

    @classmethod
    def _write_ihsa_spreadsheet(cls, workbook, experiment_name, experiment_path):
        for domain in sorted(os.listdir(experiment_path)):
            domain_path = os.path.join(experiment_path, domain)
            for setting in sorted(os.listdir(domain_path)):
                worksheet = workbook.add_worksheet(
                    f"{experiment_name}_{cls.DOMAINS[domain]}_{cls.SETTINGS[setting]}"
                )
                stats_out_path = os.path.join(domain_path, setting, "stats.out")
                if os.path.exists(stats_out_path):
                    with open(os.path.join(domain_path, setting, "stats.out")) as f:
                        cls._write_ihsa_stats(json.load(f), worksheet)

    @classmethod
    def _write_ihsa_stats(cls, stats, worksheet):
        worksheet.write(0, 0, "Task")
        cell = 1
        for field in cls.IHSA_SPREADSHEET_DOMAIN_FIELDS:
            worksheet.write(0, cell, field.heading)
            cell += field.size

        # Stats for each domain
        row = 1
        for domain_name, domain_stats in stats["domains"].items():
            if len(domain_stats) > 0:
                worksheet.write(row, 0, domain_name)
                cell = 1
                for field in cls.IHSA_SPREADSHEET_DOMAIN_FIELDS:
                    stat = domain_stats[field.name]
                    if isinstance(stat, list):
                        for i in range(len(stat)):
                            stat_i = "-" if math.isnan(stat[i]) else stat[i]
                            worksheet.write(row, cell, stat_i)
                            cell += 1
                    else:
                        worksheet.write(row, cell, stat)
                        cell += 1
            row += 1

        # General stats
        general_stats = stats["general"]
        for field in cls.IHSA_SPREADSHEET_GENERAL_FIELDS:
            worksheet.write(row, 0, field.heading)
            cell = 1
            stat = general_stats[field.name]
            if isinstance(stat, list):
                for field_value in stat:
                    worksheet.write(row, cell, field_value)
                    cell += 1
            else:
                worksheet.write(row, cell, stat)
                cell += 1
            row += 1

    @classmethod
    def _write_flat_comparison_spreadsheets(cls, workbook, in_results_path):
        worksheet = workbook.add_worksheet("flat")

        # Create heading
        worksheet.write(0, 0, "Task")
        cell = 1
        for algorithm_name in ["IHSA (Non-Flat)", "IHSA (Flat)", "DeepSynth", "JIRP", "LRM"]:
            worksheet.write(0, cell, algorithm_name)
            for field in cls.FLAT_COMPARISON_IHSA_SPREADSHEET_FIELDS:
                worksheet.write(1, cell, field.heading)
                cell += field.size

        row = 2
        for domain, tasks in cls.FLAT_COMPARISON_TASKS.items():
            for task in tasks:
                worksheet.write(row, 0, task)
                cell = 1

                # Get non-flat stats
                setting = "open_plan" if domain == "craftworld" else "wo_black"
                cell = cls._write_ihsa_flat_comparison_stats(
                    worksheet, row, cell,
                    os.path.join(in_results_path, common.FOLDER_DEFAULT_SETTINGS, domain, setting),
                    task
                )

                # Get flat with our method
                if task == WaterWorldTasks.RG.value:  # RG is already flat
                    cell = cls._write_ihsa_flat_comparison_stats(
                        worksheet, row, cell,
                        os.path.join(in_results_path, common.FOLDER_DEFAULT_SETTINGS, domain, setting),
                        task
                    )
                else:
                    cell = cls._write_ihsa_flat_comparison_stats(
                        worksheet, row, cell,
                        os.path.join(in_results_path, common.FOLDER_FLAT, "with_goal_base", domain, task),
                        task
                    )

                # Get flat with other methods
                for algorithm in ["deepsynth", "jirp", "lrm"]:
                    cell = cls._write_baseline_flat_comparison_stats(
                        worksheet, row, cell,
                        os.path.join(in_results_path, common.FOLDER_FLAT, algorithm, domain, task)
                    )

                row += 1

    @classmethod
    def _write_ihsa_flat_comparison_stats(cls, worksheet, row, cell, folder_path, task):
        return cls._write_flat_comparison_stats(
            worksheet,
            row,
            cell,
            cls._get_ihsa_task_stats(folder_path, task),
            cls.FLAT_COMPARISON_IHSA_SPREADSHEET_FIELDS
        )

    @classmethod
    def _write_baseline_flat_comparison_stats(cls, worksheet, row, cell, folder_path):
        return cls._write_flat_comparison_stats(
            worksheet,
            row,
            cell,
            cls._get_baseline_stats(folder_path),
            cls.FLAT_COMPARISON_BASELINE_SPREADSHEET_FIELDS
        )

    @classmethod
    def _write_flat_comparison_stats(cls, worksheet, row, cell, stats, fields):
        for field in fields:
            if isinstance(stats[field.name], list):
                for i in range(len(stats[field.name])):
                    worksheet.write(row, cell, stats[field.name][i])
                    cell += 1
            else:
                worksheet.write(row, cell, stats[field.name])
                cell += 1
        return cell

    @classmethod
    def _get_ihsa_task_stats(cls, folder_path, task):
        return read_json_file(os.path.join(folder_path, "stats.out"))["domains"][task]

    @classmethod
    def _get_ihsa_general_stats(cls, folder_path):
        return read_json_file(os.path.join(folder_path, "stats.out"))["general"]

    @classmethod
    def _get_baseline_stats(cls, folder_path):
        return read_json_file(os.path.join(folder_path, "stats.out"))

    @classmethod
    def _print_ablation_improvements(cls, in_results_path):
        cls._print_restricted_callable_improvements(in_results_path)
        cls._print_exploration_improvements(in_results_path)

    @classmethod
    def _print_restricted_callable_improvements(cls, in_results_path):
        print("Restricted Callable RM Set Ablation")

        default_path = os.path.join(in_results_path, common.FOLDER_DEFAULT_SETTINGS)
        restricted_path = os.path.join(in_results_path, common.FOLDER_RESTRICTIONS)

        def _print_improvements(default_stats, restricted_stats):
            print(
                f"\t\tLearning is {default_stats['ilasp_time'][0] / restricted_stats['ilasp_time'][0]}x faster with "
                f"restricted set of callable RMs."
            )
            print(
                f"\t\tLearning with a restricted set of callable RMs requires "
                f"{100 * (1 - restricted_stats['ilasp_calls'][0] / default_stats['ilasp_calls'][0])}% less calls to the "
                f"learner."
            )

        for setting in ["open_plan", "open_plan_lava", "four_rooms", "four_rooms_lava"]:
            print(f"\tCraftWorld - {setting}")
            _print_improvements(
                cls._get_ihsa_general_stats(os.path.join(default_path, "craftworld", setting)),
                cls._get_ihsa_general_stats(os.path.join(restricted_path, "craftworld", setting))
            )

        for setting in ["wo_black", "w_black"]:
            print(f"\tWaterWorld - {setting}")
            _print_improvements(
                cls._get_ihsa_general_stats(os.path.join(default_path, "waterworld", setting)),
                cls._get_ihsa_general_stats(os.path.join(restricted_path, "waterworld", setting))
            )

    @classmethod
    def _print_exploration_improvements(cls, in_results_path):
        print("Exploration with Actions Only")

        default_path = os.path.join(in_results_path, common.FOLDER_DEFAULT_SETTINGS)
        exploration_path = os.path.join(in_results_path, common.FOLDER_EXPLORATION, "acTrue_fFalse_autFalse")

        def _print_improvements(domain, setting, tasks):
            improvements = []
            for i, task in enumerate(tasks):
                print(f"\t\t{task}")
                default_stats = cls._get_ihsa_task_stats(os.path.join(default_path, domain, setting), task)
                exploration_stats = cls._get_ihsa_task_stats(os.path.join(exploration_path, domain, setting), task)
                if exploration_stats['ep_level_to_first_aut'][0] > 0:
                    improvements.append(exploration_stats['ep_level_to_first_aut'][0] / default_stats['ep_level_to_first_aut'][0])
                    print(
                        f"\t\t\tExploration without options requires {improvements[-1]}x more episodes to learn the "
                        f"first automaton."
                    )
                else:
                    print(
                        "\t\t\tExploration without options did not discover enough examples to learn the first "
                        "automaton."
                    )
            print(f"\t\tAverage")
            print(
                f"\t\t\tExploration without options requires {np.mean(improvements)}x more episodes to learn the first "
                f"automaton."
            )

        for setting in ["open_plan", "open_plan_lava", "four_rooms", "four_rooms_lava"]:
            print(f"\tCraftWorld - {setting}")
            _print_improvements("craftworld", setting, [
                CraftWorldTasks.BOOK.value, CraftWorldTasks.MAP.value, CraftWorldTasks.MILK_BUCKET.value,
                CraftWorldTasks.BOOK_AND_QUILL.value, CraftWorldTasks.MILK_BUCKET_AND_SUGAR.value,
                CraftWorldTasks.CAKE.value
            ])

        for setting in ["wo_black", "w_black"]:
            print(f"\tWaterWorld - {setting}")
            _print_improvements("waterworld", setting, [
                WaterWorldTasks.RG_BC.value, WaterWorldTasks.BC_MY.value, WaterWorldTasks.RG_MY.value,
                WaterWorldTasks.RGB.value, WaterWorldTasks.CMY.value, WaterWorldTasks.RGB_CMY.value
            ])

    @classmethod
    def _generate_tex_tables(cls, in_results_path, out_path):
        print(f"Generating LaTeX tables in {out_path}...")

        for experiment_folder in [
            common.FOLDER_DEFAULT_SETTINGS, common.FOLDER_RESTRICTIONS,
            os.path.join(common.FOLDER_EXPLORATION, "acTrue_fFalse_autFalse")
        ]:
            setting_name = experiment_folder.split("/")[0].split("-")[1]

            # CraftWorld
            for setting in ["open_plan", "open_plan_lava", "four_rooms", "four_rooms_lava"]:
                cls._write_tex_table(
                    read_json_file(os.path.join(
                        in_results_path, experiment_folder, "craftworld", setting, "stats.out"
                    )),
                    cls.CW_TABLE_HEADINGS,
                    os.path.join(out_path, f"{setting_name}_craftworld_{setting}.tex")
                )

            # WaterWorld
            for setting in ["wo_black", "w_black"]:
                cls._write_tex_table(
                    read_json_file(os.path.join(
                        in_results_path, experiment_folder, "waterworld", setting, "stats.out"
                    )),
                    cls.WW_TABLE_HEADINGS,
                    os.path.join(out_path, f"{setting_name}_waterworld_{setting}.tex")
                )

    @classmethod
    def _write_tex_table(cls, stats, tasks, out_tex):
        with open(out_tex, 'w') as f:
            include_dends = stats["domains"][list(tasks.keys())[0]]["dend_ex_num"][0] > 0.0
            num_example_cols = 3 if include_dends else 2

            fields = [
                TableField("Task", "domain", 1),
                TableField("\# G", "num_runs_found_goal", 1),
                TableField("\# L", "num_runs_learned_automata", 1),
                TableField("Time (s.)", "ilasp_time", 1),
                TableField("Calls", "ilasp_calls", 1),
                TableField("States", "num_states", 1),
                TableField("Edges", "num_edges", 1),
                TableField("Ep. First HRM", "ep_level_to_first_aut", 1),
                TableField("\# Examples", "ex_num", num_example_cols),
                TableField("", None, 1),
                TableField("Example Length", "ex_length", num_example_cols),
            ]

            num_cols = sum([x.size for x in fields])

            f.write("\\begin{tabular}{l" + "r" * (num_cols - 1) + "}\n")
            f.write("\\toprule[1.5pt]\n")

            first_field = True
            for field in fields:
                if not first_field:
                    f.write("& ")
                f.write("\\multicolumn{" + str(field.size) + "}{c}{" + field.heading + "} ")
                first_field = False

            f.write("\\\\\n")

            if include_dends:
                f.write("\\cmidrule{9-11} \\cmidrule{13-15}\n")
            else:
                f.write("\\cmidrule{9-10} \\cmidrule{12-13}\n")

            first_field = True
            for field in fields:
                if not first_field:
                    f.write("&")
                if field.name is not None:
                    if field.name == "ep_level_to_first_aut":
                        f.write("\\multicolumn{1}{c}{$(\\times 10^2)$}")
                    elif field.name.startswith("ex"):
                        if field.size == 3:
                            f.write("\\multicolumn{1}{c}{G} & \\multicolumn{1}{c}{D} & \\multicolumn{1}{c}{I}")
                        elif field.size == 2:
                            f.write("\\multicolumn{1}{c}{G} & \\multicolumn{1}{c}{I}")
                first_field = False

            f.write("\\\\\n")
            f.write("\\midrule\n")

            for task in tasks.keys():
                task_stats = stats["domains"][task]

                for field_id, field in enumerate(fields):
                    if field.name is not None:
                        if field.name == "domain":
                            f.write("\\textsc{" + tasks[task] + "}")
                        elif field.name.startswith("ex"):
                            goal_stat = task_stats[f"goal_{field.name}"]
                            dend_stat = task_stats[f"dend_{field.name}"]
                            inc_stat = task_stats[f"inc_{field.name}"]

                            f.write(f"{goal_stat[0]} ({goal_stat[1]}) & ")
                            if include_dends:
                                f.write(f"{dend_stat[0]} ({dend_stat[1]}) & ")
                            f.write(f"{inc_stat[0]} ({inc_stat[1]})")
                        else:
                            stat = task_stats[field.name]
                            if field.name == "ep_level_to_first_aut":
                                stat[0] = round(0.01 * stat[0], 1)
                                stat[1] = round(0.01 * stat[1], 1)

                            if isinstance(stat, list) and len(stat) == 2:
                                f.write(f"{stat[0]} ({stat[1]})")
                            else:
                                f.write(str(stat))

                    if field_id == len(fields) - 1:
                        f.write("\\\\\n")
                    else:
                        f.write(" & ")

            f.write("\\midrule[1.5pt]\n")

            general_stats = stats["general"]
            num_cols = 15 if include_dends else 13

            f.write("\\multicolumn{" + str(num_cols) + "}{l}{")
            f.write("\\textbf{Completed Runs} = " + str(general_stats["num_completed_runs"]) + "\\hfill")
            f.write("\\textbf{Total Time (s.)} = " + str(general_stats["ilasp_time"][0]) + " (" + str(general_stats["ilasp_time"][1]) + ")\\hfill")
            f.write("\\textbf{Total Calls} = " + str(general_stats["ilasp_calls"][0]) + " (" + str(general_stats["ilasp_calls"][1]) + ")}\\\\\n")

            f.write("\\bottomrule[1.5pt]\n")
            f.write("\\end{tabular}")


if __name__ == "__main__":
    ExperimentsResultSummarizer.run()
