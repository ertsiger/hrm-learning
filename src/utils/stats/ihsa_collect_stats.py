import numpy as np
from utils.stats.collect_stats import StatsCollector
from utils.file_utils import read_json_file, write_json_obj


class IHSAStatsCollector(StatsCollector):
    @classmethod
    def _export_stats(cls, stats_files, output_file):
        write_json_obj({
            "general": cls._get_general_stats(stats_files),
            "domains": {
                domain_name: cls._get_domain_stats(domain_name, stats_files)
                for domain_name in cls._get_domain_names(stats_files[0])
            }
        }, output_file, sort_keys=True, indent=4)

    @classmethod
    def _get_domain_names(cls, stats_file):
        return list(read_json_file(stats_file)["domains"].keys())

    @classmethod
    def _get_general_stats(cls, stats_files):
        num_completed_runs = 0
        total_running_time = []
        ilasp_calls, ilasp_time = [], []

        for stats_file in stats_files:
            stats = read_json_file(stats_file)
            if stats["interrupted"]:
                continue
            num_completed_runs += 1
            total_running_time.append(stats["total_running_time"])
            ilasp_calls.append(stats["ilasp"]["calls"])
            ilasp_time.append(stats["ilasp"]["time"])

        return {
            "num_completed_runs": num_completed_runs,
            "total_running_time": cls._mean_sem(total_running_time),
            "ilasp_calls": cls._mean_sem(ilasp_calls),
            "ilasp_time": cls._mean_sem(ilasp_time)
        }

    @classmethod
    def _get_domain_stats(cls, domain_name, stats_files):
        num_runs_found_example = 0
        num_runs_learned_automata = 0
        ilasp_calls, ilasp_time = [], []
        goal_count, goal_lengths = [], []
        dend_count, dend_lengths = [], []
        inc_count, inc_lengths = [], []
        ep_first_automaton, ep_level_to_first_aut, ep_level_to_first_ex = [], [], []
        num_states, num_edges = [], []

        for stats_file in stats_files:
            stats = read_json_file(stats_file)
            if stats["interrupted"]:
                continue
            d_stats = stats["domains"][domain_name]
            if d_stats["episodes"]["first_example"] > 0:
                num_runs_found_example += 1
            if len(d_stats["episodes"]["learned_automaton"]) == 0:
                continue
            num_runs_learned_automata += 1
            ilasp_calls.append(d_stats["ilasp"]["calls"])
            ilasp_time.append(d_stats["ilasp"]["time"])
            goal_count.append(d_stats["goal_examples"]["count"])
            goal_lengths.append(cls._mean_sem(d_stats["goal_examples"]["lengths"])[0])
            dend_count.append(d_stats["dend_examples"]["count"])
            dend_lengths.append(cls._mean_sem(d_stats["dend_examples"]["lengths"])[0])
            inc_count.append(d_stats["inc_examples"]["count"])
            inc_lengths.append(cls._mean_sem(d_stats["inc_examples"]["lengths"])[0])
            ep_first_automaton.append(d_stats["episodes"]["learned_automaton"][0])
            ep_level_to_first_aut.append(d_stats["episodes"]["learned_automaton"][0] - d_stats["episodes"]["level_started"])
            ep_level_to_first_ex.append(d_stats["episodes"]["first_example"] - d_stats["episodes"]["level_started"])
            num_states.append(d_stats["automaton"]["states"])
            num_edges.append(d_stats["automaton"]["edges"])

        return {
            "num_runs_found_goal": num_runs_found_example,
            "num_runs_learned_automata": num_runs_learned_automata,
            "ilasp_calls": cls._mean_sem(ilasp_calls),
            "ilasp_time": cls._mean_sem(ilasp_time),
            "total_ex_num": cls._mean_sem(np.sum([goal_count, dend_count, inc_count], axis=0)),
            # "total_ex_length": _mean_std(goal_lengths + dend_lengths + inc_lengths),
            "goal_ex_num": cls._mean_sem(goal_count),
            "goal_ex_length": cls._mean_sem(goal_lengths),
            "dend_ex_num": cls._mean_sem(dend_count),
            "dend_ex_length": cls._mean_sem(dend_lengths),
            "inc_ex_num": cls._mean_sem(inc_count),
            "inc_ex_length": cls._mean_sem(inc_lengths),
            "ep_first_automaton": cls._mean_sem(ep_first_automaton),
            "ep_level_to_first_aut": cls._mean_sem(ep_level_to_first_aut),
            "ep_level_to_first_ex": cls._mean_sem(ep_level_to_first_ex),
            "num_states": cls._mean_sem(num_states),
            "num_edges": cls._mean_sem(num_edges)
        }


if __name__ == "__main__":
    IHSAStatsCollector.run()
