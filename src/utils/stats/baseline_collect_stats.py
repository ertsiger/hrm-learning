from utils.file_utils import read_json_file, write_json_obj
from utils.stats.collect_stats import StatsCollector


class BaselineStatsCollector(StatsCollector):
    @classmethod
    def _export_stats(cls, stats_files, output_file):
        num_completed_runs = 0
        num_states = []
        num_edges = []
        num_traces = []
        num_calls = []
        learning_time = []

        for stats_file in stats_files:
            stats = read_json_file(stats_file)
            if not stats["interrupted"]:
                num_completed_runs += 1
                num_states.append(stats["num_states"])
                num_edges.append(stats["num_edges"])
                num_traces.append(stats["num_traces"])
                num_calls.append(stats["num_calls"])
                learning_time.append(stats["learning_time"])

        write_json_obj({
            "num_completed_runs": num_completed_runs,
            "num_states": cls._mean_sem(num_states),
            "num_edges": cls._mean_sem(num_edges),
            "num_traces": cls._mean_sem(num_traces),
            "num_calls": cls._mean_sem(num_calls),
            "learning_time": cls._mean_sem(learning_time)
        }, output_file, sort_keys=True, indent=4)


if __name__ == "__main__":
    BaselineStatsCollector.run()
