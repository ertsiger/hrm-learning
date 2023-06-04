import argparse
import gym
from ilasp.ilasp_common import set_ilasp_env_variables
from ilasp.generator.ilasp_task_generator import generate_ilasp_task
from ilasp.parser.ilasp_solution_parser import parse_ilasp_solutions
from ilasp.solver.ilasp_solver import solve_ilasp_task
import os
import sys
from utils import file_utils


def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("task_config", help="json file containing number of states, observables and examples")
    parser.add_argument("task_filename", help="filename of the ILASP task")
    parser.add_argument("solution_filename", help="filename of the ILASP task solution")
    parser.add_argument("plot_filename", help="filename of the automaton plot")
    parser.add_argument("--symmetry_breaking_enable", "-s", action="store_true", help="whether to enable symmetry breaking")
    return parser


if __name__ == "__main__":
    """
    Isolation of the code for running ILASP (created for debugging purposes). The code takes a JSON configuration file,
    and the name of the ILASP task, the solution and the plotting of the solution (the last three will be automatically
    created). This debugging code was solely designed for CraftWorld: the background automata being passes below are
    from the CraftWorld environment. The `example.json` configuration file in this repository is used to learn the
    automaton for the task BookQuill.
    """

    args = get_argparser().parse_args()
    config = file_utils.read_json_file(args.task_config)
    set_ilasp_env_variables(os.path.join(os.path.dirname(sys.argv[0]), ".."))

    background_automata = []
    for env_name in [
        "CraftWorldBucket-v0", "CraftWorldSugar-v0", "CraftWorldBatter-v0", "CraftWorldPaper-v0",
        "CraftWorldCompass-v0", "CraftWorldLeather-v0", "CraftWorldQuill-v0", "CraftWorldMilkBucket-v0",
        "CraftWorldBook-v0", "CraftWorldMap-v0"
    ]:
        # Create an environment to get is root automaton.
        env = gym.make(
            env_name, params={
                "grid_params": {"width": 7, "height": 7, "grid_type": "open_plan"},
                "include_deadends": False, "environment_seed": 0
            }
        )

        # Each root automaton is made callable (see True in the appended pair).
        background_automata.append((env.get_hierarchy().get_root_automaton(), True))

    generate_ilasp_task(
        config["automaton_name"], config["num_states"], "u_acc", "u_rej", config["observables"],
        config["goal_examples"], config["deadend_examples"], config["inc_examples"], background_automata, ".",
        args.task_filename, args.symmetry_breaking_enable, config["observables"], config["max_disjunction_size"],
        config["learn_acyclic"], config["avoid_learning_only_negative"], True, config["use_mutex_saturation"],
        len(config["deadend_examples"]) > 0
    )

    solve_ilasp_task(args.task_filename, args.solution_filename, {}, timeout=60 * 60 * 3)
    automaton = parse_ilasp_solutions(config["automaton_name"], args.solution_filename)
    automaton.plot(".", args.plot_filename)
