import os
from ilasp.ilasp_common import CALL_STR, LEAF_AUTOMATON, N_TRANSITION_STR, OBS_STR, generate_injected_statement
from ilasp.generator.utils.ilasp_task_generator_state import generate_state_set
from ilasp.parser.ilasp_parser_utils import parse_call_rule, parse_negative_transition_rule
from ilasp.solver.ilasp_solver import solve_ilasp_task, ILASP_OPERATION_SEARCH_SPACE

MAX_PENALTY = 100
TMP_OUTPUT_FILENAME = "tmp_task.las"
TMP_SEARCH_SPACE_FILENAME = "search_space.txt"


def get_hypothesis_space(automaton_name, num_states, accepting_state, rejecting_state, observables, background_automata,
                         output_folder, max_disj_size):
    tmp_output_path = os.path.join(output_folder, TMP_OUTPUT_FILENAME)
    tmp_search_space_path = os.path.join(output_folder, TMP_SEARCH_SPACE_FILENAME)

    # generate an ILASP task from which we can derivate the entire search space, which is needed
    # to add the injections afterwards
    _generate_base_ilasp_task(automaton_name, num_states, accepting_state, rejecting_state, observables,
                              background_automata, tmp_output_path, max_disj_size)

    # get the search space for the given ILASP task
    solve_ilasp_task(tmp_output_path, tmp_search_space_path, {}, operation=ILASP_OPERATION_SEARCH_SPACE)

    # return the new hypothesis space with the injections to pos and neg atoms
    return _get_hypothesis_space_with_injections(tmp_search_space_path)


def _generate_base_ilasp_task(automaton_name, num_states, accepting_state, rejecting_state, observables,
                              background_automata, output_path, max_disj_size):
    with open(output_path, 'w') as f:
        task = _generate_mode_bias(automaton_name, num_states, accepting_state, rejecting_state, observables,
                                   background_automata, max_disj_size)
        task += _generate_connected_hypothesis_rules(automaton_name, num_states, accepting_state, rejecting_state,
                                                     background_automata, max_disj_size)
        f.write(task)


def _generate_automata_constants(automaton_name, background_automata):
    stmt = [f"#constant(automaton, {automaton_name})."]
    for bk_automaton, is_callable in background_automata:
        if is_callable:
            stmt.append(f"#constant(background_automaton, {bk_automaton.get_name()}).")
    return "\n".join(stmt) + "\n"


def _generate_state_constants(num_states, accepting_state, rejecting_state):
    state_constants = [f"#constant(state, {s}).\n" for s in generate_state_set(num_states, accepting_state, rejecting_state)]
    return "".join(state_constants) + '\n'


def _generate_observables_constants(observables):
    observables_constants = [f"#constant(observable, \"{o}\").\n" for o in observables]
    return "".join(observables_constants) + '\n'


def _generate_edge_indices_constants(max_disj_size):
    edge_indices_constants = [f"#constant(edge_id, {i}).\n" for i in range(1, max_disj_size + 1)]
    return "".join(edge_indices_constants) + '\n'


def _generate_mode_bias(automaton_name, num_states, accepting_state, rejecting_state, observables, background_automata,
                        max_disj_size):
    # generate constants
    stmt = _generate_automata_constants(automaton_name, background_automata)
    stmt += _generate_state_constants(num_states, accepting_state, rejecting_state)
    stmt += _generate_observables_constants(observables)
    stmt += _generate_edge_indices_constants(max_disj_size)

    # maximum number of variables in each rule (each transition rule can only refer to one particular index)
    stmt += "#maxv(1).\n\n"
    stmt += f"#modeh({N_TRANSITION_STR}(const(state), const(state), const(edge_id), const(automaton), var(step))).\n"
    stmt += f"#modeb({OBS_STR}(const(observable), var(step))).\n\n"

    # rule for forcing observables to be mentioned in every rule
    stmt += f"#bias(\":- not body({OBS_STR}(_, _)), not body(naf({OBS_STR}(_, _))).\").\n\n"

    # avoid learning rules to transition to itself
    stmt += f"#bias(\":- head({N_TRANSITION_STR}(X, X, _, _, _)).\").\n"

    # avoid learning transitions from accepting and rejecting states
    stmt += f"#bias(\":- head({N_TRANSITION_STR}({accepting_state}, _, _, _, _)).\").\n"
    if rejecting_state is not None:
        stmt += f"#bias(\":- head({N_TRANSITION_STR}({rejecting_state}, _, _, _, _)).\").\n"
    stmt += "\n"
    return stmt


def _generate_connected_hypothesis_rules(automaton_name, num_states, accepting_state, rejecting_state,
                                         background_automata, max_disj_size):
    """
    Adds auxiliary 'ed' facts of cost 2 to minimize the number of transitions and also calls to automata in the
    background.
    """
    hypothesis_rules = []
    states = generate_state_set(num_states, accepting_state, rejecting_state)
    for s1 in [s for s in states if s != accepting_state and s != rejecting_state]:  # no edges from terminal states
        for s2 in [s for s in states if s != s1]:  # no loop edges
            for i in range(1, max_disj_size + 1):
                for bk_automaton, is_callable in background_automata:
                    if is_callable:
                        hypothesis_rules.append(f"2 ~ {CALL_STR}({s1}, {s2}, {i}, {automaton_name}, {bk_automaton.get_name()}).")
                hypothesis_rules.append(f"2 ~ {CALL_STR}({s1}, {s2}, {i}, {automaton_name}, {LEAF_AUTOMATON}).")
    return "\n".join(hypothesis_rules) + '\n'


def _get_hypothesis_space_with_injections(hypothesis_space_filename):
    hypothesis_space = [f"#max_penalty({MAX_PENALTY}).\n"]
    with open(hypothesis_space_filename) as f:
        counter = 0
        for line in f:
            line = line.strip()
            hypothesis_space.append(line)

            line = line.strip("2 ~ ")  # all rules have cost 2
            if line.startswith(N_TRANSITION_STR):
                hypothesis_space.append(_get_negative_transition_injection(parse_negative_transition_rule(line), counter))
            elif line.startswith(CALL_STR):
                hypothesis_space.append(_get_call_injection(parse_call_rule(line), counter))

            counter += 1
    return "\n".join(hypothesis_space) + "\n\n"


def _get_negative_transition_injection(parsed_transition, counter):
    injection_str = ""
    for p in parsed_transition.pos:
        injection_str += _get_negative_transition_injection_helper(parsed_transition, p, True, counter)
    for n in parsed_transition.neg:
        injection_str += _get_negative_transition_injection_helper(parsed_transition, n, False, counter)
    return injection_str


def _get_negative_transition_injection_helper(parsed_transition, symbol, is_pos_obs, counter):
    predicate_name = "injected_"
    if is_pos_obs:  # remember it is a negative transition and, thus, observables have to be inverted
        predicate_name += "neg"
    else:
        predicate_name += "pos"
    return generate_injected_statement(
        f"{predicate_name}({parsed_transition.src}, {parsed_transition.dst}, {parsed_transition.edge}, {parsed_transition.automaton}, {symbol}) "
        f":- active({counter}).")


def _get_call_injection(parsed_call, counter):
    return generate_injected_statement(
        f"injected_{CALL_STR}({parsed_call.src}, {parsed_call.dst}, {parsed_call.edge}, {parsed_call.src_automaton}, {parsed_call.dst_automaton}) "
        f":- active({counter}).")
