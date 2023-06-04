from ilasp.generator.utils.ilasp_task_generator_state import generate_state_statements_for_automaton
from ilasp.ilasp_common import generate_injected_block, CALL_STR, LEAF_AUTOMATON, N_TRANSITION_STR, OBS_STR


def generate_background_automata(background_automata):
    stmt = ""
    for bk_automaton, _ in background_automata:
        stmt += _generate_background_automaton(bk_automaton)
    return stmt


def _generate_background_automaton(automaton):
    stmt = generate_state_statements_for_automaton(automaton)

    automaton_rules, injected_automaton_rules = [], []

    automaton_edges = automaton.get_edges()
    for from_state in automaton_edges:
        for to_state in automaton_edges[from_state]:
            edge_id = 1
            for condition in automaton_edges[from_state][to_state]:
                dnf_formula, called_automaton = _get_formula_and_call_from_condition(condition)
                if dnf_formula.is_true():
                    _add_call_fact(from_state, to_state, edge_id, automaton.get_name(), called_automaton,
                                   automaton_rules, injected_automaton_rules)
                    edge_id += 1
                else:
                    for disjunct in dnf_formula.get_formula():
                        _add_call_fact(from_state, to_state, edge_id, automaton.get_name(), called_automaton,
                                       automaton_rules, injected_automaton_rules)
                        _add_transition_rule(from_state, to_state, edge_id, automaton.get_name(), disjunct,
                                             automaton_rules, injected_automaton_rules)
                        edge_id += 1

    stmt += "\n".join(automaton_rules) + "\n\n"
    stmt += generate_injected_block(injected_automaton_rules) + '\n'
    return stmt


def _get_formula_and_call_from_condition(condition):
    if condition.is_call():
        return condition.get_context(), condition.get_called_automaton()
    else:
        return condition.get_formula(), LEAF_AUTOMATON


def _add_call_fact(from_state, to_state, edge_id, from_automaton, called_automaton, automaton_rules, injected_automaton_rules):
    call_rule = f"{CALL_STR}({from_state}, {to_state}, {edge_id}, {from_automaton}, {called_automaton})."
    automaton_rules.append(call_rule)
    injected_automaton_rules.append(f"injected_{call_rule}")


def _add_transition_rule(from_state, to_state, edge_id, automaton_name, formula, automaton_rules, injected_automaton_rules):
    for literal in formula.get_formula():
        if formula.is_negative_literal(literal):
            prefix, injected_prefix = "", "neg"
        else:
            prefix, injected_prefix = "not ", "pos"
        obs = formula.get_symbol_from_literal(literal)
        automaton_rules.append(f"{N_TRANSITION_STR}({from_state}, {to_state}, {edge_id}, {automaton_name}, T) "
                               f":- {prefix}{OBS_STR}(\"{obs}\", T), step(T).")
        injected_automaton_rules.append(
            f"injected_{injected_prefix}({from_state}, {to_state}, {edge_id}, {automaton_name}, {obs}).")
