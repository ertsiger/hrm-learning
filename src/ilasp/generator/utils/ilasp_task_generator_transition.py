from ilasp.ilasp_common import CALL_STR, CONNECTED_STR, LEAF_AUTOMATON, N_TRANSITION_STR, TRANSITION_STR, \
    generate_injected_block, generate_injected_statement
from ilasp.generator.utils.ilasp_task_generator_example import get_longest_example_length


def generate_timestep_statements(goal_examples, dend_examples, inc_examples):
    stmt = f"all_steps(0..{get_longest_example_length(goal_examples, dend_examples, inc_examples)}).\n"
    stmt += "step(T) :- all_steps(T), last(U), T<U+1.\n\n"
    return stmt


def _generate_acceptance_rejection_rules(automaton_name):
    stmt = f"accept :- last(T), satisfied({automaton_name}, 0, T+1).\n"
    stmt += f"reject :- last(T), failed({automaton_name}, 0, T+1).\n"
    return stmt + "\n"


def generate_transition_statements(automaton_name, learn_acyclic, avoid_learning_only_negative, remove_deadend_states,
                                   accepting_state, rejecting_state, use_mutex_saturation):
    stmt = f"reachable(u0, {automaton_name}, 0, 0).\n"
    stmt += f"reachable(u0, M, T, T) :- state(u0, M), M!={automaton_name}, step(T).\n"
    stmt += f"reachable(X, M, T0, T+1) :- reachable(X, M, T0, T), not pre_sat(X, M, T), step(T), X=u0, M={automaton_name}.\n"

    stmt += f"reachable(X, M, T0, T+1) :- reachable(X, M, T0, T), not pre_sat(X, M, T), step(T), T0<T, X!={accepting_state}.\n"

    # loop accepting state of the root (the others are not) - not that in the rule above, the rej states all loop
    stmt += f"reachable(X, M, T0, T+1) :- reachable(X, M, T0, T), not pre_sat(X, M, T), step(T), X={accepting_state}, M={automaton_name}.\n"

    stmt += "reachable(Y, M, T0, T2) :- reachable(X, M, T0, T), delta(X, Y, M, T, T2).\n\n"

    stmt += f"satisfied(M, T0, TE) :- reachable({accepting_state}, M, T0, TE).\n"
    stmt += f"satisfied({LEAF_AUTOMATON}, T, T+1) :- step(T).\n\n"

    stmt += f"failed(M, T0, TE) :- reachable({rejecting_state}, M, T0, TE).\n\n"

    stmt += f"{TRANSITION_STR}(X, Y, E, M, M2, T) :- not {N_TRANSITION_STR}(X, Y, E, M, T), " \
            f"{CALL_STR}(X, Y, E, M, M2), step(T).\n"

    stmt += f"pre_sat(X, M, T) :- {TRANSITION_STR}(X, _, _, M, {LEAF_AUTOMATON}, T).\n"
    stmt += f"pre_sat(X, M, T) :- {TRANSITION_STR}(X, _, _, M, M2, T), pre_sat(u0, M2, T), M2!={LEAF_AUTOMATON}.\n\n"

    stmt += f"delta(X, Y, M, T, T2) :- {TRANSITION_STR}(X, Y, _, M, M2, T), satisfied(M2, T, T2).\n\n"

    stmt += f"delta(X, {rejecting_state}, M, T, T2) :- {TRANSITION_STR}(X, _, _, M, M2, T), failed(M2, T, T2).\n\n"

    stmt += _generate_acceptance_rejection_rules(automaton_name)

    # helper predicate (similar to what we had in previous work)
    stmt += f"{CONNECTED_STR}(X, Y, E, M) :- {CALL_STR}(X, Y, E, M, _).\n"
    stmt += generate_injected_statement(f"injected_{CONNECTED_STR}(X, Y, E, M) :- injected_{CALL_STR}(X, Y, E, M, _).")
    stmt += "\n\n"

    # avoid solutions with states different from the accepting and rejecting states without outgoing transitions
    if remove_deadend_states:
        stmt += f"has_outgoing_edges(X, M) :- {CONNECTED_STR}(X, _, _, M).\n"
        stmt += f":- state(X, M), not has_outgoing_edges(X, M), X!={accepting_state}, X!={rejecting_state}.\n\n"

    # no two deltas to different states can be satisfied at the same time (this is not needed given the enforcement of
    # mutual exclusivity)
    # stmt += ":- delta(X, Y, _, M, T, _), delta(X, Z, _, M, T, _), Y!=Z.\n\n"

    # an edge cannot be labeled by two calls simultaneously
    stmt += f":- {CALL_STR}(X, Y, E, M, M2), {CALL_STR}(X, Y, E, M, M3), M2!=M3.\n\n"

    if learn_acyclic:  # restrictions to force acyclicity
        stmt += f"path(X, Y, M) :- {CONNECTED_STR}(X, Y, _, M).\n"
        stmt += f"path(X, Y, M) :- {CONNECTED_STR}(X, Z, _, M), path(Z, Y, M).\n"
        stmt += ":- path(X, Y, M), path(Y, X, M).\n\n"

    # extra edge constraints (e.g., determinism)
    stmt += _generate_injected_constraints(automaton_name, avoid_learning_only_negative, use_mutex_saturation)
    return stmt


def _generate_injected_constraints(automaton_name, avoid_learning_only_negative, use_mutex_saturation):
    # the same observation cannot appear positive and negative in the same edge
    # they are actually not needed since ILASP attempts to find a minimal hypothesis and this case clearly
    # will appear only in non-minimal hypothesis (since the transition is cannot be taken, it is equivalent
    # to the loop transition used when no outgoing formulas are satisfied)
    stmts = [":- injected_pos(X, Y, E, M, O), injected_neg(X, Y, E, M, O)."]

    if use_mutex_saturation:
        stmts.extend(_generate_injected_mutex_saturation_constraints(automaton_name))
    else:
        stmts.extend(_generate_injected_mutex_constraints())

    # to enforce that automata cannot be reused for enforcing one path only from the initial state of the called automaton
    # deprecated: we must not use this to reuse rejecting states from lower levels!
    # stmts.append(":- injected_pos(X, Y, E, M, O), injected_call(X, Y, E, M, M2), injected_flat_pos(u0, _, _, M2, _, O).")
    # stmts.append(":- injected_pos(X, Y, E, M, O), injected_call(X, Y, E, M, M2), injected_flat_neg(u0, _, _, M2, _, O).")
    # stmts.append(":- injected_neg(X, Y, E, M, O), injected_call(X, Y, E, M, M2), injected_flat_pos(u0, _, _, M2, _, O).")
    # stmts.append(":- injected_neg(X, Y, E, M, O), injected_call(X, Y, E, M, M2), injected_flat_neg(u0, _, _, M2, _, O).")

    # avoid learning calls to the leaf automaton without conditions
    stmts.append(f":- not injected_pos(X, Y, E, M, _), not injected_neg(X, Y, E, M, _), injected_call(X, Y, E, M, {LEAF_AUTOMATON}).")

    # avoid learning purely negative formulas for the calls to the leaf automaton
    if avoid_learning_only_negative:
        stmts.append(":- injected_neg(X, Y, E, M, _), not injected_pos(X, Y, E, M, _), "
                       f"injected_call(X, Y, E, M, {LEAF_AUTOMATON}).")

    return generate_injected_block(stmts) + '\n'


def _generate_injected_mutex_constraints():
    stmts = []
    stmts.append(f"injected_flat_ed(X, Y, E, M, ()) :- injected_call(X, Y, E, M, {LEAF_AUTOMATON}).")
    stmts.append(f"injected_flat_ed(X, Y, E, M, ((u0, Y2, E2, M2), EG)) :- injected_call(X, Y, E, M, M2), M2 != {LEAF_AUTOMATON}, "
                                                                         "injected_flat_ed(u0, Y2, E2, M2, (EG)).")

    stmts.append(f"injected_flat_pos(X, Y, E, M, (), O) :- injected_call(X, Y, E, M, {LEAF_AUTOMATON}), injected_pos(X, Y, E, M, O).")
    stmts.append(f"injected_flat_pos(X, Y, E, M, ((u0, Y2, E2, M2), EG), O) :- injected_call(X, Y, E, M, M2), M2 != {LEAF_AUTOMATON}, "
                                                                             "injected_flat_pos(u0, Y2, E2, M2, (EG), O).")
    stmts.append(f"injected_flat_pos(X, Y, E, M, ((u0, Y2, E2, M2), EG), O) :- injected_call(X, Y, E, M, M2), M2 != {LEAF_AUTOMATON}, "
                                                                             "injected_pos(X, Y, E, M, O), "
                                                                             "injected_flat_ed(u0, Y2, E2, M2, (EG)).")

    stmts.append(f"injected_flat_neg(X, Y, E, M, (), O) :- injected_call(X, Y, E, M, {LEAF_AUTOMATON}), injected_neg(X, Y, E, M, O).")
    stmts.append(f"injected_flat_neg(X, Y, E, M, ((u0, Y2, E2, M2), EG), O) :- injected_call(X, Y, E, M, M2), M2 != {LEAF_AUTOMATON}, "
                                                                             "injected_flat_neg(u0, Y2, E2, M2, (EG), O).")
    stmts.append(f"injected_flat_neg(X, Y, E, M, ((u0, Y2, E2, M2), EG), O) :- injected_call(X, Y, E, M, M2), M2 != {LEAF_AUTOMATON}, "
                                                                             "injected_neg(X, Y, E, M, O), "
                                                                             "injected_flat_ed(u0, Y2, E2, M2, (EG)).")

    stmts.append("injected_mutex(X, Y, EY, EGY, Z, EZ, EGZ, M) :- injected_flat_pos(X, Y, EY, M, EGY, O), "
                                                                 "injected_flat_neg(X, Z, EZ, M, EGZ, O).")
    stmts.append("injected_mutex(X, Y, EY, EGY, Z, EZ, EGZ, M) :- injected_flat_neg(X, Y, EY, M, EGY, O), "
                                                                 "injected_flat_pos(X, Z, EZ, M, EGZ, O).")

    stmts.append(":- not injected_mutex(X, Y, EY, EGY, Z, EZ, EGZ, M), injected_flat_ed(X, Y, EY, M, EGY), "
                    "injected_flat_ed(X, Z, EZ, M, EGZ), Y != Z.")
    stmts.append(":- not injected_mutex(X, Y, EY, EGY, Y, EZ, EGZ, M), injected_flat_ed(X, Y, EY, M, EGY), "
                    "injected_flat_ed(X, Y, EZ, M, EGZ), EY != EZ, injected_call(X, Y, EY, M, MY), "
                    "injected_call(X, Y, EZ, M, MZ), MY != MZ.")
    return stmts


def _generate_injected_mutex_saturation_constraints(automaton_name):
    stmts = []
    stmts.append("injected_sat_id(1;2).")

    stmts.append(f"injected_root_point(X, M) : injected_call(X, _, _, M, _), M={automaton_name}.")

    stmts.append("injected_edge_mtx((X, Y, E, M, M2), SatID) : injected_call(X, Y, E, M, M2) :- injected_root_point(X, M), injected_sat_id(SatID).")
    stmts.append(f"injected_edge_mtx((u0, Y2, E2, M2, M3), SatID) : injected_call(u0, Y2, E2, M2, M3) :- injected_edge_mtx((_, _, _, _, M2), SatID), M2 != {LEAF_AUTOMATON}.")

    stmts.append("injected_pos_obv(O, ID) :- injected_edge_mtx((X, Y, E, M, _), ID), injected_pos(X, Y, E, M, O).")
    stmts.append("injected_neg_obv(O, ID) :- injected_edge_mtx((X, Y, E, M, _), ID), injected_neg(X, Y, E, M, O).")

    stmts.append("injected_saturate :- injected_pos_obv(O, 1), injected_neg_obv(O, 2).")
    stmts.append("injected_saturate :- injected_pos_obv(O, 2), injected_neg_obv(O, 1).")
    stmts.append("injected_saturate :- injected_edge_mtx((X, Y, _, M, M2), 1), injected_edge_mtx((X, Y, _, M, M2), 2), injected_root_point(X, M).")

    stmts.append(f"injected_root_point(X, M) :- injected_call(X, _, _, M, _), injected_saturate, M={automaton_name}.")
    stmts.append(f"injected_edge_mtx((X, Y, E, M, M2), SatID) :- injected_call(X, Y, E, M, M2), M={automaton_name}, injected_sat_id(SatID), injected_saturate.")
    stmts.append("injected_edge_mtx((u0, Y, E, M, M2), SatID) :- injected_call(u0, Y, E, M, M2), injected_sat_id(SatID), injected_saturate.")

    stmts.append(":- not injected_saturate.")
    return stmts
