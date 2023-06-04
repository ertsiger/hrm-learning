from ilasp.ilasp_common import generate_injected_block
from ilasp.generator.utils.ilasp_task_generator_state import generate_state_set, get_state_priorities


def generate_symmetry_breaking_statements(automaton_name, num_states, accepting_state, rejecting_state, observables,
                                          background_automata, max_disj_size):
    stmt = _generate_injected_state_statements(num_states, accepting_state, rejecting_state)
    stmt += _generate_injected_state_priority_statements(num_states, accepting_state, rejecting_state)
    stmt += _generate_injected_observable_statements(observables, background_automata)
    stmt += _generate_bfs_symmetry_breaking_rules_alternative(automaton_name, max_disj_size)
    return stmt


def _generate_injected_state_statements(num_states, accepting_state, rejecting_state):
    states = generate_state_set(num_states, accepting_state, rejecting_state)
    return generate_injected_block([f"injected_state({s})." for s in states]) + '\n'


def _generate_injected_state_priority_statements(num_states, accepting_state, rejecting_state):
    state_priorities = get_state_priorities(num_states, accepting_state, rejecting_state, False)
    return generate_injected_block([f"injected_state_id({s}, {i})." for s, i in state_priorities]) + '\n'


def _generate_injected_observable_statements(observables, background_automata):
    obs_list = sorted(observables) + sorted([bk_automaton.get_name()
                                             for bk_automaton, is_callable in background_automata if is_callable])
    stmts = [f"injected_obs_id({obs_list[i]}, {i + 1})." for i in range(len(obs_list))]
    stmts.append(f"injected_num_obs({len(obs_list)}).")
    stmts.append(f"injected_valid_label(1..{2 * len(observables) + len(background_automata)}).")
    return generate_injected_block(stmts) + '\n'


def _generate_bfs_symmetry_breaking_rules_alternative(automaton_name, max_disj_size):
    # helper methods for comparing state ids
    stmts = ["injected_state_leq(X, Y) :- injected_state_id(X, XID), injected_state_id(Y, YID), XID<=YID.",
             "injected_state_lt(X, Y) :- injected_state_id(X, XID), injected_state_id(Y, YID), XID<YID."]

    # edge ids
    stmts.append(f"injected_edge_id(1..{max_disj_size}).")

    # use lower edge ids first
    stmts.append(f":- injected_ed(X, Y, E, {automaton_name}), not injected_ed(X, Y, E-1, {automaton_name}), injected_edge_id(E), E>1.")

    # 1. choose an ordering between edges from a given state
    # 2. impose transitive relationships
    # 3. impose that an edge with a lower edge id is lower than another edge to the same state with a higher edge id
    stmts.append(f"1 {{ injected_ed_lt(X, (Y, E), (YP, EP)) ; injected_ed_lt(X, (YP, EP), (Y, E)) }} 1 :- injected_ed(X, Y, E, {automaton_name}), injected_ed(X, YP, EP, {automaton_name}), (Y, E) < (YP, EP).")
    stmts.append(":- injected_ed_lt(X, Edge1, Edge2), injected_ed_lt(X, Edge2, Edge3), not injected_ed_lt(X, Edge1, Edge3), Edge1 != Edge3.")
    stmts.append(f":- injected_ed_lt(X, (Y, E), (Y, EP)), injected_ed(X, Y, E, {automaton_name}), injected_ed(X, Y, EP, {automaton_name}), E>EP.")

    # parenting relationship
    # 1. define edges only for those states with ids (accepting and rejecting states are excluded)
    # 2. the parent is the node with lowest id with an edge to Y
    # 3. all nodes have a parent
    # 4. BFS ordering
    stmts.append(f"injected_ed_sb(X, Y, E) :- injected_ed(X, Y, E, {automaton_name}), injected_state_id(Y, _).")
    stmts.append("injected_pa(X, Y) :- injected_ed_sb(X, Y, _), injected_state_lt(X, Y), "
                 "#false : injected_ed_sb(Z, Y, _), injected_state_lt(Z, X).")
    stmts.append(":- injected_state_id(Y, YID), YID > 0, not injected_pa(_, Y).")
    stmts.append(":- injected_pa(X, Y), injected_ed_sb(XP, YP, _), injected_state_lt(XP, X), injected_state_leq(Y, YP).")

    # if X is the parent of Y, there is a smallest edge from X to Y (there is no destination YP such that Y<YP, but
    # the edge (X, YP) is lower than Y), no edge from X violates the order w.r.t. Y
    stmts.append("injected_state_ord(Y) :- injected_ed_sb(X, Y, E), injected_pa(X, Y), "
                 "#false : injected_ed_sb(X, YP, EP), injected_state_lt(Y, YP), injected_ed_lt(X, (YP, EP), (Y, E)).")
    stmts.append(":- injected_pa(_, Y), not injected_state_ord(Y).")

    # map pos and neg into integer labels
    stmts.append(f"injected_label(X, (Y, E), OID) :- injected_pos(X, Y, E, {automaton_name}, O), injected_obs_id(O, OID).")
    stmts.append(f"injected_label(X, (Y, E), OID+N) :- injected_neg(X, Y, E, {automaton_name}, O), injected_obs_id(O, OID), injected_num_obs(N).")
    stmts.append(f"injected_label(X, (Y, E), OID) :- injected_call(X, Y, E, {automaton_name}, O), injected_obs_id(O, OID).")

    # observation ordering should follow edge ordering
    stmts.append("injected_label_lt(X, Edge1, Edge2, L) :- injected_ed_lt(X, Edge1, Edge2), not injected_label(X, Edge1, L), injected_label(X, Edge2, L).")
    stmts.append("injected_label_lt(X, Edge1, Edge2, L+1) :- injected_label_lt(X, Edge1, Edge2, L), injected_valid_label(L+1).")
    stmts.append(":- injected_ed_lt(X, Edge1, Edge2), injected_label(X, Edge1, L), not injected_label(X, Edge2, L), not injected_label_lt(X, Edge1, Edge2, L).")

    return generate_injected_block(stmts) + '\n'
