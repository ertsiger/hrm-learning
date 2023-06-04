def generate_state_set(num_states, accepting_state, rejecting_state):
    states = [f"u{i}" for i in range(num_states - 2)]
    states.append(accepting_state)
    if rejecting_state is not None:
        states.append(rejecting_state)
    return states


def generate_state_statements(automaton_name, num_states, accepting_state, rejecting_state):
    states = generate_state_set(num_states, accepting_state, rejecting_state)
    return _generate_state_statements(automaton_name, states)


def generate_state_statements_for_automaton(automaton):
    return _generate_state_statements(automaton.get_name(), automaton.get_states())


def _generate_state_statements(automaton_name, states):
    return "".join([f"state({s}, {automaton_name}).\n" for s in states]) + '\n'


def get_state_priorities(num_states, accepting_state, rejecting_state, use_terminal_priority):
    states = generate_state_set(num_states, accepting_state, rejecting_state)
    states_with_priority = states if use_terminal_priority \
                                  else [s for s in states if s != accepting_state and s != rejecting_state]
    state_priorities = []
    for s, priority in zip(states_with_priority, range(len(states_with_priority))):
        state_priorities.append((s, priority))
    return state_priorities

