from gym_hierarchical_subgoal_automata.automata.hierarchical_automaton import HierarchicalAutomaton
from ilasp.generator.utils.ilasp_task_generator_example import generate_examples
from ilasp.generator.utils.ilasp_task_generator_hypothesis import get_hypothesis_space
from ilasp.generator.utils.ilasp_task_generator_automata import generate_background_automata
from ilasp.generator.utils.ilasp_task_generator_state import generate_state_statements
from ilasp.generator.utils.ilasp_task_generator_transition import generate_timestep_statements, generate_transition_statements
from ilasp.generator.utils.ilasp_task_generator_symmetry_breaking import generate_symmetry_breaking_statements
import os
from typing import List, Tuple


def generate_ilasp_task(automaton_name, num_states, accepting_state, rejecting_state, hyp_observables, goal_examples,
                        dend_examples, inc_examples, background_automata: List[Tuple[HierarchicalAutomaton, bool]],
                        output_folder, output_filename, symmetry_breaking_enable, symmetry_breaking_observables,
                        max_disj_size, learn_acyclic, avoid_learning_only_negative, remove_deadend_states,
                        use_mutex_saturation, use_rejecting_state):
    if num_states < 3:
        raise ValueError("The number of states should be >= 3.")

    with open(os.path.join(output_folder, output_filename), 'w') as f:
        task = _generate_ilasp_task_str(automaton_name, num_states, accepting_state, rejecting_state, hyp_observables,
                                        goal_examples, dend_examples, inc_examples, background_automata, output_folder,
                                        symmetry_breaking_enable, symmetry_breaking_observables, max_disj_size,
                                        learn_acyclic, avoid_learning_only_negative, remove_deadend_states,
                                        use_mutex_saturation, use_rejecting_state)
        f.write(task)


def _generate_ilasp_task_str(automaton_name, num_states, accepting_state, rejecting_state, hyp_observables, goal_examples,
                             dend_examples, inc_examples, background_automata: List[Tuple[HierarchicalAutomaton, bool]],
                             output_folder, symmetry_breaking_enable, symmetry_breaking_observables, max_disj_size,
                             learn_acyclic, avoid_learning_only_negative, remove_deadend_states, use_mutex_saturation,
                             use_rejecting_state):
    task = generate_state_statements(automaton_name, num_states, accepting_state, rejecting_state)
    task += generate_timestep_statements(goal_examples, dend_examples, inc_examples)
    task += generate_background_automata(background_automata)
    task += generate_transition_statements(automaton_name, learn_acyclic, avoid_learning_only_negative,
                                           remove_deadend_states, accepting_state, rejecting_state, use_mutex_saturation)

    # The rejecting state will not be in the rules of the hypothesis space if use_rejecting_state is False
    root_rejecting_state = rejecting_state if use_rejecting_state else None
    task += get_hypothesis_space(automaton_name, num_states, accepting_state, root_rejecting_state, hyp_observables,
                                 background_automata, output_folder, max_disj_size)

    if symmetry_breaking_enable:
        task += generate_symmetry_breaking_statements(automaton_name, num_states, accepting_state, rejecting_state,
                                                      symmetry_breaking_observables, background_automata, max_disj_size)
    task += generate_examples(goal_examples, dend_examples, inc_examples)
    return task

