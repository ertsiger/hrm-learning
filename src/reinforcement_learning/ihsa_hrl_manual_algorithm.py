from gym_hierarchical_subgoal_automata.automata.condition import FormulaCondition
from reinforcement_learning.ihsa_hrl_algorithm import IHSAAlgorithmHRL, FormulaBank
from typing import List


class IHSAAlgorithmHRLManual(IHSAAlgorithmHRL):
    """
    Implementation of an agent whose decisions are controlled by the user (i.e., which edge to take and what actions to
    take). Purely used for debugging the option selection algorithm and termination of options in a controlled manner.
    """
    def _choose_egreedy_condition(self, task_id, state, automaton, automaton_state, context, hierarchy):
        conditions = automaton.get_outgoing_conditions(automaton_state)
        cond_index = -1
        while cond_index not in range(0, len(conditions)):
            conditions_str = [str(cond) for cond in conditions]
            cond_index = int(input(f"Choose a condition ({conditions_str}): "))
        return conditions[cond_index]

    def _get_greedy_condition(self, task_id, state, automaton, automaton_state, context, hierarchy, ignore_rej_edges):
        pass

    def _choose_action(self, domain_id, task_id, state, hierarchy, hierarchy_state):
        # Override the method to show the task first
        task = self._get_task(domain_id, task_id)
        task.render()
        return super(IHSAAlgorithmHRLManual, self)._choose_action(domain_id, task_id, state, hierarchy, hierarchy_state)

    def _choose_egreedy_action(self, task, state, q_function, epsilon):
        action = -1
        while action not in range(0, task.action_space.n):
            action = int(input(f"Choose an action (0-{task.action_space.n - 1}): "))
        return action

    def _update_q_functions(self, domain_id, task_id, current_state, action, next_state, is_terminal, observations):
        pass

    def _get_q_function(self, task_id, formula_condition: FormulaCondition):
        pass

    def _get_formula_q_function_step_count(self, task_id, formula_condition: FormulaCondition):
        pass

    def _inc_formula_q_function_step_count(self, task_id, formula_condition: FormulaCondition):
        pass

    def _init_hierarchy_meta_q_functions_for_hierarchy_state(self, domain_id, task_id, automaton_name, state_name,
                                                             context):
        pass

    def _init_meta_q_functions_for_automaton(self, domain_id, task_id, automaton_name, force_init=False):
        pass

    def _update_meta_q_functions(
        self, domain_id, task_id, next_state, is_terminal, option, hierarchy, observation, satisfied_automata
    ):
        pass

    def _get_automaton_state_step_count(self, task_id, automaton_name, automaton_state, context):
        pass

    def _inc_automaton_state_step_count(self, task_id, automaton_name, automaton_state, context):
        pass

    def _reset_automaton_step_counters(self, automaton_name):
        pass

    def _export_policy_banks(self):
        pass

    def _export_meta_functions(self, automaton):
        pass

    def _on_initial_observation(self, observation):
        pass

    def _import_policy_banks(self):
        pass

    def _get_policy_banks(self) -> List[FormulaBank]:
        pass

    def _get_policy_bank(self, task_id) -> FormulaBank:
        pass
