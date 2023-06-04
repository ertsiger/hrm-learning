from gym_hierarchical_subgoal_automata.automata.condition import CallCondition, FormulaCondition
from gym_hierarchical_subgoal_automata.automata.logic import DNFFormula
import numpy as np
from reinforcement_learning.ihsa_hrl_algorithm import FormulaBank, IHSAAlgorithmHRL
from typing import Dict, List, Tuple
from utils import file_utils, math_utils


class FormulaBankTabular(FormulaBank):
    def __init__(self, params, ignore_empty_observations):
        super(FormulaBankTabular, self).__init__(params, ignore_empty_observations)
        self._q_functions: Dict[FormulaCondition, np.ndarray] = {}

    def _get_q_function_keys(self):
        return list(self._q_functions.keys())

    def _has_q_function(self, formula_condition: FormulaCondition):
        return formula_condition in self._q_functions

    def _copy_q_function(self, from_condition: FormulaCondition, to_condition: FormulaCondition, task):
        super(FormulaBankTabular, self)._copy_q_function(from_condition, to_condition, task)
        self._q_functions[to_condition] = self._q_functions[from_condition].copy()

    def _rm_q_function(self, formula_condition: FormulaCondition):
        super(FormulaBankTabular, self)._rm_q_function(formula_condition)
        self._q_functions.pop(formula_condition)

    def _init_q_function(self, formula_condition: FormulaCondition, task):
        super(FormulaBankTabular, self)._init_q_function(formula_condition, task)
        self._q_functions[formula_condition] = np.zeros(
            (task.observation_space.n, task.action_space.n),
            dtype=np.float32
        )

    def update_q_functions(self, task, state, action, next_state, is_terminal, is_goal_achieved, observation):
        """
        Given a low level experience in the environment <current_state, action, next_state, is_terminal, observation>,
        the Q-functions of all the subgoals are updated. This is a form of intra-option learning. Note that each
        Q-function is updated with a reward that depends on the formula itself (a local objective). The Q-functions are
        updated using Q-Learning.

        TODO: We could move this to on_step_performed in the future, should be equivalent and we would avoid
         distributing the code in different parts.
        """
        super(FormulaBankTabular, self).update_q_functions(task, state, action, next_state, is_terminal, is_goal_achieved, observation)

        current_pair = (state, action)

        for formula in self._get_subgoals_to_update():
            formula_q_function = self._q_functions[formula]
            reward, is_terminal_local = self._get_subgoal_pseudoreward(formula, observation, is_terminal, is_goal_achieved)
            if is_terminal_local:
                delta = reward
            else:
                next_action_value = formula_q_function[next_state].max()
                delta = reward + self._discount_rate * next_action_value

            formula_q_function[current_pair] += self._learning_rate * (delta - formula_q_function[current_pair])
            self._q_function_update_counter[formula] += 1
        self._inc_num_update_calls()

    def get_q_function(self, formula_condition: FormulaCondition):
        root_formula_condition = self.get_root(formula_condition).get_formula_condition()
        return self._q_functions[root_formula_condition]

    def _export_bank_helper(self, bank, export_path):
        file_utils.dump_pickle_obj(bank, export_path)

    def _import_bank_helper(self, import_path):
        return file_utils.load_pickle_obj(import_path)

    def _load_q_function(self, formula_condition: FormulaCondition, q_function):
        self._q_functions[formula_condition] = q_function


class IHSAAlgorithmHRLTabular(IHSAAlgorithmHRL):
    """
    Implements the options' policies using Tabular SMDP Q-Learning.
    """
    EXPORT_MODEL_EXTENSION = "pkl"

    def __init__(self, params):
        super().__init__(params)

        # Q-functions for selecting among options
        self._meta_q_functions: Dict[int, Dict[str, Dict[Tuple[str, DNFFormula], np.ndarray]]] = {task_id: {} for task_id in range(self.num_tasks)}
        self._init_meta_q_functions()

        # Q-functions for the formulas
        self._formula_banks: List[FormulaBankTabular] = [
            FormulaBankTabular(params, self.ignore_empty_observations)
            for _ in range(self.num_tasks)
        ]

        self._automaton_step_counter: Dict[int, Dict[str, Dict[Tuple[str, FormulaCondition], int]]] = {task_id: {} for task_id in range(self.num_tasks)}
        self._init_formula_q_functions()

    '''
    Formula Options Management
    '''
    def _get_policy_bank(self, task_id):
        return self._formula_banks[task_id]

    def _get_policy_banks(self) -> List[FormulaBank]:
        return self._formula_banks

    def _export_policy_banks(self):
        for task_id in range(self.num_tasks):
            self._formula_banks[task_id].export_bank(
                self._get_formula_bank_model_path(IHSAAlgorithmHRLTabular.EXPORT_MODEL_EXTENSION, task_id)
            )

    def _import_policy_banks(self):
        for task_id in range(self.num_tasks):
            self._formula_banks[task_id].import_bank(
                self._get_formula_bank_model_path(IHSAAlgorithmHRLTabular.EXPORT_MODEL_EXTENSION, task_id)
            )

    def _on_initial_observation(self, observation):
        for task_id in range(self.num_tasks):
            # We could pick any task here (e.g., 0 which will always exist), but we can leverage the fact that we
            # iterate the different task ids.
            self._formula_banks[task_id].on_task_observation(observation, self._get_task(0, task_id))

    '''
    Meta Q-Functions Management
    '''
    def _get_greedy_condition(self, task_id, state, automaton, automaton_state, context, hierarchy, ignore_rej_edges):
        # From a given automaton state, get the outgoing conditions and the resulting automaton states. Note that the
        # conditions are not obtained taking the context into account. If we are interested in appending the context to
        # the returned condition later on, we will have to do it separately.
        conditions, tgt_automaton_states = zip(*automaton.get_outgoing_conditions(automaton_state))

        # Take the Q-values for each of the previous outgoing conditions under a given context. Unlike the previous
        # step, here we do take the context into account.
        q_values = self._meta_q_functions[task_id][automaton.get_name()][(automaton_state, context)][state, :]

        # Mask values for transitions to rejecting state with highly negative values.
        if ignore_rej_edges:
            q_values[np.array(tgt_automaton_states) == automaton.get_reject_state()] = 0.1 * IHSAAlgorithmHRLTabular.Q_VALUE_NEG_MASK

        # Compute a mask of the unsat conditions and mask the corresponding q-values.
        # TODO: This is kind of inefficient because the first operation and this one are similar. I thought of putting
        #  everything together but I didn't want to 'contaminate' the first operation with the parameters required by
        #  this one.
        sat_conditions, _ = zip(*automaton.get_outgoing_conditions_with_terminating_paths(
            automaton_state, context, self._get_policy_bank(task_id).get_observations(), hierarchy, False
        ))
        unsat_mask = np.array([c not in sat_conditions for c in conditions])
        q_values[unsat_mask] = IHSAAlgorithmHRLTabular.Q_VALUE_NEG_MASK

        # Take the index of the condition with the highest Q-value.
        max_q_idx = math_utils.randargmax(q_values)

        # Use the previous index to return the condition to pursue and the resulting automaton state if we successfully
        # achieve it.
        return conditions[max_q_idx], tgt_automaton_states[max_q_idx]

    def _get_automaton_state_step_count(self, task_id, automaton_name, automaton_state, context):
        if automaton_name not in self._automaton_step_counter[task_id] or \
           (automaton_state, context) not in self._automaton_step_counter[task_id][automaton_name]:
            return 0
        return self._automaton_step_counter[task_id][automaton_name][(automaton_state, context)]

    def _inc_automaton_state_step_count(self, task_id, automaton_name, automaton_state, context):
        if automaton_name not in self._automaton_step_counter[task_id]:
            self._automaton_step_counter[task_id][automaton_name] = {}
        if (automaton_state, context) not in self._automaton_step_counter[task_id][automaton_name]:
            self._automaton_step_counter[task_id][automaton_name][(automaton_state, context)] = 0
        self._automaton_step_counter[task_id][automaton_name][(automaton_state, context)] += 1

    def _reset_automaton_step_counters(self, automaton_name):
        for task_id in range(self.num_tasks):
            if automaton_name in self._automaton_step_counter[task_id]:
                self._automaton_step_counter[task_id][automaton_name].clear()

    def _init_hierarchy_meta_q_functions_for_hierarchy_state(self, domain_id, task_id, automaton_name, state_name, context):
        # The q-functions for other automata (!= root) should not be always reset, only when necessary (i.e., the entry
        # does not exist). That's why force_init=False (default value).
        self._init_meta_q_functions_for_automaton(domain_id, task_id, automaton_name)

        # Add an entry for the (state, context) pair in the automaton Q-function.
        automaton_q_function = self._meta_q_functions[task_id][automaton_name]
        state_ctx = (state_name, context)
        if state_ctx not in automaton_q_function:
            automaton_q_function[state_ctx] = np.zeros(
                (
                    self.tasks[self.environment_names[domain_id]][task_id].observation_space.n,
                    self._get_hierarchy(domain_id).get_automaton(automaton_name).get_num_outgoing_conditions(state_name, context)
                ),
                dtype=np.float32
            )

    def _init_meta_q_functions_for_automaton(self, domain_id, task_id, automaton_name, force_init=False):
        task_q_function = self._meta_q_functions[task_id]
        if force_init or automaton_name not in task_q_function:
            if self.training_enable:
                task_q_function[automaton_name] = {}
            else:
                task_q_function[automaton_name] = file_utils.load_pickle_obj(self._get_metacontroller_file_path(
                    automaton_name, IHSAAlgorithmHRLTabular.EXPORT_MODEL_EXTENSION, task_id
                ))

    def _update_meta_q_functions(self, domain_id, task_id, next_state, is_terminal, option, hierarchy, observation, satisfied_automata):
        """
        Updates the Q-function of a given metacontroller at a triplet <automaton, automaton state, context> given a
        high-level experience <state, option, next state, is terminal, observation, satisfied automata>. The option
        itself contains important information to perform the update: the starting state, the number of steps, the
        automaton, the automaton state, and the target automaton state (i.e. the automaton state we would be at if the
        option runs successfully - achieves its goal). SMDP Q-learning is used to perform the update.
        """
        meta_q_function = self._meta_q_functions[task_id][option.get_automaton_name()]

        automaton = hierarchy.get_automaton(option.get_automaton_name())
        automaton_state = option.get_automaton_state()

        # The updated automaton state is not the one in the hierarchy but inside the same automaton where the option
        # acts
        next_automaton_state = self._get_next_automaton_state(option, observation, satisfied_automata)

        # Compute the reward obtained by the option.
        delta = self._get_option_pseudoreward(option, hierarchy, observation, satisfied_automata)

        # If the next environment state is not terminal and the next automaton state is not terminal, then we have to
        # compute the bootstrapping term.
        if not is_terminal and not automaton.is_terminal_state(next_automaton_state):
            next_context = self._get_next_context(automaton_state, option.get_context(), next_automaton_state)

            assert not automaton.is_deadend_state(next_automaton_state)

            # Get the next condition and get its id based on the outgoing edges of the next automaton state. Note that
            # the next condition is contextless; therefore, we must not pass the next_context to the method that gives
            # us the id of the next condition.
            next_condition, _ = self._get_greedy_condition(task_id, next_state, automaton, next_automaton_state, next_context, hierarchy, False)
            next_condition_id = automaton.get_edge_id_from_state_for_condition(next_automaton_state, next_condition)

            discount = self._meta_discount_rate ** option.get_num_steps()
            delta += discount * meta_q_function[(next_automaton_state, next_context)][(next_state, next_condition_id)]

        assert not automaton.is_deadend_state(automaton_state)

        # Get the id of the condition in the run option. Note that we pass the base condition, which does not include
        # the context: remember that the conditions we get from _choose_egreedy_condition are contextless (thus, we must
        # pass the contextless version of the condition to get the id). Form a pair using this id and the environment
        # state in which the option started.
        state_cond_pair = (
            option.get_start_state(),
            automaton.get_edge_id_from_state_for_condition(automaton_state, option.get_base_condition())
        )
        delta -= meta_q_function[(automaton_state, option.get_context())][state_cond_pair]
        meta_q_function[(automaton_state, option.get_context())][state_cond_pair] += self._meta_learning_rate * delta

    def _export_meta_functions(self, automaton):
        for task_id in range(self.num_tasks):
            file_utils.dump_pickle_obj(
                self._meta_q_functions[task_id][automaton.get_name()],
                self._get_metacontroller_file_path(automaton.get_name(), IHSAAlgorithmHRLTabular.EXPORT_MODEL_EXTENSION, task_id)
            )
