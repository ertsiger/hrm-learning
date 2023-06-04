from abc import abstractmethod
from gym_hierarchical_subgoal_automata.automata.common import CallStackItem, SatisfiedCall
from gym_hierarchical_subgoal_automata.automata.condition import CallCondition, EdgeCondition, FormulaCondition
from gym_hierarchical_subgoal_automata.automata.formula_tree import FormulaTree
from gym_hierarchical_subgoal_automata.automata.logic import DNFFormula, TRUE
import numpy as np
import os
import random
from reinforcement_learning.ihsa_base_algorithm import IHSAAlgorithmBase
from reinforcement_learning.option import Option
from typing import Dict, List, Optional, Set, Tuple
from utils import file_utils
from utils.container_utils import get_param


class FormulaBank(FormulaTree):
    LEARNING_RATE = "learning_rate"
    DISCOUNT_RATE = "discount_rate"

    PSEUDOREWARD_DEADEND = "pseudoreward_deadend"                          # amount of reward given for reaching deadend MDP states
    PSEUDOREWARD_CONDITION_SATISFIED = "pseudoreward_condition_satisfied"  # amount of reward when the condition of an option is satisfied
    PSEUDOREWARD_AFTER_STEP = "pseudoreward_after_step"                    # reward given after every step for a formula

    FORMULA_UPDATE_SEL_NUM = "formula_update_sel_num"    # number of formula Q-functions to be updated after each step

    def __init__(self, params, ignore_empty_observations):
        super(FormulaBank, self).__init__()

        self._learning_rate = get_param(params, FormulaBank.LEARNING_RATE, 0.1)
        self._discount_rate = get_param(params, FormulaBank.DISCOUNT_RATE, 0.99)

        self._ignore_empty_observations = ignore_empty_observations

        # Pseudorewards to update the q-functions of the conditions
        self._pseudoreward_deadend_state = get_param(params, FormulaBank.PSEUDOREWARD_DEADEND, 0.0)
        self._pseudoreward_condition_satisfied = get_param(params, FormulaBank.PSEUDOREWARD_CONDITION_SATISFIED, 1.0)
        self._pseudoreward_after_step = get_param(params, FormulaBank.PSEUDOREWARD_AFTER_STEP, 0.0)

        self._formula_update_sel_num = get_param(params, FormulaBank.FORMULA_UPDATE_SEL_NUM, 6)
        self._num_update_calls = 0
        assert self._formula_update_sel_num is None or self._formula_update_sel_num >= 1

        # Keeps track the number of steps for which a subgoal has run, and how many times it has been updated.
        self._q_function_step_counter: Dict[FormulaCondition, int] = {}
        self._q_function_update_counter: Dict[FormulaCondition, int] = {}

        # A set of the active subgoals in the automata
        self._active_subgoals: Set[FormulaCondition] = set()

        # A list derived from the set of active but replacing each of those formula conditions by the roots in the bank
        # and discarding the unsatisfiable ones
        self._active_root_sat_subgoals: List[FormulaCondition] = []

    def add_q_function(self, task, formula_condition: FormulaCondition):
        if not self.contains(formula_condition):
            self.add_formula(formula_condition)
            self._update_bank(task)

    def _update_bank(self, task):
        # If a formula that had a q-function is not a root anymore, take its root and check whether it has a Q-function.
        # If it doesn't, we take the Q-function of the formula, copy it and later remove the entry from the bank.
        for formula_condition in self._get_q_function_keys():
            formula_node = self.formula_nodes[formula_condition]
            if not formula_node.is_root():
                root_formula_condition = self.get_root(formula_condition).get_formula_condition()
                if not self._has_q_function(root_formula_condition):
                    self._copy_q_function(formula_condition, root_formula_condition, task)
                self._rm_q_function(formula_condition)

        # If some root node still does not have a q-function
        for child_node in self.root_node.children:
            formula_condition = child_node.get_formula_condition()
            if not self._has_q_function(formula_condition):
                self._init_q_function(formula_condition, task)

    @abstractmethod
    def _get_q_function_keys(self):
        """
        Returns the formula conditions associated to a Q-function.
        """
        pass

    @abstractmethod
    def _has_q_function(self, formula_condition: FormulaCondition):
        """
        Returns True if the formula_condition is associated to a Q-function. Else, returns False.
        """
        pass

    @abstractmethod
    def _copy_q_function(self, from_condition: FormulaCondition, to_condition: FormulaCondition, task):
        self._q_function_step_counter[to_condition] = self._q_function_step_counter[from_condition]
        self._q_function_update_counter[to_condition] = self._q_function_update_counter[from_condition]

    @abstractmethod
    def _rm_q_function(self, formula_condition: FormulaCondition):
        self._q_function_step_counter.pop(formula_condition)
        self._q_function_update_counter.pop(formula_condition)

    @abstractmethod
    def _init_q_function(self, formula_condition: FormulaCondition, task):
        self._q_function_step_counter[formula_condition] = 0
        self._q_function_update_counter[formula_condition] = 0

    @abstractmethod
    def update_q_functions(self, task, state, action, next_state, is_terminal, is_goal_achieved, observation):
        self.on_task_observation(observation, task)

    def _inc_num_update_calls(self):
        # Intuitively, this could be in the update_q_functions BUT even if that function is called, we don't always
        # perform an update (e.g., if the replay buffer still does not have a minimum size)
        self._num_update_calls += 1

    def _get_subgoal_pseudoreward(self, formula: FormulaCondition, observation: Set[str], is_terminal, is_goal_achieved):
        """
        Returns the pseudoreward used to update the policy that aims to achieve the formula condition passed as a
        parameter. It also returns a boolean indicating if a local terminal state (i.e., a terminal state according to
        the local policy being followed) has been reached.
        """
        valid_observation = not self._ignore_empty_observations or len(observation) > 0

        if formula.is_satisfied(observation) and valid_observation:
            return self._pseudoreward_condition_satisfied, True
        elif is_terminal and not is_goal_achieved:
            # Note that if the MDP state is terminal and the goal is not achieved, the state is a deadend state.
            # Important: in previous versions we also assumed that the local termination indicator was true if the state
            # was terminal and it was the goal. However, this introduces problems in the policy updates when performing
            # intra-option learning since it considers that the goal is reached even if it's not, thus removing the
            # bootstrapping term from the Q-learning update.
            return self._pseudoreward_deadend_state, True
        return self._pseudoreward_after_step, False

    def _get_subgoals_to_update(self):
        """
        Selects a subset of subgoals whose Q-functions will be updated. These can be either selected uniformly at
        random (if the temperature is 0), or based on the number of times they have been updated (those that have been
        updated the least, have a higher probability). See subclasses of this class to check its use. Note that only
        those formulas that are satisfiable and for which there exists at least one observation that satisfies them
        are considered for an update.

        TODO: Maybe we can extend the criteria by taking into account the TD-error, or only update the subgoals of the
         latest task we selected.
        """
        base_conditions = self._active_root_sat_subgoals
        if self._formula_update_sel_num is None or self._formula_update_sel_num >= len(base_conditions):
            return base_conditions
        update_counters = np.array([self._q_function_update_counter[condition] for condition in base_conditions], dtype=np.float32)
        score = self._num_update_calls + 1 - update_counters  # Use +1 to avoid nullifying probabilities in the following step
        probs = score / np.sum(score)
        return np.random.choice(base_conditions, self._formula_update_sel_num, replace=False, p=probs)

    @abstractmethod
    def get_q_function(self, formula_condition: FormulaCondition):
        pass

    def get_num_steps(self, formula_condition: FormulaCondition):
        root_formula_condition = self.get_root(formula_condition).get_formula_condition()
        return self._q_function_step_counter[root_formula_condition]

    def inc_num_steps(self, formula_condition: FormulaCondition):
        root_formula_condition = self.get_root(formula_condition).get_formula_condition()
        self._q_function_step_counter[root_formula_condition] += 1

    def export_bank(self, export_path):
        self._export_bank_helper({
            "bank": {
                key.get_formula_condition(): self.get_q_function(key.get_formula_condition())
                for key in self.root_node.children
            },
            "observations": self.observations
        }, export_path)

    @abstractmethod
    def _export_bank_helper(self, bank, export_path):
        pass

    def import_bank(self, import_path):
        bank = self._import_bank_helper(import_path)
        for obs in bank["observations"]:
            # No need to call on_task_observation here since giving the observations beforehand is going to impose the
            # required tree ordering when inserting the formulas.
            self.on_observation(obs)
        for formula_condition in bank["bank"]:
            if not self.contains(formula_condition):
                raise RuntimeError(f"Error: The formula {formula_condition} must be contained in the formula bank before"
                                   "before loading it.")
            if not self.formula_nodes[formula_condition].is_root():
                raise RuntimeError(f"Error: The node for {formula_condition} must be a root node to load its Q-function.")
            self._load_q_function(formula_condition, bank["bank"][formula_condition])

    @abstractmethod
    def _import_bank_helper(self, import_path):
        pass

    @abstractmethod
    def _load_q_function(self, formula_condition: FormulaCondition, q_function):
        pass

    def set_active_subgoals(self, active_subgoals: Set[FormulaCondition]):
        self._active_subgoals = active_subgoals
        self._set_active_root_sat_subgoals()

    def on_task_observation(self, observation: Set[str], task):
        is_new = self.on_observation(observation)
        if is_new:
            self._update_bank(task)
            self._set_active_root_sat_subgoals()

    def _set_active_root_sat_subgoals(self):
        root_sat_subgoals = set()
        self.get_root_sat_subgoals(self._active_subgoals, root_sat_subgoals)
        self._active_root_sat_subgoals = sorted(root_sat_subgoals)

    def get_root_sat_subgoals(self, subgoals: Set[FormulaCondition], root_sat_subgoals: Set[FormulaCondition]):
        for subgoal in subgoals:
            root_formula = self.get_root_formula(subgoal)
            if root_formula.get_formula().is_satisfiable() and self._is_formula_condition_sat_by_seen_obs(root_formula):
                root_sat_subgoals.add(root_formula)


class IHSAAlgorithmHRL(IHSAAlgorithmBase):
    """
    Base class for the algorithms using the options framework to exploit the structure of a hierarchy of automata. This
    class implements high-level methods for choosing, terminating and restoring (from the stack) hierarchies of options
    (i.e., options from higher to lower levels of abstraction). The specific methods for tabular and function
    approximation methods inherit from this class.
    """
    META_LEARNING_RATE = "meta_learning_rate"  # learning rate for the metacontrollers
    META_DISCOUNT_RATE = "meta_discount_rate"  # discount rate for the metacontrollers

    META_PSEUDOREWARD_DEADEND = "meta_pseudoreward_deadend"                          # amount of reward given for reaching deadend MDP states
    META_PSEUDOREWARD_CONDITION_SATISFIED = "meta_pseudoreward_condition_satisfied"  # amount of reward when the condition of an option is satisfied
    META_PSEUDOREWARD_AFTER_STEP = "meta_pseudoreward_after_step"                    # reward given after every step for a formula

    EXPLORATION_RATE_PER_SUBGOAL_INIT = "exploration_rate_subgoal_init"      # initial exploration rate for each subgoal
    EXPLORATION_RATE_PER_SUBGOAL_END = "exploration_rate_subgoal_end"        # final exploration rate for each subgoal
    EXPLORATION_RATE_PER_SUBGOAL_STEPS = "exploration_rate_subgoal_steps"    # amount of steps throughout which each explorarion rate is decreased

    EXPLORATION_RATE_AUTOMATON_ENABLE = "exploration_rate_automaton_enable"  # whether to have a different exploration rate for each <automaton, automaton state, context>
    EXPLORATION_RATE_AUTOMATON_INIT = "exploration_rate_automaton_init"      # initial exploration rate for each triplet
    EXPLORATION_RATE_AUTOMATON_END = "exploration_rate_automaton_end"        # final exploration rate for each triplet
    EXPLORATION_RATE_AUTOMATON_STEPS = "exploration_rate_automaton_steps"    # amount of steps throughout which each explorarion rate is decreased

    OPTION_EXPLORATION_INCLUDE_AUTOMATA = "option_exploration_include_automata"  # whether to use automata options to explore the environment when a counterexample hasn't been observed yet
    OPTION_EXPLORATION_INCLUDE_FORMULAS = "option_exploration_include_formulas"  # whether to use formula options to explore the environment when a counterexample hasn't been observed yet
    OPTION_EXPLORATION_INCLUDE_ACTIONS = "option_exploration_include_actions"    # whether to use actions to explore the environment when a counterexample hasn't been observed yet

    MODELS_META_FOLDER = "meta"    # folder inside the custom models folder where the metacontrollers are saved
    MODELS_BANK_FOLDER = "banks"   # folder inside the custom models folder where the banks of policies are saved
    MODELS_BANK_FILENAME = "bank"  # name of the file containing all the lowest level policies

    PLOT_FORMULAS_FOLDER_NAME = "plot_formulas"
    PLOT_METACONTROLLERS_FOLDER_NAME = "plot_meta"

    Q_VALUE_NEG_MASK = -100000000000.0  # a negative Q-value mask for options or actions that must not be chosen when performing argmax

    def __init__(self, params):
        super().__init__(params)

        # Rates for the metacontrollers (by default, the same as the rates for the lowest level policies)
        self._meta_learning_rate = get_param(params, IHSAAlgorithmHRL.META_LEARNING_RATE, 0.1)
        self._meta_discount_rate = get_param(params, IHSAAlgorithmHRL.META_DISCOUNT_RATE, 0.99)

        # Pseudo-rewards given to the metacontrollers
        self._meta_pseudoreward_deadend_state = get_param(params, IHSAAlgorithmHRL.META_PSEUDOREWARD_DEADEND, 0.0)
        self._meta_pseudoreward_condition_satisfied = get_param(params, IHSAAlgorithmHRL.META_PSEUDOREWARD_CONDITION_SATISFIED, 1.0)
        self._meta_pseudoreward_after_step = get_param(params, IHSAAlgorithmHRL.META_PSEUDOREWARD_AFTER_STEP, 0.0)

        # Exploration rates for each subgoal. The step counters for each subgoal are kept in the subclasses.
        self._exploration_rate_subgoal_init = get_param(params, IHSAAlgorithmHRL.EXPLORATION_RATE_PER_SUBGOAL_INIT, 1.0)
        self._exploration_rate_subgoal_end = get_param(params, IHSAAlgorithmHRL.EXPLORATION_RATE_PER_SUBGOAL_END, 0.1)
        self._exploration_rate_subgoal_steps = get_param(params, IHSAAlgorithmHRL.EXPLORATION_RATE_PER_SUBGOAL_STEPS, 300000)

        # Exploration rates for each <automaton, automaton state, context> tuple.
        self._exploration_rate_automaton_init = get_param(params, IHSAAlgorithmHRL.EXPLORATION_RATE_AUTOMATON_INIT, 1.0)
        self._exploration_rate_automaton_end = get_param(params, IHSAAlgorithmHRL.EXPLORATION_RATE_AUTOMATON_END, 0.1)
        self._exploration_rate_automaton_steps = get_param(params, IHSAAlgorithmHRL.EXPLORATION_RATE_AUTOMATON_STEPS, 5000)

        # Whether to use existing options to explore the environment when the current automaton only consists of an
        # initial state (this should allow to make the search for the first counterexample faster). Note that if
        # automata and formula options are not included, primitive actions are going to be included by default.
        self._option_exploration_include_automata = get_param(params, IHSAAlgorithmHRL.OPTION_EXPLORATION_INCLUDE_AUTOMATA, True)
        self._option_exploration_include_formulas = get_param(params, IHSAAlgorithmHRL.OPTION_EXPLORATION_INCLUDE_FORMULAS, True)
        self._option_exploration_include_actions = \
            get_param(params, IHSAAlgorithmHRL.OPTION_EXPLORATION_INCLUDE_ACTIONS, True) or \
            (not self._option_exploration_include_automata and not self._option_exploration_include_formulas)
        self._option_exploration_base_condition = None
        self._option_exploration_hierarchy = None
        self._option_exploration_hierarchy_state = None

        # Option related structures and corresponding call to initialize them
        self._selected_option: Dict[int, Dict[int, Optional[Option]]] = {}
        self._init_option_tracking_structs()

        # Cache indicating whether an automaton is satisfiable (there is a path from its initial state to its accepting
        # state).
        self._is_automaton_satisfiable = {
            self._get_hierarchy(domain_id).get_root_automaton_name(): False
            for domain_id in range(self.num_domains)
        }

        # Remove old plot folders
        file_utils.rm_dirs(self._get_formula_plot_folders())
        file_utils.rm_dirs(self._get_metacontroller_plot_folders())

    '''
    Helper methods for obtaining paths to load/save policies or plots
    '''
    def _get_metacontrollers_model_folder(self):
        return os.path.join(self.models_folder, IHSAAlgorithmHRL.MODELS_META_FOLDER)

    def _get_metacontroller_file_path(self, automaton_name, extension, task_id=None):
        metacontroller_task = f"-{task_id}" if task_id is not None else ""
        return os.path.join(
            self._get_metacontrollers_model_folder(),
            f"{automaton_name}{metacontroller_task}.{extension}"
        )

    def _get_formula_bank_model_folder(self):
        return os.path.join(self.models_folder, IHSAAlgorithmHRL.MODELS_BANK_FOLDER)

    def _get_formula_bank_model_path(self, extension, task_id=None):
        formula_bank_name_task = f"-{task_id}" if task_id is not None else ""
        return os.path.join(
            self._get_formula_bank_model_folder(),
            f"{IHSAAlgorithmHRL.MODELS_BANK_FILENAME}{formula_bank_name_task}.{extension}"
        )

    def _get_formula_plot_folders(self):
        return [self._get_formula_plot_folder(domain_id) for domain_id in range(self.num_domains)]

    def _get_formula_plot_folder(self, domain_id):
        env_name = self.environment_names[domain_id]
        return os.path.join(self.export_folder_names.get(env_name), IHSAAlgorithmHRL.PLOT_FORMULAS_FOLDER_NAME)

    def _get_metacontroller_plot_folders(self):
        return [self._get_metacontroller_plot_folder(domain_id) for domain_id in range(self.num_domains)]

    def _get_metacontroller_plot_folder(self, domain_id):
        env_name = self.environment_names[domain_id]
        return os.path.join(self.export_folder_names.get(env_name), IHSAAlgorithmHRL.PLOT_METACONTROLLERS_FOLDER_NAME)

    '''
    Formula Options Management
    '''
    def _get_q_function(self, task_id, formula_condition: FormulaCondition):
        """
        Returns the Q-function corresponding to the passed formula condition in the specified task.
        """
        return self._get_policy_bank(task_id).get_q_function(formula_condition)

    def _init_formula_q_functions(self):
        """
        Add entries for the formulas in each domain hierarchy (note that since they are initialized for the first
        time, policies for existing formulas are not going to be reused).
        """
        for domain_id in range(self.num_domains):
            self._init_hierarchy_formula_q_functions(domain_id)

        if self.training_enable:
            self._update_all_subgoals()
        else:
            self._import_policy_banks()

    def _init_hierarchy_formula_q_functions(self, domain_id):
        """
        Adds an entry in the bank of formula Q-functions for each possible formula in the hierarchy given by the passed
        domain. Note that these formulas are not restricted to the ones that are found in a single automaton, but also
        take into account the contexts of the calls that propagate through the hierarchy.
        """

        # Get all the possible formula conditions in the hierarchy. Note that all these conditions are going to be
        # either TRUE (which will rarely happen in practice given its limited applicability) or a DNF with one disjunct.
        # IMPORTANT: the set is sorted for reproducibility (specially in the case of DQN, the order in which this
        # container is iterated affects the initialization of the networks).
        subgoals = set()
        self._get_hierarchy(domain_id).get_subgoals(
            subgoals, ignore_rejecting_transitions=False, only_satisfiable=False)
        subgoals = sorted(subgoals)

        for task_id in range(self.num_tasks):
            task = self._get_task(domain_id, task_id)
            for formula_condition in subgoals:
                self._get_policy_bank(task_id).add_q_function(task, formula_condition)

            # If the automata are not being learned, we use existing knowledge of what are the actual observations that
            # are going to appear in the environment (otherwise we don't know which of the formulas we are given are
            # actually applicable or not). This step is likely to be repeated many times at initialization time, but it
            # is fine (it's not going to be done during training).
            if not self.interleaved_automaton_learning:
                for obs in task.get_possible_observations():
                    self._get_policy_bank(task_id).on_task_observation(obs, task)

    @abstractmethod
    def _get_policy_bank(self, task_id) -> FormulaBank:
        pass

    @abstractmethod
    def _get_policy_banks(self) -> List[FormulaBank]:
        pass

    def _get_formula_q_function_step_count(self, task_id, formula_condition: FormulaCondition):
        return self._get_policy_bank(task_id).get_num_steps(formula_condition)

    def _inc_formula_q_function_step_count(self, task_id, formula_condition: FormulaCondition):
        self._get_policy_bank(task_id).inc_num_steps(formula_condition)

    def _update_q_functions(self, domain_id, task_id, state, action, next_state, is_terminal, observation):
        task = self._get_task(domain_id, task_id)
        self._get_policy_bank(task_id).update_q_functions(
            task, state, action, next_state, is_terminal, task.is_goal_achieved(), observation
        )

    '''
    Meta Q-Functions Management
    '''
    def _init_meta_q_functions(self):
        """
        Add entries for the meta Q-functions in the hierarchy.
        """
        for domain_id in range(self.num_domains):
            self._init_hierarchy_meta_q_functions(domain_id)

    def _init_hierarchy_meta_q_functions(self, domain_id):
        """
        Initializes the meta Q-functions of the hierarchy representing the passed domain for each task.
        """
        hierarchy = self._get_hierarchy(domain_id)

        # Get all possible hierarchy states (i.e. tuples <automaton, state, context>) in the hierarchy.
        # ATTENTION! We need to sort the set afterwards for reproducibility in the deep approach.
        possible_hierarchy_states = set()
        hierarchy.get_hierarchy_states(possible_hierarchy_states)
        possible_hierarchy_states = sorted(possible_hierarchy_states)

        for task_id in range(self.num_tasks):
            if self.interleaved_automaton_learning:  # (re)set the root automaton (e.g., if it has been relearned)
                self._init_meta_q_functions_for_automaton(domain_id, task_id, hierarchy.get_root_automaton_name(), True)

            # This loop is a bit overkill for DQN because we don't need the states and contexts, but it is nice to have
            # something general even if a bit of repetition occurs (in the end, it is not that expensive to perform).
            for automaton_name, state_name, context in possible_hierarchy_states:
                self._init_hierarchy_meta_q_functions_for_hierarchy_state(domain_id, task_id, automaton_name, state_name, context)

    @abstractmethod
    def _init_hierarchy_meta_q_functions_for_hierarchy_state(self, domain_id, task_id, automaton_name, state_name, context):
        """
        Initializes a Q-function for a given (state, context) pair for a given automaton (in the tabular case) or a
        Q-function for the passed automaton (in the DQN case).
        """
        pass

    @abstractmethod
    def _init_meta_q_functions_for_automaton(self, domain_id, task_id, automaton_name, force_init=False):
        """
        Initializes the container for the Q-functions of a given automaton (for the tabular case), or the the neural
        network associated to the automaton (for the DQN case).
        """
        pass

    @abstractmethod
    def _update_meta_q_functions(self, domain_id, task_id, next_state, is_terminal, option, hierarchy, observation, satisfied_automata):
        """
        Applies SMDP Q-Learning update to the function for the given (domain, task) pair.
        """
        pass

    '''
    Learning Loop
    '''
    def _run_episode(self, domain_id, task_id):
        # Overrides the base method to set the option to None beforehand. The option will not be automatically set to
        # None if the episode is interrupted because an automaton is learnt.
        self._init_option_tracking_struct_domain_task(domain_id, task_id)
        return super()._run_episode(domain_id, task_id)

    def _is_option_ineligible(self, domain_id, hierarchy, hierarchy_state):
        """
        Returns True if an exploratory option must be chosen. That is, if:
          - An exploratory option is already being used (still active).
          - The current automaton state is a deadend state (this could be an accepting state, a rejecting state, or the
            initial state of an automaton that has still not been refined).
          - There is no satisfiable outgoing condition from the state (this subsumes the previous one but I keep it by
            now just in case...).
          - The automaton is not satisfiable (there is not a feasible path from the initial state to the accepting
            state). Note that the previous case (the deadend state case) is covered by this one EXCEPT if the automaton
            we are using is not the target one (therefore, we might have not finished exactly in the accepting state and
            we are looping in it).
        """
        automaton = hierarchy.get_automaton(hierarchy_state.automaton_name)

        return self._is_exploratory_option_active() or \
               len(automaton.get_outgoing_conditions_with_terminating_paths(
                   hierarchy_state.state_name,
                   hierarchy_state.context,
                   self._get_policy_bank(0).get_observations(),
                   hierarchy, False
               )) == 0

    def _choose_action(self, domain_id, task_id, state, hierarchy, hierarchy_state):
        # Choose an exploratory option if one is already being used or if the current root automaton is unsatisfiable
        # (i.e., no option can be chosen). Else, just choose an option according to the current hierarchy state.
        use_exploratory_option = self._is_option_ineligible(domain_id, hierarchy, hierarchy_state)
        if use_exploratory_option:
            option = self._choose_exploratory_option(domain_id, task_id, state, hierarchy_state.automaton_name, hierarchy_state.state_name)
        else:
            option = self._choose_option(domain_id, task_id, state, hierarchy, hierarchy_state, False)

        # If the option is an action, return it. This can happen in case the exploratory option is a primitive action.
        if not isinstance(option, Option):
            return option

        # Set the option being followed. Note that we place this below the 'if' above: we only set this for
        # non-primitive action options (which are only taken for smart exploration).
        self._selected_option[domain_id][task_id] = option

        # The bottom option has to be a satisfiable formula option, whose associated formula is either TRUE (it will
        # rarely be the case) or a DNF formula with a single disjunct.
        option_condition = option.get_condition()
        assert not option_condition.is_call() and option_condition.get_formula().is_satisfiable() and \
               (option_condition.get_formula().is_true() or len(option_condition.get_formula()) == 1)

        # Determine the exploration factor value (if we use option exploration, we will use the greedy policy associated
        # to the formula). DO NOT be confused with the variable 'use_exploratory_option': these options used for
        # 'smart' exploration must be used greedily.
        if use_exploratory_option:
            exploration_factor = 0.0
        else:
            exploration_factor = self._get_formula_exploration_rate(task_id, option_condition)

        # A primitive action is selected according to the policy of the formula option
        action = self._choose_egreedy_action(
            self._get_task(domain_id, task_id),
            state,
            self._get_q_function(task_id, option_condition),
            exploration_factor
        )
        assert not isinstance(action, Option)
        return action

    def _is_exploratory_option_active(self):
        return self._option_exploration_base_condition is not None

    def _get_formula_exploration_rate(self, task_id, formula_condition: FormulaCondition):
        return self._get_annealed_exploration_rate(
            self._get_formula_q_function_step_count(task_id, formula_condition),
            self._exploration_rate_subgoal_init,
            self._exploration_rate_subgoal_end,
            self._exploration_rate_subgoal_steps
        )

    '''
    Option Management
    '''
    def _choose_option(self, domain_id, task_id, state, hierarchy, hierarchy_state, is_exploratory_option):
        """
        Selects a hierarchy of options (a sequence of nested options of increasingly lower level) in an environment
        state given the place of the agent in the hierarchy (the hierarchy state).
        """
        automaton = hierarchy.get_automaton(hierarchy_state.automaton_name)
        automaton_state = hierarchy_state.state_name
        context = TRUE  # Accumulated DNF Context formula given by the calls

        # While the chosen option is not a formula...
        option = self._selected_option[domain_id][task_id]
        while option is None or option.get_condition().is_call():
            if is_exploratory_option:
                # If the base option was chosen for exploration, then select greedily and ignore edges to rejecting
                # states (we are interested in reaching tasks' goals, so we ignore these edges just in case the
                # metacontroller policies do not generalize to the entire state space).
                condition, tgt_automaton_state = self._get_greedy_condition(task_id, state, automaton, automaton_state,
                                                                            context, hierarchy, True)
            else:
                # Choose a condition to pursue using the (epsilon-)greedy policy and create the option.
                condition, tgt_automaton_state = self._choose_egreedy_condition(task_id, state, automaton, automaton_state,
                                                                                context, hierarchy)
            option = Option(condition, context, state, automaton.get_name(), automaton_state, tgt_automaton_state, option)

            # If the condition is a call, then we have to keep on selecting options (until we select a formula).
            if condition.is_call():
                automaton = hierarchy.get_automaton(condition.get_called_automaton())
                automaton_state = automaton.get_initial_state()
                context = context.logic_and(condition.get_context())

            # The context is local, so it is going to be either a DNF with a single disjunct or TRUE. This is because
            # the agent is selecting between conditions whose associated DNFs have a single disjunct or are TRUE.
            assert context.is_true() or len(context) == 1
        return option

    def _choose_egreedy_condition(self, task_id, state, automaton, automaton_state, context, hierarchy) -> Tuple[EdgeCondition, str]:
        """
        Returns a tuple containing (i) an edge condition to pursue (a call or a formula available from the passed
        automaton state under the passed context) and (ii) the automaton state to which the agent will transition if
        such condition is eventually satisfied. This tuple is chosen following an epsilon-greedy policy.
        """
        if self.training_enable:
            prob = np.random.uniform(0, 1)
            if prob <= self._get_automaton_exploration_rate(task_id, automaton.get_name(), automaton_state, context):
                return self._get_random_condition(task_id, automaton, automaton_state, context, hierarchy)
        return self._get_greedy_condition(task_id, state, automaton, automaton_state, context, hierarchy, False)

    def _choose_exploratory_option(self, domain_id, task_id, start_state, automaton_name, automaton_state):
        """
        Chooses a random option from the set of automata options, formula options and primitive actions. This is only
        used in case that the current automaton state is a dead-end (i.e., it has no outgoing edges, like when a new
        automaton for a task is to be learnt).

        If the selected option is not None, it means that it is under execution, so we keep running it. If there is not
        an option selected and the base condition we originally chose is an automaton (i.e., a call condition), we need
        to check whether we have reached the accepting state there. If so, then the attributes of the condition being
        pursued are set to None; else, we choose a new option starting from where we left. If none of the previous cases
        applies, we choose a new exploratory option.
        """
        if self._selected_option[domain_id][task_id] is not None:
            return self._selected_option[domain_id][task_id]
        return self._choose_new_exploratory_option(domain_id, task_id, start_state, automaton_name, automaton_state)

    def _choose_new_exploratory_option(self, domain_id, task_id, start_state, automaton_name, automaton_state):
        exploratory_options = []

        # Primitive actions. We only include them if they are required or if the current hierarchy level is the
        # minimum one, which means that policies might have not been learned yet so we still need something to explore.
        # TODO: This strategy will have to change if we use continuous actions (maybe we can just take a primitive
        #  action with some probability).
        if self._option_exploration_include_actions or self._curriculum.is_current_level_min():
            task = self._get_task(domain_id, task_id)
            for action in range(task.action_space.n):
                exploratory_options.append(action)

        # Formula options. Note that options are created only for subgoals in domains we have already mastered. Using
        # subgoals from domains being currently learned can be dangerous since their policies are more likely to not be
        # learned yet and, thus, the greedy policy will probably get stuck somewhere and not allowing to observe
        # counterexamples. The formulas that are satisfied by the current observation are not included (they are useless
        # if we are already satisfying them, we won't explore at all!).
        if self._option_exploration_include_formulas:
            exploratory_options.extend(filter(
                lambda x: not x.is_satisfied(self._get_task(domain_id, task_id).get_observation()),
                self._get_mastered_domain_subgoals(task_id)
            ))

        # Automata options. Note that these options are only created for automata in lower hierarchical levels (the
        # automata in the current level might not be learned yet (in either terms of structure or policies).
        # Do not add as possible exploratory options, those automata that reach a terminal state by processing
        # the current observation.
        automaton_name_to_hierarchy = {}
        if self._option_exploration_include_automata:
            for mastered_domain_id in self._curriculum.get_mastered_domains():
                mastered_hierarchy = self._get_hierarchy(mastered_domain_id)
                init_hierarchy_state = self._get_initial_hierarchy_state_successor(
                    mastered_hierarchy, self._get_task(domain_id, task_id).get_observation()
                )
                if not mastered_hierarchy.is_terminal_state(init_hierarchy_state):
                    automaton_name_to_hierarchy[mastered_hierarchy.get_root_automaton_name()] = mastered_hierarchy
                    exploratory_options.append(CallCondition(mastered_hierarchy.get_root_automaton_name(), TRUE))

        # Choose a condition uniformly at random
        if len(exploratory_options) == 0:
            raise RuntimeError("Error: The set of exploratory options cannot be empty.")

        self._option_exploration_base_condition = random.choice(exploratory_options)
        if isinstance(self._option_exploration_base_condition, EdgeCondition):
            if self._option_exploration_base_condition.is_call():
                self._option_exploration_hierarchy = automaton_name_to_hierarchy[self._option_exploration_base_condition.get_called_automaton()]
                self._option_exploration_hierarchy_state = self._get_initial_hierarchy_state_successor(
                    self._option_exploration_hierarchy,
                    self._get_task(domain_id, task_id).get_observation()
                )
                return self._choose_option(domain_id, task_id, start_state, self._option_exploration_hierarchy,
                                           self._option_exploration_hierarchy_state, True)
            return Option(self._option_exploration_base_condition, TRUE, start_state, automaton_name, automaton_state,
                          automaton_state, None, 0)
        return self._option_exploration_base_condition

    def _get_automaton_exploration_rate(self, task_id, automaton, automaton_state, context):
        return self._get_annealed_exploration_rate(
            self._get_automaton_state_step_count(task_id, automaton, automaton_state, context),
            self._exploration_rate_automaton_init,
            self._exploration_rate_automaton_end,
            self._exploration_rate_automaton_steps
        )

    @abstractmethod
    def _get_automaton_state_step_count(self, task_id, automaton_name, automaton_state, context):
        pass

    @abstractmethod
    def _inc_automaton_state_step_count(self, task_id, automaton_name, automaton_state, context):
        pass

    @abstractmethod
    def _reset_automaton_step_counters(self, automaton_name):
        pass

    def _get_random_condition(self, task_id, automaton, automaton_state, context, hierarchy) -> Tuple[EdgeCondition, str]:
        """
        Returns a tuple containing (i) an edge condition to pursue (a call or a formula available from the passed
        automaton state) and (ii) the automaton state to which the agent will transition if such condition is eventually
        satisfied. This tuple is chosen uniformly at random.

        Important: the returned condition is contextless (if we are interested, we may append a context later on).
        """
        return random.choice(automaton.get_outgoing_conditions_with_terminating_paths(
            automaton_state, context, self._get_policy_bank(task_id).get_observations(), hierarchy, False
        ))

    @abstractmethod
    def _get_greedy_condition(self, task_id, state, automaton, automaton_state, context, hierarchy, ignore_rej_edges) \
            -> Tuple[EdgeCondition, str]:
        """
        Returns a tuple containing (i) an edge condition to pursue (a call or a formula available from the passed
        automaton state) and (ii) the automaton state to which the agent will transition if such condition is eventually
        satisfied. This tuple is chosen using a greedy policy.

        Important: the returned condition is contextless, the passed 'context' parameter is only used to select the
        contextless option. If we are interested, we can append the context later on. Note that we can do this since the
        context is going to be either TRUE or a formula with one disjunct; therefore, the number of outgoing conditions
        would be the same if we were considering the context or not.
        """
        pass

    def _on_performed_step(self, domain_id, task_id, next_state, reward, is_terminal, observation, observation_changed,
                           hierarchy, hierarchy_state, next_hierarchy_state, episode_length):
        # Take current option being run
        current_option = self._selected_option[domain_id][task_id]

        # If the automaton state is a dead-end, we might have chosen an option to explore (useful for finding
        # counterexamples if we are learning the automata)
        if self._is_exploratory_option_active():
            self._on_performed_step_exploratory_option(current_option, domain_id, task_id, next_state, is_terminal,
                                                       observation, observation_changed, episode_length)
        else:
            self._on_performed_step_standard_option(current_option, domain_id, task_id, next_state, is_terminal,
                                                    observation, hierarchy, hierarchy_state, next_hierarchy_state,
                                                    episode_length)

    def _on_performed_step_exploratory_option(self, current_option, domain_id, task_id, next_state, is_terminal,
                                              observation, observation_changed, episode_length):
        if isinstance(self._option_exploration_base_condition, EdgeCondition):
            if self._option_exploration_base_condition.is_call():
                # If the option is a call to an automaton, we check what would the next hierarchy state be and unroll
                # the option accordingly (check which options in the option hierarchy terminate). Then, we get the
                # actual option according to the stack. This is very similar to the process in which a "standard" option
                # is used but without involving updates of Q-functions.
                next_hierarchy_state = self._get_next_hierarchy_state(self._option_exploration_hierarchy,
                                                                      self._option_exploration_hierarchy_state,
                                                                      observation, observation_changed)
                option, terminated_options = self._unroll_option(current_option, self._option_exploration_hierarchy_state,
                                                                 next_hierarchy_state, is_terminal, episode_length)
                if not is_terminal and episode_length < self.max_episode_length:
                    option = self._get_option_from_stack(option, next_hierarchy_state.stack, terminated_options)
                self._option_exploration_hierarchy_state = next_hierarchy_state

                if self._option_exploration_hierarchy.is_terminal_state(self._option_exploration_hierarchy_state):
                    option = None
                    self._on_exploratory_option_finished()
                else:
                    self._selected_option[domain_id][task_id] = option  # Need to do this for the following call
                    option = self._choose_option(domain_id, task_id, next_state, self._option_exploration_hierarchy,
                                                 self._option_exploration_hierarchy_state, True)
            else:
                # If the option is a formula, then it ends if the formula is satisfied by the observation
                if self._option_exploration_base_condition.get_formula().is_satisfied(observation):
                    option = None
                    self._on_exploratory_option_finished()
                else:
                    option = current_option
            self._selected_option[domain_id][task_id] = option
        else:
            self._on_exploratory_option_finished()

    def _on_exploratory_option_finished(self):
        self._option_exploration_base_condition = None
        self._option_exploration_hierarchy = None
        self._option_exploration_hierarchy_state = None

    def _on_performed_step_standard_option(self, current_option, domain_id, task_id, next_state, is_terminal,
                                           observation, hierarchy, hierarchy_state, next_hierarchy_state,  episode_length):
        # Update the current sequence of selected options
        current_option.update()

        # Increase the number of performed steps for the given formula condition. We don't want the steps when
        # evaluating the greedy policy to affect the exploration, that's why we check whether the model is being trained
        # or not.
        if self.training_enable:
            self._inc_formula_q_function_step_count(task_id, current_option.get_condition())

        # Unroll the option: get the "bottom" active option (connected to its parents) and the options that have been
        # terminated
        current_option, terminated_options = self._unroll_option(
            current_option, hierarchy_state, next_hierarchy_state, is_terminal, episode_length
        )

        # Update the meta q-functions for the terminated options
        if self.training_enable:
            for terminated_option in terminated_options:
                self._update_meta_q_functions(
                    domain_id, task_id, next_state, is_terminal, terminated_option, hierarchy, observation,
                    next_hierarchy_state.satisfied_calls
                )
                self._inc_automaton_state_step_count(
                    task_id, terminated_option.get_automaton_name(), terminated_option.get_automaton_state(),
                    terminated_option.get_context()
                )

        # Add options in the stack that do not appear in our option 'sequence' (the chain of options that we currently
        # keep)
        if not is_terminal and episode_length < self.max_episode_length:
            current_option = self._get_option_from_stack(current_option, next_hierarchy_state.stack, terminated_options)

        self._selected_option[domain_id][task_id] = current_option

    def _unroll_option(self, option: Option, hierarchy_state, next_hierarchy_state, is_terminal, episode_length) -> Tuple[Option, List[Option]]:
        """
        Returns the option hierarchy that results from terminating options in it according to the state of the
        environment or the state in the hierarchy. It also returns a list of the options that have been terminated.
        """
        # All selected options terminate if the MDP state is terminal or the maximum episode length has been surpassed
        if is_terminal or episode_length >= self.max_episode_length:
            return self._unroll_all_options(option)

        current_option = option
        terminated_options = []
        unroll_completed = False

        # The current option becomes None when the option at the first hierarchy level has been unrolled. Note that here
        # we just check local termination (global termination is checked at the beginning of this method). The unrolling
        # process terminates when the current option being checked does not terminate (if that option does not terminate,
        # the higher level ones do no terminate either).
        while not unroll_completed and current_option is not None:
            option_condition = current_option.get_condition()
            if option_condition.is_call():
                # A call option terminates when it does not appear in the stack.
                in_stack, _ = self._is_option_in_stack(current_option, next_hierarchy_state.stack)
                if not in_stack:
                    terminated_options.append(current_option)
                    current_option = current_option.get_parent()
                else:
                    unroll_completed = True
            else:
                # A formula option terminates whenever the state changes
                if hierarchy_state != next_hierarchy_state:
                    terminated_options.append(current_option)
                    current_option = current_option.get_parent()
                else:
                    unroll_completed = True

        return current_option, terminated_options

    def _unroll_all_options(self, option) -> Tuple[Optional[Option], List[Option]]:
        """
        Terminates all options in the option hierarchy passed as a parameter and returns the resulting option hierarchy
        (None) and the list of terminated options.
        """
        current_option = option
        terminated_options = []
        while current_option is not None:
            terminated_options.append(current_option)
            current_option = current_option.get_parent()
        return current_option, terminated_options

    def _is_option_in_stack(self, option: Option, stack: List[CallStackItem]):
        """
        Checks if an option appears in the stack and returns the boolean result (True if appears, False else) and the
        index within the stack (None if it does not appear there). An option appears in the stack if there is an item
        such that (i) shares the same automaton state in which the option started, (ii) has the same automaton name in
        which the option is running, (iii) the called automaton in the stack is the same than the one the option has
        selected AND under the "same" context (the context under which the automaton is called in the stack contains
        the context selected by the option), and (iv) its context contains the context of the option (i.e., the option
        context's DNF formula is part of the stack item's context).
        """
        option_condition = option.get_condition()
        for i in range(len(stack)):
            stack_item = stack[i]
            stack_condition = stack_item.call_condition
            if stack_item.from_state_name == option.get_automaton_state() and \
               stack_item.automaton_name == option.get_automaton_name() and \
               stack_condition.get_called_automaton() == option_condition.get_called_automaton() and \
               stack_condition.get_context().contains(option_condition.get_context()) and \
               stack_item.context.contains(option.get_context()):
                return True, i
        return False, None

    def _get_option_from_stack(self, option: Option, new_stack: List[CallStackItem], terminated_options: List[Option]) -> Option:
        """
        Returns a new option hierarchy derived from the stack passed as a parameter.
        """
        new_option = option

        # Build new option sequence/hierarchy from the stack of the update hierarchy state
        if new_option is None:
            # The new option is None if all options in our option chain have been terminated. In this case we add one
            # option for each item in the stack (see that the starting index is 0). It is possible that the stack is
            # empty (this happens if the option chain consisted of a formula option ony).
            new_option = self._get_option_from_stack_helper(new_option, new_stack, 0, terminated_options)
        else:
            # Find the option in the stack and also the position where it appears. The assertion below verifies that:
            #  1. if the current option after unrolling is a call and has not been terminated, it must be in the stack
            #     and something must have terminated
            #  2. if the current option is a formula, it cannot be in the stack (formulas are never there) and nothing
            #     has terminated
            # The new options are those in the stack appearing after (or in stack terms, above) the stack item
            # corresponding to the current option (see stack_index + 1).
            in_stack, stack_index = self._is_option_in_stack(new_option, new_stack)
            assert (new_option.get_condition().is_call() and in_stack and len(terminated_options) > 0) or \
                   (not new_option.get_condition().is_call() and not in_stack and len(terminated_options) == 0)
            if in_stack:
                new_option = self._get_option_from_stack_helper(new_option, new_stack, stack_index + 1, terminated_options)
        return new_option

    def _get_option_from_stack_helper(self, option, new_stack, starting_stack_index, terminated_options):
        """
        The new option is built by appending options to the current one. Note that we go from shallower levels in the
        hierarchy to deeper levels (i.e. from deeper levels in the stack to shallower). All the options have the same
        starting MDP state and number of steps as the option we terminated the last (the shallowest terminated option in
        our previous option chain).
        """
        new_option = option
        assert len(terminated_options) > 0
        last_terminated_option = terminated_options[-1]  # The option last added to the array was the shallowest
        context = last_terminated_option.get_context()   # The starting context is that of the last terminated option
        for i in range(starting_stack_index, len(new_stack)):
            stack_item = new_stack[i]

            call_condition = stack_item.call_condition
            if len(call_condition.get_context()) > 1:
                # The call conditions in the stack might have contexts given by DNF formulas with more than one
                # disjunct. This means that both disjuncts were satisfied by an observation the agent saw. Therefore, we
                # could choose any of those disjuncts to represent our 'simulated' option. The simplest strategy is to
                # choose one uniformly at random, which is what we do here.
                call_condition = random.choice([c for c in call_condition.decompose()])

            new_option = Option(call_condition, context, last_terminated_option.get_start_state(),
                                stack_item.automaton_name, stack_item.from_state_name, stack_item.to_state_name,
                                new_option, last_terminated_option.get_num_steps())
            if call_condition.is_call():
                # Augment the context according to the call condition we have taken. Note that we use the call condition
                # we have derived before from the original call condition in the stack (i.e., we use call_condition
                # instead of stack_item.call_condition). This has to be done in order to keep the consistency across
                # options and because the contexts in the options have to be either TRUE or a DNF with a single
                # disjunct.
                context = context.logic_and(call_condition.get_context())

        return new_option

    def _init_option_tracking_structs(self):
        self._selected_option.clear()
        for domain_id in range(self.num_domains):
            self._selected_option[domain_id] = {}
            for task_id in range(self.num_tasks):
                self._init_option_tracking_struct_domain_task(domain_id, task_id)

    def _init_option_tracking_struct_domain_task(self, domain_id, task_id):
        self._selected_option[domain_id][task_id] = None

    def _get_option_pseudoreward(self, option: Option, hierarchy, observation, satisfied_automata):
        """
        The final reward for each option is the discounted sum of rewards such that reward penalty is given for each
        step, a positive reward is given if the accept state is reached in the automaton where the option is selected
        and another penalty if a rejecting state is reached (checking the is_terminal would be wrong, it has to be
        something local to our automaton)..
        """
        automaton = hierarchy.get_automaton(option.get_automaton_name())
        next_automaton_state = self._get_next_automaton_state(option, observation, satisfied_automata)

        acc_reward = np.sum(self._meta_pseudoreward_after_step * (self._meta_discount_rate ** np.arange(option.get_num_steps())))
        if automaton.is_accept_state(next_automaton_state):
            final_reward = self._meta_pseudoreward_condition_satisfied
        elif automaton.is_reject_state(next_automaton_state):
            final_reward = self._meta_pseudoreward_deadend_state
        else:
            final_reward = self._meta_pseudoreward_after_step
        return acc_reward + (self._meta_discount_rate ** option.get_num_steps()) * final_reward

    def _get_next_automaton_state(self, option: Option, observation: Set[str], satisfied_calls: List[SatisfiedCall]):
        """
        Returns the next automaton state after a given option has terminated according to the passed observation and
        list of satisfied calls (i.e. automata that have just been successfully executed in the last step). The next
        automaton state is going to be the target state indicated in the option if the latter run successfully. Else,
        the state is going to be the same one at which the option started.
        """
        if self._is_option_successful(option, observation, satisfied_calls):
            return option.get_target_automaton_state()
        return option.get_automaton_state()

    def _is_option_successful(self, option: Option, observation: Set[str], satisfied_calls: List[SatisfiedCall]):
        """
        Returns True if an option achieved its purpose: (i) if its condition is a call, the purpose was to complete the
        execution of an automaton (thus, the passed list of satisfied calls is checked), or (2) if its condition is a
        formula, the purpose is to satisfy that formula (thus, we check whether the passed observation satisfies it).
        """
        condition = option.get_condition()
        return (condition.is_call() and self._is_call_condition_sat(option.get_automaton_name(), condition, satisfied_calls)) or \
               (not condition.is_call() and condition.is_satisfied(observation))

    def _is_call_condition_sat(self, automaton_name, call_condition: CallCondition, satisfied_calls: List[SatisfiedCall]):
        """
        Returns True if the passed CallCondition appears in the list of satisfied calls. This occurs if at least one of
        the disjuncts in the context of the CallCondition appears in that of the condition of the satisfied call. Note
        that while the passed call condition will be the full DNF associated to a given edge, the satisfied call might
        only mention that DNF partially (for example, one of the disjuncts). For instance, the call condition could be
        i|s (m1), and the satisfied call could be i (m1) or s (m1).
        """
        for sat_call in satisfied_calls:
            if sat_call.automaton_name == automaton_name:
                # If the automaton making the call corresponds to this automaton, we check whether the call condition in
                # the satisfied call is associated to the same automaton than that of the passed call condition AND the
                # context associated to the satisfied call contains the context of the passed condition. The passed
                # condition should be TRUE or have a single disjunct, while the one in the call may have multiple disjuncts
                # (all those that were satisfied by a given observation).
                # Note we can return the result here since there cannot be two calls two the same automaton that have
                # been simultaneously satisfied (these hierarchies do not have cyclic dependencies nor recursion).
                return sat_call.call_condition.get_called_automaton() == call_condition.get_called_automaton() and \
                       sat_call.call_condition.get_context().contains(call_condition.get_context())
        return False

    def _get_next_context(self, automaton_state, context: DNFFormula, next_automaton_state) -> DNFFormula:
        """
        Returns the next context after moving from one automaton state to another (not necessarily different). If the
        states are the same, the context does not change. Else, the context becomes TRUE (the one that came from higher
        level automata does not apply anymore, it only applies to the first outgoing transition once the called automaton
        starts running).
        """
        return context if automaton_state == next_automaton_state else TRUE

    '''
    Automaton Learning Management (what to do when an automaton is learned for a given domain)
    '''
    def _on_automaton_learned(self, domain_id):
        # Update the background knowledge if needed
        super()._on_automaton_learned(domain_id)

        # The option we used for exploration has finished (if any)
        self._on_exploratory_option_finished()

        # Some of the q-functions might need to be reset since they are adapted to the automaton at hand
        self._reset_q_functions(domain_id)

        # Reset the step counters for all automaton states in the learned automaton
        self._reset_automaton_step_counters(self._get_hierarchy(domain_id).get_root_automaton().get_name())

        # The subgoal cache is updated to capture the subgoals in the newly learned automaton
        self._update_all_subgoals()

    def _on_domains_learned(self, domain_ids):
        super()._on_domains_learned(domain_ids)

        # Update the subgoal cache. The subgoals of the next level in the hierarchy are added to the list. This is
        # specially important if the automata in the next level are handcrafted; else, if the automata are learned, the
        # subgoal list will be updated once an automaton is learned.
        self._update_all_subgoals()

    def _reset_q_functions(self, domain_id):
        """
        Rebuild Q-functions when an automaton is learned.
        """
        self._init_hierarchy_formula_q_functions(domain_id)
        self._init_hierarchy_meta_q_functions(domain_id)

    def _update_all_subgoals(self):
        """
        Sets a sorted (in order to have reproducible results) list of all formulas that occur across active domains
        (i.e., domains which have some probability to be chosen in the curriculum). These are used to know which are the
        current relevant formulas whose policies have to be updated (should be less than those in the policy bank).

        NOTE: In the past there was a separate method that iterated through the automata in each hierarchy and called
        the method "get_hierarchy_subgoals" with context TRUE. This was done to get the subgoals without a context. We
        want to keep these unaltered formulas in case a given automaton is called without a context. However, note that
        we don't need to do that if all the automata we are interested in are the root of each domain hierarchy, so we
        will be using context TRUE initially for each of them!

        NOTE 2: In previous versions we didn't check only the formulas of the active domains, but also of other domains.
        In the case where automata are handcrafted, this caused the important formulas at the first level to be sampled
        with low probability (this is if we update a subset of formulas at each step) even if they were the important
        ones to learn at that time.
        """
        subgoals = set()
        for domain_id in self._curriculum.get_active_domains():
            self._get_hierarchy(domain_id).get_subgoals(
                subgoals, ignore_rejecting_transitions=False, only_satisfiable=True)

        # TODO: Having the same set across banks is a bit inefficient for tabular (we perform the same operations for as
        #  many grids as we have).
        for policy_bank in self._get_policy_banks():
            policy_bank.set_active_subgoals(subgoals)

    def _get_mastered_domain_subgoals(self, task_id):
        """
        Returns a sorted list of the subgoals appearing in domains that have already been mastered (i.e., whose
        hierarchy level is lower than the current one). The subgoals will only be the ones appearing on edges to
        non-rejecting states. The subgoals we will use correspond to satisfiable formulas in our policy bank.
        """
        subgoals = set()
        for domain_id in self._curriculum.get_mastered_domains():
            self._get_hierarchy(domain_id).get_subgoals(
                subgoals, ignore_rejecting_transitions=True, only_satisfiable=True
            )

        # Get filtered subgoals: take roots of the subgoals and discard unsatisfiable ones
        root_sat_subgoals = set()
        self._get_policy_bank(task_id).get_root_sat_subgoals(subgoals, root_sat_subgoals)
        return sorted(root_sat_subgoals)

    '''
    Model Management
    '''
    def _export_models(self):
        # Export the hierarchies
        super()._export_models()

        # Export metacontroller policies
        file_utils.mkdir(self._get_metacontrollers_model_folder())
        for automaton in self._get_all_used_automata():
            self._export_meta_functions(automaton)

        # Export lowest level policies
        file_utils.mkdir(self._get_formula_bank_model_folder())
        self._export_policy_banks()

    def _get_all_used_automata(self):
        used_automata = set()
        for domain_id in range(self.num_domains):
            hierarchy = self._get_hierarchy(domain_id)
            root_automaton = hierarchy.get_root_automaton()
            used_automata.add(root_automaton)
            for dependency in self._dependency_graph.get_dependencies(root_automaton.get_name()):
                used_automata.add(hierarchy.get_automaton(dependency))
        return used_automata

    @abstractmethod
    def _export_policy_banks(self):
        pass

    @abstractmethod
    def _export_meta_functions(self, automaton):
        pass

    @abstractmethod
    def _import_policy_banks(self):
        pass
