from abc import abstractmethod
from gym_hierarchical_subgoal_automata.automata.hierarchical_automaton import HierarchicalAutomaton
from gym_hierarchical_subgoal_automata.automata.hierarchy import Hierarchy
from ilasp.generator.ilasp_task_generator import generate_ilasp_task
from ilasp.parser import ilasp_solution_parser
from ilasp.solver.ilasp_solver import solve_ilasp_task
import numpy as np
import os
from reinforcement_learning.learning_algorithm import LearningAlgorithm, InterruptLearningException
from typing import Dict, List, Set, Tuple
from utils import file_utils
from utils.container_utils import get_param
from utils.dependency_graph import DependencyGraph


class CoveredExampleError(Exception):
    def __init__(self, example):
        super().__init__("Error: An example that an automaton is currently covering cannot be uncovered afterwards."
                         f"Example: {example}.")


class IHSAAlgorithmBase(LearningAlgorithm):
    """
    Generic class for the algorithms performing interleaving between RL and automata learning.
    """
    INITIAL_STATE_NAME = "u0"
    ACCEPTING_STATE_NAME = "u_acc"
    REJECTING_STATE_NAME = "u_rej"

    TRAINING_MODE = "training_mode"            # which training mode is enabled (see below)
    TRAINING_MODE_LEARN = "learn"              # train policies and learn automata in an interleaved way
    TRAINING_MODE_HANDCRAFTED = "handcrafted"  # train policies using a handcrafted hierarchy (loaded from the tasks themselves)

    ILASP_FLAGS = "ilasp_flags"
    LEARNING_TIME_BUDGET = "learning_time_budget"                      # budget time that ILASP has for finding the automaton
    STARTING_NUM_STATES_FIELD = "starting_num_states"                  # number of states that the starting automaton has
    FILTER_RESTRICTED_OBSERVABLES = "filter_restricted_observables"    # whether to remove observables not in the restricted set from the observations
    HYP_RESTRICT_OBSERVABLES = "hyp_restrict_observables"              # only the restricted set of observables can be used in the hypothesis space of the learning task
    HYP_RESTRICT_DEPENDENCIES = "hyp_restrict_dependencies"            # restrict the hypothesis space of each automaton learning task by only providing the required dependencies provided in the provided dependencies
    MAX_DISJUNCTION_SIZE = "max_disjunction_size"                      # maximum number of edges from one state to another
    LEARN_ACYCLIC_GRAPH = "learn_acyclic_graph"                        # whether the target automata has cycles or not
    SYMMETRY_BREAKING_ENABLE = "symmetry_breaking_enable"              # whether the symmetry breaking method is enable in automaton learning
    AVOID_LEARNING_ONLY_NEGATIVE = "avoid_learning_only_negative"      # whether to avoid learning labels made only of negative literals (e.g., ~n)
    REMOVE_DEADEND_STATES = "remove_deadend_states"                    # whether to remove deadend states from the learned automata
    USE_MUTEX_SATURATION = "use_mutex_saturation"                      # whether to use the constraints using saturation for enforcing mutual exclusivity
    ROOT_REJECTING_STATE_DEEPENING = "root_rejecting_state_deepening"  # whether to perform iterative deepening on the root's rejecting state (i.e., try learning an automaton without it and, if the task becomes UNSAT, add it)

    MIN_GOAL_EXAMPLES = "min_goal_examples"                            # minimum number of goal examples that must have been observed to learn an automaton
    USE_TOP_SHORTEST_GOAL_EXAMPLES = "use_top_shortest_goal_examples"  # whether to use the top shortest goal examples to learn the first automaton
    NUM_TOP_SHORTEST_GOAL_EXAMPLES = "num_top_shortest_goal_examples"  # how many examples to use from the top shortest ones

    AUTOMATON_TASK_FOLDER = "automaton_tasks"  # folder where the automaton learning tasks are saved
    AUTOMATON_TASK_FILENAME = "task-%d.las"    # filename pattern of the automaton learning tasks

    AUTOMATON_SOLUTION_FOLDER = "automaton_solutions"  # folder where the solutions to automaton learning tasks are saved
    AUTOMATON_SOLUTION_FILENAME = "solution-%d.txt"    # filename pattern of the solutions to automaton learning tasks

    AUTOMATON_PLOT_FOLDER = "automaton_plots"          # folder where the graphical solutions to automaton learning tasks are saved
    AUTOMATON_PLOT_FILENAME = "plot-%d.png"            # filename pattern of the graphical solutions to automaton learning tasks

    MODELS_HIERARCHIES_FOLDER = "hierarchies"  # name of the folder inside the custom models folder where hierarchies are exported

    def __init__(self, params):
        super().__init__(params)

        self.training_mode = get_param(params, IHSAAlgorithmBase.TRAINING_MODE, IHSAAlgorithmBase.TRAINING_MODE_HANDCRAFTED)

        # Interleaved automaton learning params
        self.interleaved_automaton_learning = self.training_enable and self.training_mode == IHSAAlgorithmBase.TRAINING_MODE_LEARN
        self.ilasp_flags = get_param(params, IHSAAlgorithmBase.ILASP_FLAGS, {})
        self.domain_running_time = np.zeros(self.num_domains, dtype=np.float32)
        self.learning_time_remaining = get_param(params, IHSAAlgorithmBase.LEARNING_TIME_BUDGET, 120)
        self.num_starting_states = get_param(params, IHSAAlgorithmBase.STARTING_NUM_STATES_FIELD, 3)
        self.num_automaton_states = self.num_starting_states * np.ones(self.num_domains, dtype=np.int)
        self.max_disjunction_size = get_param(params, IHSAAlgorithmBase.MAX_DISJUNCTION_SIZE, 1)
        self.learn_acyclic_graph = get_param(params, IHSAAlgorithmBase.LEARN_ACYCLIC_GRAPH, True)
        self.symmetry_breaking_enable = get_param(params, IHSAAlgorithmBase.SYMMETRY_BREAKING_ENABLE, True)
        self.avoid_learning_only_negative = get_param(params, IHSAAlgorithmBase.AVOID_LEARNING_ONLY_NEGATIVE, True)
        self._remove_deadend_states = get_param(params, IHSAAlgorithmBase.REMOVE_DEADEND_STATES, True)
        self._use_mutex_saturation = get_param(params, IHSAAlgorithmBase.USE_MUTEX_SATURATION, False)
        self._root_rejecting_state_deepening_enabled = get_param(params, IHSAAlgorithmBase.ROOT_REJECTING_STATE_DEEPENING, False)
        self._use_root_rejecting_state = np.zeros(self.num_domains, dtype=np.bool)
        self._domains_to_relearn: Set[int] = set([])  # Domains that have to be relearned because some dependency has been relearned

        min_goal_examples = get_param(params, IHSAAlgorithmBase.MIN_GOAL_EXAMPLES, 1)
        if isinstance(min_goal_examples, list):
            # The read min_goal_examples contains the minimum number of goal examples for each hierarchical level
            self._min_goal_examples = np.array([
                min_goal_examples[self._curriculum.get_domain_hierarchy_level(domain_id) - 1]
                for domain_id in range(self.num_domains)
            ])
        else:
            # All domains have the same number of goal examples
            self._min_goal_examples = np.ones(self.num_domains, dtype=np.int32) * min_goal_examples
        assert self._min_goal_examples.min() >= 1  # Always require at least one goal example

        use_top_shortest_goal_examples = get_param(params, IHSAAlgorithmBase.USE_TOP_SHORTEST_GOAL_EXAMPLES, False)
        if use_top_shortest_goal_examples:
            num_top_shortest_goal_examples = get_param(params, IHSAAlgorithmBase.NUM_TOP_SHORTEST_GOAL_EXAMPLES, 1)

            if isinstance(num_top_shortest_goal_examples, list):
                # The read num_top_shortest_goal_examples contains the number of starting goal examples used for each
                # hierarchical level
                self._num_top_shortest_goal_examples = np.array([
                    num_top_shortest_goal_examples[self._curriculum.get_domain_hierarchy_level(domain_id) - 1]
                    for domain_id in range(self.num_domains)
                ])
            else:
                # All domains take the same number of examples
                self._num_top_shortest_goal_examples = np.ones(self.num_domains, dtype=np.int32) * num_top_shortest_goal_examples

            assert self._num_top_shortest_goal_examples.min() >= 1
            assert all(np.less_equal(self._num_top_shortest_goal_examples, self._min_goal_examples))
        else:
            self._num_top_shortest_goal_examples = np.ones(self.num_domains, dtype=np.int32) * self._min_goal_examples.min()
        self._pre_goal_examples = [set() for _ in range(self.num_domains)]

        # Background knowledge (the curriculum determines whether automata are added here to be reused by other
        # automaton learning tasks)
        self._bk_automata: Dict[str, HierarchicalAutomaton] = {}

        self._dependency_graph = DependencyGraph()
        self._init_dependency_graph(get_param(params, IHSAAlgorithmBase.ENV_CONFIG))

        # Restrictions on the observables and the automata in the hypothesis space
        self.hypothesis_restrict_observables = get_param(params, IHSAAlgorithmBase.HYP_RESTRICT_OBSERVABLES, False)
        self.filter_restricted_observables = get_param(params, IHSAAlgorithmBase.FILTER_RESTRICTED_OBSERVABLES, False)

        self._bk_dependency_graph = DependencyGraph()
        self._hypothesis_restrict_dependencies = get_param(params, IHSAAlgorithmBase.HYP_RESTRICT_DEPENDENCIES, False)
        if self._hypothesis_restrict_dependencies:
            self._init_bk_dependency_graph(get_param(params, IHSAAlgorithmBase.ENV_CONFIG))

        # Set of automata per domain
        self._hierarchies: Dict[str, Hierarchy] = {}
        self._init_hierarchies(params)

        # Sets of examples (goal, deadend and incomplete)
        self.goal_examples = [set() for _ in range(self.num_domains)]
        self.dend_examples = [set() for _ in range(self.num_domains)]
        self.inc_examples = [set() for _ in range(self.num_domains)]

        # Keep track of the number of learnt automata per domain
        self.automaton_counters = np.zeros(self.num_domains, dtype=np.int)
        self.automaton_learning_episodes = [[] for _ in range(self.num_domains)]

        # Keep track of the first episode in which a counterexample was observed for each domain
        self._first_counterexample_episodes = np.zeros(self.num_domains, dtype=np.int)

        if self.interleaved_automaton_learning:  # If the tasks are learnt, remove previous folders if they exist
            file_utils.rm_dirs(self.get_automaton_task_folders())
            file_utils.rm_dirs(self.get_automaton_solution_folders())
            file_utils.rm_dirs(self.get_automaton_plot_folders())

    '''
    Learning Loop (main loop, what happens when an episode ends, changes or was not completed)
    '''
    def _run_episode(self, domain_id, task_id):
        task = self._get_task(domain_id, task_id)  # get the task to learn

        if self._is_relearn_pending(domain_id):
            self._relearn_pending_automaton(task, domain_id)

        # initialize reward and steps counters, histories and reset the task to its initial state
        total_reward, episode_length = 0, 0
        observation_history = []
        current_state = task.reset()

        # get initial observations and initialise histories
        initial_observation = self._get_task_observation(task)
        self._update_observation_history(observation_history, initial_observation)
        self._on_initial_observation(initial_observation)  # TODO: maybe we need to ignore this call if the observation is empty and the ignore_empty_obs is enabled

        # get actual initial automaton state (performs verification that there is only one possible initial state!)
        current_hierarchy_state = self._get_initial_hierarchy_state_successor_for_domain(domain_id, initial_observation)

        # update the automaton if the initial state achieves the goal and the example is not covered
        if self.interleaved_automaton_learning:
            updated_automaton = self._perform_hierarchy_learning(task, domain_id, current_hierarchy_state, observation_history)
            if updated_automaton:  # get the actual initial state as done before
                current_hierarchy_state = self._get_initial_hierarchy_state_successor_for_domain(domain_id, initial_observation)

        # whether the episode execution must be stopped (an automaton is learnt in the middle)
        interrupt_episode = False
        hierarchy = self._get_hierarchy(domain_id)

        while not task.is_terminal() and episode_length < self.max_episode_length and not interrupt_episode:
            action = self._choose_action(domain_id, task_id, current_state, hierarchy, current_hierarchy_state)
            next_state, reward, is_terminal, _ = task.step(action)
            observation = self._get_task_observation(task)

            # update the observation history and get update hierarchy state
            observation_changed = self._update_observation_history(observation_history, observation)
            next_hierarchy_state = self._get_next_hierarchy_state(
                self._get_hierarchy(domain_id), current_hierarchy_state, observation, observation_changed
            )

            if self.training_enable:
                self._update_q_functions(domain_id, task_id, current_state, action, next_state, is_terminal, observation)

            # The episode has to be interrupted if an automaton is learnt
            if self._is_counterexample(task, domain_id, next_hierarchy_state):
                if self.interleaved_automaton_learning:
                    interrupt_episode = self._perform_hierarchy_learning(task, domain_id, next_hierarchy_state, observation_history)
                else:
                    # TODO: Maybe it would be a bit more correct to continue the episode. However, this might be helpful
                    #  to detect if a handcrafted machine OR a learned machine being evaluated is not accurate to what
                    #  we observe in the environment.
                    interrupt_episode = True

            if not interrupt_episode:
                hierarchy = self._get_hierarchy(domain_id)
                total_reward += reward
                episode_length += 1

                self._on_performed_step(
                    domain_id, task_id, next_state, reward, is_terminal, observation, observation_changed,
                    hierarchy, current_hierarchy_state, next_hierarchy_state, episode_length
                )

            # update current environment and automaton states and increase episode length
            current_state = next_state
            current_hierarchy_state = next_hierarchy_state

        completed_episode = not interrupt_episode

        return completed_episode, total_reward, episode_length, task.is_terminal(), observation_history

    def _on_incomplete_episode(self, domain_id):
        # If the episode was interrupted, log the learning episode
        self.automaton_learning_episodes[domain_id].append(self._curriculum.get_current_episode())

        # Tell the curriculum that the episode was interrupted in order to set appropriately for the next episode
        self._curriculum.on_learning_interrupted()

    def _on_domains_learned(self, domain_ids):
        for domain_id in domain_ids:
            self._update_background_automata(domain_id)

    @abstractmethod
    def _on_initial_observation(self, observation):
        pass

    @abstractmethod
    def _choose_action(self, domain_id, task_id, current_state, hierarchy, current_hierarchy_state):
        pass

    @abstractmethod
    def _on_performed_step(
        self, domain_id, task_id, next_state, reward, is_terminal, observation, observation_changed, hierarchy,
        current_hierarchy_state, next_hierarchy_state, episode_length
    ):
        pass

    @abstractmethod
    def _update_q_functions(self, domain_id, task_id, current_state, action, next_state, is_terminal, observations):
        pass

    '''
    Greedy Policy Evaluation
    '''
    def _evaluate_greedy_policies(self):
        # we do not want automata to be learned during the evaluation of a policy
        tmp_interleaved_automaton_learning = self.interleaved_automaton_learning
        self.interleaved_automaton_learning = False
        super()._evaluate_greedy_policies()
        self.interleaved_automaton_learning = tmp_interleaved_automaton_learning

    '''
    Task and History Management Methods (getting observations)
    '''
    def _get_task_observation(self, task):
        obs = task.get_observation()
        if self.filter_restricted_observables:
            return obs.intersection(task.get_restricted_observables())
        return obs

    def _update_observation_history(self, observation_history, observation):
        # update histories only if the observation is non-empty
        if self.ignore_empty_observations and len(observation) == 0:
            return False

        observation_t = self._get_observation_as_ordered_tuple(observation)

        if self.use_compressed_traces:
            observation_changed = len(observation_history) == 0 or observation_t != observation_history[-1]
            if observation_changed:
                observation_history.append(observation_t)
            return observation_changed
        else:
            observation_history.append(observation_t)
            return True  # all observations are relevant if the traces are uncompressed

    '''
    Automata Management Methods (setters, getters, associated rewards)
    '''
    def _get_hierarchy(self, domain_id):
        env_name = self.environment_names[domain_id]
        return self._hierarchies[env_name]

    def _set_hierarchy(self, domain_id, hierarchy):
        self._hierarchies[self.environment_names[domain_id]] = hierarchy

    def _get_next_hierarchy_state(self, hierarchy, current_hierarchy_state, observation, observation_changed):
        # The automata have to be navigated with compressed traces if specified (just when a change occurs)
        if (self.ignore_empty_observations and len(observation) == 0) or (self.use_compressed_traces and not observation_changed):
            return current_hierarchy_state
        return hierarchy.get_next_hierarchy_state(current_hierarchy_state, observation)

    def _get_initial_hierarchy_state_successor_for_domain(self, domain_id, observation):
        hierarchy = self._get_hierarchy(domain_id)
        return self._get_initial_hierarchy_state_successor(hierarchy, observation)

    def _get_initial_hierarchy_state_successor(self, hierarchy, observation):
        initial_state = hierarchy.get_initial_state()
        return self._get_next_hierarchy_state(hierarchy, initial_state, observation, True)

    def _init_hierarchies(self, params):
        if self.training_enable:
            if self.training_mode == IHSAAlgorithmBase.TRAINING_MODE_LEARN:
                self._init_basic_hierarchies(params)
            elif self.training_mode == IHSAAlgorithmBase.TRAINING_MODE_HANDCRAFTED:
                self._init_handcrafted_hierarchies()
            else:
                raise RuntimeError(f"Error: Training mode \"{self.training_mode}\" is not unknown.")
        else:
            self._load_hierarchies()

    def _init_basic_hierarchies(self, params):
        """
        Creates an initial hierarchy for each environment such that their root automaton does not accept nor reject
        anything. Each hierarchy consists only of its particular root.
        """
        for env in get_param(params, LearningAlgorithm.ENV_CONFIG):
            # The initial automaton is an automaton that doesn't accept nor reject anything
            automaton = HierarchicalAutomaton(env.get(LearningAlgorithm.ENV_CONFIG_AUTOMATON_NAME))
            automaton.add_state(IHSAAlgorithmBase.INITIAL_STATE_NAME)
            automaton.set_initial_state(IHSAAlgorithmBase.INITIAL_STATE_NAME)

            hierarchy = Hierarchy()
            hierarchy.set_root_automaton(automaton)
            self._hierarchies[env.get(LearningAlgorithm.ENV_CONFIG_NAME)] = hierarchy

    def _set_learned_hierarchy(self, learned_automaton, domain_id):
        hierarchy = Hierarchy()

        # Set initial, accepting and rejecting states in the automaton
        learned_automaton.set_initial_state(IHSAAlgorithmBase.INITIAL_STATE_NAME)
        learned_automaton.set_accept_state(IHSAAlgorithmBase.ACCEPTING_STATE_NAME)
        learned_automaton.set_reject_state(IHSAAlgorithmBase.REJECTING_STATE_NAME)
        hierarchy.set_root_automaton(learned_automaton)

        root_automaton_name = learned_automaton.get_name()
        for automaton_name in self._dependency_graph.get_dependencies(root_automaton_name):
            assert automaton_name != root_automaton_name
            hierarchy.add_automaton(self._bk_automata.get(automaton_name))

        self._set_hierarchy(domain_id, hierarchy)

    def _update_background_automata(self, domain_id):
        hierarchy = self._get_hierarchy(domain_id)
        self._bk_automata[hierarchy.get_root_automaton_name()] = hierarchy.get_root_automaton()

    def _init_handcrafted_hierarchies(self):
        """
        Initializes the dictionary of hierarchies with handcrafted hierarchies for each of the environments (domains) we
        consider. Note that any of the tasks for a given environment can be used to get the handcrafted hierarchy, so we
        use index 0 (it will work for any number of tasks higher or equal than 1).
        """
        self._hierarchies = {
            env_name: self.tasks[env_name][0].get_hierarchy()
            for env_name in self.environment_names
        }

    def _load_hierarchies(self):
        """
        Loads hierarchies for the specified environments stored in the models folder.
        """
        self._hierarchies = {
            env_name: file_utils.load_pickle_obj(f"{self.models_folder}/hierarchies/{env_name}.pkl")
            for env_name in self.environment_names
        }

    def _init_dependency_graph(self, environments):
        for env in environments:
            self._dependency_graph.add_node(env.get(IHSAAlgorithmBase.ENV_CONFIG_AUTOMATON_NAME))

        self._add_banned_dependencies(environments)

        for bk_automaton_name in self._bk_automata:
            self._dependency_graph.add_node(bk_automaton_name)
            self._update_automaton_dependencies(self._dependency_graph, self._bk_automata[bk_automaton_name])

    def _add_banned_dependencies(self, environments):
        for env_id in range(len(environments)):
            env_h_level = environments[env_id].get(IHSAAlgorithmBase.ENV_CONFIG_HIERARCHY_LEVEL)
            env_automaton_name = environments[env_id].get(IHSAAlgorithmBase.ENV_CONFIG_AUTOMATON_NAME)
            for env2_id in range(len(environments)):
                env2 = environments[env2_id]
                if env_h_level <= env2.get(IHSAAlgorithmBase.ENV_CONFIG_HIERARCHY_LEVEL):
                    self._dependency_graph.add_banned_dependency(env_automaton_name,
                                                                 env2.get(IHSAAlgorithmBase.ENV_CONFIG_AUTOMATON_NAME))

    def _update_automaton_dependencies(self, dependency_graph, automaton):
        # Cleaning dependencies is important when updating the dynamic dependency graph (not the BK one)
        dependency_graph.clean_dependencies(automaton.get_name())
        for called_automaton in automaton.get_called_automaton_names():
            dependency_graph.add_dependency(automaton.get_name(), called_automaton)

    def _init_bk_dependency_graph(self, environments):
        """
        Adds the dependencies provided in the configuration file to the background dependency graph. This allows
        checking afterwards that only appropriate automata are added to the background knowledge of the learning tasks
        of other automata.
        """
        # Add the names of the automata to be learned to the graph
        for env in environments:
            self._bk_dependency_graph.add_node(env[IHSAAlgorithmBase.ENV_CONFIG_AUTOMATON_NAME])

        # Add the dependencies according to the "dependencies" field in the configuration file
        # TODO: check levels to see if they make sense
        for env in environments:
            dependant_automaton_name = env.get(IHSAAlgorithmBase.ENV_CONFIG_AUTOMATON_NAME)
            dependencies = env.get(IHSAAlgorithmBase.ENV_CONFIG_DEPENDENCIES, [])
            for dependency_automaton_name in dependencies:
                self._bk_dependency_graph.add_dependency(dependant_automaton_name, dependency_automaton_name)

    '''
    Automata Learning Methods (example update, task generation/solving/parsing)
    '''
    def _on_automaton_learned(self, domain_id):
        hierarchy = self._get_hierarchy(domain_id)
        if hierarchy.get_root_automaton_name() in self._bk_automata:
            # if the automaton was in the background knowledge, then update it :)
            self._update_background_automata(domain_id)

    def _is_counterexample(self, task, domain_id, hierarchy_state):
        return self._is_goal_counterexample(task, domain_id, hierarchy_state) or \
               self._is_dend_counterexample(task, domain_id, hierarchy_state) or \
               self._is_inc_counterexample(task, domain_id, hierarchy_state)

    def _is_goal_counterexample(self, task, domain_id, hierarchy_state):
        return task.is_terminal() and \
               task.is_goal_achieved() and \
               not self._get_hierarchy(domain_id).is_accept_state(hierarchy_state)

    def _is_dend_counterexample(self, task, domain_id, hierarchy_state):
        return task.is_terminal() and \
               not task.is_goal_achieved() and \
               not self._get_hierarchy(domain_id).is_reject_state(hierarchy_state)

    def _is_inc_counterexample(self, task, domain_id, hierarchy_state):
        return not task.is_terminal() and \
               self._get_hierarchy(domain_id).is_terminal_state(hierarchy_state)

    def _perform_hierarchy_learning(self, task, domain_id, hierarchy_state, observation_history):
        """
        Updates the set of examples based on the current observed trace. In case the set of example is updated, it
        makes a call to the automata learner. Returns True if a new automaton has been learnt, False otherwise.
        """
        updated_examples = self._update_examples(task, domain_id, hierarchy_state, observation_history)
        if updated_examples:
            if self.debug:
                print(f"Updating automaton {domain_id}... The last counterexample is: {observation_history}.")
            self._update_hierarchy(task, domain_id)
            return True  # whether a new automaton has been learnt
        return False

    def _update_examples(self, task, domain_id, hierarchy_state, observation_history):
        """
        Updates the set of examples. Returns True if the set of examples is ready to learn an automaton from it, and
        False otherwise.
         - If the trace (observation_history) is a goal trace, then the set is always updated. However, we will only
           allow to learn a new automaton (i.e., return True) if we have observed the minimum number of goal examples.
         - If the trace is a dead-end or incomplete trace, we will only update the set if we have already observed the
           minimum number of goal examples. If this condition applies, then we will just notify the set has changed and
           allow learning a new automaton.
        """
        if self._is_goal_counterexample(task, domain_id, hierarchy_state):
            # Update the episode of the first counter example. Note that we assume that the first counterexample
            # is always going to be a goal trace.
            self._update_first_example_episode(domain_id)

            if not self._is_min_goal_examples_observed(domain_id):
                try:
                    self._update_example_set(self._pre_goal_examples[domain_id], observation_history)
                except CoveredExampleError:
                    # Maybe we have found the same counterexample while randomly looking for goal examples, in
                    # such case DO NOT raise the exception.
                    pass

            if self._is_min_goal_examples_observed(domain_id):
                if len(self.goal_examples[domain_id]) == 0:
                    top_shortest_pre_goal_examples = sorted(
                        self._pre_goal_examples[domain_id],
                        key=lambda x: (len(x), x)  # In case of a tie in length, use the sequence to break the ties
                    )[:self._num_top_shortest_goal_examples[domain_id]]
                    self.goal_examples[domain_id].update(set(top_shortest_pre_goal_examples))
                else:
                    self._update_example_set(self.goal_examples[domain_id], observation_history)
                return True
        elif self._is_dend_counterexample(task, domain_id, hierarchy_state) and self._is_min_goal_examples_observed(domain_id):
            self._update_example_set(self.dend_examples[domain_id], observation_history)
            if not self._root_rejecting_state_deepening_enabled:  # Use the rejecting state from now on if the deepening is not enabled
                self._use_root_rejecting_state[domain_id] = True
            return True
        elif self._is_inc_counterexample(task, domain_id, hierarchy_state) and self._is_min_goal_examples_observed(domain_id):
            self._update_example_set(self.inc_examples[domain_id], observation_history)
            return True
        return False

    def _update_first_example_episode(self, domain_id):
        if not self._is_first_example_observed(domain_id):
            self._first_counterexample_episodes[domain_id] = self._curriculum.get_current_episode()

    def _is_first_example_observed(self, domain_id):
        return self._first_counterexample_episodes[domain_id] > 0

    def _is_min_goal_examples_observed(self, domain_id):
        return len(self._pre_goal_examples[domain_id]) >= self._min_goal_examples[domain_id]

    def _update_example_set(self, example_set, observation_history):
        """Updates the a given example set with the corresponding history of observations depending on whether
        compressed traces are used or not to learn the automata. An exception is thrown if a trace is readded."""
        history_tuple = tuple(observation_history)
        if history_tuple not in example_set:
            example_set.add(history_tuple)
        else:
            raise CoveredExampleError(history_tuple)

    def _update_hierarchy(self, task, domain_id):
        self.automaton_counters[domain_id] += 1  # increment the counter of the number of aut. learnt for a domain
        self._generate_ilasp_task(task, domain_id)  # generate the automata learning task

        solver_success, running_time = self._solve_ilasp_task(domain_id)  # run the task solver

        # Update timeout and time run for the domain
        self.learning_time_remaining -= running_time
        self.domain_running_time[domain_id] += running_time

        if solver_success:
            ilasp_solution_filename = os.path.join(self.get_automaton_solution_folder(domain_id),
                                                   IHSAAlgorithmBase.AUTOMATON_SOLUTION_FILENAME % self.automaton_counters[domain_id])
            root_automaton_name = self._get_hierarchy(domain_id).get_root_automaton_name()
            candidate_automaton = self._parse_ilasp_solutions(root_automaton_name, ilasp_solution_filename)

            if candidate_automaton.get_num_states() > 0:
                # add the dependants of the learned automaton as automata to be relearned next
                self._update_relearn_pending(root_automaton_name)

                # update the dependency graph and add dependencies to the hierarchy
                self._update_automaton_dependencies(self._dependency_graph, candidate_automaton)

                # create a new hierarchy from the automaton and set it for the domain_id
                self._set_learned_hierarchy(candidate_automaton, domain_id)

                # plot the new automaton
                candidate_automaton.plot(self.get_automaton_plot_folder(domain_id),
                                         IHSAAlgorithmBase.AUTOMATON_PLOT_FILENAME % self.automaton_counters[domain_id])

                self._on_automaton_learned(domain_id)
            else:
                if self.debug:
                    print(f"The learning task for automaton {domain_id} with {self.num_automaton_states[domain_id]} "
                          f"states including {1 if self._use_root_rejecting_state[domain_id] else 0} rejecting "
                          f"state is UNSAT.")

                # If the task is UNSATISFIABLE and the deepening is enabled, the following cases arise:
                #  1. If we haven't observed any deadend examples, then it means that the number of states we are using
                #     is insufficient. Therefore, we increase the number of states and relearn the automaton.
                #  2. If we have observed at least one deadend example, then we need to check whether edges to rejecting
                #     states were being learned. If so, then the situation is analogous to case (1), so we learn a new
                #     automaton with one more state and we will exclude an explicit rejecting state in the root by now.
                #     Otherwise, we will allow to learn an automaton with the same set of states but allowing edges to
                #     be directed to the rejecting state of the root.
                # If the deepening is not enabled, we just increase the number of states (whether to use a rejecting
                # state or not is determined when the set of deadend examples is updated).
                if self._root_rejecting_state_deepening_enabled:
                    if len(self.dend_examples[domain_id]) == 0:
                        self._increase_num_automaton_states(domain_id)
                    else:
                        if self._use_root_rejecting_state[domain_id]:
                            self._increase_num_automaton_states(domain_id)
                            self._use_root_rejecting_state[domain_id] = False
                        else:
                            self._use_root_rejecting_state[domain_id] = True
                else:
                    self._increase_num_automaton_states(domain_id)

                if self.debug:
                    print(f"(Re)Updating automaton {domain_id} with {self.num_automaton_states[domain_id]} states "
                          f"including {1 if self._use_root_rejecting_state[domain_id] else 0} rejecting state.")

                self._update_hierarchy(task, domain_id)
        else:
            raise InterruptLearningException("Error: Couldn't find an automaton under the specified timeout!")

    def _increase_num_automaton_states(self, domain_id):
        self.num_automaton_states[domain_id] += 1

    def _generate_ilasp_task(self, task, domain_id):
        file_utils.mkdir(self.get_automaton_task_folder(domain_id))

        ilasp_task_filename = IHSAAlgorithmBase.AUTOMATON_TASK_FILENAME % self.automaton_counters[domain_id]

        # The set of observables that will be used to label the edges of the automaton is set here. Note that it is
        # different from the set of observables below, which is used for the symmetry breaking.
        if self.hypothesis_restrict_observables:
            hyp_observables = task.get_restricted_observables()
        else:
            hyp_observables = task.get_observables()

        # If the traces are being filtered using the set of restricted observables, then the symmetry breaking only
        # requires these observables
        if self.filter_restricted_observables:
            sb_observables = task.get_restricted_observables()
        else:
            sb_observables = task.get_observables()

        # The sets of examples are sorted to make sure that ILASP produces the same solution for the same sets (ILASP
        # can produce different hypothesis for the same set of examples but given in different order)
        generate_ilasp_task(self._get_hierarchy(domain_id).get_root_automaton_name(), self.num_automaton_states[domain_id],
                            IHSAAlgorithmBase.ACCEPTING_STATE_NAME, IHSAAlgorithmBase.REJECTING_STATE_NAME, hyp_observables,
                            sorted(self.goal_examples[domain_id]), sorted(self.dend_examples[domain_id]),
                            sorted(self.inc_examples[domain_id]), self._get_background_automata(domain_id),
                            self.get_automaton_task_folder(domain_id), ilasp_task_filename, self.symmetry_breaking_enable,
                            sb_observables, self.max_disjunction_size, self.learn_acyclic_graph,
                            self.avoid_learning_only_negative, self._remove_deadend_states, self._use_mutex_saturation,
                            self._use_root_rejecting_state[domain_id])

    def _get_background_automata(self, domain_id) -> List[Tuple[HierarchicalAutomaton, bool]]:
        root_automaton_name = self._get_hierarchy(domain_id).get_root_automaton_name()
        bk_automata_local = []
        for bk_automaton_name in self._bk_automata:
            # An automaton will be added to the background knowledge if:
            #  - Does not form a circular dependency (i.e., it cannot depend on the automaton being learned).
            #  - The dependency is not banned (i.e., the learned automaton cannot depend on the candidate BK one).
            #  - If we restrict dependencies by following the ones provided as an input, then the automaton will be
            #    added if it forms such a dependency.
            if not self._dependency_graph.is_dependant_on(bk_automaton_name, root_automaton_name) and \
               not self._dependency_graph.is_banned_dependency(root_automaton_name, bk_automaton_name) and \
               (not self._hypothesis_restrict_dependencies or self._bk_dependency_graph.is_dependant_on(root_automaton_name,
                                                                                                        bk_automaton_name)):
                is_callable = True
                if self._hypothesis_restrict_dependencies:
                    # The automata on which the root directly depends will appear in the calls of the hypothesis space,
                    # and those which indirectly depend will only appear in the BK but cannot be called
                    is_callable = self._bk_dependency_graph.is_directly_dependant_on(root_automaton_name,
                                                                                     bk_automaton_name)
                bk_automata_local.append((self._bk_automata[bk_automaton_name], is_callable))
        bk_automata_local.sort(key=lambda x: x[0].get_name())  # sorting is important for reproducibility
        return bk_automata_local

    def _solve_ilasp_task(self, domain_id):
        file_utils.mkdir(self.get_automaton_solution_folder(domain_id))

        ilasp_task_filename = os.path.join(self.get_automaton_task_folder(domain_id),
                                           IHSAAlgorithmBase.AUTOMATON_TASK_FILENAME % self.automaton_counters[domain_id])

        ilasp_solution_filename = os.path.join(self.get_automaton_solution_folder(domain_id),
                                               IHSAAlgorithmBase.AUTOMATON_SOLUTION_FILENAME % self.automaton_counters[domain_id])

        return solve_ilasp_task(ilasp_task_filename, ilasp_solution_filename, self.ilasp_flags, timeout=self.learning_time_remaining)

    def _parse_ilasp_solutions(self, root_automaton_name, last_automaton_filename):
        return ilasp_solution_parser.parse_ilasp_solutions(root_automaton_name, last_automaton_filename)

    def _is_relearn_pending(self, domain_id):
        return domain_id in self._domains_to_relearn

    def _relearn_pending_automaton(self, task, domain_id):
        if self.debug:
            print(f"Relearning automaton for domain {domain_id}...")
        self.num_automaton_states[domain_id] = self.num_starting_states  # restart state set
        self._update_hierarchy(task, domain_id)  # learn the new automaton
        self._domains_to_relearn.remove(domain_id)  # remove it from the pending list

    def _update_relearn_pending(self, learned_automaton_name):
        for domain_id in range(self.num_domains):
            if self._dependency_graph.is_dependant_on(self._get_hierarchy(domain_id).get_root_automaton_name(),
                                                      learned_automaton_name):
                self._domains_to_relearn.add(domain_id)

    '''
    Logging and Messaging Management Methods
    '''
    def _restore_uncheckpointed_files(self):
        super()._restore_uncheckpointed_files()
        self._remove_uncheckpointed_files()

    def _remove_uncheckpointed_files(self):
        """Removes files which were generated after the last checkpoint."""
        for domain_id in range(self.num_domains):
            counter = self.automaton_counters[domain_id]
            self._remove_uncheckpointed_files_helper(self.get_automaton_task_folder(domain_id), "task-", ".las", counter)
            self._remove_uncheckpointed_files_helper(self.get_automaton_solution_folder(domain_id), "solution-", ".txt", counter)
            self._remove_uncheckpointed_files_helper(self.get_automaton_plot_folder(domain_id), "plot-", ".png", counter)

    def _remove_uncheckpointed_files_helper(self, folder, prefix, extension, automaton_counter):
        if file_utils.path_exists(folder):
            files_to_remove = [os.path.join(folder, x) for x in os.listdir(folder)
                               if x.startswith(prefix) and int(x[len(prefix):-len(extension)]) > automaton_counter]
            file_utils.rm_files(files_to_remove)

    def _update_stats(self, stats):
        stats["mode"] = self.training_mode

        if self.interleaved_automaton_learning:
            stats["ilasp"] = {
                "calls": self.automaton_counters.sum().item(),
                "time": self.domain_running_time.sum().item()
            }

            stats["domains"] = {}
            for domain_id in range(self.num_domains):
                stats["domains"][self.environment_names[domain_id]] = self._get_domain_stats(domain_id)

    def _get_domain_stats(self, domain_id):
        stats = {
            "episodes": {
                "level_started": self._curriculum.get_episode_level_started(domain_id),
                "learned_automaton": self.automaton_learning_episodes[domain_id],
                "first_example": self._first_counterexample_episodes[domain_id].item() if self._is_first_example_observed(domain_id) else -1,
            },
            "ilasp": {
                "calls": self.automaton_counters[domain_id].item(),
                "time": self.domain_running_time[domain_id].item()
            },
            "automaton": {
                "states": self._get_hierarchy(domain_id).get_root_automaton().get_num_states(),
                "edges": self._get_hierarchy(domain_id).get_root_automaton().get_num_edges()
            },
            "goal_examples": {
                "count": len(self.goal_examples[domain_id]),
                "lengths": [len(x) for x in self.goal_examples[domain_id]]
            },
            "dend_examples": {
                "count": len(self.dend_examples[domain_id]),
                "lengths": [len(x) for x in self.dend_examples[domain_id]]
            },
            "inc_examples": {
                "count": len(self.inc_examples[domain_id]),
                "lengths": [len(x) for x in self.inc_examples[domain_id]]
            }
        }

        return stats

    '''
    Model Management
    '''
    def _export_models(self):
        file_utils.mkdir(self.models_folder)
        self._export_hierarchies()

    def _export_hierarchies(self):
        """
        Exports the hierarchy for each domain into separate .pkl files.
        """
        file_utils.mkdir(self._get_hierarchies_model_folder())
        for domain_id in range(self.num_domains):
            file_utils.dump_pickle_obj(
                self._get_hierarchy(domain_id),
                self._get_hierarchies_model_file(domain_id)
            )

    '''
    File Management Methods
    '''
    def get_automaton_task_folders(self):
        return [self.get_automaton_task_folder(domain_id) for domain_id in range(self.num_domains)]

    def get_automaton_task_folder(self, domain_id):
        env_name = self.environment_names[domain_id]
        return os.path.join(self.export_folder_names.get(env_name), IHSAAlgorithmBase.AUTOMATON_TASK_FOLDER)

    def get_automaton_solution_folders(self):
        return [self.get_automaton_solution_folder(domain_id) for domain_id in range(self.num_domains)]

    def get_automaton_solution_folder(self, domain_id):
        env_name = self.environment_names[domain_id]
        return os.path.join(self.export_folder_names.get(env_name), IHSAAlgorithmBase.AUTOMATON_SOLUTION_FOLDER)

    def get_automaton_plot_folders(self):
        return [self.get_automaton_plot_folder(domain_id) for domain_id in range(self.num_domains)]

    def get_automaton_plot_folder(self, domain_id):
        env_name = self.environment_names[domain_id]
        return os.path.join(self.export_folder_names.get(env_name), IHSAAlgorithmBase.AUTOMATON_PLOT_FOLDER)

    def _get_hierarchies_model_folder(self):
        return os.path.join(self.models_folder, IHSAAlgorithmBase.MODELS_HIERARCHIES_FOLDER)

    def _get_hierarchies_model_file(self, domain_id):
        return os.path.join(self._get_hierarchies_model_folder(), f"{self.environment_names[domain_id]}.pkl")
