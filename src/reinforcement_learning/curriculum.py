import numpy as np
from scipy.special import softmax


class Curriculum:
    HIERARCHY_LEVEL = "hierarchy_level"
    SCORING_PIERROT = "pierrot"
    SCORING_ANDREAS = "andreas"

    def __init__(self, num_domains, num_tasks, domains_learned_callback, env_config, curriculum_weight, curriculum_temp,
                 curriculum_threshold, curriculum_scoring_method):
        self._num_domains = num_domains
        self._num_tasks = num_tasks
        self._domains_learned_callback = domains_learned_callback  # Function to call once the domains in a level are considered learned

        # Information about the hierarchy levels of each domain, the hierarchy level in which we currently are, the
        # min. and max. hierarchy levels
        self._hierarchy_levels = self._load_hierarchy_levels(env_config)
        self._current_hierarchy_level = self._hierarchy_levels.min()
        self._min_hierarchy_level = self._hierarchy_levels.min()
        self._max_hierarchy_level = self._hierarchy_levels.max()

        # Keep track of the level change history
        self._hierarchy_level_history = []

        # Keep track of the current domain, task and episode
        self._current_domain = 0
        self._current_task = 0
        self._current_episode = 0

        # Parameters for the selection of next domain and task:
        #  - Running average of the return for a (domain, task) pair.
        #  - The coefficient used for the running average R_{i+1} = \beta * R_i + (1 - \beta) * r_i.
        #  - Temperature for the softmax.
        #  - Minimum return threshold across tasks in the same level to consider a level mastered.
        #  - Method used to get scores from the returns, which are used to compute the probability afterwards.
        #  - Last episode in which an automaton was learned for each domain (deprecated by now).
        self._average_returns = np.zeros((num_domains, num_tasks))
        self._beta = curriculum_weight
        self._tau = curriculum_temp
        self._threshold_level_complete = curriculum_threshold
        self._curriculum_scoring_method = curriculum_scoring_method

    def _load_hierarchy_levels(self, env_config):
        hierarchy_levels = np.zeros(self._num_domains, dtype=np.int)
        for env_id in range(self._num_domains):
            assert Curriculum.HIERARCHY_LEVEL in env_config[env_id]
            hierarchy_levels[env_id] = env_config[env_id][Curriculum.HIERARCHY_LEVEL]
        return hierarchy_levels

    def update(self, domain_id, task_id, total_return):
        # Update the running average of undiscounted returns
        self._update_average_returns(domain_id, task_id, total_return)

        # If not all hierarchy levels have been completed, take the minimum return for the current level and check
        # whether it surpasses the threshold (if it does, set the current level as learned)
        if self._current_hierarchy_level <= self._max_hierarchy_level:
            if self._get_current_level_min_return() > self._threshold_level_complete:
                self._on_current_level_completed()
            else:
                pass  # TODO: Maybe we should backtrack if we find that the threshold for lower levels is no longer
                      #  surpassed, thus removing automata from the background knowledge.

    def _get_current_level_min_return(self):
        # Get the indices for the current level and return the minimum average reward of a task in this level
        return self._average_returns[self._get_current_level_indices()].min()

    def _on_current_level_completed(self):
        # Get the indices of the domains in the level we just completed
        current_level_indices = self._get_current_level_indices()

        # Increase the current hierarchy level until we found that some domain has this level or we reach the maximum
        # possible level
        old_level = self._current_hierarchy_level
        self._current_hierarchy_level += 1
        while self._current_hierarchy_level < self._max_hierarchy_level and self._current_hierarchy_level not in self._hierarchy_levels:
            self._current_hierarchy_level += 1
        self._on_level_changed(old_level, self._current_hierarchy_level)

        # Invoke the callback passing the current level domain ids (this needs to be called after increasing the level
        # because the callback requires the hierarchy level to be updated)
        self._domains_learned_callback(current_level_indices)

    def _on_level_changed(self, old_level, new_level):
        """
        Records the change of hierarchical level in a list.
        """
        self._hierarchy_level_history.append((self._current_episode, old_level.item(), new_level.item()))

    def get_history(self):
        return self._hierarchy_level_history

    def get_episode_level_started(self, domain_id):
        level = self.get_domain_hierarchy_level(domain_id)
        if level == self._min_hierarchy_level:
            return 1
        for episode, _, new_level in self._hierarchy_level_history:
            if level == new_level:
                return episode
        return -1

    def _get_current_level_indices(self):
        return np.where(self._hierarchy_levels == self._current_hierarchy_level)[0]

    def _update_average_returns(self, domain_id, task_id, total_return):
        self._average_returns[domain_id, task_id] = \
            self._beta * self._average_returns[domain_id, task_id] + (1 - self._beta) * total_return

    def get_current_domain(self):
        return self._current_domain

    def get_current_task(self):
        return self._current_task

    def get_current_episode(self):
        return self._current_episode

    def is_current_level_min(self):
        """
        Returns True if the current level is the minimum one.
        """
        return self._current_hierarchy_level == self._min_hierarchy_level

    def on_learning_interrupted(self):
        # Set the current hierarchy level to that of the current domain. This occurs when an automaton is learned: we
        # want to set the level to the current one to make sure it is mastered again before going back to the higher
        # levels which may use its automaton. Also, restart the average return to avoid having a biased estimate of it
        # with the old automaton.
        new_hierarchy_level = self._hierarchy_levels[self._current_domain]

        if self._current_hierarchy_level != new_hierarchy_level:
            self._on_level_changed(self._current_hierarchy_level, new_hierarchy_level)
            self._current_hierarchy_level = new_hierarchy_level

        self._average_returns[self._current_domain, :] = 0.0

    def on_episode_start(self):
        self._current_episode += 1
        self._select_domain()
        self._select_task()

    def _select_domain(self):
        # Take the indices of the tasks that are currently being considered (any task whose level is lower or equal than
        # the current one)
        current_domain_indices = np.where(self._hierarchy_levels <= self._current_hierarchy_level)[0]

        # Compute the minimum return across tasks for each domain
        avg_task_return_per_domain = self._average_returns[current_domain_indices].min(axis=1)

        # Select a domain (the domain whose average return across tasks is lower will be more likely to be selected),
        # note that we use current_domain_indices because it contains the actually selectable indices of the domains we
        # are considering!
        self._current_domain = current_domain_indices[self._stochastic_selection(avg_task_return_per_domain)]

    def _select_task(self):
        self._current_task = self._stochastic_selection(self._average_returns[self._current_domain])

    def _stochastic_selection(self, average_returns):
        """
        Compute the choice probability for each domain using the formula in "Learning Compositional Neural Programs with
        Recursive Tree Search and Planning" (Pierrot et al., 2019) or in "Modular Multitask Reinforcement Learning with
        Policy Sketches" (Andreas et al., 2017), and select an index according to these probabilities.
        """
        if self._curriculum_scoring_method == Curriculum.SCORING_PIERROT:
            probs = self._pierrot_selection(average_returns)
        elif self._curriculum_scoring_method == Curriculum.SCORING_ANDREAS:
            probs = self._andreas_selection(average_returns)
        else:
            raise RuntimeError(f"Error: Unknown curriculum scoring method '{self._curriculum_scoring_method}'.")

        return np.random.choice(len(average_returns), p=probs)

    def _pierrot_selection(self, average_returns):
        # Since we will later apply a softmax, we must ensure that we won't divide by 0, so we add a small constant
        score = 1 / (average_returns + 1e-7)
        return softmax(self._tau * score)

    def _andreas_selection(self, average_returns):
        score = 1 - average_returns
        return score / np.sum(score)

    def is_active_domain(self, domain_id):
        """
        Returns True if the level of the passed domain is lower or equal than the current one (i.e., it is available to
        be selected in the curriculum).
        """
        return self._hierarchy_levels[domain_id] <= self._current_hierarchy_level

    def get_active_domains(self):
        """
        Returns an array with the ids of the domains whose hierarchy level is lower or equal than the current one (i.e.,
        they are available to be selected in the curriculum).
        """
        return np.arange(self._num_domains)[self._hierarchy_levels <= self._current_hierarchy_level]

    def get_mastered_domains(self):
        """
        Returns an array with the ids of the domains whose hierarchy level is lower than the current one (i.e., domains
        that are considered to be mastered, where we are already performing well).
        """
        return np.arange(self._num_domains)[self._hierarchy_levels < self._current_hierarchy_level]

    def get_domain_hierarchy_level(self, domain_id):
        return self._hierarchy_levels[domain_id]
