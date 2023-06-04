import random
from reinforcement_learning.ihsa_base_algorithm import IHSAAlgorithmBase


class IHSAAlgorithmRandom(IHSAAlgorithmBase):
    """
    Implementation of an agent choosing a random action at each step.
    """
    def _choose_action(self, domain_id, task_id, current_state, hierarchy, current_hierarchy_state):
        task = self._get_task(domain_id, task_id)
        return random.choice(task.action_space.n)

    def _on_performed_step(
        self, domain_id, task_id, next_state, reward, is_terminal, observation, observation_changed, hierarchy,
        current_hierarchy_state, next_hierarchy_state, episode_length
    ):
        pass

    def _build_q_functions(self):
        pass

    def _on_initial_observation(self, observation):
        pass

    def _update_q_functions(self, domain_id, task_id, current_state, action, next_state, is_terminal, observations):
        pass

    def _build_experience_replay_buffers(self):
        pass

    def _on_automaton_learned(self, domain_id):
        pass

    def _export_models(self):
        pass
