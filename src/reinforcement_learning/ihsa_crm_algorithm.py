import collections
from gym_hierarchical_subgoal_automata.automata.common import HierarchyState
from gym_hierarchical_subgoal_automata.automata.logic import TRUE
import numpy as np
import os
from reinforcement_learning.ihsa_cross_product_algorithm import IHSAAlgorithmCrossProduct
from reinforcement_learning.model import MinigridCRMDQN, WaterWorldCRMDQN
import torch
from typing import Optional


Experience = collections.namedtuple(
    "Experience",
    field_names=["state", "automaton_state", "action", "reward", "is_terminal", "next_state", "next_automaton_state"]
)


class IHSAAlgorithmCRM(IHSAAlgorithmCrossProduct):
    """
    Implementation of the CRM (Counterfactual Experiences for Reward Machine) algorithm.

    Paper: Toro Icarte et al. 2022. "Reward Machines: Exploiting Reward Function Structure in Reinforcement Learning".
    """
    def __init__(self, params):
        super().__init__(params)

        self._q_function: Optional[torch.nn.Module] = None
        self._tgt_q_function: Optional[torch.nn.Module] = None
        self._init_q_functions()

        self._optimizer = torch.optim.RMSprop(self._q_function.parameters(), lr=self._learning_rate)

    def _init_q_functions(self):
        self._q_function = self._create_q_function()
        if not self.training_enable:
            self._q_function.load_state_dict(torch.load(self._get_model_path()))
        self._tgt_q_function = self._create_q_function()
        self._tgt_q_function.load_state_dict(self._q_function.state_dict())

    def _create_q_function(self):
        automaton = self._get_hierarchy(0).get_root_automaton()
        task = self._get_task(0, 0)

        if self.env_name == IHSAAlgorithmCRM.ENV_NAME_CRAFTWORLD:
            q_function = MinigridCRMDQN(
                task.observation_space.shape, automaton.get_num_states(), task.action_space.n, self._use_maxpool
            )
        elif self.env_name == IHSAAlgorithmCRM.ENV_NAME_WATERWORLD:
            q_function = WaterWorldCRMDQN(
                task.observation_space.shape, automaton.get_num_states(), task.action_space.n, self._layer_size
            )
        else:
            raise RuntimeError(f"Error: Unknown environment '{self.env_name}'.")
        q_function.to(self.device)
        return q_function

    def _update_q_functions(self, domain_id, task_id, current_state, action, next_state, is_terminal, observations):
        self._add_buffer_experiences(current_state, action, next_state, observations)
        if len(self._er_buffer) >= self._er_start_size:
            self._update_q_functions_helper(
                self._er_buffer.sample(self._er_batch_size)
            )

    def _add_buffer_experiences(self, current_state, action, next_state, observations):
        hierarchy = self._get_hierarchy(0)
        automaton = hierarchy.get_root_automaton()

        for automaton_state in automaton.get_states():
            if not automaton.is_terminal_state(automaton_state):
                next_automaton_state = hierarchy.get_next_hierarchy_state(
                    HierarchyState(automaton_state, automaton.get_name(), TRUE, [], []),
                    observations
                ).state_name

                self._er_buffer.append(Experience(
                    current_state,
                    automaton.get_state_embedding(automaton_state),
                    action,
                    1.0 if automaton.is_accept_state(next_automaton_state) else 0.0,
                    automaton.is_terminal_state(next_automaton_state),
                    next_state,
                    automaton.get_state_embedding(next_automaton_state)
                ))

    def _update_q_functions_helper(self, experience_batch):
        states, automaton_states, actions, rewards, is_terminal, next_states, next_automaton_states = zip(*experience_batch)

        states_v = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        automaton_states_v = torch.tensor(np.array(automaton_states), dtype=torch.float32, device=self.device)
        actions_v = torch.tensor(np.array(actions), dtype=torch.long, device=self.device)
        rewards_v = torch.tensor(np.array(rewards), dtype=torch.float32, device=self.device)
        is_terminal_v = torch.tensor(np.array(is_terminal), dtype=torch.bool, device=self.device)
        next_states_v = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        next_automaton_states_v = torch.tensor(np.array(next_automaton_states), dtype=torch.float32, device=self.device)

        state_action_values = self._q_function(states_v, automaton_states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            if self._use_double_dqn:
                next_state_actions = self._q_function(next_states_v, next_automaton_states_v).max(1)[1]
                next_state_action_values = self._tgt_q_function(next_states_v, next_automaton_states_v).gather(1, next_state_actions.unsqueeze(-1)).squeeze(-1)
            else:
                next_state_action_values = self._tgt_q_function(next_states_v, next_automaton_states_v).max(1)[0]
            next_state_action_values[is_terminal_v] = 0.0

        expected_state_action_values = rewards_v + self._discount_rate * next_state_action_values
        loss = torch.nn.MSELoss()(state_action_values, expected_state_action_values)

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        self._q_function_update_counter += 1
        if self._q_function_update_counter % self._tgt_update_freq == 0:
            self._tgt_q_function.load_state_dict(self._q_function.state_dict())

    def _choose_action(self, domain_id, task_id, current_state, hierarchy, current_hierarchy_state):
        automaton = hierarchy.get_root_automaton()
        automaton_state_v = torch.tensor(
            np.array([automaton.get_state_embedding(current_hierarchy_state.state_name)]),
            dtype=torch.float32,
            device=self.device
        )

        return self._choose_egreedy_action(
            self._get_task(domain_id, task_id),
            current_state,
            lambda state: self._q_function(state, automaton_state_v),
            self._get_exploration_rate()
        )

    def _export_models(self):
        super()._export_models()
        torch.save(self._q_function.state_dict(), self._get_model_path())

    def _get_model_path(self):
        return os.path.join(self.models_folder, "model.pt")
