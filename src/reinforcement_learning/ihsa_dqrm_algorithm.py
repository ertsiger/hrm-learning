import collections
from gym_hierarchical_subgoal_automata.automata.common import HierarchyState
from gym_hierarchical_subgoal_automata.automata.logic import TRUE
import numpy as np
import os
import random
from reinforcement_learning.ihsa_cross_product_algorithm import IHSAAlgorithmCrossProduct
from reinforcement_learning.model import WaterWorldDQN, MinigridFormulaDQN
import torch
from typing import Dict
from utils.container_utils import get_param


Experience = collections.namedtuple(
    "Experience",
    field_names=["state", "action", "next_state", "observations"]
)


class IHSAAlgorithmDQRM(IHSAAlgorithmCrossProduct):
    """
    Implementation of the DQRM (Deep Q-Learning for Reward Machines) algorithm.

    Papers:
        - Toro Icarte et al. 2018. "Using Reward Machines for High-Level Task Specification and Decomposition in Reinforcement Learning".
        - Toro Icarte et al. 2022. "Reward Machines: Exploiting Reward Function Structure in Reinforcement Learning".
    """
    NUM_UPDATED_Q_FUNCTIONS_PER_STEP = "num_updated_q_functions"

    def __init__(self, params):
        super().__init__(params)

        self._num_updated_q_functions = get_param(params, IHSAAlgorithmDQRM.NUM_UPDATED_Q_FUNCTIONS_PER_STEP, 4)

        self._q_functions: Dict[str, torch.nn.Module] = {}
        self._tgt_q_functions: Dict[str, torch.nn.Module] = {}
        self._optimizers: Dict[str, torch.optim.Optimizer] = {}
        self._update_counters: Dict[str, int] = {}
        self._init_q_functions()

    def _init_q_functions(self):
        task = self._get_task(0, 0)
        for state in self._get_hierarchy(0).get_root_automaton().get_states():
            self._q_functions[state] = self._create_q_function(task)
            if not self.training_enable:
                self._q_functions[state].load_state_dict(torch.load(self._get_models_path())[state])
            self._tgt_q_functions[state] = self._create_q_function(task)
            self._tgt_q_functions[state].load_state_dict(self._q_functions[state].state_dict())
            self._optimizers[state] = torch.optim.RMSprop(self._q_functions[state].parameters(), lr=self._learning_rate)
            self._update_counters[state] = 0

    def _create_q_function(self, task):
        if self.env_name == IHSAAlgorithmDQRM.ENV_NAME_CRAFTWORLD:
            q_function = MinigridFormulaDQN(
                task.observation_space.shape, task.action_space.n, (16, 32, 32), [256], self._use_maxpool
            )
        elif self.env_name == IHSAAlgorithmDQRM.ENV_NAME_WATERWORLD:
            q_function = WaterWorldDQN(task.observation_space.shape[0], task.action_space.n, 3 * [self._layer_size])
        else:
            raise RuntimeError(f"Error: Unknown environment '{self.env_name}'.")
        q_function.to(self.device)
        return q_function

    def _choose_action(self, domain_id, task_id, current_state, hierarchy, current_hierarchy_state):
        return self._choose_egreedy_action(
            self._get_task(domain_id, task_id),
            current_state,
            self._q_functions[current_hierarchy_state.state_name],
            self._get_exploration_rate()
        )

    def _update_q_functions(self, domain_id, task_id, current_state, action, next_state, is_terminal, observations):
        self._er_buffer.append(Experience(current_state, action, next_state, observations))

        if len(self._er_buffer) >= self._er_start_size:
            self._update_q_functions_helper(
                self._er_buffer.sample(self._er_batch_size)
            )

    def _update_q_functions_helper(self, experience_batch):
        states, actions, next_states, observations = zip(*experience_batch)

        states_v = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions_v = torch.tensor(np.array(actions), dtype=torch.long, device=self.device)
        next_states_v = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)

        for automaton_state in self._get_random_automaton_state_set():
            state_action_values = self._q_functions[automaton_state](states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
            loss = torch.nn.MSELoss()(
                state_action_values,
                self._get_expected_state_action_values(automaton_state, next_states_v, observations)
            )

            optimizer = self._optimizers[automaton_state]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self._update_counters[automaton_state] += 1
            if self._update_counters[automaton_state] % self._tgt_update_freq == 0:
                self._tgt_q_functions[automaton_state].load_state_dict(self._q_functions[automaton_state].state_dict())

    def _get_expected_state_action_values(self, automaton_state, next_states_v, observations):
        hierarchy = self._get_hierarchy(0)
        automaton = hierarchy.get_root_automaton()

        def _get_update_info(obs):
            next_automaton_state = hierarchy.get_next_hierarchy_state(
                HierarchyState(automaton_state, automaton.get_name(), TRUE, [], []), obs
            ).state_name
            return (
                next_automaton_state,
                1.0 if automaton.is_accept_state(next_automaton_state) else 0.0,
                automaton.is_terminal_state(next_automaton_state)
            )

        next_automaton_states, rewards, is_terminal = zip(*[_get_update_info(obs) for obs in observations])
        rewards_v = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        is_terminal_v = torch.tensor(is_terminal, dtype=torch.bool, device=self.device)
        next_state_action_values = torch.zeros(self._er_batch_size, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            distinct_next_automaton_states = sorted(set(next_automaton_states))
            for next_automaton_state in distinct_next_automaton_states:
                if automaton.is_terminal_state(next_automaton_state):
                    continue  # Avoid doing forward passes

                q_function = self._q_functions[next_automaton_state]
                tgt_q_function = self._tgt_q_functions[next_automaton_state]
                indices = np.array(next_automaton_states) == next_automaton_state

                if self._use_double_dqn:
                    next_state_actions = q_function(next_states_v).max(1)[1]
                    next_state_action_values[indices] = tgt_q_function(next_states_v).gather(1, next_state_actions.unsqueeze(-1)).squeeze(-1)[indices]
                else:
                    next_state_action_values[indices] = tgt_q_function(next_states_v).max(1)[0][indices]

        next_state_action_values[is_terminal_v] = 0.0
        return rewards_v + self._discount_rate * next_state_action_values

    def _get_random_automaton_state_set(self):
        automaton = self._get_hierarchy(0).get_root_automaton()
        states = [s for s in automaton.get_states() if not automaton.is_terminal_state(s)]
        if self._num_updated_q_functions > len(states):
            return states
        return random.sample(states, self._num_updated_q_functions)

    def _export_models(self):
        super()._export_models()
        torch.save({
            state: q_function.state_dict()
            for state, q_function in self._q_functions.items()
        }, self._get_models_path())

    def _get_models_path(self):
        return os.path.join(self.models_folder, "models.pt")
