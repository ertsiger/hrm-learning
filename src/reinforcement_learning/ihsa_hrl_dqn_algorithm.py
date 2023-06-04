import collections
from gym_hierarchical_subgoal_automata.automata.condition import FormulaCondition
from gym_hierarchical_subgoal_automata.automata.logic import DNFFormula
import numpy as np
from reinforcement_learning.ihsa_hrl_algorithm import FormulaBank, IHSAAlgorithmHRL
from reinforcement_learning.model import MlpDQN, MinigridFormulaDQN, MinigridMetaDQN, WaterWorldFormulaDQN, WaterWorldMetaDQN
from reinforcement_learning.replay import ExperienceBuffer
import torch
from typing import Dict, Set, Tuple
from utils.container_utils import get_param
from utils.math_utils import randargmax

OPTIMIZER_ADAM = "adam"
OPTIMIZER_RMSPROP = "rmsprop"
OPTIMIZER_SGD = "sgd"
OPTIMIZER_SGD_MOMENTUM = "sgd_momentum"

# Experience tuple used to update the DQNs of the formula options
Experience = collections.namedtuple(
    "Experience",
    field_names=["state", "action", "next_state", "is_terminal", "is_goal_achieved", "observations"]
)

# Experience tuple used to update the DQNs of the metacontrollers
OptionExperience = collections.namedtuple(
    "OptionExperience",
    field_names=["state", "automaton_state", "context", "option", "next_state", "next_automaton_state", "next_context",
                 "reward", "num_steps"]
)


def init_optimizer(optimizer_class, model: torch.nn.Module, learning_rate):
    if optimizer_class == OPTIMIZER_ADAM or optimizer_class is None:
        return torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_class == OPTIMIZER_RMSPROP:
        return torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    elif optimizer_class == OPTIMIZER_SGD:
        return torch.optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer_class == OPTIMIZER_SGD_MOMENTUM:
        return torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    raise RuntimeError(f"Error: Unknown optimizer '{optimizer_class}'.")


def clip_grad_value(model: torch.nn.Module, grad_clipping_val: float):
    # From https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    # for param in model.parameters():
    #     param.grad.data.clamp_(-1, 1)
    torch.nn.utils.clip_grad_value_(model.parameters(), grad_clipping_val)


class FormulaBankDQN(FormulaBank):
    ER_BUFFER_SIZE = "er_buffer_size"  # size of the ER buffer for the lowest level policies
    ER_START_SIZE = "er_start_size"    # size of the ER for lowest level policies after which learning starts
    ER_BATCH_SIZE = "er_batch_size"    # size of the batches sampled from the ER buffers

    TGT_UPDATE_FREQUENCY = "tgt_update_freq"  # how many steps happen between target DQN updates (lowest level policies)

    NUM_CONV_CHANNELS = "dqn_num_conv_channels"    # number of output channels for each conv layer
    NUM_LINEAR_OUT_UNITS = "dqn_linear_out_units"  # number of output units for all layers (except for the last)
    USE_MAX_POOL = "dqn_use_max_pool"              # whether to use maxpooling

    def __init__(self, params, ignore_empty_observations, env_name, observation_format, optimizer_class, use_double_dqn,
                 use_grad_clipping, grad_clipping_val, device):
        super(FormulaBankDQN, self).__init__(params, ignore_empty_observations)

        self._env_name = env_name
        self._observation_format = observation_format
        self._optimizer_class = optimizer_class
        self._use_double_dqn = use_double_dqn
        self._use_grad_clipping = use_grad_clipping
        self._grad_clipping_val = grad_clipping_val
        self._device = device

        if self._env_name == IHSAAlgorithmHRLDQN.ENV_NAME_CRAFTWORLD:
            self._num_conv_channels = get_param(params, FormulaBankDQN.NUM_CONV_CHANNELS, (16, 32, 32))
            self._num_linear_out_units = get_param(params, FormulaBankDQN.NUM_LINEAR_OUT_UNITS, (64))
            self._use_max_pool = get_param(params, FormulaBankDQN.USE_MAX_POOL, False)
        elif self._env_name == IHSAAlgorithmHRLDQN.ENV_NAME_WATERWORLD:
            self._num_linear_out_units = get_param(params, FormulaBankDQN.NUM_LINEAR_OUT_UNITS, (1024, 1024, 1024))  # (64, 64, 64, 64, 64, 64)

        self._q_functions: Dict[FormulaCondition, torch.nn.Module] = {}
        self._tgt_q_functions: Dict[FormulaCondition, torch.nn.Module] = {}
        self._optimizers: Dict[FormulaCondition, torch.optim.Optimizer] = {}

        self._er_buffer = ExperienceBuffer(get_param(params, FormulaBankDQN.ER_BUFFER_SIZE, 500000))
        self._er_start_size = get_param(params, FormulaBankDQN.ER_START_SIZE, 100000)
        self._er_batch_size = get_param(params, FormulaBankDQN.ER_BATCH_SIZE, 32)

        self._tgt_update_freq = get_param(params, FormulaBankDQN.TGT_UPDATE_FREQUENCY, 1500)

    def _get_q_function_keys(self):
        return list(self._q_functions.keys())

    def _has_q_function(self, formula_condition: FormulaCondition):
        return formula_condition in self._q_functions

    def _copy_q_function(self, from_condition: FormulaCondition, to_condition: FormulaCondition, task):
        if not self._has_q_function(to_condition):
            self._init_q_function(to_condition, task)
        super(FormulaBankDQN, self)._copy_q_function(from_condition, to_condition, task)
        self._q_functions[to_condition].load_state_dict(self._q_functions[from_condition].state_dict())
        self._tgt_q_functions[to_condition].load_state_dict(self._tgt_q_functions[from_condition].state_dict())
        self._optimizers[to_condition].load_state_dict(self._optimizers[from_condition].state_dict())

    def _rm_q_function(self, formula_condition: FormulaCondition):
        super(FormulaBankDQN, self)._rm_q_function(formula_condition)
        self._q_functions.pop(formula_condition)
        self._tgt_q_functions.pop(formula_condition)
        self._optimizers.pop(formula_condition)

    def _init_q_function(self, formula_condition: FormulaCondition, task):
        super(FormulaBankDQN, self)._init_q_function(formula_condition, task)

        self._q_functions[formula_condition] = self._create_dqn(task)
        self._q_functions[formula_condition].to(self._device)

        self._tgt_q_functions[formula_condition] = self._create_dqn(task)
        self._tgt_q_functions[formula_condition].to(self._device)
        self._tgt_q_functions[formula_condition].load_state_dict(self._q_functions[formula_condition].state_dict())

        self._optimizers[formula_condition] = init_optimizer(
            self._optimizer_class, self._q_functions[formula_condition], self._learning_rate
        )

    def _create_dqn(self, task):
        if self._env_name == IHSAAlgorithmHRLDQN.ENV_NAME_CRAFTWORLD:
            if self._observation_format == IHSAAlgorithmHRLDQN.STATE_FORMAT_ONE_HOT:
                return MlpDQN(task.observation_space.n, task.action_space.n)
            elif self._observation_format == IHSAAlgorithmHRLDQN.STATE_FORMAT_FULL_OBS:
                return MinigridFormulaDQN(task.observation_space.shape, task.action_space.n, self._num_conv_channels,
                                          self._num_linear_out_units, self._use_max_pool)
        elif self._env_name == IHSAAlgorithmHRLDQN.ENV_NAME_WATERWORLD:
            return WaterWorldFormulaDQN(task.observation_space.shape, task.action_space.n, self._num_linear_out_units)
        else:
            raise RuntimeError(f"Error: Unknown environment '{self._env_name}'.")

    def update_q_functions(self, task, state, action, next_state, is_terminal, is_goal_achieved, observation):
        super(FormulaBankDQN, self).update_q_functions(task, state, action, next_state, is_terminal, is_goal_achieved, observation)

        # Add experience to the replay buffer
        self._er_buffer.append(Experience(state, action, next_state, is_terminal, is_goal_achieved, observation))

        if len(self._er_buffer) >= self._er_start_size:
            self._update_q_functions_helper(
                self._er_buffer.sample(self._er_batch_size)
            )
            self._inc_num_update_calls()

    def _update_q_functions_helper(self, experience_batch):
        states, actions, next_states, is_terminal, is_goal_achieved, observations = zip(*experience_batch)

        states_v = torch.tensor(np.array(states), dtype=torch.float32, device=self._device)
        actions_v = torch.tensor(np.array(actions), dtype=torch.long, device=self._device)
        next_states_v = torch.tensor(np.array(next_states), dtype=torch.float32, device=self._device)

        for condition in self._get_subgoals_to_update():
            # Get the rewards and terminal flags for each batch item and unpack them into different containers.
            rewards, is_terminal_local = zip(*[
                self._get_subgoal_pseudoreward(condition, observations[i], is_terminal[i], is_goal_achieved[i])
                for i in range(self._er_batch_size)
            ])

            rewards_v = torch.tensor(np.array(rewards), dtype=torch.float32, device=self._device)
            is_terminal_v = torch.tensor(np.array(is_terminal_local), dtype=torch.bool, device=self._device)

            net = self._q_functions[condition]
            target_net = self._tgt_q_functions[condition]

            state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)

            with torch.no_grad():
                if self._use_double_dqn:
                    next_state_actions = net(next_states_v).max(1)[1]
                    next_state_action_values = target_net(next_states_v).gather(1, next_state_actions.unsqueeze(-1)).squeeze(-1)
                else:
                    next_state_action_values = target_net(next_states_v).max(1)[0]
                next_state_action_values[is_terminal_v] = 0.0

            expected_state_action_values = rewards_v + self._discount_rate * next_state_action_values
            loss = torch.nn.MSELoss()(state_action_values, expected_state_action_values)

            self._optimizers[condition].zero_grad()
            loss.backward()
            if self._use_grad_clipping:
                clip_grad_value(net, self._grad_clipping_val)
            self._optimizers[condition].step()

            self._q_function_update_counter[condition] += 1
            if self._q_function_update_counter[condition] % self._tgt_update_freq == 0:
                target_net.load_state_dict(net.state_dict())

    def get_q_function(self, formula_condition: FormulaCondition):
        root_formula_condition = self.get_root(formula_condition).get_formula_condition()
        return self._q_functions[root_formula_condition]

    def _export_bank_helper(self, bank_obj, export_path):
        torch.save(bank_obj, export_path)

    def _import_bank_helper(self, import_path):
        return torch.load(import_path)

    def _load_q_function(self, formula_condition: FormulaCondition, q_function):
        self._q_functions[formula_condition] = q_function


class IHSAAlgorithmHRLDQN(IHSAAlgorithmHRL):
    """
    Implements the options' policies using DQNs.
    """
    USE_DOUBLE_DQN = "use_double_dqn"            # whether double DQN is used instead of simple DQN
    USE_GRAD_CLIPPING = "dqn_use_grad_clipping"  # whether to the clip gradients
    GRAD_CLIPPING_VAL = "dqn_grad_clipping_val"  # value at which the gradient is clipped
    OPTIMIZER_CLASS = "dqn_optimizer"            # which optimizer to use (adam, rmsprop)

    META_ER_BUFFER_SIZE = "meta_er_buffer_size"  # size of the ER for the higher level policies over options
    META_ER_START_SIZE = "meta_er_start_size"    # size of the ER for the higher level policies after which learning starts
    META_ER_BATCH_SIZE = "meta_er_batch_size"    # size of the batches sampled from the ER buffers

    META_TGT_UPDATE_FREQUENCY = "meta_tgt_update_freq"  # how many steps happen between target DQN updates (metacontrollers)

    META_NUM_CONV_CHANNELS = "meta_dqn_num_conv_channels"
    META_NUM_LINEAR_OUT_UNITS = "meta_dqn_num_linear_out_units"
    META_USE_MAX_POOL = "meta_dqn_use_max_pool"

    EXPORT_MODEL_EXTENSION = "pt"

    def __init__(self, params):
        super().__init__(params)

        # General DQN parameters
        self._use_double_dqn = get_param(params, IHSAAlgorithmHRLDQN.USE_DOUBLE_DQN, True)
        self._use_grad_clipping = get_param(params, IHSAAlgorithmHRLDQN.USE_GRAD_CLIPPING, False)
        self._grad_clipping_val = get_param(params, IHSAAlgorithmHRLDQN.GRAD_CLIPPING_VAL, 1.0)
        self._optimizer_class = get_param(params, IHSAAlgorithmHRLDQN.OPTIMIZER_CLASS, OPTIMIZER_RMSPROP)

        if self.env_name == IHSAAlgorithmHRLDQN.ENV_NAME_CRAFTWORLD:
            self._num_conv_channels = get_param(params, IHSAAlgorithmHRLDQN.META_NUM_CONV_CHANNELS, (16, 32, 32))
            self._use_max_pool = get_param(params, IHSAAlgorithmHRLDQN.META_USE_MAX_POOL, False)
        elif self.env_name == IHSAAlgorithmHRLDQN.ENV_NAME_WATERWORLD:
            self._num_linear_out_units = get_param(params, IHSAAlgorithmHRLDQN.META_NUM_LINEAR_OUT_UNITS, (256, 256))  # (64, 64, 64, 64, 64, 64)

        # Frequency with which the target networks are updated
        self._meta_tgt_net_update_freq = get_param(params, IHSAAlgorithmHRLDQN.META_TGT_UPDATE_FREQUENCY, 500)

        # Experience replay parameters
        self._meta_er_batch_size = get_param(params, IHSAAlgorithmHRLDQN.META_ER_BATCH_SIZE, 32)
        self._meta_er_buffer_size = get_param(params, IHSAAlgorithmHRLDQN.META_ER_BUFFER_SIZE, 5000)
        self._meta_er_start_size = get_param(params, IHSAAlgorithmHRLDQN.META_ER_START_SIZE, 5000)

        # Q-functions for the formulas
        self._formula_bank = FormulaBankDQN(
            params, self.ignore_empty_observations, self.env_name, self.observation_format, self._optimizer_class,
            self._use_double_dqn, self._use_grad_clipping, self._grad_clipping_val, self.device
        )
        self._init_formula_q_functions()

        # Q-functions for the metacontrollers
        self._meta_q_functions: Dict[str, torch.nn.Module] = {}
        self._tgt_meta_q_functions: Dict[str, torch.nn.Module] = {}
        self._meta_optimizers: Dict[str, torch.optim.Optimizer] = {}
        self._tgt_meta_update_counter: Dict[str, int] = {}
        self._meta_replay_buffers: Dict[str, ExperienceBuffer] = {}
        self._automaton_step_counter: Dict[str, Dict[Tuple[str, FormulaCondition], int]] = {}
        self._init_meta_q_functions()
        self._init_meta_replay_buffers()

    '''
    Formula Options Management
    '''
    def _get_policy_bank(self, task_id):
        return self._formula_bank

    def _get_policy_banks(self):
        return [self._formula_bank]

    def _export_policy_banks(self):
        self._formula_bank.export_bank(self._get_formula_bank_model_path(IHSAAlgorithmHRLDQN.EXPORT_MODEL_EXTENSION))

    def _import_policy_banks(self):
        self._formula_bank.import_bank(self._get_formula_bank_model_path(IHSAAlgorithmHRLDQN.EXPORT_MODEL_EXTENSION))

    def _on_initial_observation(self, observation):
        # We just pick a task that will exist for any execution here (all of them should have the same state and action
        # spaces).
        self._formula_bank.on_task_observation(observation, self._get_task(0, 0))

    '''
    Meta Q-Functions Management
    '''
    def _get_greedy_condition(self, task_id, state, automaton, automaton_state, context, hierarchy, ignore_rej_edges):
        # We assume the set of observables is the same across all domains and tasks, so we just take it from any task to
        # compute the formula embeddings later.
        observables = self._get_task(0, task_id).get_observables()

        state_v = torch.tensor(np.array([state]), dtype=torch.float32, device=self.device)
        automaton_state_v = torch.tensor(
            np.array([automaton.get_state_embedding(automaton_state)]),
            dtype=torch.float32,
            device=self.device
        )
        context_v = torch.tensor(np.array([context.get_embedding(observables)]), dtype=torch.float32, device=self.device)
        unsat_mask_v = torch.tensor(
            np.array([automaton.get_automaton_subgoals_sat_mask(
                automaton_state, context, self._get_policy_bank(task_id).get_observations(), hierarchy
            )]),
            dtype=torch.bool,
            device=self.device
        ).logical_not()

        if ignore_rej_edges:
            rej_mask_v = torch.tensor(
                np.array([automaton.get_rejecting_outgoing_conditions_mask(automaton_state)]),
                dtype=torch.bool,
                device=self.device
            )
        else:
            rej_mask_v = torch.zeros(unsat_mask_v.shape, dtype=torch.bool, device=self.device)

        with torch.no_grad():
            q_values = self._get_state_option_values(
                self._meta_q_functions[automaton.get_name()], state_v, automaton_state_v, context_v
            )

            # Apply masks. The final one, is the most important to be respected since it avoids picking options that are
            # not available in the current automaton state (the ones above just express preferences).
            q_values[rej_mask_v] = 0.1 * IHSAAlgorithmHRLDQN.Q_VALUE_NEG_MASK
            q_values[unsat_mask_v] = IHSAAlgorithmHRLDQN.Q_VALUE_NEG_MASK

        option_index = randargmax(q_values.cpu().numpy())
        option = automaton.get_automaton_subgoals()[option_index]

        cond_to_target_autom_state = {cond: state for cond, state in automaton.get_outgoing_conditions(automaton_state)}
        tgt_automaton_state = cond_to_target_autom_state[option]

        return option, tgt_automaton_state

    def _get_automaton_state_step_count(self, task_id, automaton_name, automaton_state, context):
        if automaton_name not in self._automaton_step_counter or (automaton_state, context) not in self._automaton_step_counter[automaton_name]:
            return 0
        return self._automaton_step_counter[automaton_name][(automaton_state, context)]

    def _inc_automaton_state_step_count(self, task_id, automaton_name, automaton_state, context):
        if automaton_name not in self._automaton_step_counter:
            self._automaton_step_counter[automaton_name] = {}
        if (automaton_state, context) not in self._automaton_step_counter[automaton_name]:
            self._automaton_step_counter[automaton_name][(automaton_state, context)] = 0
        self._automaton_step_counter[automaton_name][(automaton_state, context)] += 1

    def _reset_automaton_step_counters(self, automaton_name):
        if automaton_name in self._automaton_step_counter:
            self._automaton_step_counter[automaton_name].clear()

    def _init_hierarchy_meta_q_functions_for_hierarchy_state(self, domain_id, task_id, automaton_name, state_name, context):
        # The q-functions for automata different from the root should not be always reset, only when necessary (i.e.,
        # the entry does not exist, hence the default force_init value).
        self._init_meta_q_functions_for_automaton(domain_id, task_id, automaton_name)

    def _init_meta_q_functions_for_automaton(self, domain_id, task_id, automaton_name, force_init=False):
        if force_init or automaton_name not in self._meta_q_functions:
            task = self._get_task(domain_id, task_id)
            automaton = self._get_hierarchy(domain_id).get_automaton(automaton_name)

            # Only create the Q-function if there are options to choose from
            if automaton.get_num_automaton_subgoals() > 0:
                self._meta_q_functions[automaton_name] = self._create_meta_q_function(task, automaton)
                if not self.training_enable:
                    self._meta_q_functions[automaton_name].load_state_dict(torch.load(
                        self._get_metacontroller_file_path(automaton_name, IHSAAlgorithmHRLDQN.EXPORT_MODEL_EXTENSION)
                    ))
                self._meta_q_functions[automaton_name].to(self.device)

                self._tgt_meta_q_functions[automaton_name] = self._create_meta_q_function(task, automaton)
                self._tgt_meta_q_functions[automaton_name].to(self.device)
                self._tgt_meta_q_functions[automaton_name].load_state_dict(self._meta_q_functions[automaton_name].state_dict())

                self._meta_optimizers[automaton_name] = init_optimizer(
                    self._optimizer_class,
                    self._meta_q_functions[automaton_name],
                    self._meta_learning_rate
                )
                self._tgt_meta_update_counter[automaton_name] = 0

    def _create_meta_q_function(self, task, automaton):
        if self.env_name == IHSAAlgorithmHRLDQN.ENV_NAME_CRAFTWORLD:
            if self.observation_format == IHSAAlgorithmHRLDQN.STATE_FORMAT_ONE_HOT:
                return MlpDQN(
                    task.observation_space.n + automaton.get_num_states() + len(task.get_observables()),
                    automaton.get_num_automaton_subgoals()
                )
            elif self.observation_format == IHSAAlgorithmHRLDQN.STATE_FORMAT_FULL_OBS:
                return MinigridMetaDQN(task.observation_space.shape, automaton.get_num_states(), len(task.get_observables()),
                                       automaton.get_num_automaton_subgoals(), self._num_conv_channels, self._use_max_pool)
        elif self.env_name == IHSAAlgorithmHRLDQN.ENV_NAME_WATERWORLD:
            return WaterWorldMetaDQN(task.observation_space.shape, automaton.get_num_states(), len(task.get_observables()),
                                     automaton.get_num_automaton_subgoals(), self._num_linear_out_units)
        else:
            raise RuntimeError(f"Error: Unknown environment '{self.env_name}'.")

    def _init_meta_replay_buffers(self):
        for domain_id in range(self.num_domains):
            for automaton_name in self._get_hierarchy(domain_id).get_automata_names():
                self._meta_replay_buffers[automaton_name] = ExperienceBuffer(self._meta_er_buffer_size)

    def _reset_meta_experience_replay_and_target_update_counter(self, domain_id):
        """
        Removes all accumulated experiences in the replay buffers for a given domain and restarts the target update
        counter. When a new automaton is learned, the learned policies over options are no longer useful: we will need
        to learn from other experiences, so the accumulated ones are not worth keeping.
        """
        root_automaton_name = self._get_hierarchy(domain_id).get_root_automaton().get_name()
        self._meta_replay_buffers[root_automaton_name].clear()
        self._tgt_meta_update_counter[root_automaton_name] = 0

    def _update_meta_q_functions(self, domain_id, task_id, next_state, is_terminal, option, hierarchy, observation, satisfied_automata):
        self._update_meta_replay_buffer(task_id, next_state, is_terminal, option, hierarchy, observation, satisfied_automata)

        replay_buffer = self._meta_replay_buffers[option.get_automaton_name()]
        if len(replay_buffer) >= self._meta_er_start_size:
            self._update_deep_meta_q_functions_from_batch(
                domain_id, task_id, option.get_automaton_name(), replay_buffer.sample(self._meta_er_batch_size)
            )

    def _update_meta_replay_buffer(self, task_id, next_state, is_terminal, option, hierarchy, observation, satisfied_automata):
        automaton = hierarchy.get_automaton(option.get_automaton_name())
        automaton_subgoals = automaton.get_automaton_subgoals()

        # If there are no subgoals to choose from, then there is nothing to learn so we can keep the buffer empty.
        #  - This should correspond to the case in which we have an automaton without edges (i.e., the initial state
        #    itself is a dead-end), and in this case we just choose an action at random.
        # If the option has not been successful (i.e., it has not achieved its associated subgoal), we don't add the
        # experience to the replay buffer. Otherwise, we add a source of non-stationarity (the same option succeeds,
        # reaches dead-ends, finishes with the end of an episode, ...).
        if len(automaton_subgoals) == 0 or not self._is_option_successful(option, observation, satisfied_automata):
            return

        # The updated automaton state is not the one in the hierarchy but in the same automaton where the option acts.
        automaton_state = option.get_automaton_state()
        next_automaton_state = self._get_next_automaton_state(option, observation, satisfied_automata)
        next_context = self._get_next_context(automaton_state, option.get_context(), next_automaton_state)

        experience = OptionExperience(
            option.get_start_state(),
            automaton_state,
            option.get_context(),
            automaton_subgoals.index(option.get_base_condition()),  # Use base condition since subgoals are contextless,
            next_state,
            next_automaton_state,
            next_context,
            self._get_option_pseudoreward(option, hierarchy, observation, satisfied_automata),
            option.get_num_steps()
        )
        self._meta_replay_buffers[automaton.get_name()].append(experience)

    def _update_deep_meta_q_functions_from_batch(self, domain_id, task_id, automaton_name, experience_batch):
        observables = self._get_task(domain_id, task_id).get_observables()  # All domains should have the same set of observables (assumption)
        hierarchy = self._get_hierarchy(domain_id)
        automaton = hierarchy.get_automaton(automaton_name)
        observations = self._get_policy_bank(task_id).get_observations()

        # Unpack the batch
        states, automaton_states, contexts, option_ids, next_states, next_automaton_states, next_contexts, rewards, \
        num_steps = zip(*experience_batch)

        # Transform the arrays into tensors. Note that the SAT masks are not added to the buffer, but derived from the
        # buffer's content instead. While we could add these masks to the buffer, we don't because they might change
        # during learning as we see more and more observations. If we keep all of them in the buffer, we are training
        # using non-stationary data (i.e., two option experiences could be exactly the same except for the mask). Thus,
        # it's better in learning terms to recompute these masks every time even if it's a bit inefficient. The need for
        # this motivates the need for the automaton states and contexts to be stored using their normal form instead of
        # their embeddings directly.
        # In the case of the is_terminal, we trust on the automaton to indicate this rather than obeying what the
        # environment says. If the automaton is wrong, it will be a counterexample anyways and we will learn another one.
        automaton_states_e, next_automaton_states_e = [], []
        contexts_e, next_contexts_e = [], []
        is_terminal_e = []
        next_options_sat_masks = []
        for i in range(self._meta_er_batch_size):
            automaton_states_e.append(automaton.get_state_embedding(automaton_states[i]))
            next_automaton_states_e.append(automaton.get_state_embedding(next_automaton_states[i]))
            contexts_e.append(contexts[i].get_embedding(observables))
            next_contexts_e.append(next_contexts[i].get_embedding(observables))
            is_terminal_e.append(automaton.is_terminal_state(next_automaton_states[i]))
            next_options_sat_masks.append(automaton.get_automaton_subgoals_sat_mask(
                next_automaton_states[i], next_contexts[i], observations, hierarchy
            ))

        states_v = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        options_v = torch.tensor(np.array(option_ids), dtype=torch.long, device=self.device)
        next_states_v = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        rewards_v = torch.tensor(np.array(rewards), dtype=torch.float32, device=self.device)
        num_steps_v = torch.tensor(np.array(num_steps), dtype=torch.float32, device=self.device)
        automaton_states_v = torch.tensor(np.array(automaton_states_e), dtype=torch.float32, device=self.device)
        contexts_v = torch.tensor(np.array(contexts_e), dtype=torch.float32, device=self.device)
        next_automaton_states_v = torch.tensor(np.array(next_automaton_states_e), dtype=torch.float32, device=self.device)
        is_terminal_v = torch.tensor(np.array(is_terminal_e), dtype=torch.bool, device=self.device)
        next_contexts_v = torch.tensor(np.array(next_contexts_e), dtype=torch.float32, device=self.device)
        next_options_sat_masks_v = torch.tensor(np.array(next_options_sat_masks), dtype=torch.float32, device=self.device)

        # Get Q-values for all options at the current state
        net = self._meta_q_functions[automaton_name]

        state_option_values = self._get_state_option_values(net, states_v, automaton_states_v, contexts_v)
        state_option_values = state_option_values.gather(1, options_v.unsqueeze(-1)).squeeze(-1)

        # Get target Q-values masking those options not available at the given (state, automaton state) with very
        # negative values in order not to select them
        target_net = self._tgt_meta_q_functions[automaton_name]

        with torch.no_grad():
            if self._use_double_dqn:
                next_state_option_values = self._get_state_option_values(net, next_states_v, next_automaton_states_v, next_contexts_v)
                next_state_option_values += IHSAAlgorithmHRLDQN.Q_VALUE_NEG_MASK * (1.0 - next_options_sat_masks_v)
                next_state_options = next_state_option_values.max(1)[1]
                next_state_option_values = self._get_state_option_values(target_net, next_states_v, next_automaton_states_v, next_contexts_v)
                next_state_option_values = next_state_option_values.gather(1, next_state_options.unsqueeze(-1)).squeeze(-1)
            else:
                next_state_option_values = self._get_state_option_values(target_net, next_states_v, next_automaton_states_v, next_contexts_v)
                next_state_option_values += IHSAAlgorithmHRLDQN.Q_VALUE_NEG_MASK * (1.0 - next_options_sat_masks_v)
                next_state_option_values = next_state_option_values.max(1)[0]
            next_state_option_values[is_terminal_v] = 0.0

        # SMDP Q-learning discount
        discount = self._meta_discount_rate ** num_steps_v

        expected_state_action_values = rewards_v + discount * next_state_option_values
        loss = torch.nn.MSELoss()(state_option_values, expected_state_action_values)

        self._meta_optimizers[automaton_name].zero_grad()
        loss.backward()
        if self._use_grad_clipping:
            clip_grad_value(net, self._grad_clipping_val)
        self._meta_optimizers[automaton_name].step()

        self._tgt_meta_update_counter[automaton_name] += 1
        if self._tgt_meta_update_counter[automaton_name] % self._meta_tgt_net_update_freq == 0:
            target_net.load_state_dict(net.state_dict())

    def _get_state_option_values(self, net, states_v, automaton_states_v, contexts_v):
        if self.observation_format == IHSAAlgorithmHRLDQN.STATE_FORMAT_ONE_HOT:
            return net(torch.cat((states_v, automaton_states_v, contexts_v)))
        elif self.observation_format == IHSAAlgorithmHRLDQN.STATE_FORMAT_FULL_OBS:
            return net(states_v, automaton_states_v, contexts_v)
        else:
            raise RuntimeError(f"Error: Unknown observation format '{self.observation_format}'.")

    def _export_meta_functions(self, automaton):
        # We need to perform this check since the meta Q-function is only created
        # once we have learned a first automaton (or we use the handcrafted automata setting)
        if automaton.get_name() in self._meta_q_functions:
            torch.save(
                self._meta_q_functions[automaton.get_name()].state_dict(),
                self._get_metacontroller_file_path(automaton.get_name(), IHSAAlgorithmHRLDQN.EXPORT_MODEL_EXTENSION)
            )

    '''
    Automaton Learning Management (what to do when an automaton is learned for a given domain)
    '''
    def _reset_q_functions(self, domain_id):
        """
        Rebuild Q-functions and reset the experience replay buffer for a domain whose root automaton has just been
        (re)learned. The experience replay and the policies cannot be kept since they depend on the structure of the
        automaton.
        """
        super()._reset_q_functions(domain_id)
        self._reset_meta_experience_replay_and_target_update_counter(domain_id)
