from reinforcement_learning.ihsa_base_algorithm import IHSAAlgorithmBase
from reinforcement_learning.replay import ExperienceBuffer
from utils.container_utils import get_param


class IHSAAlgorithmCrossProduct(IHSAAlgorithmBase):
    """
    Base class for method based on learning a policy over the cross product of the environment's state space and the
    state space of the exploited automaton (or reward machine).
    """
    # General parameters
    LEARNING_RATE = "learning_rate"
    DISCOUNT_RATE = "discount_rate"
    USE_DOUBLE_DQN = "use_double_dqn"
    TGT_UPDATE_FREQUENCY = "tgt_update_freq"

    # Exploration parameters
    EXPLORATION_RATE_ANNEALING_INIT = "exploration_rate_annealing_init"              # the degree of exploration at the beginning of training
    EXPLORATION_RATE_ANNEALING_END = "exploration_rate_annealing_end"                # the minimum degree of exploration
    EXPLORATION_RATE_ANNEALING_DURATION = "exploration_rate_annealing_duration"      # for how long is the exploration rate decreased
    EXPLORATION_RATE_ANNEALING_TIMESCALE = "exploration_rate_annealing_timescale"    # whether the annealing is based on the number of episodes or steps
    EXPLORATION_RATE_ANNEALING_TIMESCALE_STEPS = "steps"
    EXPLORATION_RATE_ANNEALING_TIMESCALE_EPISODES = "episodes"

    # Experience replay parameters
    ER_BUFFER_SIZE = "er_buffer_size"
    ER_START_SIZE = "er_start_size"
    ER_BATCH_SIZE = "er_batch_size"
    SCALE_ER_NUM_STATES = "scale_er_num_states"  # Whether the batch size and buffer size are scaled by number of states in the HRM

    # CraftWorld parameters
    USE_MAX_POOL = "use_max_pool"

    # WaterWorld parameters
    LAYER_SIZE = "layer_size"

    def __init__(self, params):
        super().__init__(params)

        if self.num_domains > 1:
            raise RuntimeError("Error: The algorithm is currently only applicable to a single domain.")
        if len(self._get_hierarchy(domain_id=0).get_automata_names()) > 1:
            raise RuntimeError("Error: The hierarchy consists of more than 1 automata.")
        if self.interleaved_automaton_learning:
            raise RuntimeError("Error: Only handcrafted automata are supported at the moment.")

        self._learning_rate = get_param(params, IHSAAlgorithmCrossProduct.LEARNING_RATE, 0.1)
        self._discount_rate = get_param(params, IHSAAlgorithmCrossProduct.DISCOUNT_RATE, 0.99)
        self._use_double_dqn = get_param(params, IHSAAlgorithmCrossProduct.USE_DOUBLE_DQN, True)
        self._tgt_update_freq = get_param(params, IHSAAlgorithmCrossProduct.TGT_UPDATE_FREQUENCY, 1500)
        self._q_function_update_counter = 0

        scale_er_by_num_states = get_param(params, IHSAAlgorithmCrossProduct.SCALE_ER_NUM_STATES, False)
        scale = self._get_hierarchy(domain_id=0).get_root_automaton().get_num_states() if scale_er_by_num_states else 1
        self._er_buffer = ExperienceBuffer(scale * get_param(params, IHSAAlgorithmCrossProduct.ER_BUFFER_SIZE, 500000))
        self._er_start_size = get_param(params, IHSAAlgorithmCrossProduct.ER_START_SIZE, 100000)
        self._er_batch_size = scale * get_param(params, IHSAAlgorithmCrossProduct.ER_BATCH_SIZE, 32)

        self._total_num_steps = 0
        self._exploration_rate_init = get_param(params, IHSAAlgorithmCrossProduct.EXPLORATION_RATE_ANNEALING_INIT, 1.0)
        self._exploration_rate_final = get_param(params, IHSAAlgorithmCrossProduct.EXPLORATION_RATE_ANNEALING_END, 0.1)
        self._exploration_rate_annealing_timescale = get_param(
            params, IHSAAlgorithmCrossProduct.EXPLORATION_RATE_ANNEALING_TIMESCALE,
            IHSAAlgorithmCrossProduct.EXPLORATION_RATE_ANNEALING_TIMESCALE_EPISODES
        )
        self._exploration_rate_annealing_duration = get_param(
            params, IHSAAlgorithmCrossProduct.EXPLORATION_RATE_ANNEALING_DURATION, 5000
        )

        if self.env_name == IHSAAlgorithmCrossProduct.ENV_NAME_CRAFTWORLD:
            self._use_maxpool = get_param(params, IHSAAlgorithmCrossProduct.USE_MAX_POOL, False)
        elif self.env_name == IHSAAlgorithmCrossProduct.ENV_NAME_WATERWORLD:
            self._layer_size = get_param(params, IHSAAlgorithmCrossProduct.LAYER_SIZE, 512)

    def _get_exploration_rate(self):
        if self._exploration_rate_annealing_timescale == IHSAAlgorithmCrossProduct.EXPLORATION_RATE_ANNEALING_TIMESCALE_STEPS:
            return self._get_annealed_exploration_rate(
                self._total_num_steps, self._exploration_rate_init, self._exploration_rate_final,
                self._exploration_rate_annealing_duration
            )
        elif self._exploration_rate_annealing_timescale == IHSAAlgorithmCrossProduct.EXPLORATION_RATE_ANNEALING_TIMESCALE_EPISODES:
            return self._get_annealed_exploration_rate(
                self._curriculum.get_current_episode(), self._exploration_rate_init, self._exploration_rate_final,
                self._exploration_rate_annealing_duration
            )
        raise RuntimeError(f"Error: Unknown timescale for exploration '{self._exploration_rate_annealing_timescale}'.")

    def _on_performed_step(
        self, domain_id, task_id, next_state, reward, is_terminal, observation, observation_changed, hierarchy,
        current_hierarchy_state, next_hierarchy_state, episode_length
    ):
        self._total_num_steps += 1

    def _on_initial_observation(self, observation):
        pass
