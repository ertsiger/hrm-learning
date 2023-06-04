from config.generators.rl.rl_config_generator import RLConfigGenerator
from reinforcement_learning.dqn_base_algorithm import DQNBaseAlgorithm
from reinforcement_learning.learning_algorithm import LearningAlgorithm


class DQNBaseRLConfigGenerator(RLConfigGenerator):
    @classmethod
    def _add_extra_args(cls, parser):
        group = parser.add_argument_group("dqn_base")
        group.add_argument(
            "--lr",
            type=float, default=5e-4,
            help="value of the learning rate (default: 5e-4)"
        )
        group.add_argument(
            "--discount",
            type=float, default=0.99,
            help="value of the discount factor (default: 0.99)"
        )
        group.add_argument(
            "--net_update_freq",
            type=int, default=1,
            help="with what frequency [timescale determined by --update_timescale] to update the network "
                 "(default: 1 step)"
        )
        group.add_argument(
            "--tgt_net_update_freq",
            type=int, default=1500,
            help="with what frequency [timescale determined by --update_timescale] to synchronize the target network "
                 "(default: 1500 steps)"
        )
        group.add_argument(
            "--update_timescale",
            default="step", choices=["episode", "step"],
            help="with what frequency updates are performed (after each step or episode)"
        )
        group.add_argument(
            "--batch_size",
            type=int, default=32,
            help="how many experiences to sample for each update of the network (default: 32, use 1 for DRQN)"
        )
        group.add_argument(
            "--er_size",
            type=int, default=500000,
            help="number of steps stores in the experience replay buffer (default: 500000, use around 1000 for DRQN)"
        )
        group.add_argument(
            "--er_start",
            type=int, default=100000,
            help="number of steps after which the network starts to be updated (default: 100000, use 100 for DRQN)"
        )
        group.add_argument(
            "--exp_annealing_duration",
            type=int, default=30000,
            help="for how long the annealing is performed (note that it depends on the chosen scale) (default: 30000 episodes)"
        )
        group.add_argument(
            "--exp_annealing_scale",
            default="episodes",
            choices=["episodes", "steps"],
            help="throughout what unit the annealing of the exploration factor is performed from 1 to 0.1 (default: episodes)"
        )

    @classmethod
    def _set_algorithm_config(cls, config, args):
        config[DQNBaseAlgorithm.LEARNING_RATE] = args.lr
        config[DQNBaseAlgorithm.DISCOUNT_RATE] = args.discount
        config[DQNBaseAlgorithm.NET_UPDATE_FREQUENCY] = args.net_update_freq
        config[DQNBaseAlgorithm.TGT_UPDATE_FREQUENCY] = args.tgt_net_update_freq
        config[DQNBaseAlgorithm.UPDATE_TIMESCALE] = args.update_timescale
        config[DQNBaseAlgorithm.ER_BATCH_SIZE] = args.batch_size
        config[DQNBaseAlgorithm.ER_BUFFER_SIZE] = args.er_size
        config[DQNBaseAlgorithm.ER_START_SIZE] = args.er_start
        config[DQNBaseAlgorithm.EXPLORATION_RATE_ANNEALING_DURATION] = args.exp_annealing_duration
        config[DQNBaseAlgorithm.EXPLORATION_RATE_ANNEALING_TIMESCALE] = args.exp_annealing_scale

        if args.domain == LearningAlgorithm.ENV_NAME_CRAFTWORLD:
            config[DQNBaseAlgorithm.USE_MAX_POOL] = args.cw_use_max_pool