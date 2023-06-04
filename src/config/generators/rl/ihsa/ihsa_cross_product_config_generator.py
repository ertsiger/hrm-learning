from config.generators.rl.ihsa.ihsa_config_generator import IHSARLConfigGenerator
from reinforcement_learning.ihsa_cross_product_algorithm import IHSAAlgorithmCrossProduct


class IHSACrossProductConfigGenerator(IHSARLConfigGenerator):
    @classmethod
    def _add_ihsa_rl_args(cls, parser):
        group = parser.add_argument_group("ihsa_cross_product")
        group.add_argument(
            "--lr",
            type=float, default=5e-4,
            help="value of learning rate (default: 5e-4)"
        )
        group.add_argument(
            "--discount",
            type=float, default=0.9,
            help="value of the discount factor (default: 0.9)"
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
        group.add_argument(
            "--er_size",
            type=int, default=500000,
            help="size of the experience replay buffer (default: 500000)"
        )
        group.add_argument(
            "--er_start",
            type=int, default=100000,
            help="size of the experience replay buffer after which learning will start (default: 100000)"
        )
        group.add_argument(
            "--batch_size",
            type=int, default=32,
            help="batch size for update of lowest level policies (default: 32)"
        )
        group.add_argument(
            "--tgt_net_update_freq",
            type=int, default=1500,
            help="every how many steps the target net is updated (default: 1500)"
        )
        group.add_argument(
            "--scale_er_num_states",
            action="store_true",
            help="whether the batch size and replay buffer size are multiplied by the number of states in the flat HRM"
        )
        group.add_argument(
            "--ww_layer_size",
            type=int, default=512,
            help="size of the linear layers used in the WaterWorld model (default: 512)"
        )

    @classmethod
    def _set_rl_config(cls, config, args):
        config[IHSAAlgorithmCrossProduct.LEARNING_RATE] = args.lr
        config[IHSAAlgorithmCrossProduct.DISCOUNT_RATE] = args.discount
        config[IHSAAlgorithmCrossProduct.ER_BUFFER_SIZE] = args.er_size
        config[IHSAAlgorithmCrossProduct.ER_START_SIZE] = args.er_start
        config[IHSAAlgorithmCrossProduct.ER_BATCH_SIZE] = args.batch_size
        config[IHSAAlgorithmCrossProduct.TGT_UPDATE_FREQUENCY] = args.tgt_net_update_freq
        config[IHSAAlgorithmCrossProduct.USE_MAX_POOL] = args.cw_use_max_pool
        config[IHSAAlgorithmCrossProduct.SCALE_ER_NUM_STATES] = args.scale_er_num_states
        config[IHSAAlgorithmCrossProduct.LAYER_SIZE] = args.ww_layer_size
        config[IHSAAlgorithmCrossProduct.EXPLORATION_RATE_ANNEALING_DURATION] = args.exp_annealing_duration
        config[IHSAAlgorithmCrossProduct.EXPLORATION_RATE_ANNEALING_TIMESCALE] = args.exp_annealing_scale
