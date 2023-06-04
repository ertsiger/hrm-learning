from config.generators.rl.ihsa.ihsa_config_generator import IHSARLConfigGenerator
from reinforcement_learning.ihsa_hrl_algorithm import IHSAAlgorithmHRL
from reinforcement_learning.ihsa_hrl_dqn_algorithm import FormulaBank, FormulaBankDQN, IHSAAlgorithmHRLDQN
from reinforcement_learning.learning_algorithm import LearningAlgorithm


class IHSAHRLConfigGenerator(IHSARLConfigGenerator):
    @classmethod
    def _add_ihsa_rl_args(cls, parser):
        group = parser.add_argument_group("ihsa_hrl")
        group.add_argument(
            "--lr",
            type=float, default=5e-4,
            help="value of learning rate for the lowest level policies (default: 5e-4)"
        )
        group.add_argument(
            "--meta_lr",
            type=float, default=5e-4,
            help="value of the learning rate for the metacontrollers (default: 5e-4)"
        )
        group.add_argument(
            "--discount",
            type=float, default=0.9,
            help="value of the discount factor for the lowest level policies (default: 0.9)"
        )
        group.add_argument(
            "--meta_discount",
            type=float, default=0.99,
            help="value of the discount factor for the metacontrollers (default: 0.99)"
        )
        group.add_argument(
            "--rw_goal",
            type=float, default=1.0,
            help="reward given for reaching a local goal (satisfy a formula for the lowest level policies, or reach the"
                 "accepting state in an automaton) (default: 1.0)"
        )
        group.add_argument(
            "--rw_step",
            type=float, default=0.0,
            help="reward used to update the lowest level policies after each step (default: 0.0)"
        )
        group.add_argument(
            "--rw_deadend",
            type=float, default=0.0,
            help="reward used to update the lowest level policies upon reaching a deadend (default: 0.0)"
        )
        group.add_argument(
            "--er_size",
            type=int, default=500000,
            help="size of the experience replay buffer for the lowest level policies (default: 500000)"
        )
        group.add_argument(
            "--er_start",
            type=int, default=100000,
            help="size of the experience replay buffer after which learning will start for the lowest level policies"
                 " (default: 100000)"
        )
        group.add_argument(
            "--meta_er_size",
            type=int, default=10000,
            help="size of the experience replay buffer for the metacontrollers (default: 10000)"
        )
        group.add_argument(
            "--meta_er_start",
            type=int, default=1000,
            help="size of the experience replay buffer after which learning will start for the metacontrollers "
                 "(default: 1000)"
        )
        group.add_argument(
            "--tgt_net_update_freq",
            type=int, default=1500,
            help="after how many steps the target net is updated for the lowest level policies (default: 1500)"
        )
        group.add_argument(
            "--meta_tgt_net_update_freq",
            type=int, default=500,
            help="after how many steps the target net is updated for the metacontrollers (default: 500)"
        )
        group.add_argument(
            "--batch_size",
            type=int, default=32,
            help="batch size for update of lowest level policies (default: 32)"
        )
        group.add_argument(
            "--meta_batch_size",
            type=int, default=32,
            help="batch size for update of the metacontrollers (default: 32)"
        )
        group.add_argument(
            "--exp_subgoal_annealing_steps",
            type=int, default=1000000,
            help="for how many steps each subgoal's exploration factor is annealed from 1 to 0.1 (default: 1000000)"
        )
        group.add_argument(
            "--exp_automaton_annealing_steps",
            type=int, default=10000,
            help="for how many steps each automaton's exploration factor is annealed from 1 to 0.1 (default: 10000)"
        )
        group.add_argument(
            "--use_exploratory_formula_options",
            action="store_true",
            help="whether to use formula options for exploration when the current automaton state has no outgoing edges"
        )
        group.add_argument(
            "--use_exploratory_automata_options",
            action="store_true",
            help="whether to use automata options for exploration when the current automaton state has no outgoing edges"
        )
        group.add_argument(
            "--use_exploratory_action_options",
            action="store_true",
            help="whether to use primitive actions for exploration when the current automaton state has no outgoing edges"
        )
        group.add_argument(
            "--use_grad_clipping",
            action="store_true",
            help="whether to clip the gradients of the loss like in the DQN paper"
        )
        group.add_argument(
            "--grad_clipping_val",
            default=1.0,
            help="value with which the gradients of the loss are clipped (default: 1.0 if gradient clipping is enabled)"
        )
        group.add_argument(
            "--opt",
            default="rmsprop",
            help="the optimizer to be used when training neural networks (e.g. adam, rmsprop, sgd, sgd_momentum) (default: rmsprop)"
        )
        group.add_argument(
            "--q_select_num",
            type=int, default=None,
            help="the number of formula Q-functions to update at each step (default: all available)"
        )

    @classmethod
    def _set_rl_config(cls, config, args):
        config["algorithm"] = "ihsa-hrl"

        # General RL parameters
        config[FormulaBank.LEARNING_RATE] = args.lr
        config[IHSAAlgorithmHRL.META_LEARNING_RATE] = args.meta_lr
        config[FormulaBank.DISCOUNT_RATE] = args.discount
        config[IHSAAlgorithmHRL.META_DISCOUNT_RATE] = args.meta_discount

        # Exploration
        config[IHSAAlgorithmHRL.EXPLORATION_RATE_PER_SUBGOAL_STEPS] = args.exp_subgoal_annealing_steps
        config[IHSAAlgorithmHRL.EXPLORATION_RATE_AUTOMATON_STEPS] = args.exp_automaton_annealing_steps

        # Pseudoreward functions
        config[FormulaBank.PSEUDOREWARD_CONDITION_SATISFIED] = args.rw_goal
        config[FormulaBank.PSEUDOREWARD_DEADEND] = args.rw_deadend
        config[FormulaBank.PSEUDOREWARD_AFTER_STEP] = args.rw_step
        config[IHSAAlgorithmHRL.META_PSEUDOREWARD_CONDITION_SATISFIED] = args.rw_goal
        config[IHSAAlgorithmHRL.META_PSEUDOREWARD_DEADEND] = args.rw_deadend
        config[IHSAAlgorithmHRL.META_PSEUDOREWARD_AFTER_STEP] = args.rw_step

        # Subsets of formulas to update
        config[FormulaBank.FORMULA_UPDATE_SEL_NUM] = args.q_select_num

        # Function approximation case (state format in general)
        if args.state_format == IHSAAlgorithmHRL.STATE_FORMAT_FULL_OBS:
            config[FormulaBankDQN.ER_BUFFER_SIZE] = args.er_size
            config[FormulaBankDQN.ER_START_SIZE] = args.er_start
            config[IHSAAlgorithmHRLDQN.META_ER_BUFFER_SIZE] = args.meta_er_size
            config[IHSAAlgorithmHRLDQN.META_ER_START_SIZE] = args.meta_er_start
            config[FormulaBankDQN.TGT_UPDATE_FREQUENCY] = args.tgt_net_update_freq
            config[IHSAAlgorithmHRLDQN.META_TGT_UPDATE_FREQUENCY] = args.meta_tgt_net_update_freq
            config[FormulaBankDQN.ER_BATCH_SIZE] = args.batch_size
            config[IHSAAlgorithmHRLDQN.META_ER_BATCH_SIZE] = args.meta_batch_size

            if args.domain == LearningAlgorithm.ENV_NAME_CRAFTWORLD:
                config[FormulaBankDQN.NUM_CONV_CHANNELS] = tuple(args.cw_num_conv_channels)
                config[FormulaBankDQN.NUM_LINEAR_OUT_UNITS] = args.cw_linear_out_units
                config[FormulaBankDQN.USE_MAX_POOL] = args.cw_use_max_pool
                config[IHSAAlgorithmHRLDQN.META_NUM_CONV_CHANNELS] = tuple(args.cw_num_conv_channels)
                config[IHSAAlgorithmHRLDQN.META_USE_MAX_POOL] = args.cw_use_max_pool
            elif args.domain == LearningAlgorithm.ENV_NAME_WATERWORLD:
                config[FormulaBankDQN.NUM_LINEAR_OUT_UNITS] = args.ww_linear_out_units
                config[IHSAAlgorithmHRLDQN.META_NUM_LINEAR_OUT_UNITS] = args.ww_meta_linear_out_units

            config[IHSAAlgorithmHRLDQN.USE_GRAD_CLIPPING] = args.use_grad_clipping
            if args.use_grad_clipping:
                config[IHSAAlgorithmHRLDQN.GRAD_CLIPPING_VAL] = args.grad_clipping_val
            config[IHSAAlgorithmHRLDQN.OPTIMIZER_CLASS] = args.opt
        elif args.state_format != IHSAAlgorithmHRL.STATE_FORMAT_TABULAR:
            raise RuntimeError(f"Error: Unknown state format {args.state_format}.")

        # Exploration using learned options
        config[IHSAAlgorithmHRL.OPTION_EXPLORATION_INCLUDE_FORMULAS] = args.use_exploratory_formula_options
        config[IHSAAlgorithmHRL.OPTION_EXPLORATION_INCLUDE_AUTOMATA] = args.use_exploratory_automata_options
        config[IHSAAlgorithmHRL.OPTION_EXPLORATION_INCLUDE_ACTIONS] = args.use_exploratory_action_options


if __name__ == "__main__":
    IHSAHRLConfigGenerator.run()

