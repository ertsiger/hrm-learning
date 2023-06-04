from abc import abstractmethod
from config.generators.rl.rl_config_generator import RLConfigGenerator
from reinforcement_learning.learning_algorithm import LearningAlgorithm
from reinforcement_learning.ihsa_base_algorithm import IHSAAlgorithmBase


class IHSARLConfigGenerator(RLConfigGenerator):
    @classmethod
    def _add_extra_args(cls, parser):
        group = parser.add_argument_group("ihsa_general")
        group.add_argument(
            "--mode",
            required=True,
            choices=["train-learn", "train-handcrafted", "test"],
            help="whether to train with automata learning (train-learn), train with handcrafted automata "
                 "(train-handcrafted) or test policies from the passed models folder (test)"
        )

        cls._add_ihsa_automaton_learning_args(parser)
        cls._add_ihsa_rl_args(parser)

    @classmethod
    def _add_ihsa_automaton_learning_args(cls, parser):
        group = parser.add_argument_group("ihsa_automaton_learning")
        group.add_argument(
            "--res_obs_enable", "-r",
            action="store_true",
            help="whether the hypothesis space only contains observables required by the task"
        )
        group.add_argument(
            "--filter_obs_enable",
            action="store_true",
            help="whether the traces only contain the observables required by the task"
        )
        group.add_argument(
            "--res_dep_enable",
            action="store_true",
            help="whether to only add to the hypothesis space those automata known to be required (i.e., known dependencies)"
        )
        group.add_argument(
            "--use_mutex_saturation",
            action="store_true",
            help="whether to use saturation for the mutual exclusivity constraints"
        )
        group.add_argument(
            "--root_rej_state_deepening",
            action="store_true",
            help="whether to use iterative deepening on the root rejecting state"
        )
        group.add_argument(
            "--min_goal_examples",
            type=int, nargs="+", default=[1],
            help="minimum number of goal examples that must be observed in order to learn an automaton"
        )
        group.add_argument(
            "--use_top_shortest_goal_examples",
            action="store_true",
            help="whether to use the top shortest goal examples for learning an automaton"
        )
        group.add_argument(
            "--num_top_shortest_goal_examples",
            type=int, nargs="+", default=[1],
            help="the number of shortest examples used to learn an automaton"
        )
        group.add_argument(
            "--ilasp_timeout",
            type=int, default=7200,
            help="ilasp total timeout in seconds (time used to learn automata) (default: 7200 seconds)"
        )
        group.add_argument(
            "--allow_suboptimal",
            action="store_true",
            help="whether the found hypotheses are allowed to be suboptimal"
        )
        group.add_argument(
            "--clingo_params",
            nargs='*', default=[],
            help="clingo parameters to use when ILASP runs (surround each parameters with quotes and add a space before"
                 " the -- to have them recognized, e.g. \" --opt-strat=usc,stratify\")"
        )

    @classmethod
    @abstractmethod
    def _add_ihsa_rl_args(cls, parser):
        raise NotImplementedError

    @classmethod
    def _set_algorithm_config(cls, config, args):
        if args.mode.startswith("train-"):
            config[LearningAlgorithm.TRAINING_ENABLE] = True
            config[IHSAAlgorithmBase.TRAINING_MODE] = IHSAAlgorithmBase.TRAINING_MODE_HANDCRAFTED if args.mode == "train-handcrafted" else IHSAAlgorithmBase.TRAINING_MODE_LEARN
        else:
            config[LearningAlgorithm.TRAINING_ENABLE] = False

        if args.mode == "train-learn":
            cls._set_interleaved_config(config, args)

        cls._set_rl_config(config, args)

    @classmethod
    def _set_interleaved_config(cls, config, args):
        config[IHSAAlgorithmBase.LEARNING_TIME_BUDGET] = args.ilasp_timeout
        config[IHSAAlgorithmBase.ILASP_FLAGS] = {
            "ilasp_version": "2",
            "ilasp_simplify_contexts": True,
            "ilasp_max_body_literals": 1,
            "clingo_params": args.clingo_params,
            "ilasp_find_optimal": not args.allow_suboptimal
        }
        config[IHSAAlgorithmBase.AVOID_LEARNING_ONLY_NEGATIVE] = True
        config[IHSAAlgorithmBase.LEARN_ACYCLIC_GRAPH] = True
        config[IHSAAlgorithmBase.SYMMETRY_BREAKING_ENABLE] = True
        config[IHSAAlgorithmBase.MAX_DISJUNCTION_SIZE] = 1
        config[IHSAAlgorithmBase.HYP_RESTRICT_OBSERVABLES] = args.res_obs_enable
        config[IHSAAlgorithmBase.FILTER_RESTRICTED_OBSERVABLES] = args.filter_obs_enable
        config[IHSAAlgorithmBase.HYP_RESTRICT_DEPENDENCIES] = args.res_dep_enable
        config[IHSAAlgorithmBase.USE_MUTEX_SATURATION] = args.use_mutex_saturation
        config[IHSAAlgorithmBase.ROOT_REJECTING_STATE_DEEPENING] = args.root_rej_state_deepening
        config[IHSAAlgorithmBase.MIN_GOAL_EXAMPLES] = args.min_goal_examples if len(args.min_goal_examples) > 1 else args.min_goal_examples[0]
        config[IHSAAlgorithmBase.USE_TOP_SHORTEST_GOAL_EXAMPLES] = args.use_top_shortest_goal_examples
        config[IHSAAlgorithmBase.NUM_TOP_SHORTEST_GOAL_EXAMPLES] = args.num_top_shortest_goal_examples if len(args.num_top_shortest_goal_examples) > 1 else args.num_top_shortest_goal_examples[0]

    @classmethod
    @abstractmethod
    def _set_rl_config(cls, config, args):
        raise NotImplementedError
