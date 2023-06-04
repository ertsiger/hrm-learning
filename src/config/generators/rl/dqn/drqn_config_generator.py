from config.generators.rl.dqn.dqn_base_config_generator import DQNBaseRLConfigGenerator
from reinforcement_learning.drqn_algorithm import DRQNAlgorithm


class DRQNConfigGenerator(DQNBaseRLConfigGenerator):
    @classmethod
    def _add_extra_args(cls, parser):
        super()._add_extra_args(parser)

        group = parser.add_argument_group("drqn")
        group.add_argument(
            "--er_seq_length",
            type=int, default=32,
            help="length of the sequences sampled for each episode (default: 32)"
        )
        group.add_argument(
            "--lstm_method",
            default=DRQNAlgorithm.LSTM_METHOD_STATE,
            choices=[DRQNAlgorithm.LSTM_METHOD_STATE, DRQNAlgorithm.LSTM_METHOD_STATE_AND_OBS, DRQNAlgorithm.LSTM_METHOD_OBS],
            help="what is the input to the LSTM module in the DRQN"
        )
        group.add_argument(
            "--lstm_hidden_size",
            type=int, default=256,
            help="the hidden size of the LSTM (default: 256)"
        )

    @classmethod
    def _set_algorithm_config(cls, config, args):
        super()._set_algorithm_config(config, args)
        config["algorithm"] = "drqn"
        config[DRQNAlgorithm.ER_SEQ_LENGTH] = args.er_seq_length
        config[DRQNAlgorithm.LSTM_METHOD] = args.lstm_method
        config[DRQNAlgorithm.LSTM_HIDDEN_SIZE] = args.lstm_hidden_size


if __name__ == "__main__":
    DRQNConfigGenerator.run()
