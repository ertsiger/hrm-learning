from config.generators.rl.dqn.dqn_base_config_generator import DQNBaseRLConfigGenerator


class DQNConfigGenerator(DQNBaseRLConfigGenerator):
    @classmethod
    def _set_algorithm_config(cls, config, args):
        super()._set_algorithm_config(config, args)
        config["algorithm"] = "dqn"


if __name__ == "__main__":
    DQNConfigGenerator.run()
