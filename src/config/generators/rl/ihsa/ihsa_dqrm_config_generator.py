from config.generators.rl.ihsa.ihsa_cross_product_config_generator import IHSACrossProductConfigGenerator


class IHSADQRMConfigGenerator(IHSACrossProductConfigGenerator):
    @classmethod
    def _set_rl_config(cls, config, args):
        super()._set_rl_config(config, args)
        config["algorithm"] = "ihsa-dqrm"


if __name__ == "__main__":
    IHSADQRMConfigGenerator.run()
