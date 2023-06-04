from config.generators.rl.ihsa.ihsa_cross_product_config_generator import IHSACrossProductConfigGenerator


class IHSACRMConfigGenerator(IHSACrossProductConfigGenerator):
    @classmethod
    def _set_rl_config(cls, config, args):
        super()._set_rl_config(config, args)
        config["algorithm"] = "ihsa-crm"


if __name__ == "__main__":
    IHSACRMConfigGenerator.run()
