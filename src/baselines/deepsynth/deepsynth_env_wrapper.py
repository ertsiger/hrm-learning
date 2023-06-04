from baselines.common.env_wrapper import EnvWrapper


class DeepSynthEnvWrapper(EnvWrapper):
    def get_events(self):
        obs = self.get_observation()
        if len(obs) == 0:
            return "e"
        return "".join(sorted(obs))
