from baselines.common.env_wrapper import EnvWrapper


class JIRPEnvWrapper(EnvWrapper):
    def get_events(self):
        obs = self.get_observation()
        if len(obs) == 0:
            return ""
        return "".join(sorted(obs))
