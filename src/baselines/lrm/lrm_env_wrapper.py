from baselines.common.env_wrapper import EnvWrapper


class LRMEnvWrapper:
    def __init__(self, args):
        self.env = EnvWrapper(args)

    def execute_action(self, action):
        _, reward, done, _ = self.env.step(action)
        return reward, done

    def restart(self):
        return self.env.reset()

    def get_actions(self):
        return self.env.get_actions()

    def get_perfect_rm(self):
        if self.env.domain == "craftworld":
            return self._get_cw_perfect_rm()
        return None

    def _get_cw_perfect_rm(self):
        root = self.env.get_hierarchy().get_root_automaton()

        state_map = {
            state: idx
            for idx, state in enumerate(root.get_states())
        }

        perfect_rm = {}

        edges = root.get_edges()
        for from_state in edges:
            for to_state in edges[from_state]:
                for cond in edges[from_state][to_state]:
                    pos_lit = cond.get_formula().get_formula()[0].get_pos_literals()
                    perfect_rm[(state_map[from_state], pos_lit.pop())] = state_map[to_state]

        return perfect_rm

    def get_events(self):
        obs = self.env.get_observation()
        if len(obs) == 0:
            return "e"
        return "".join(sorted(obs))
