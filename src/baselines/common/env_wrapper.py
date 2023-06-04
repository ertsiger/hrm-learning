import gym
from utils.rl_utils import get_environment_class


class EnvWrapper:
    def __init__(self, args):
        self.domain = args.domain
        self.task = args.task

        self.envs = [
            self._make_env(args, seed)
            for seed in range(args.seed, args.seed + args.num_instances)
        ]

        self.env = None
        self.env_idx = None

    def _make_env(self, args, seed):
        return gym.make(get_environment_class(args.task), params={
            "use_flat_hierarchy": True,
            "grid_params": {  # only for craftworld
                "grid_type": "open_plan",
                "size": 7, "height": 7, "width": 7,
                "use_lava_walls": False,
                "use_lava": args.use_lava,
                "right_rooms_even": False,
                "random_door_placement": False,
                "max_objs_per_class": 1
            },
            "environment_seed": seed,
            "random_restart": False
        })

    def step(self, action):
        return self.env.step(action)

    def get_observation(self):
        return self.env.get_observation()

    def get_observables(self):
        return self.env.get_observables()

    def get_actions(self):
        return list(range(0, self.envs[0].action_space.n))

    def reset(self):
        if self.env_idx is None:
            self.env_idx = 0
        else:
            self.env_idx = (self.env_idx + 1) % len(self.envs)
        self.env = self.envs[self.env_idx]
        return self.env.reset()

    def get_hierarchy(self):
        return self.envs[0].get_hierarchy()
