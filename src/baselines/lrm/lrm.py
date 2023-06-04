from baselines.common.baseline import Baseline
from baselines.lrm.reward_machine import RewardMachine
from baselines.lrm.lrm_env_wrapper import LRMEnvWrapper
from gym_hierarchical_subgoal_automata.automata.hierarchical_automaton import HierarchicalAutomaton
from gym_hierarchical_subgoal_automata.automata.logic import DNFFormula
import random
import stopit
import time

"""
- Pseudo-code:
    - Run 'n' random episodes until completion or timeout
    - Learn an RM using those traces
    - Learn a policy for the learned RM until reaching a contradition
    - Add the contradicting trace and relearn the RM
    - Relearn a policy for the new RM from stratch
"""


class LRM(Baseline):
    def __init__(self, args):
        super(LRM, self).__init__(args)

        self.env = LRMEnvWrapper(args)

        # Defaults used in original code
        self.rm_init_episodes = 1000
        self.rm = RewardMachine(
            U_max=args.lrm_max_states, preprocess_trace=True, tabu_list_max=10000, n_workers=16, rm_lr_steps=100,
            perfect_rm=self.env.get_perfect_rm(), use_perfect_rm=False
        )

        self.actions = self.env.get_actions()
        self.train_rewards = []
        self.rm_scores = []

    def run_algorithm(self):
        if self.episode < self.rm_init_episodes:
            self._collect_random_traces()

        if self.episode == self.rm_init_episodes:
            self._learn_rm()

        if self.time_left > 0:
            self._train()

    def get_num_states(self):
        return len(self.rm.get_states()) if self.rm.get_states() is not None else 0

    def get_num_edges(self):
        return len(self.rm.delta_u) if self.rm.delta_u is not None else 0

    def get_num_traces(self):
        return len(self.rm.traces)

    def get_hierarchical_automaton(self):
        automaton = HierarchicalAutomaton("m0")
        for state in self.rm.get_states():
            automaton.add_state(f"u{state}")
            print(state)
        for from_state, obs in self.rm.delta_u:
            automaton.add_formula_edge(f"u{from_state}", f"u{self.rm.delta_u[(from_state, obs)]}", DNFFormula([[obs]]))
        return automaton

    def _collect_random_traces(self):
        print("Collecting random traces...")
        while self.episode < self.rm_init_episodes:
            # running an episode using a random policy
            self.env.restart()
            trace = [(self.env.get_events(), 0.0)]
            reward_total, steps_total = 0, 0
            for _ in range(self.max_episode_length):
                # executing a random action
                a = random.choice(self.actions)
                reward, done = self.env.execute_action(a)
                o2_events = self.env.get_events()
                reward_total += reward
                steps_total += 1
                trace.append((o2_events, reward))

                # checking if the episode finishes
                if done:
                    self.rm.add_terminal_observations(o2_events)
                    break

            self.episode += 1

            # adding this trace to the set of traces that we use to learn the rm
            self.rm.add_trace(trace)

    def _learn_rm(self):
        print("Learning a reward machines...")

        start = time.time()

        with stopit.ThreadingTimeout(self.time_left) as to_ctx_mgr:  # https://pypi.org/project/stopit/
            same_rm, info = self.rm.learn_the_reward_machine()

        self.time_left -= (time.time() - start)
        self.num_calls += 1

        if to_ctx_mgr.state == to_ctx_mgr.TIMED_OUT:
            return True

        self.rm_scores.append((self.episode,) + info)
        self.get_hierarchical_automaton().plot(self.output_dir, f"automaton_{self.num_calls}.png")

        return False

    def _train(self):
        timeout = self.time_left < 0
        # Start learning a policy for the current rm
        while self.episode < self.num_episodes and not timeout:
            self.env.restart()
            o1_events = self.env.get_events()
            u1 = self.rm.get_initial_state()
            trace = [(o1_events, 0.0)]
            reward_total, steps_total = 0, 0
            add_trace = False

            for _ in range(self.max_episode_length):
                a = random.choice(self.actions)

                # executing a random action
                reward, done = self.env.execute_action(a)
                o2_events = self.env.get_events()
                u2 = self.rm.get_next_state(u1, o2_events)

                # updating the number of steps and total reward
                trace.append((o2_events, reward))
                reward_total += reward
                steps_total += 1

                # updating the current RM if needed
                self.rm.update_rewards(u1, o2_events, reward)
                if done:
                    self.rm.add_terminal_observations(o2_events)
                if self.rm.is_observation_impossible(u1, o1_events, o2_events):
                    # if o2 is impossible according to the current RM,
                    # then the RM has a bug and must be relearned
                    add_trace = True

                # checking if the episode finishes or the agent reaches the maximum number of training steps
                if done:  # or finish_learning:
                    break

                # Moving to the next state
                o1_events, u1 = o2_events, u2

            self.episode += 1

            # Testing
            print(f"Episode: {self.episode}\tReward: {reward_total}\tSteps: {steps_total}")
            self.train_rewards.append((self.episode, reward_total, steps_total))

            # If the trace isn't correctly predicted by the reward machine,
            # we add the trace and relearn the machine
            if add_trace and self.episode < self.num_episodes:
                print("Relearning the reward machine...")
                self.rm.add_trace(trace)
                timeout = self._learn_rm()

            if self.episode % 100 == 0 or timeout:
                self.make_checkpoint(self.episode)

        # return the training rewards
        return self.train_rewards, self.rm_scores, self.rm.get_info()
