INITIAL_STATE = 0
TERMINAL_STATE = -1

MAX_RM_STATES_N = 6
UPDATE_X_EVERY_N = 100
EQV_THRESHOLD = 0.7 # probability threshold for concluding two
                    # states in two different RMs are equivalent

M = 100000.0
EXACT_EPSILON = 1e-7
MINIMIZATION_EPSILON = 0.15

NOISE_PROB = 0.1
NOISE = 0.1

TERMINATION = True
X_PRUNE_MIN_SIZE = 3
X_PRUNE_MAX = 0

DISCRETE_NOISE_P = 0.02 # Probability that an episode will be corrupted
REWARD_FLIP_P = 0.2 # (In a corrupted trace) probability of flipping a reward
UPDATE_X_EVERY_N_DISCRETE_NOISE = 1000

REGISTER_MEAN_REWARD_EVERY_N_STEP = 100

REPORT = False
