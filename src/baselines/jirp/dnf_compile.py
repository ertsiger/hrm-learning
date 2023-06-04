import itertools
from baselines.jirp.reward_machine_utils import evaluate_dnf

def compile_dnf(formula):
    """
    evaluate_dnf is extremely slow, using a naive truth-table approach improves on it
    """
    chars = set()
    for char in formula:
        if char.isalnum():
            chars.add(char)
    true_for = set()
    for r in range(0, len(chars)+1):
        for true_props in itertools.combinations(chars, r):
            true_props_joined = "".join(true_props)
            if evaluate_dnf(formula, true_props_joined):
                true_for.add(frozenset(true_props))
    return (true_for, frozenset(chars))

def evaluate_dnf_compiled(compiled, true_props):
    true_for, chars = compiled
    return frozenset.intersection(frozenset(true_props), chars) in true_for
