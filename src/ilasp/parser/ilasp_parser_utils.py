from collections import namedtuple
from ilasp.ilasp_common import CALL_STR, N_TRANSITION_STR, CONNECTED_STR, OBS_STR

N_TRANSITION_STR_PREFIX = f"{N_TRANSITION_STR}("
CALL_STR_PREFIX = f"{CALL_STR}("
CALL_STR_SUFFIX = ")."
POS_FLUENT_STR_PREFIX = f"{OBS_STR}("
NEG_FLUENT_STR_PREFIX = f"not {POS_FLUENT_STR_PREFIX}"

ParsedNegativeTransitionRule = namedtuple("ParsedNegativeTransitionRule",
                                          field_names=["src", "dst", "edge", "automaton", "pos", "neg"])
ParsedCallRule = namedtuple("ParsedCallRule", field_names=["src", "dst", "edge", "src_automaton", "dst_automaton"])


def parse_call_rule(call_str):
    from_state, to_state, edge_id, from_machine, to_machine = [x.strip() for x in call_str[len(CALL_STR_PREFIX):-len(CALL_STR_SUFFIX)].split(",")]
    return ParsedCallRule(from_state, to_state, edge_id, from_machine, to_machine)


def _parse_head_body_rule(rule_str):
    head, body = rule_str.split(":-")
    head, body = head.strip(), body.strip()
    return head, body


def parse_negative_transition_rule(transition_str):
    head, body = _parse_head_body_rule(transition_str)
    from_state, to_state, edge_id, automaton = _parse_negative_transition_head(head)
    pos_fluents, neg_fluents = _parse_negative_transition_body(body)
    return ParsedNegativeTransitionRule(from_state, to_state, edge_id, automaton, pos_fluents, neg_fluents)


def _parse_negative_transition_head(transition_head_str):
    parsed_head = [x.strip() for x in transition_head_str[len(N_TRANSITION_STR_PREFIX):].split(",")]
    return parsed_head[0], parsed_head[1], parsed_head[2], parsed_head[3]


def _parse_negative_transition_body(transition_body_str):
    fluents = transition_body_str.split(");")
    pos_fluents, neg_fluents = [], []

    for fluent in fluents:
        fluent = fluent.strip()
        if fluent.startswith(POS_FLUENT_STR_PREFIX):
            character = _parse_fluent(fluent, True)
            pos_fluents.append(character)
        elif fluent.startswith(NEG_FLUENT_STR_PREFIX):
            character = _parse_fluent(fluent, False)
            neg_fluents.append(character)

    return pos_fluents, neg_fluents


def _parse_fluent(fluent_str, is_positive):
    prefix = POS_FLUENT_STR_PREFIX
    if not is_positive:
        prefix = NEG_FLUENT_STR_PREFIX
    return fluent_str[len(prefix):].split(",")[0].strip("\"")

