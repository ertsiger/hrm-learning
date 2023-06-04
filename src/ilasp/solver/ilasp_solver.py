import timeit
from utils.container_utils import get_param

ILASP_BINARY_NAME = "ILASP"
CLINGO_BINARY_NAME = "clingo"
CLINGO_OPT_MODE_IGNORE_BINARY_NAME = "clingo_optmode_ignore.rb"
TIMEOUT_ERROR_CODE = 124

ILASP_OPERATION_SOLVE = "solve"
ILASP_OPERATION_SEARCH_SPACE = "search_space"

ILASP_PARAM_MAX_BODY_LITERALS = "ilasp_max_body_literals"  # Maximum number of literals that a learnt rule can have
ILASP_PARAM_VERSION = "ilasp_version"                      # ILASP version to run
ILASP_PARAM_SIMPLIFY_CONTEXTS = "ilasp_simplify_contexts"  # Simplify the representation of contexts
ILASP_PARAM_FIND_OPTIMAL = "ilasp_find_optimal"            # Whether to find an optimal hypothesis
ILASP_PARAM_CLINGO_PARAMS = "clingo_params"                # Clingo parameters


def solve_ilasp_task(ilasp_problem_filename, output_filename, ilasp_flags, timeout=60*35, operation=ILASP_OPERATION_SOLVE):
    if timeout <= 0.0:
        raise RuntimeError("Error: The ILASP timeout must be higher than 0.0.")

    with open(output_filename, 'w') as f:
        arguments = []
        if timeout is not None:
            arguments = ["timeout", str(timeout)]

        # other flags: -d -ni -ng -np
        arguments.extend([
            ILASP_BINARY_NAME,
            f"--version={get_param(ilasp_flags, ILASP_PARAM_VERSION, '2')}",
            "--strict-types",
            "-nc",  # omit constraints from the search space
            f"-ml={get_param(ilasp_flags, ILASP_PARAM_MAX_BODY_LITERALS, 1)}",  # maybe we can just set to 1
            ilasp_problem_filename
        ])

        if get_param(ilasp_flags, ILASP_PARAM_SIMPLIFY_CONTEXTS, True):
            arguments.append("--simple")

        clingo_params = get_param(ilasp_flags, ILASP_PARAM_CLINGO_PARAMS, [])
        find_optimal = get_param(ilasp_flags, ILASP_PARAM_FIND_OPTIMAL, True)
        if len(clingo_params) > 0 or not find_optimal:
            arguments.append("--clingo")
            binary_name = CLINGO_BINARY_NAME if find_optimal else CLINGO_OPT_MODE_IGNORE_BINARY_NAME
            arguments.append(f"{' '.join([CLINGO_OPT_MODE_IGNORE_BINARY_NAME, *clingo_params])}")

        if operation == ILASP_OPERATION_SEARCH_SPACE:
            arguments.append("-s")

        # useful source: https://stackoverflow.com/questions/16701310/get-how-much-time-python-subprocess-spends
        return_code_arr = []
        running_time = timeit.timeit("return_code_arr.append(subprocess.call(arguments, stdout=f))", number=1,
                                     setup="import subprocess", globals=locals())
        return_code = return_code_arr.pop()
        return return_code != TIMEOUT_ERROR_CODE, running_time
