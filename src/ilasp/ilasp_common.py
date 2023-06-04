import os

CALL_STR = "call"
CONNECTED_STR = "ed"
TRANSITION_STR = "phi"
N_TRANSITION_STR = f"n_{TRANSITION_STR}"
OBS_STR = "obs"
LEAF_AUTOMATON = "mL"


def generate_injected_statements(stmts):
    return "\n".join([generate_injected_statement(stmt) for stmt in stmts]) + '\n'


def generate_injected_block(stmts):
    return generate_injected_statement('\n\t' + "\n\t".join(stmts) + '\n') + '\n'


def generate_injected_statement( stmt):
    return "#inject(\"" + stmt + "\")."


def set_ilasp_env_variables(root_folder_path):
    _set_env_variable(root_folder_path, "bin", "PATH")
    _set_env_variable(root_folder_path, "lib", "LD_LIBRARY_PATH")


def _set_env_variable(root_folder_path, dir, env_var):
    final_path = os.path.join(root_folder_path, dir)
    os.environ.putenv(env_var, final_path + os.pathsep + os.environ.get(env_var, ""))
