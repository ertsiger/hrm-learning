def get_param(param_dict, param_name, default_value=None):
    if param_dict is not None and param_name in param_dict:
        return param_dict[param_name]
    return default_value


def sort_by_ord(input_list):
    input_list.sort(key=lambda s: ord(str(s).lower()))
