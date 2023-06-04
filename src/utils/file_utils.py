import json
import os
import pickle
import shutil


def copy(src, dst):
    shutil.copy(src, dst)


def mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def rm_dir(dir_name):
    if path_exists(dir_name):
        shutil.rmtree(dir_name, ignore_errors=True)


def rm_dirs(dir_list):
    for dir_name in dir_list:
        rm_dir(dir_name)


def rm_file(filename):
    if os.path.exists(filename):
        os.remove(filename)


def rm_files(file_list):
    for filename in file_list:
        rm_file(filename)


def is_file_empty(path):
    return os.path.getsize(path) == 0


def path_exists(path):
    return os.path.exists(path)


def read_json_file(filepath):
    with open(filepath) as f:
        return json.load(f)


def write_json_obj(obj, filepath, **kwargs):
    with open(filepath, 'w') as f:
        json.dump(obj, f, **kwargs)


def dump_pickle_obj(obj, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle_obj(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)
