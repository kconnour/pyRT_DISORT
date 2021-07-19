import os


def get_module_path():
    return os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                        '../..'))
