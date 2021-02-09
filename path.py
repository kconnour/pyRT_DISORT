import os


def get_project_path() -> str:
    """ Get the absolute path to the project.

    Returns
    -------
    project_path: str
        The project path.
    """
    abs_path_of_this_file = os.path.realpath(__file__)
    dir_of_this_file = os.path.dirname(abs_path_of_this_file)
    return os.path.abspath(os.path.join(dir_of_this_file, ''))
