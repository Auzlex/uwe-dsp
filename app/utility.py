"""
    script name: utility.py
    script description: This script contains a number of functions that are used commonly in the project.
"""
import os

def get_file_path(file_name):
    """
        This function returns the file path of the file that is passed in.
        It takes in a file name and returns the file path.
    """
    return os.path.join(os.path.dirname(__file__), file_name)

def get_root_dir_path():
    """
        This function returns the root path of the project.
    """
    return os.path.dirname(os.path.realpath(__file__))
