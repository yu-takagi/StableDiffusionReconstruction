"""
Utility functions for preprocessing
"""


import inspect
from datetime import datetime


def print_start_msg():
    """
    Print process starting message
    """
    print("%s Running %s"  % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                              inspect.currentframe().f_back.f_code.co_name))


def print_finish_msg():
    """
    Print process finishing message
    """
    print("%s DONE"  % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
