"""
Merge functions for lagged timeseries
"""

from typing import List
from ledge.datatypes import Series


def latest(series_list: List[Series]) -> Series:
    """
    Skip older lag values. Prefer series without lag set.
    """

    ...


def mix_alpha(series_list: List[Series]) -> Series:
    """
    Use alpha to weigh the last lag value
    """

    ...


def mean(series_list: List[Series]) -> Series:
    """
    Take mean of all the values that we have
    """

    ...
