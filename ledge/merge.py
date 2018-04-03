"""
Merge functions for lagged timeseries
"""

import xarray as xr
import numpy as np
from typing import List, Union
from ledge.datatypes import Truth, Loss


Series = Union[Truth, Loss]


def get_right_envelope(ds: xr.Dataset) -> xr.DataArray:
    """
    Get an envelope of non nan values from the right
    """

    array = mat.to_array()
    n_cols = array.shape[0]

    col_idx = n_cols - 1 - np.flip(np.isnan(array), axis=0).argmin(dim="variable")
    return array.isel(variable = col_idx)


def get_lag(series: Series) -> int:
    """
    Parse lag value from the series
    """

    return ("lag" in series.attrs and series.attrs["lag"]) or np.inf


def sort_lags(series_list: List[Series]) -> List[Series]:
    """
    Sort series according to lags. Latest goes to the end.
    """

    return sorted(series_list, key=get_lag)


def merge_lags(series_list: List[Series]) -> xr.Dataset:
    """
    Create a left joined dataset
    """

    # Add dummy names based on lags
    for series in series_list:
        series.name = get_lag(series)

    return xr.merge(series_list, join="left")


def latest(series_list: List[Series]) -> Series:
    """
    Skip older lag values. Prefer series without lag set.
    """

    dataset = merge_lags(sort_lags(series_list))
    return get_right_envelope(dataset)


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
