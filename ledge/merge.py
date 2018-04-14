"""
Merge functions for lagged timeseries
"""

import xarray as xr
import numpy as np
from typing import List, Union
from ledge.datatypes import Truth, Loss
from ledge.utils import get_lag


Series = Union[Truth, Loss]


def _get_right_envelope(ds: xr.Dataset) -> xr.DataArray:
    """
    Get an envelope of non nan values from the right
    """

    array = ds.to_array()
    n_cols = array.shape[0]

    col_idx = n_cols - 1 - np.flip(np.isnan(array), axis=0).argmin(dim="variable")
    return array.isel(variable = col_idx).rename({ "variable": "lag" })


def _merge_lags(series_list: List[Series]) -> xr.Dataset:
    """
    Create a left joined dataset
    """

    return xr.merge([ser.rename(get_lag(ser)) for ser in series_list], join="left")


def latest(series_list: List[Series]) -> Series:
    """
    Skip older lag values. Prefer series without lag set.
    """

    dataset = _merge_lags(sorted(series_list, key=get_lag))
    latest_series = _get_right_envelope(dataset)
    latest_series.attrs = series_list[0].attrs
    latest_series.attrs.pop("lag", None)
    return latest_series


def zero(series_list: List[Series]) -> Series:
    """
    Do nothing. Return the series with lowest lag value
    """

    return sorted(series_list, key=get_lag)[0]


def mix_alpha(series_list: List[Series]) -> Series:
    """
    Use alpha to weigh the last lag value
    """

    raise NotImplementedError()


def mean(series_list: List[Series]) -> Series:
    """
    Take mean of all the values that we have
    """

    raise NotImplementedError()
