"""
Common utilities for tests
"""

import xarray as xr


def make_series(values, timepoints, lag=None) -> xr.DataArray:
    """
    Helper for creating a fully specified series
    """

    attrs = { "lag": lag } if lag is not None else {}
    coords = { "timepoints": timepoints }
    return xr.DataArray(values, attrs=attrs, dims="timepoints", coords=coords)
