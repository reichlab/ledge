"""
Common utilities for tests
"""

import xarray as xr


def make_series(values, timepoints, lag=None, extra_attrs=None) -> xr.DataArray:
    """
    Helper for creating a fully specified series
    """

    attrs = { "lag": lag } if lag is not None else {}

    if extra_attrs is not None:
        attrs = { **attrs, **extra_attrs }

    coords = { "timepoints": timepoints }
    return xr.DataArray(values, attrs=attrs, dims="timepoints", coords=coords)


def make_weights(values, models) -> xr.DataArray:
    """
    Create xarray weights
    """

    return xr.DataArray(values, dims="model", coords = { "model": models })
