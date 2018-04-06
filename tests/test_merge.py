import xarray as xr
import numpy as np
from ledge import merge


def test_latest():
    times = 7
    lags = 4

    series_list = [
        xr.DataArray(np.ones((times - lag,)) * lag, attrs={ "lag": lag }, dims="timepoints", coords={ "timepoints" : range(times - lag) })
        for lag in range(lags)
    ]

    expected = xr.DataArray([3, 3, 3, 3, 2, 1, 0], dims="timepoints", coords={ "timepoints": range(times) })

    assert np.all(expected == merge.latest(series_list))
