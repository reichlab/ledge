import xarray as xr
import numpy as np
from ledge import merge


def test_latest():
    times = 7
    max_lag = 4

    series_list = []
    for lag in range(max_lag + 1):
        mat = np.ones((times - lag,)) * lag
        attrs = { "lag": lag }
        coords = { "timepoints": range(times - lag) }
        series_list.append(xr.DataArray(mat, attrs=attrs, dims="timepoints", coords=coords))

    expected = xr.DataArray([4, 4, 4, 3, 2, 1, 0], dims="timepoints", coords={ "timepoints": range(times) })

    assert np.all(expected == merge.latest(series_list))


def test_earliest():
    times = 7
    max_lag = 4

    series_list = []
    for lag in range(max_lag + 1):
        mat = np.ones((times - (max_lag - lag),)) * lag
        attrs = { "lag": lag }
        coords = { "timepoints": range(times - (max_lag - lag)) }
        series_list.append(xr.DataArray(mat, attrs=attrs, dims="timepoints", coords=coords))

    expected = xr.DataArray([0, 0, 0, 1, 2, 3, 4], dims="timepoints", coords={ "timepoints": range(times) })

    assert np.all(expected == merge.earliest(series_list))
