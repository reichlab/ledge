import xarray as xr
import numpy as np
from ledge import merge
from utils import make_series


def test_latest():
    times = 7
    max_lag = 4

    series_list = []
    for lag in range(max_lag + 1):
        values = np.ones((times - lag,)) * lag
        series_list.append(make_series(values, range(times - lag), lag))

    expected = make_series([4, 4, 4, 3, 2, 1, 0], range(times))

    assert np.all(expected == merge.latest(series_list))


def test_earliest():
    times = 7
    max_lag = 4

    series_list = []
    for lag in range(max_lag + 1):
        values = np.ones((times - (max_lag - lag),)) * lag
        series_list.append(make_series(values, range(times - (max_lag - lag))))

    expected = make_series([0, 0, 0, 1, 2, 3, 4], range(times))

    assert np.all(expected == merge.earliest(series_list))
