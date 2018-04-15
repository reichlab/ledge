import xarray as xr
import numpy as np
from ledge import fill
from utils import make_series


def test_diff_mean_noninc():
    """
    Check if the non incremental version works fine
    """

    times = 10

    input_list = [
        make_series(np.zeros(times), range(times), 0),
        make_series(np.linspace(0, times - 3, times - 2), range(times - 2), 1),
        make_series(np.linspace(1, times - 3, times - 3), range(times - 3), 2)
    ]

    expected_output_list = [
        make_series(np.zeros(times), range(times), 0),
        make_series(np.concatenate([np.linspace(0, times - 3, times - 2), [3.5] * 2]), range(times), 1),
        make_series(np.concatenate([np.linspace(1, times - 3, times - 3), [4.0] * 3]), range(times), 2),
    ]

    output = fill.diff_mean(input_list)

    for o, eo in zip(output, expected_output_list):
        assert np.all(o == eo)


def test_diff_mean_inc():
    """
    Check for the incremental version
    """

    times = 10

    input_list = [
        make_series(np.zeros(times), range(times), 0),
        make_series(np.linspace(0, times - 3, times - 2), range(times - 2), 1),
        make_series(np.linspace(1, times - 3, times - 3), range(times - 3), 2)
    ]

    expected_output_list = [
        make_series(np.zeros(times), range(times), 0),
        make_series(np.concatenate([np.linspace(0, times - 3, times - 2), [3.5] * 2]), range(times), 1),
        make_series(np.concatenate([np.linspace(1, times - 3, times - 3), [8, 4.5, 4.5]]), range(times), 2),
    ]

    output = fill.diff_mean(input_list, incremental=True)

    for o, eo in zip(output, expected_output_list):
        assert np.all(o == eo)
