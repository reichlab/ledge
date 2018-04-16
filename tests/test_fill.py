import xarray as xr
import numpy as np
from ledge import fill
from utils import make_series

TIMES = 10
INPUT_LIST = [
    make_series(np.zeros(TIMES), range(TIMES), 0),
    make_series(np.linspace(0, TIMES - 3, TIMES - 2), range(TIMES - 2), 1),
    make_series(np.linspace(1, TIMES - 3, TIMES - 3), range(TIMES - 3), 2)
]

def test_diff_mean_noninc():
    """
    Check if the non incremental version works fine
    """

    expected_output_list = [
        make_series(np.zeros(TIMES), range(TIMES), 0),
        make_series(np.concatenate([np.linspace(0, TIMES - 3, TIMES - 2), [3.5] * 2]), range(TIMES), 1),
        make_series(np.concatenate([np.linspace(1, TIMES - 3, TIMES - 3), [4.0] * 3]), range(TIMES), 2),
    ]

    output = fill.diff_mean(INPUT_LIST)

    for o, eo in zip(output, expected_output_list):
        assert np.allclose(o, eo)


def test_diff_mean_inc():
    """
    Check for the incremental version
    """

    expected_output_list = [
        make_series(np.zeros(TIMES), range(TIMES), 0),
        make_series(np.concatenate([np.linspace(0, TIMES - 3, TIMES - 2), [3.5] * 2]), range(TIMES), 1),
        make_series(np.concatenate([np.linspace(1, TIMES - 3, TIMES - 3), [8, 4.5, 4.5]]), range(TIMES), 2),
    ]

    output = fill.diff_mean(INPUT_LIST, inc=True)

    for o, eo in zip(output, expected_output_list):
        assert np.allclose(o, eo)


def test_diff_linear_noninc():
    expected_output_list = [
        make_series(np.zeros(TIMES), range(TIMES), 0),
        make_series(np.concatenate([np.linspace(0, TIMES - 3, TIMES - 2), [4.6666666666666661] * 2]), range(TIMES), 1),
        make_series(np.concatenate([np.linspace(1, TIMES - 3, TIMES - 3), [5.0] * 3]), range(TIMES), 2),
    ]

    output = fill.diff_linear(INPUT_LIST)

    for o, eo in zip(output, expected_output_list):
        assert np.allclose(o, eo)


def test_diff_linear_inc():
    expected_output_list = [
        make_series(np.zeros(TIMES), range(TIMES), 0),
        make_series(np.concatenate([np.linspace(0, TIMES - 3, TIMES - 2), [4.6666666666666661] * 2]), range(TIMES), 1),
        make_series(np.concatenate([np.linspace(1, TIMES - 3, TIMES - 3), [8, 5.6666666666666661, 5.6666666666666661]]), range(TIMES), 2),
    ]

    output = fill.diff_linear(INPUT_LIST, inc=True)
    print(output)
    print(expected_output_list)

    for o, eo in zip(output, expected_output_list):
        assert np.allclose(o, eo)
