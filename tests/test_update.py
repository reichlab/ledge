import xarray as xr
import numpy as np
from ledge import update
from utils import make_weights, make_series
from functools import partial


MODELS = list("abcde")


def test_mixer():
    """
    Test for mixer
    """

    # noop defaults to uniform weights
    bias_fn = update.noop
    main_fn = partial(update.pick, index=2)

    mix_fn = update.create_mixer([main_fn, bias_fn], [0.3, 0.7])

    losses = [
        make_series(np.random.rand(10), range(10), extra_attrs={ "model": model })
        for model in MODELS
    ]
    weights = mix_fn(losses)

    expected_weights = make_weights([0.14, 0.14, 0.44, 0.14, 0.14], MODELS)

    assert np.allclose(expected_weights, weights)
