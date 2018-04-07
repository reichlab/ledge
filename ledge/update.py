"""
Functions for updating weights
"""

import xarray as xr
import numpy as np
from typing import List
from ledge.datatypes import Loss, Weight


def ftl(losses: List[Loss], init_weights=None) -> Weight:
    """
    Follow the leader update. Give full weight to the model with least loss.
    """

    models = [loss.attrs["model"] for loss in losses]
    best_idx = np.argmin([loss.sum() for loss in losses])
    weights = [1 if i == best_idx else 0 for i in range(len(losses))]

    return xr.DataArray(weights, dims="model", coords={ "model": models })


def ftpl(losses: List[Loss]) -> Weight:
    """
    Follow the perturbed leader update.
    """

    raise NotImplementedError()


def fixed_share(losses: List[Loss], eta: float, alpha: float, init_weights=None) -> Weight:
    r"""
    Fixed share update.
    """

    raise NotImplementedError()


def variable_share(losses: List[Loss]) -> Weight:
    r"""
    Variable share update.
    """
    raise NotImplementedError()


def mw(losses: List[Loss], eta: float, init_weights=None) -> Weight:
    r"""
    Multiplicative weight update. :math:`w_i(t + 1) = w_i(t) (1 - \eta m_i(t))`
    """

    raise NotImplementedError()


def hedge(losses: List[Loss], eta: float, init_weights=None) -> Weight:
    r"""
    Exponential weight update. :math:`w_i(t + 1) = w_i(t) e^{- \eta m_i(t)}`
    """

    raise NotImplementedError()
