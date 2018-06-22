"""
This file is generated using an accompanying org file.
Do not edit manually.
"""

import xarray as xr
from typing import List, Callable
from ledge.datatypes import Loss, Weight
from ledge.utils import uniform_weights
import matplotlib.pyplot as plt

def c_profile(ax, losses: List[Loss],
              xlabel="Timepoints", ylabel="Loss",
              title="Cumulative loss profile"):
    """
    Plot cumulative profile for given losses
    """

    losses = sorted(losses, key=sum, reverse=True)
    c_losses = [l.cumsum() for l in losses]

    for cl, l in zip(c_losses, losses):
        ax.plot(l.timepoints, cl, label=l.attrs['model'])
        total = float(cl[-1])
        ax.annotate(f"{total:.4f}", xy=(l.timepoints[-1], total))

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
