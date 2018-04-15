"""
Imputation functions for dataset
"""

import xarray as xr
from typing import List, Union
from ledge.datatypes import Truth, Loss


Series = Union[Truth, Loss]


def diff_mean(series_list: List[Series], incremental=False) -> List[Series]:
    """
    Fill in the series using mean of the diff between a `base` series.
    If incremental is true, the base series is the previous series. If false,
    the base is the first series.
    """

    if len(series_list) < 2:
        raise ValueError("At least 2 series needed")

    base_series = series_list[0]
    output = [base_series]

    for i in range(1, len(series_list)):
        if incremental:
            base_series = output[i - 1]

        diff = (series_list[i] - base_series).mean()
        fallback = base_series + diff

        output.append(series_list[i].combine_first(fallback))

    return output
