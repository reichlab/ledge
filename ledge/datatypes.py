"""
Types that we work with
"""

import xarray as xr

"""
A series is an xarray with optional attrs["lag"] set to the lag value
"""
Series = xr.DataArray
