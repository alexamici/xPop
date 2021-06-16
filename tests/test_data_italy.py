import os.path

import pandas as pd
import xarray as xr

from xpop.data import italy

DATA = os.path.join(os.path.dirname(__file__), "data")
ISTAT_DEATHS = os.path.join(DATA, "comuni_giornaliero_31marzo21.csv")


def test_istat_deaths_to_pandas():
    res = italy.istat_deaths_to_pandas(ISTAT_DEATHS)

    assert isinstance(res, pd.DataFrame)
    assert "month_day" in res.columns
    assert "age" in res.columns


def test_read_istat_deaths():
    _, res = italy.read_istat_deaths(ISTAT_DEATHS)

    assert isinstance(res, xr.DataArray)
    assert "age" in res.coords
    assert "location" in res.coords
    assert "time" in res.coords


def test_istat_deaths_to_italy_year():
    _, ds = italy.read_istat_deaths(ISTAT_DEATHS)
    res = italy.istat_deaths_to_italy_year(ds)

    assert isinstance(res, xr.DataArray)
