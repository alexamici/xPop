import numpy as np
import pandas as pd
import xarray as xr


def istat_deaths_to_pandas(path):
    istat = pd.read_csv(path, encoding="8859", na_values="n.d.", dtype={"GE": str})

    # make a date index from GE
    def ge2month_day(x):
        return f"{x[:2]}-{x[2:]}"

    month_day = istat["GE"].map(ge2month_day).values
    istat["month_day"] = month_day

    def cl_eta2age(x):
        if x <= 1:
            return x
        elif x <= 21:
            age = x - 1
            return age * 5
        else:
            raise ValueError(f"unknown age class {x}")

    istat["age"] = istat["CL_ETA"].apply(cl_eta2age)

    return istat


def read_istat_deaths(path):
    istat = istat_deaths_to_pandas(path).rename(columns={"NOME_COMUNE": "location"})

    data = None

    for yy in range(11, 21):
        print(yy)
        tmp = istat.groupby(["month_day", "age", "location"]).agg(
            **{
                "f": (f"F_{yy}", sum),
                "m": (f"M_{yy}", sum),
            }
        )
        if yy % 4 != 0:
            tmp = tmp.drop(index="02-29")
        tmp = tmp.reset_index()
        tmp["time"] = tmp["month_day"].map(lambda x: np.datetime64(f"20{yy}-{x}"))
        tmp = tmp.set_index(["time", "age", "location"]).drop(columns="month_day")
        xtmp = tmp.to_xarray().to_array("sex").fillna(0)
        if data is None:
            data = xtmp
        else:
            data = xr.concat([data, xtmp], dim="time", fill_value=0)

    coords = {
        "region": (
            "location",
            "Italy / " + istat.groupby(["location"])["NOME_REGIONE"].first(),
        ),
        "province": (
            "location",
            istat.groupby(["location"])["NOME_PROVINCIA"].first(),
        ),
    }
    data = data.assign_coords(coords)
    return istat, data


def istat_deaths_to_italy_year(istat):
    deaths_italy = istat.sum("location")
    deaths_italy = deaths_italy.resample(time="Y", label="left", loffset="1D").sum()
    deaths_italy = deaths_italy.assign_coords(year=deaths_italy.time.dt.year)
    return deaths_italy.swap_dims(time="year").drop_vars("time")
