import functools

import numpy as np
import xarray as xr
from scipy import stats

from . import data


class FertilityModelItaly:
    def __init__(self, age_mean=32.0, age_range=8.0, tfr=1.28, sex_ratio=1.05):
        age = np.arange(100)
        tfr_by_age = stats.norm.pdf(age, loc=age_mean, scale=age_range)
        tfr_by_age[age <= 15] = 0
        tfr_by_age[age >= 46] = 0
        tfr_by_age = tfr_by_age / tfr_by_age.sum() * tfr
        tfr_by_age = (
            tfr_by_age[:, None] * np.array([1, sex_ratio]).T[None, :] / (1 + sex_ratio)
        )
        self.tfr_by_age = xr.DataArray(
            tfr_by_age, coords={"age": age, "sex": ["f", "m"]}, dims=("age", "sex")
        )

    def __call__(self, population_by_age):
        return population_by_age * self.tfr_by_age


class MortalityModelItaly:
    def __init__(self):
        age = np.arange(100)

        pod_by_age_f = 2 ** (age / 7.3) * 0.000015
        pod_by_age_f[67:] = 2 ** (age[67:] / 5.05) * 0.0000008

        pod_by_age_m = 2 ** (age / 7.3) * 0.000026
        pod_by_age_m[77:] = 2 ** (age[77:] / 12) * 0.00048

        pod_by_age = np.array([pod_by_age_f, pod_by_age_m])

        self.pod_by_age = xr.DataArray(
            pod_by_age, coords={"age": age, "sex": ["f", "m"]}, dims=("age", "sex")
        )

    def __call__(self, population_by_age, newborns):
        return population_by_age * self.pod_by_age, newborns * 0.005


class MigrationModelItaly:
    def __call__(self, population_by_age):
        return population_by_age * 0, population_by_age * 0


def next_step(
    population: xr.DataArray, fertility_model, mortality_model, migration_model
):
    births = fertility_model(population).sum("age")
    population_deaths, newborn_deaths = mortality_model(population, births)
    inflow, outflow = migration_model(population)

    next_population = population - population_deaths + inflow - outflow
    next_population = next_population.assign_coords(year=population.year + 1)

    next_population = np.maximum(next_population, 0)
    next_births = max(births - newborn_deaths, 0)

    return next_population.shift(age=1, fill_value=next_births)
