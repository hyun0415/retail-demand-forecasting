from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ForecastSplit:
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def forecast_split(last_origin: pd.Timestamp, days: int = 7, horizon: int = 7) -> ForecastSplit:
    test_end = pd.Timestamp(last_origin)
    test_start = test_end - pd.Timedelta(days=days - 1)
    validation_end = test_start - pd.Timedelta(days=horizon + 1)
    validation_start = validation_end - pd.Timedelta(days=days - 1)
    return ForecastSplit(validation_start, validation_end, test_start, test_end)


def complete_evaluation_series(frame: pd.DataFrame, split: ForecastSplit, horizons: tuple[int, ...]) -> pd.DataFrame:
    keys = ["store_nbr", "family"]
    expected = (split.validation_end - split.validation_start).days + 1
    expected *= len(horizons)
    valid = frame.loc[frame["date"].between(split.validation_start, split.validation_end)]
    test = frame.loc[frame["date"].between(split.test_start, split.test_end)]
    valid_counts = valid.groupby(keys, observed=True).size().rename("valid_rows")
    test_counts = test.groupby(keys, observed=True).size().rename("test_rows")
    counts = pd.concat([valid_counts, test_counts], axis=1).fillna(0)
    return counts.loc[counts.eq(expected).all(axis=1)].reset_index()[keys]
