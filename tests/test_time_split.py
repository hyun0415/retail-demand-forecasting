import pandas as pd

from retail_forecast.time_split import complete_evaluation_series, forecast_split


def test_seven_day_validation_gap_and_test() -> None:
    split = forecast_split(pd.Timestamp("2017-08-08"))
    assert split.validation_start == pd.Timestamp("2017-07-19")
    assert split.validation_end == pd.Timestamp("2017-07-25")
    assert split.test_start == pd.Timestamp("2017-08-02")
    assert split.test_end == pd.Timestamp("2017-08-08")
    assert split.validation_end + pd.Timedelta(days=7) < split.test_start


def test_cohort_requires_complete_validation_and_test_rows() -> None:
    split = forecast_split(pd.Timestamp("2017-08-08"))
    origins = list(pd.date_range(split.validation_start, split.validation_end))
    origins += list(pd.date_range(split.test_start, split.test_end))
    frame = pd.MultiIndex.from_product(
        [[1, 2], ["A"], origins, range(1, 8)],
        names=["store_nbr", "family", "date", "forecast_horizon"],
    ).to_frame(index=False)
    frame = frame.loc[~((frame["store_nbr"] == 2) & (frame["date"] == split.test_end) & (frame["forecast_horizon"] == 7))]
    result = complete_evaluation_series(frame, split, tuple(range(1, 8)))
    assert result.to_dict("records") == [{"store_nbr": 1, "family": "A"}]
