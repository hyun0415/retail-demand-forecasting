import numpy as np
import pandas as pd

from retail_forecast.data import FavoritaData
from retail_forecast.features import build_feature_table
from retail_forecast.pipeline import build_decision_summary, temporal_split


def make_sample_data() -> FavoritaData:
    dates = pd.date_range("2026-01-01", periods=70)
    sales = pd.MultiIndex.from_product(
        [[1, 2], ["BEVERAGES", "GROCERY I"], dates],
        names=["store_nbr", "family", "date"],
    ).to_frame(index=False)
    sales["sales"] = np.arange(len(sales), dtype="float32")
    sales["onpromotion"] = 0
    stores = pd.DataFrame(
        {
            "store_nbr": [1, 2],
            "city": ["Quito", "Guayaquil"],
            "state": ["Pichincha", "Guayas"],
            "type": ["A", "B"],
            "cluster": [1, 2],
        }
    )
    holidays = pd.DataFrame(
        {
            "date": [dates[20]],
            "type": ["Holiday"],
            "locale": ["National"],
            "locale_name": ["Ecuador"],
            "description": ["Sample holiday"],
            "transferred": [False],
        }
    )
    transactions = sales[["store_nbr", "date"]].drop_duplicates()
    transactions["transactions"] = 100

    return FavoritaData(sales, stores, holidays, transactions)


def test_future_targets_match_each_horizon() -> None:
    data = make_sample_data()
    frame = build_feature_table(
        data,
        horizons=(1, 7),
        lags=(1, 7),
        rolling_windows=(7,),
    )
    row = frame.loc[
        (frame["store_nbr"] == 1)
        & (frame["family"] == "BEVERAGES")
        & (frame["date"] == pd.Timestamp("2026-01-20"))
        & (frame["forecast_horizon"] == 7)
    ].iloc[0]
    expected = data.sales.loc[
        (data.sales["store_nbr"] == 1)
        & (data.sales["family"] == "BEVERAGES")
        & (data.sales["date"] == pd.Timestamp("2026-01-27")),
        "sales",
    ].iloc[0]

    assert row["target_date"] == pd.Timestamp("2026-01-27")
    assert row["target_sales"] == expected


def test_weekly_seasonal_feature_uses_known_history() -> None:
    data = make_sample_data()
    frame = build_feature_table(data, (1, 7), (1, 7), (7,))
    row = frame.loc[
        (frame["store_nbr"] == 1)
        & (frame["family"] == "BEVERAGES")
        & (frame["date"] == pd.Timestamp("2026-01-20"))
        & (frame["forecast_horizon"] == 1)
    ].iloc[0]

    assert row["seasonal_reference_date"] == pd.Timestamp("2026-01-14")
    assert row["seasonal_reference_date"] <= row["date"]
    expected_dates = [
        pd.Timestamp("2026-01-14") - pd.Timedelta(days=7 * offset)
        for offset in range(4)
    ]
    expected = data.sales.loc[
        (data.sales["store_nbr"] == 1)
        & (data.sales["family"] == "BEVERAGES")
        & (data.sales["date"].isin(expected_dates)),
        "sales",
    ].mean()
    assert row["sales_same_weekday_4w_mean"] == expected
    assert "current_oil_price" not in frame.columns


def test_temporal_split_has_no_date_overlap() -> None:
    frame = build_feature_table(make_sample_data(), (1, 7), (1, 7), (7,))
    train, valid, cutoff = temporal_split(frame, validation_days=14)

    assert train["target_date"].max() < cutoff
    assert valid["date"].min() >= cutoff
    assert set(valid["forecast_horizon"].unique()) == {1, 7}


def test_origin_window_is_applied_before_horizon_expansion() -> None:
    frame = build_feature_table(
        make_sample_data(),
        horizons=(1, 7),
        lags=(1, 7),
        rolling_windows=(7,),
        origin_days=10,
    )

    assert frame["date"].nunique() == 10
    assert frame["date"].max() - frame["date"].min() == pd.Timedelta(days=9)


def test_decision_summary_aggregates_daily_forecasts() -> None:
    predictions = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-01-01")] * 7,
            "store_nbr": [1] * 7,
            "family": ["BEVERAGES"] * 7,
            "forecast_horizon": range(1, 8),
            "prediction": np.arange(1, 8, dtype=float),
            "target_sales": np.arange(2, 9, dtype=float),
        }
    )

    summary = build_decision_summary(predictions).iloc[0]

    assert summary["prediction_sum_1_3"] == 6
    assert summary["prediction_sum_1_7"] == 28
