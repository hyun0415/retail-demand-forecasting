import numpy as np
import pandas as pd

from retail_forecast.data import FavoritaData
from retail_forecast.features import build_feature_table
from retail_forecast.pipeline import temporal_split


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
    oil = pd.DataFrame({"date": dates, "dcoilwtico": 50.0})
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

    return FavoritaData(sales, stores, oil, holidays, transactions)


def test_future_target_matches_exact_horizon() -> None:
    data = make_sample_data()
    frame = build_feature_table(data, horizon=7, lags=(1, 7), rolling_windows=(7,))
    row = frame.loc[
        (frame["store_nbr"] == 1)
        & (frame["family"] == "BEVERAGES")
        & (frame["date"] == pd.Timestamp("2026-01-20"))
    ].iloc[0]
    expected = data.sales.loc[
        (data.sales["store_nbr"] == 1)
        & (data.sales["family"] == "BEVERAGES")
        & (data.sales["date"] == pd.Timestamp("2026-01-27")),
        "sales",
    ].iloc[0]

    assert row["target_date"] == pd.Timestamp("2026-01-27")
    assert row["target_sales"] == expected


def test_temporal_split_has_no_date_overlap() -> None:
    frame = build_feature_table(make_sample_data(), 7, (1, 7), (7,))
    train, valid, cutoff = temporal_split(frame, validation_days=14)

    assert train["target_date"].max() < cutoff
    assert valid["target_date"].min() >= cutoff
