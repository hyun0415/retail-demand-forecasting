import numpy as np
import pandas as pd

from retail_forecast.data import FavoritaData


KEY_COLUMNS = ["store_nbr", "family"]
NON_FEATURE_COLUMNS = [
    "date",
    "target_date",
    "seasonal_reference_date",
    "target_sales",
]


def add_sales_history(
    frame: pd.DataFrame,
    lags: tuple[int, ...],
    rolling_windows: tuple[int, ...],
) -> pd.DataFrame:
    frame = frame.sort_values(KEY_COLUMNS + ["date"]).copy()
    grouped = frame.groupby(KEY_COLUMNS, observed=True)["sales"]

    for lag in lags:
        frame[f"sales_lag_{lag}"] = grouped.shift(lag).astype("float32")

    shifted = grouped.shift(1)
    history_groups = shifted.groupby(
        [frame["store_nbr"], frame["family"]], observed=True
    )

    for window in rolling_windows:
        frame[f"sales_mean_{window}"] = (
            history_groups.rolling(window, min_periods=1)
            .mean()
            .reset_index(level=[0, 1], drop=True)
            .astype("float32")
        )

    return frame


def add_future_targets(
    frame: pd.DataFrame,
    horizons: tuple[int, ...],
    origin_days: int | None = None,
    origin_start: pd.Timestamp | None = None,
    origin_end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    labels = frame[KEY_COLUMNS + ["date", "sales", "onpromotion"]].rename(
        columns={
            "date": "target_date",
            "sales": "target_sales",
            "onpromotion": "target_onpromotion",
        }
    )
    max_horizon = max(horizons)
    last_complete_origin = frame["date"].max() - pd.Timedelta(days=max_horizon)
    origins = frame.loc[frame["date"].le(last_complete_origin)].copy()
    if origin_days is not None:
        first_origin = last_complete_origin - pd.Timedelta(days=origin_days - 1)
        origins = origins.loc[origins["date"].ge(first_origin)].copy()
    if origin_start is not None:
        origins = origins.loc[origins["date"].ge(origin_start)].copy()
    if origin_end is not None:
        origins = origins.loc[origins["date"].le(origin_end)].copy()
    labels = labels.loc[
        labels["target_date"].between(
            origins["date"].min() + pd.Timedelta(days=min(horizons)),
            origins["date"].max() + pd.Timedelta(days=max_horizon),
        )
    ]
    targets = []

    for horizon in horizons:
        horizon_frame = origins.copy()
        horizon_frame["forecast_horizon"] = np.int8(horizon)
        horizon_frame["target_date"] = horizon_frame["date"] + pd.Timedelta(days=horizon)
        targets.append(
            horizon_frame.merge(labels, on=KEY_COLUMNS + ["target_date"], how="left")
        )

    result = pd.concat(targets, ignore_index=True)
    seasonal_columns = []

    for week in range(1, 5):
        reference_column = f"seasonal_reference_date_{week}w"
        sales_column = f"sales_same_weekday_{week}w"
        seasonal_columns.append(sales_column)
        result[reference_column] = result["target_date"] - pd.Timedelta(days=7 * week)
        seasonal_history = frame.loc[
            frame["date"].between(
                result[reference_column].min(), result[reference_column].max()
            ),
            KEY_COLUMNS + ["date", "sales"],
        ]
        seasonal = seasonal_history.rename(
            columns={"date": reference_column, "sales": sales_column}
        )
        result = result.merge(
            seasonal,
            on=KEY_COLUMNS + [reference_column],
            how="left",
        )

    result["seasonal_reference_date"] = result["seasonal_reference_date_1w"]
    result["sales_same_weekday_last_week"] = result["sales_same_weekday_1w"]
    result["sales_same_weekday_4w_mean"] = result[seasonal_columns].mean(axis=1)
    temporary_columns = [
        *(f"seasonal_reference_date_{week}w" for week in range(1, 5)),
        *seasonal_columns,
    ]

    return result.drop(columns=temporary_columns)


def build_store_holidays(
    holidays: pd.DataFrame,
    stores: pd.DataFrame,
) -> pd.DataFrame:
    holidays = holidays.loc[~holidays["transferred"]].copy()
    holidays["is_holiday"] = (~holidays["type"].eq("Work Day")).astype("int8")
    holidays["is_event"] = holidays["type"].eq("Event").astype("int8")
    holidays["is_earthquake"] = (
        holidays["description"].str.contains("Terremoto", case=False, na=False).astype("int8")
    )

    national = stores[["store_nbr"]].merge(
        holidays.loc[holidays["locale"].eq("National")], how="cross"
    )
    regional = stores[["store_nbr", "state"]].merge(
        holidays.loc[holidays["locale"].eq("Regional")],
        left_on="state",
        right_on="locale_name",
    )
    local = stores[["store_nbr", "city"]].merge(
        holidays.loc[holidays["locale"].eq("Local")],
        left_on="city",
        right_on="locale_name",
    )
    columns = ["store_nbr", "date", "is_holiday", "is_event", "is_earthquake"]

    return (
        pd.concat([national[columns], regional[columns], local[columns]])
        .groupby(["store_nbr", "date"], as_index=False)
        .max()
        .rename(columns={"date": "target_date"})
    )


def add_external_features(frame: pd.DataFrame, data: FavoritaData) -> pd.DataFrame:
    transactions = data.transactions.rename(
        columns={"transactions": "current_transactions"}
    )
    holidays = build_store_holidays(data.holidays, data.stores)

    frame = frame.merge(data.stores, on="store_nbr", how="left")
    frame = frame.merge(transactions, on=["store_nbr", "date"], how="left")
    frame = frame.merge(holidays, on=["store_nbr", "target_date"], how="left")

    frame["target_year"] = frame["target_date"].dt.year.astype("int16")
    frame["target_month"] = frame["target_date"].dt.month.astype("int8")
    frame["target_day"] = frame["target_date"].dt.day.astype("int8")
    frame["target_dayofweek"] = frame["target_date"].dt.dayofweek.astype("int8")
    frame["target_weekofyear"] = frame["target_date"].dt.isocalendar().week.astype("int8")
    frame["target_is_weekend"] = frame["target_dayofweek"].isin([5, 6]).astype("int8")
    frame[["is_holiday", "is_event", "is_earthquake"]] = frame[
        ["is_holiday", "is_event", "is_earthquake"]
    ].fillna(0)

    return frame


def build_feature_table(
    data: FavoritaData,
    horizons: tuple[int, ...],
    lags: tuple[int, ...],
    rolling_windows: tuple[int, ...],
    origin_days: int | None = None,
    history_frame: pd.DataFrame | None = None,
    origin_start: pd.Timestamp | None = None,
    origin_end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    frame = history_frame if history_frame is not None else add_sales_history(data.sales, lags, rolling_windows)
    frame = add_future_targets(frame, horizons, origin_days, origin_start, origin_end)
    frame = add_external_features(frame, data)
    frame = frame.dropna(
        subset=[
            "target_sales",
            "sales_same_weekday_last_week",
            "sales_same_weekday_4w_mean",
            *(f"sales_lag_{lag}" for lag in lags),
        ]
    )
    frame["target_sales"] = frame["target_sales"].clip(lower=0).astype("float32")

    return frame.reset_index(drop=True)


def get_feature_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column not in NON_FEATURE_COLUMNS]


def get_categorical_columns(frame: pd.DataFrame) -> list[str]:
    candidates = ["store_nbr", "family", "city", "state", "type", "cluster"]
    return [column for column in candidates if column in frame.columns]


def prepare_model_frame(
    frame: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
    model_name: str,
) -> pd.DataFrame:
    features = frame[feature_columns].copy()

    for column in categorical_columns:
        features[column] = features[column].astype("string").fillna("unknown")
        if model_name == "lightgbm":
            features[column] = features[column].astype("category")

    numeric_columns = features.columns.difference(categorical_columns)
    features[numeric_columns] = features[numeric_columns].replace([np.inf, -np.inf], np.nan)

    return features
