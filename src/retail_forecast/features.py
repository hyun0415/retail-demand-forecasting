import numpy as np
import pandas as pd

from retail_forecast.data import FavoritaData


KEY_COLUMNS = ["store_nbr", "family"]
NON_FEATURE_COLUMNS = ["date", "target_date", "target_sales"]


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


def add_future_target(frame: pd.DataFrame, horizon: int) -> pd.DataFrame:
    labels = frame[KEY_COLUMNS + ["date", "sales", "onpromotion"]].rename(
        columns={
            "date": "target_date",
            "sales": "target_sales",
            "onpromotion": "target_onpromotion",
        }
    )
    frame = frame.copy()
    frame["target_date"] = frame["date"] + pd.Timedelta(days=horizon)

    return frame.merge(labels, on=KEY_COLUMNS + ["target_date"], how="left")


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
    oil = data.oil.sort_values("date").copy()
    oil["dcoilwtico"] = oil["dcoilwtico"].interpolate().bfill().ffill()
    oil = oil.rename(columns={"date": "target_date", "dcoilwtico": "target_oil_price"})

    transactions = data.transactions.rename(
        columns={"transactions": "current_transactions"}
    )
    holidays = build_store_holidays(data.holidays, data.stores)

    frame = frame.merge(data.stores, on="store_nbr", how="left")
    frame = frame.merge(transactions, on=["store_nbr", "date"], how="left")
    frame = frame.merge(oil, on="target_date", how="left")
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
    horizon: int,
    lags: tuple[int, ...],
    rolling_windows: tuple[int, ...],
) -> pd.DataFrame:
    frame = add_sales_history(data.sales, lags, rolling_windows)
    frame = add_future_target(frame, horizon)
    frame = add_external_features(frame, data)
    frame = frame.dropna(subset=["target_sales", *(f"sales_lag_{lag}" for lag in lags)])
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

