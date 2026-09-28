from pathlib import Path

import numpy as np
import pandas as pd

from retail_forecast.metrics import regression_metrics


KEYS = ["date", "target_date", "forecast_horizon", "store_nbr", "family"]
RESULT_COLUMNS = KEYS + ["target_sales", "prediction"]
CALENDAR_COLUMNS = ["dayofweek", "month"]


def load_validation_template(path: str | Path, expected_days: int = 28) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        usecols=KEYS + ["target_sales"],
        parse_dates=["date", "target_date"],
        dtype={"forecast_horizon": "int8", "store_nbr": "int16", "family": "string", "target_sales": "float64"},
    )
    if frame.empty:
        raise ValueError("ML 검증 예측 파일이 비어 있습니다.")
    origins = pd.date_range(frame["date"].min(), periods=expected_days, freq="D")
    if not frame["date"].drop_duplicates().sort_values().reset_index(drop=True).equals(pd.Series(origins)):
        raise ValueError(f"ML 검증 기준일이 연속된 {expected_days}일이 아닙니다.")
    if frame.duplicated(KEYS).any() or not frame["forecast_horizon"].between(1, 7).all():
        raise ValueError("ML 검증 키 또는 horizon이 잘못되었습니다.")
    if set(frame["forecast_horizon"].unique()) != set(range(1, 8)):
        raise ValueError("ML 검증에 1~7일 horizon이 모두 있어야 합니다.")
    if not (frame["target_date"] == frame["date"] + pd.to_timedelta(frame["forecast_horizon"], unit="D")).all():
        raise ValueError("target_date가 기준일 + horizon과 다릅니다.")
    if frame["target_sales"].isna().any():
        raise ValueError("ML 검증 정답에 결측값이 있습니다.")
    return frame.sort_values(KEYS).reset_index(drop=True)


def load_neural_panel(raw_path: str | Path, template: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    sales = pd.read_csv(
        Path(raw_path) / "train.csv",
        usecols=["date", "store_nbr", "family", "sales"],
        parse_dates=["date"],
        dtype={"store_nbr": "int16", "family": "category", "sales": "float32"},
    )
    series = template[["store_nbr", "family"]].drop_duplicates().sort_values(["store_nbr", "family"])
    series = series.reset_index(drop=True)
    series.insert(0, "unique_id", np.arange(len(series), dtype="int32"))
    observed = sales.merge(series, on=["store_nbr", "family"], how="inner")
    if observed.duplicated(["unique_id", "date"]).any():
        raise ValueError("원본 판매 데이터에 중복된 점포·상품군·날짜가 있습니다.")
    if observed["date"].max() < template["target_date"].max():
        raise ValueError("원본 판매 이력이 ML 검증 목표일보다 짧습니다.")
    dates = pd.date_range(observed["date"].min(), template["target_date"].max(), freq="D")
    index = pd.MultiIndex.from_product([series["unique_id"], dates], names=["unique_id", "ds"])
    panel = observed.set_index(["unique_id", "date"])["sales"].reindex(index).rename("y").reset_index()
    panel["available_mask"] = panel["y"].notna().astype("float32")
    panel["y"] = panel["y"].fillna(0).clip(lower=0).astype("float32")
    panel["dayofweek"] = panel["ds"].dt.dayofweek.astype("float32")
    panel["month"] = panel["ds"].dt.month.astype("float32")
    return panel, series


def future_calendar(series: pd.DataFrame, origin: pd.Timestamp, horizon: int = 7) -> pd.DataFrame:
    dates = pd.date_range(origin + pd.Timedelta(days=1), periods=horizon, freq="D")
    index = pd.MultiIndex.from_product([series["unique_id"], dates], names=["unique_id", "ds"])
    future = index.to_frame(index=False)
    future["dayofweek"] = future["ds"].dt.dayofweek.astype("float32")
    future["month"] = future["ds"].dt.month.astype("float32")
    return future


def training_before_origin(panel: pd.DataFrame, first_origin: pd.Timestamp, days: int | None = None) -> pd.DataFrame:
    train = panel.loc[panel["ds"].lt(first_origin)]
    if days is not None:
        train = train.loc[train["ds"].ge(first_origin - pd.Timedelta(days=days))]
    return train


def history_through_origin(panel: pd.DataFrame, origin: pd.Timestamp, input_size: int) -> pd.DataFrame:
    return panel.loc[panel["ds"].between(origin - pd.Timedelta(days=input_size - 1), origin)]


def format_neural_predictions(
    forecast: pd.DataFrame,
    template: pd.DataFrame,
    series: pd.DataFrame,
    origin: pd.Timestamp,
    model_name: str,
) -> pd.DataFrame:
    forecast = forecast[["unique_id", "ds", model_name]].rename(columns={"ds": "target_date", model_name: "prediction"})
    forecast = forecast.merge(series, on="unique_id", validate="many_to_one")
    forecast["date"] = origin
    forecast["forecast_horizon"] = (forecast["target_date"] - origin).dt.days.astype("int8")
    expected = template.loc[template["date"].eq(origin)]
    result = expected.merge(forecast[KEYS + ["prediction"]], on=KEYS, how="left", validate="one_to_one")
    if result["prediction"].isna().any() or len(result) != len(expected):
        raise ValueError("DL 예측 키가 ML 검증 행과 일치하지 않습니다.")
    result["prediction"] = result["prediction"].clip(lower=0).astype("float32")
    return result[RESULT_COLUMNS]


def summarize_neural_predictions(predictions: pd.DataFrame, model_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = [{"model": model_name, "scope": "daily", "forecast_horizon": 0,
             "rows": len(predictions), **regression_metrics(predictions["target_sales"], predictions["prediction"])}]
    for horizon, group in predictions.groupby("forecast_horizon"):
        rows.append({"model": model_name, "scope": "daily", "forecast_horizon": int(horizon),
                     "rows": len(group), **regression_metrics(group["target_sales"], group["prediction"])})
    decisions = predictions.pivot(
        index=["date", "store_nbr", "family"],
        columns="forecast_horizon",
        values=["target_sales", "prediction"],
    )
    decisions.columns = [f"{'actual' if value == 'target_sales' else 'prediction'}_h{horizon}" for value, horizon in decisions.columns]
    decisions = decisions.reset_index()
    for horizon in (3, 7):
        decisions[f"actual_sum_1_{horizon}"] = decisions[[f"actual_h{h}" for h in range(1, horizon + 1)]].sum(axis=1, min_count=horizon)
        decisions[f"prediction_sum_1_{horizon}"] = decisions[[f"prediction_h{h}" for h in range(1, horizon + 1)]].sum(axis=1, min_count=horizon)
    complete = decisions.dropna(subset=["actual_sum_1_7", "prediction_sum_1_7"])
    rows.append({"model": model_name, "scope": "7_day_sum", "forecast_horizon": 0,
                 "rows": len(complete), **regression_metrics(complete["actual_sum_1_7"], complete["prediction_sum_1_7"])})
    return pd.DataFrame(rows), decisions
