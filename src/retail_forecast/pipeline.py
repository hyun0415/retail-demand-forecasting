import json
import joblib
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from retail_forecast.config import ExperimentConfig
from retail_forecast.data import load_favorita_data
from retail_forecast.features import (
    build_feature_table,
    get_categorical_columns,
    get_feature_columns,
    prepare_model_frame,
)
from retail_forecast.metrics import regression_metrics
from retail_forecast.models import fit_model


def feature_cache_path(config: ExperimentConfig) -> Path:
    horizon_label = f"h{min(config.forecast_horizons)}-{max(config.forecast_horizons)}"
    lag_label = "-".join(map(str, config.lags))
    rolling_label = "-".join(map(str, config.rolling_windows))
    filename = (
        f"features_{horizon_label}_d{'all' if config.training_origin_days == 0 else config.training_origin_days}"
        f"_l{lag_label}_r{rolling_label}.parquet"
    )

    return config.processed_path / filename


def load_or_build_feature_table(
    config: ExperimentConfig,
    rebuild: bool = False,
) -> pd.DataFrame:
    cache_path = feature_cache_path(config)
    if cache_path.exists() and not rebuild:
        return pd.read_parquet(cache_path)

    data = load_favorita_data(config.raw_path)
    frame = build_feature_table(
        data,
        config.forecast_horizons,
        config.lags,
        config.rolling_windows,
        config.training_origin_days or None,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(cache_path, index=False)

    return frame


def temporal_split(
    frame: pd.DataFrame,
    validation_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    cutoff = frame["date"].max() - pd.Timedelta(days=validation_days - 1)
    train = frame.loc[frame["target_date"] < cutoff].copy()
    valid = frame.loc[frame["date"] >= cutoff].copy()

    return train, valid, cutoff


def get_feature_importance(model: Any, columns: list[str]) -> pd.DataFrame:
    values = getattr(model, "feature_importances_", np.zeros(len(columns)))

    return (
        pd.DataFrame({"feature": columns, "importance": values})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def build_decision_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    keys = ["date", "store_nbr", "family"]
    predicted = predictions.pivot(
        index=keys,
        columns="forecast_horizon",
        values="prediction",
    ).add_prefix("prediction_h")
    actual = predictions.pivot(
        index=keys,
        columns="forecast_horizon",
        values="target_sales",
    ).add_prefix("actual_h")
    summary = predicted.join(actual).reset_index()
    summary["prediction_sum_1_3"] = summary[
        [f"prediction_h{horizon}" for horizon in range(1, 4)]
    ].sum(axis=1, min_count=3)
    summary["prediction_sum_1_7"] = summary[
        [f"prediction_h{horizon}" for horizon in range(1, 8)]
    ].sum(axis=1, min_count=7)
    summary["actual_sum_1_3"] = summary[
        [f"actual_h{horizon}" for horizon in range(1, 4)]
    ].sum(axis=1, min_count=3)
    summary["actual_sum_1_7"] = summary[
        [f"actual_h{horizon}" for horizon in range(1, 8)]
    ].sum(axis=1, min_count=7)

    return summary


def save_artifacts(
    config: ExperimentConfig,
    model: Any,
    metrics: dict[str, float | str | int],
    predictions: pd.DataFrame,
    importance: pd.DataFrame,
    metrics_by_horizon: pd.DataFrame,
    decision_summary: pd.DataFrame,
) -> None:
    config.model_path.mkdir(parents=True, exist_ok=True)
    config.output_path.mkdir(parents=True, exist_ok=True)

    horizon_label = f"h{min(config.forecast_horizons)}-h{max(config.forecast_horizons)}"
    joblib.dump(model, config.model_path / f"{config.model}_{horizon_label}.joblib")
    predictions.to_csv(
        config.output_path / f"validation_predictions_{config.model}.csv",
        index=False,
    )
    importance.to_csv(
        config.output_path / f"feature_importance_{config.model}.csv",
        index=False,
    )
    metrics_by_horizon.to_csv(
        config.output_path / f"metrics_by_horizon_{config.model}.csv",
        index=False,
    )
    decision_summary.to_csv(
        config.output_path / f"decision_summary_{config.model}.csv",
        index=False,
    )

    with (config.output_path / f"metrics_{config.model}.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)


def run_experiment(
    config: ExperimentConfig,
    rebuild_features: bool = False,
    progress: bool = False,
) -> dict[str, float | str | int]:
    started = perf_counter()

    def report(stage: str) -> None:
        if progress:
            tqdm.write(f"[{perf_counter() - started:.0f}s] {stage}")

    report("Feature 데이터 읽기 시작")
    frame = load_or_build_feature_table(config, rebuild_features)
    report(f"Feature 데이터 읽기 완료: {len(frame):,}행")
    train, valid, cutoff = temporal_split(frame, config.validation_days)
    feature_columns = [
        column for column in get_feature_columns(frame)
        if column not in config.excluded_features
    ]
    categorical_columns = get_categorical_columns(frame)
    report(f"모델 입력 준비: 학습 {len(train):,}행, 검증 {len(valid):,}행")
    train_x = prepare_model_frame(train, feature_columns, categorical_columns, config.model)
    valid_x = prepare_model_frame(valid, feature_columns, categorical_columns, config.model)

    report(f"{config.model} 학습 시작")
    model = fit_model(
        config.model,
        train_x,
        train["target_sales"],
        valid_x,
        valid["target_sales"],
        categorical_columns,
        config.model_params,
        config.random_state,
        progress=progress,
    )
    report(f"{config.model} 학습 완료, 검증 예측 시작")
    predicted = np.clip(model.predict(valid_x), 0, None)
    metrics = regression_metrics(valid["target_sales"].to_numpy(), predicted)
    previous_week_metrics = regression_metrics(
        valid["target_sales"].to_numpy(),
        valid["sales_same_weekday_last_week"].to_numpy(),
    )
    four_week_metrics = regression_metrics(
        valid["target_sales"].to_numpy(),
        valid["sales_same_weekday_4w_mean"].to_numpy(),
    )
    metrics.update(
        {
            "previous_week_rmsle": previous_week_metrics["rmsle"],
            "previous_week_mae": previous_week_metrics["mae"],
            "previous_week_rmse": previous_week_metrics["rmse"],
            "previous_week_wape": previous_week_metrics["wape"],
            "previous_week_bias": previous_week_metrics["bias"],
            "same_weekday_4w_rmsle": four_week_metrics["rmsle"],
            "same_weekday_4w_mae": four_week_metrics["mae"],
            "same_weekday_4w_rmse": four_week_metrics["rmse"],
            "same_weekday_4w_wape": four_week_metrics["wape"],
            "same_weekday_4w_bias": four_week_metrics["bias"],
            "model": config.model,
            "objective": config.model_params["objective" if config.model == "lightgbm" else "loss_function"],
            "forecast_horizons": list(config.forecast_horizons),
            "validation_origin_start": cutoff.date().isoformat(),
            "train_rows": len(train),
            "validation_rows": len(valid),
        }
    )
    predictions = valid[
        [
            "date",
            "target_date",
            "forecast_horizon",
            "store_nbr",
            "family",
            "sales",
            "sales_same_weekday_last_week",
            "sales_same_weekday_4w_mean",
            "target_onpromotion",
            "target_sales",
        ]
    ].copy()
    predictions["prediction"] = predicted
    horizon_metrics = []
    for horizon, group in predictions.groupby("forecast_horizon", observed=True):
        model_metrics = regression_metrics(
            group["target_sales"].to_numpy(),
            group["prediction"].to_numpy(),
        )
        previous_week_metrics = regression_metrics(
            group["target_sales"].to_numpy(),
            group["sales_same_weekday_last_week"].to_numpy(),
        )
        four_week_metrics = regression_metrics(
            group["target_sales"].to_numpy(),
            group["sales_same_weekday_4w_mean"].to_numpy(),
        )
        horizon_metrics.append(
            {
                "forecast_horizon": int(horizon),
                **model_metrics,
                "previous_week_rmsle": previous_week_metrics["rmsle"],
                "previous_week_mae": previous_week_metrics["mae"],
                "previous_week_rmse": previous_week_metrics["rmse"],
                "previous_week_wape": previous_week_metrics["wape"],
                "previous_week_bias": previous_week_metrics["bias"],
                "same_weekday_4w_rmsle": four_week_metrics["rmsle"],
                "same_weekday_4w_mae": four_week_metrics["mae"],
                "same_weekday_4w_rmse": four_week_metrics["rmse"],
                "same_weekday_4w_wape": four_week_metrics["wape"],
                "same_weekday_4w_bias": four_week_metrics["bias"],
                "rows": len(group),
            }
        )
    metrics_by_horizon = pd.DataFrame(horizon_metrics)
    decision_summary = build_decision_summary(predictions)
    importance = get_feature_importance(model, feature_columns)
    report("결과 저장 시작")
    save_artifacts(
        config,
        model,
        metrics,
        predictions,
        importance,
        metrics_by_horizon,
        decision_summary,
    )
    report("결과 저장 완료")

    return metrics
