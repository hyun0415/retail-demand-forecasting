import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

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


def temporal_split(
    frame: pd.DataFrame,
    validation_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    cutoff = frame["target_date"].max() - pd.Timedelta(days=validation_days - 1)
    train = frame.loc[frame["target_date"] < cutoff].copy()
    valid = frame.loc[frame["target_date"] >= cutoff].copy()

    return train, valid, cutoff


def get_feature_importance(model: Any, columns: list[str]) -> pd.DataFrame:
    values = getattr(model, "feature_importances_", np.zeros(len(columns)))

    return (
        pd.DataFrame({"feature": columns, "importance": values})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def save_artifacts(
    config: ExperimentConfig,
    model: Any,
    metrics: dict[str, float | str | int],
    predictions: pd.DataFrame,
    importance: pd.DataFrame,
) -> None:
    import joblib

    config.model_path.mkdir(parents=True, exist_ok=True)
    config.output_path.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, config.model_path / f"{config.model}_h{config.forecast_horizon}.joblib")
    predictions.to_csv(config.output_path / "validation_predictions.csv", index=False)
    importance.to_csv(config.output_path / "feature_importance.csv", index=False)

    with (config.output_path / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)


def run_experiment(config: ExperimentConfig) -> dict[str, float | str | int]:
    data = load_favorita_data(config.raw_path)
    frame = build_feature_table(
        data,
        config.forecast_horizon,
        config.lags,
        config.rolling_windows,
    )
    train, valid, cutoff = temporal_split(frame, config.validation_days)
    feature_columns = get_feature_columns(frame)
    categorical_columns = get_categorical_columns(frame)
    train_x = prepare_model_frame(train, feature_columns, categorical_columns, config.model)
    valid_x = prepare_model_frame(valid, feature_columns, categorical_columns, config.model)

    model = fit_model(
        config.model,
        train_x,
        train["target_sales"],
        valid_x,
        valid["target_sales"],
        categorical_columns,
        config.model_params,
        config.random_state,
    )
    predicted = np.clip(model.predict(valid_x), 0, None)
    metrics = regression_metrics(valid["target_sales"].to_numpy(), predicted)
    naive_metrics = regression_metrics(
        valid["target_sales"].to_numpy(),
        valid["sales"].to_numpy(),
    )
    metrics.update(
        {
            "seasonal_naive_rmsle": naive_metrics["rmsle"],
            "seasonal_naive_mae": naive_metrics["mae"],
            "model": config.model,
            "forecast_horizon": config.forecast_horizon,
            "validation_start": cutoff.date().isoformat(),
            "train_rows": len(train),
            "validation_rows": len(valid),
        }
    )
    predictions = valid[["store_nbr", "family", "target_date", "target_sales"]].copy()
    predictions["prediction"] = predicted
    importance = get_feature_importance(model, feature_columns)
    save_artifacts(config, model, metrics, predictions, importance)

    return metrics
