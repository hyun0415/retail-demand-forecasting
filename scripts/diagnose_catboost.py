import argparse
import gc
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from retail_forecast.config import load_config
from retail_forecast.features import (
    get_categorical_columns,
    get_feature_columns,
    prepare_model_frame,
)
from retail_forecast.metrics import regression_metrics
from retail_forecast.models import fit_catboost
from retail_forecast.pipeline import feature_cache_path, get_feature_importance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare short CatBoost GPU MAE and RMSE runs")
    parser.add_argument("--config", default="configs/full_catboost_gpu.toml")
    parser.add_argument("--train-origin-days", type=int, default=180)
    parser.add_argument("--iterations", type=int, default=300)
    return parser.parse_args()


def load_diagnostic_frames(config, train_origin_days):
    cache_path = feature_cache_path(config)
    parts = sorted(cache_path.glob("part-*.parquet"))
    if not parts:
        raise FileNotFoundError(f"전체 이력 Feature 캐시를 찾지 못했습니다: {cache_path}")

    last_origin = pd.read_parquet(parts[-1], columns=["date"])["date"].max()
    cutoff = last_origin - pd.Timedelta(days=config.validation_days - 1)
    train_start = cutoff - pd.Timedelta(days=train_origin_days)
    frame = pd.read_parquet(cache_path, filters=[("date", ">=", train_start)])
    train = frame.loc[(frame["date"] < cutoff) & (frame["target_date"] < cutoff)]
    valid = frame.loc[frame["date"] >= cutoff]
    if train.empty or valid.empty:
        raise ValueError("진단용 학습 또는 검증 데이터가 비어 있습니다")

    feature_columns = [
        column for column in get_feature_columns(frame)
        if column not in config.excluded_features
    ]
    categorical_columns = get_categorical_columns(frame)
    train_x = prepare_model_frame(train, feature_columns, categorical_columns, "catboost")
    valid_x = prepare_model_frame(valid, feature_columns, categorical_columns, "catboost")
    train_y = train["target_sales"]
    del frame, train
    gc.collect()
    return train_x, train_y, valid_x, valid, feature_columns, categorical_columns, cutoff


def diagnostic_metrics(actual, predicted, baseline, importance, objective, cutoff, train_rows):
    high_sales = actual >= 1000
    metrics = regression_metrics(actual, predicted)
    return {
        "objective": objective,
        "validation_origin_start": cutoff.date().isoformat(),
        "train_rows": train_rows,
        "validation_rows": len(actual),
        **metrics,
        "baseline_4w_wape": regression_metrics(actual, baseline)["wape"],
        "actual_mean": float(actual.mean()),
        "prediction_mean": float(predicted.mean()),
        "prediction_p99": float(np.quantile(predicted, 0.99)),
        "prediction_max": float(predicted.max()),
        "high_sales_rows": int(high_sales.sum()),
        "high_sales_actual_mean": float(actual[high_sales].mean()),
        "high_sales_prediction_mean": float(predicted[high_sales].mean()),
        "family_importance": float(importance.loc[importance.feature.eq("family"), "importance"].sum()),
        "store_importance": float(importance.loc[importance.feature.eq("store_nbr"), "importance"].sum()),
    }


def main() -> None:
    args = parse_args()
    if args.train_origin_days < 1 or args.iterations < 1:
        raise ValueError("학습 기준일 수와 반복 횟수는 양수여야 합니다")
    config = load_config(args.config)
    if config.model != "catboost" or config.training_origin_days != 0:
        raise ValueError("전체 이력 CatBoost 설정을 사용하세요")

    train_x, train_y, valid_x, valid, columns, categorical, cutoff = load_diagnostic_frames(
        config, args.train_origin_days
    )
    output_dir = config.output_path.parent / "diagnostics" / (
        f"catboost_t{args.train_origin_days}_i{args.iterations}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    actual = valid["target_sales"].to_numpy()
    baseline = valid["sales_same_weekday_4w_mean"].to_numpy()
    rows = []

    for objective in ("MAE", "RMSE"):
        params = {**config.model_params, "loss_function": objective, "iterations": args.iterations}
        print(f"{objective} 진단 시작: 학습 {len(train_y):,}행, 검증 {len(valid):,}행", flush=True)
        model = fit_catboost(
            train_x, train_y, valid_x, valid["target_sales"], categorical,
            params, config.random_state, progress=True,
        )
        predicted = np.clip(model.predict(valid_x), 0, None)
        importance = get_feature_importance(model, columns)
        rows.append(diagnostic_metrics(
            actual, predicted, baseline, importance, objective, cutoff, len(train_y)
        ))
        result = valid[["date", "target_date", "forecast_horizon", "store_nbr", "family", "target_sales"]].copy()
        result["prediction"] = predicted
        result.to_parquet(output_dir / f"predictions_{objective.lower()}.parquet", index=False)
        importance.to_csv(output_dir / f"feature_importance_{objective.lower()}.csv", index=False)
        del model, predicted, importance, result
        gc.collect()

    comparison = pd.DataFrame(rows)
    comparison.to_csv(output_dir / "comparison.csv", index=False)
    print(comparison.to_string(index=False))
    print(f"진단 결과 저장: {output_dir}")


if __name__ == "__main__":
    main()
