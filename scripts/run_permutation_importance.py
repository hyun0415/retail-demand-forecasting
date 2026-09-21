import argparse
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from retail_forecast.config import load_config
from retail_forecast.data import load_favorita_data
from retail_forecast.features import (
    build_feature_table,
    get_categorical_columns,
    get_feature_columns,
    prepare_model_frame,
)
from retail_forecast.pipeline import temporal_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calculate validation permutation importance")
    parser.add_argument("--config", default="configs/baseline.toml")
    parser.add_argument("--sample-size", type=int, default=30_000)
    parser.add_argument("--repeats", type=int, default=3)
    return parser.parse_args()


def negative_rmsle(model, features, target) -> float:
    predicted = np.clip(model.predict(features), 0, None)
    error = np.log1p(target) - np.log1p(predicted)
    return -float(np.sqrt(np.mean(np.square(error))))


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    data = load_favorita_data(config.raw_path)
    frame = build_feature_table(
        data,
        config.forecast_horizon,
        config.lags,
        config.rolling_windows,
    )
    _, valid, _ = temporal_split(frame, config.validation_days)
    sample = valid.sample(min(args.sample_size, len(valid)), random_state=config.random_state)
    feature_columns = get_feature_columns(frame)
    categorical_columns = get_categorical_columns(frame)
    sample_x = prepare_model_frame(
        sample,
        feature_columns,
        categorical_columns,
        config.model,
    )
    model = joblib.load(config.model_path / f"{config.model}_h{config.forecast_horizon}.joblib")
    result = permutation_importance(
        model,
        sample_x,
        sample["target_sales"],
        scoring=negative_rmsle,
        n_repeats=args.repeats,
        random_state=config.random_state,
        n_jobs=-1,
    )
    importance = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    importance.to_csv(config.output_path / "permutation_importance.csv", index=False)


if __name__ == "__main__":
    main()
