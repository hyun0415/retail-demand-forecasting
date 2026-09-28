from pathlib import Path

import numpy as np
import pandas as pd

from retail_forecast.neural_data import KEYS, RESULT_COLUMNS, summarize_neural_predictions


def load_common_predictions(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, parse_dates=["date", "target_date"])
    frame = frame[RESULT_COLUMNS].copy()
    frame[["date", "target_date"]] = frame[["date", "target_date"]].apply(pd.to_datetime)
    frame["forecast_horizon"] = frame["forecast_horizon"].astype("int8")
    frame["store_nbr"] = frame["store_nbr"].astype("int16")
    frame["family"] = frame["family"].astype("string")
    if frame.duplicated(KEYS).any() or frame[RESULT_COLUMNS].isna().any().any():
        raise ValueError(f"예측 결과에 중복 키 또는 결측값이 있습니다: {path}")
    return frame.sort_values(KEYS).reset_index(drop=True)


def compare_on_common_rows(paths: dict[str, str | Path]) -> pd.DataFrame:
    frames = {name: load_common_predictions(path) for name, path in paths.items()}
    reference = next(iter(frames.values()))
    results = []
    for name, frame in frames.items():
        if not frame[KEYS].equals(reference[KEYS]):
            raise ValueError(f"{name}: 기준일·목표일·점포·상품군 검증 행이 다릅니다.")
        if not np.allclose(frame["target_sales"], reference["target_sales"], rtol=0, atol=1e-5):
            raise ValueError(f"{name}: 동일한 검증 행의 실제 판매량이 다릅니다.")
        metrics, _ = summarize_neural_predictions(frame, name)
        results.append(metrics)
    return pd.concat(results, ignore_index=True)
