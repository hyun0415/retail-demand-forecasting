from pathlib import Path
from time import perf_counter

import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import MSE
from neuralforecast.models import NHITS, TCN, TFT
from tqdm.auto import tqdm

from retail_forecast.neural_data import (
    CALENDAR_COLUMNS,
    format_neural_predictions,
    future_calendar,
    history_through_origin,
    summarize_neural_predictions,
    training_before_origin,
)


MODEL_CLASSES = {"NHITS": NHITS, "TCN": TCN, "TFT": TFT}


def make_neural_model(name: str, input_size: int, max_steps: int, batch_size: int = 32):
    common = dict(
        h=7,
        input_size=input_size,
        futr_exog_list=CALENDAR_COLUMNS,
        loss=MSE(),
        max_steps=max_steps,
        batch_size=batch_size,
        windows_batch_size=256,
        scaler_type="robust",
        random_seed=42,
        accelerator="gpu",
        devices=1,
        enable_progress_bar=True,
        logger=False,
    )
    if name == "NHITS":
        common["mlp_units"] = 3 * [[128, 128]]
    if name == "TCN":
        common["encoder_hidden_size"] = 64
        common["decoder_hidden_size"] = 64
    if name == "TFT":
        common["hidden_size"] = 32
        common["n_head"] = 4
    return MODEL_CLASSES[name](**common)


def run_neural_validation(
    name: str,
    panel: pd.DataFrame,
    series: pd.DataFrame,
    template: pd.DataFrame,
    output_path: str | Path,
    model_path: str | Path,
    input_size: int = 56,
    max_steps: int = 300,
    batch_size: int = 32,
    training_history_days: int | None = None,
) -> pd.DataFrame:
    started = perf_counter()
    first_origin = template["date"].min()
    train = training_before_origin(panel, first_origin, training_history_days)
    if train["ds"].nunique() < input_size + 14:
        raise ValueError("학습 이력이 입력 길이와 검증용 7일을 채우지 못합니다.")

    model = NeuralForecast(models=[make_neural_model(name, input_size, max_steps, batch_size)], freq="D")
    tqdm.write(f"{name}: 학습 시작 ({len(train):,}행, {train['ds'].min().date()}~{train['ds'].max().date()})")
    model.fit(df=train, val_size=7)
    tqdm.write(f"{name}: 학습 완료 ({perf_counter() - started:.0f}초)")

    predictions = []
    origins = sorted(template["date"].unique())
    for origin in tqdm(origins, desc=f"{name} 기준일 예측"):
        origin = pd.Timestamp(origin)
        history = history_through_origin(panel, origin, input_size)
        future = future_calendar(series, origin)
        forecast = model.predict(df=history, futr_df=future).reset_index()
        predictions.append(format_neural_predictions(forecast, template, series, origin, name))

    result = pd.concat(predictions, ignore_index=True)
    metrics, decisions = summarize_neural_predictions(result, name)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path / f"validation_predictions_{name.lower()}.parquet", index=False)
    metrics.to_csv(output_path / f"metrics_{name.lower()}.csv", index=False)
    decisions.to_parquet(output_path / f"decision_summary_{name.lower()}.parquet", index=False)
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path=str(model_path), save_dataset=False, overwrite=True)
    tqdm.write(f"{name}: 결과 저장 완료 ({perf_counter() - started:.0f}초)")
    return metrics
