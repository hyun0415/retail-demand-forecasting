from pathlib import Path
from time import perf_counter

import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import MSE
from neuralforecast.models import NHITS, TCN, TFT
from pytorch_lightning.callbacks import Callback
from torch.utils.data import DataLoader
from tqdm import tqdm

from retail_forecast.neural_data import (
    CALENDAR_COLUMNS,
    format_neural_predictions,
    future_calendar,
    history_through_origin,
    summarize_neural_predictions,
    training_before_origin,
)


MODEL_CLASSES = {"NHITS": NHITS, "TCN": TCN, "TFT": TFT}


class TrainingProgress(Callback):
    def __init__(self, name: str, max_steps: int):
        self.name = name
        self.max_steps = max_steps
        self.bar = None
        self.steps_per_epoch = None
        self.completed_steps = 0

    def on_train_start(self, trainer, pl_module):
        self.steps_per_epoch = int(trainer.num_training_batches)
        self.bar = tqdm(total=self.max_steps, desc=f"{self.name} 학습", unit="step", mininterval=5)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.bar.update(1)

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking or self.bar is None or not pl_module.valid_trajectories:
            return
        step, val_mse = pl_module.valid_trajectories[-1]
        tqdm.write(f"{self.name}: {step} step | 검증 MSE {val_mse:.4f}")

    def on_train_end(self, trainer, pl_module):
        self.completed_steps = int(trainer.global_step)
        self.bar.close()
        self.bar = None


def inspect_series_schedule(
    series_count: int, max_steps: int, batch_size: int = 32, windows_batch_size: int = 256
) -> pd.DataFrame:
    loader = DataLoader(range(series_count), batch_size=batch_size, shuffle=True)
    visits = [0] * series_count
    batches = iter(loader)
    for _ in range(max_steps):
        try:
            batch = next(batches)
        except StopIteration:
            batches = iter(loader)
            batch = next(batches)
        for series_id in batch.tolist():
            visits[series_id] += 1

    steps_per_epoch = len(loader)
    return pd.DataFrame([{
        "series_count": series_count,
        "series_per_batch": batch_size,
        "steps_per_epoch": steps_per_epoch,
        "steps": max_steps,
        "full_epochs": max_steps // steps_per_epoch,
        "steps_in_next_epoch": max_steps % steps_per_epoch,
        "min_series_visits": min(visits),
        "max_series_visits": max(visits),
        "windows_per_step": windows_batch_size,
        "window_draws": max_steps * windows_batch_size,
    }])


def make_neural_model(
    name: str,
    input_size: int,
    max_steps: int,
    batch_size: int = 32,
    windows_batch_size: int = 256,
    val_check_steps: int = 250,
):
    common = dict(
        h=7,
        input_size=input_size,
        futr_exog_list=CALENDAR_COLUMNS,
        loss=MSE(),
        max_steps=max_steps,
        val_check_steps=val_check_steps,
        batch_size=batch_size,
        windows_batch_size=windows_batch_size,
        scaler_type="robust",
        random_seed=42,
        accelerator="gpu",
        devices=1,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[TrainingProgress(name, max_steps)],
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
    windows_batch_size: int = 256,
    val_check_steps: int = 250,
    train_end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    started = perf_counter()
    first_origin = template["date"].min()
    if train_end is not None and train_end >= first_origin:
        raise ValueError("학습 종료일은 첫 검증 기준일보다 이전이어야 합니다.")
    if train_end is None:
        train = training_before_origin(panel, first_origin)
    else:
        if first_origin != train_end + pd.Timedelta(days=1) or template["date"].max() != train_end + pd.Timedelta(days=7):
            raise ValueError("내부 검증 기간은 첫 검증 기준일부터 연속 7일이어야 합니다.")
        train = panel.loc[panel["ds"].le(train_end + pd.Timedelta(days=7))]
    val_size = 7
    if train["ds"].nunique() < input_size + val_size + 7:
        raise ValueError("학습 이력이 입력 길이와 예측 길이를 채우지 못합니다.")

    model = NeuralForecast(
        models=[make_neural_model(name, input_size, max_steps, batch_size, windows_batch_size, val_check_steps)],
        freq="D",
    )
    gradient_end = train["ds"].max() - pd.Timedelta(days=val_size)
    tqdm.write(
        f"{name}: 학습 시작 ({len(train):,}행, 학습 정답 ~{gradient_end.date()}, "
        f"내부 검증 {(gradient_end + pd.Timedelta(days=1)).date()}~{train['ds'].max().date()})"
    )
    model.fit(df=train, val_size=val_size)
    progress = model.models[0].trainer_kwargs["callbacks"][0]
    full_epochs, extra_steps = divmod(progress.completed_steps, progress.steps_per_epoch)
    tqdm.write(
        f"{name}: 학습 완료 ({perf_counter() - started:.0f}초, "
        f"{progress.completed_steps} step = {full_epochs} epoch + {extra_steps} step; "
        f"epoch당 {progress.steps_per_epoch} step)"
    )
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    loss_curve = pd.DataFrame(model.models[0].valid_trajectories, columns=["step", "val_mse"])
    loss_curve.loc[loss_curve["step"] > 0].to_csv(
        output_path / f"validation_loss_{name.lower()}.csv", index=False
    )
    pd.DataFrame(model.models[0].train_trajectories, columns=["step", "train_mse_scaled"]).to_csv(
        output_path / f"validation_training_loss_{name.lower()}.csv", index=False
    )

    metrics = evaluate_neural_model(name, model, panel, series, template, output_path, input_size, "validation")
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path=str(model_path), save_dataset=False, overwrite=True)
    tqdm.write(f"{name}: 결과 저장 완료 ({perf_counter() - started:.0f}초)")
    return metrics


def run_neural_test(
    name: str,
    panel: pd.DataFrame,
    series: pd.DataFrame,
    template: pd.DataFrame,
    output_path: str | Path,
    model_path: str | Path,
    input_size: int = 56,
) -> pd.DataFrame:
    model = NeuralForecast.load(path=str(model_path))
    return evaluate_neural_model(name, model, panel, series, template, output_path, input_size, "test")


def evaluate_neural_model(
    name: str,
    model: NeuralForecast,
    panel: pd.DataFrame,
    series: pd.DataFrame,
    template: pd.DataFrame,
    output_path: str | Path,
    input_size: int,
    phase: str,
) -> pd.DataFrame:
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    predictions = []
    origins = sorted(template["date"].unique())
    for origin in tqdm(origins, desc=f"{name} 기준일 예측", unit="기준일", mininterval=1):
        origin = pd.Timestamp(origin)
        history = history_through_origin(panel, origin, input_size)
        future = future_calendar(series, origin)
        forecast = model.predict(df=history, futr_df=future).reset_index()
        predictions.append(format_neural_predictions(forecast, template, series, origin, name))

    result = pd.concat(predictions, ignore_index=True)
    metrics, decisions = summarize_neural_predictions(result, name)
    result.to_parquet(output_path / f"{phase}_predictions_{name.lower()}.parquet", index=False)
    metrics.to_csv(output_path / f"{phase}_metrics_{name.lower()}.csv", index=False)
    decisions.to_parquet(output_path / f"{phase}_decision_summary_{name.lower()}.parquet", index=False)
    return metrics
