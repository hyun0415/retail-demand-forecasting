import re
from typing import Any

import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor, early_stopping
from tqdm.auto import tqdm


class LightGBMProgress:
    order = 20
    before_iteration = False

    def __init__(self, bar: tqdm) -> None:
        self.bar = bar

    def __call__(self, environment: Any) -> None:
        iteration = environment.iteration + 1
        self.bar.update(iteration - self.bar.n)
        if iteration % 50 == 0 and environment.evaluation_result_list:
            _, metric, value, _ = environment.evaluation_result_list[0]
            tqdm.write(f"[{iteration}] 검증 {metric}: {value:.4f}")


class CatBoostProgress:
    def __init__(self, bar: tqdm) -> None:
        self.bar = bar
        self.buffer = ""

    def write(self, message: str) -> None:
        self.buffer += message
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            match = re.match(r"^\s*(\d+):", line)
            if match:
                iteration = int(match.group(1)) + 1
                self.bar.update(iteration - self.bar.n)
                tqdm.write(f"[{iteration}] {line.split(':', 1)[1].strip()}")

    def flush(self) -> None:
        pass


def fit_lightgbm(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    valid_x: pd.DataFrame,
    valid_y: pd.Series,
    params: dict,
    random_state: int,
    progress: bool = False,
) -> Any:
    model_params = params.copy()
    patience = model_params.pop("early_stopping_rounds", 100)
    model = LGBMRegressor(**model_params, random_state=random_state)
    callbacks = [early_stopping(patience, verbose=False)]
    with tqdm(total=model.n_estimators, desc="LightGBM 학습", unit="회", disable=not progress) as bar:
        if progress:
            callbacks.append(LightGBMProgress(bar))
        model.fit(
            train_x,
            train_y,
            eval_X=valid_x,
            eval_y=valid_y,
            eval_metric="rmse",
            callbacks=callbacks,
        )

    return model


def fit_catboost(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    valid_x: pd.DataFrame,
    valid_y: pd.Series,
    categorical_columns: list[str],
    params: dict,
    random_state: int,
    progress: bool = False,
) -> Any:
    model_params = params.copy()
    patience = model_params.pop("early_stopping_rounds", 100)
    model = CatBoostRegressor(**model_params, random_seed=random_state)
    with tqdm(total=model.get_param("iterations") or 1000, desc="CatBoost 학습", unit="회", disable=not progress) as bar:
        fit_options = {"log_cout": CatBoostProgress(bar)} if progress else {}
        model.fit(
            train_x,
            train_y,
            eval_set=(valid_x, valid_y),
            cat_features=categorical_columns,
            early_stopping_rounds=patience,
            **fit_options,
        )

    return model


def fit_model(
    model_name: str,
    train_x: pd.DataFrame,
    train_y: pd.Series,
    valid_x: pd.DataFrame,
    valid_y: pd.Series,
    categorical_columns: list[str],
    params: dict,
    random_state: int,
    progress: bool = False,
) -> Any:
    if model_name == "lightgbm":
        return fit_lightgbm(train_x, train_y, valid_x, valid_y, params, random_state, progress)

    if model_name == "catboost":
        return fit_catboost(
            train_x,
            train_y,
            valid_x,
            valid_y,
            categorical_columns,
            params,
            random_state,
            progress,
        )

    raise ValueError(f"Unsupported model: {model_name}")
