from typing import Any

import pandas as pd


def fit_lightgbm(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    valid_x: pd.DataFrame,
    valid_y: pd.Series,
    params: dict,
    random_state: int,
) -> Any:
    from lightgbm import LGBMRegressor, early_stopping

    model = LGBMRegressor(**params, random_state=random_state)
    model.fit(
        train_x,
        train_y,
        eval_set=[(valid_x, valid_y)],
        eval_metric="l1",
        callbacks=[early_stopping(100, verbose=False)],
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
) -> Any:
    from catboost import CatBoostRegressor

    model = CatBoostRegressor(**params, random_seed=random_state)
    model.fit(
        train_x,
        train_y,
        eval_set=(valid_x, valid_y),
        cat_features=categorical_columns,
        early_stopping_rounds=100,
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
) -> Any:
    if model_name == "lightgbm":
        return fit_lightgbm(train_x, train_y, valid_x, valid_y, params, random_state)

    if model_name == "catboost":
        return fit_catboost(
            train_x,
            train_y,
            valid_x,
            valid_y,
            categorical_columns,
            params,
            random_state,
        )

    raise ValueError(f"Unsupported model: {model_name}")

