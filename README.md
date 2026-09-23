# Retail Demand Forecasting

This project forecasts daily sales for each Favorita store and product family from one to seven days ahead. It supports replenishment and promotion decisions rather than treating forecasting as a leaderboard-only exercise.

[한국어 README](README_KR.md)

## Problem definition

- **Prediction unit:** store × product family × target date
- **Target:** non-negative daily sales from one to seven days ahead
- **Main model:** LightGBM
- **Comparison model:** CatBoost
- **Model structure:** one global model with `forecast_horizon=1..7`
- **Baselines:** previous-week daily profile and four-week same-weekday average
- **Validation:** the latest 28 forecast-origin dates, with later labels purged from training
- **Metrics:** RMSLE, MAE, WAPE, and Bias

Features available at forecast origin `t` are shared across seven rows, while `forecast_horizon` changes from 1 to 7 to predict `t+1` through `t+7`. Target-date inputs are limited to promotions, calendar attributes, and holidays known in advance. Daily forecasts can be aggregated over the replenishment window of each product group.

Lag and rolling features are calculated from the full history, after which only the latest 365 forecast-origin dates are expanded across horizons 1–7. The resulting training table is cached as Parquet under `data/processed` and reused across model runs.

## Data

Download the files from Kaggle's [Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data) competition and place them in `data/raw`.

Required files:

```text
train.csv
stores.csv
holidays_events.csv
transactions.csv
```

The raw data is excluded from Git because it is distributed under the Kaggle competition rules.

## Method

The pipeline combines:

- origin sales and sales lags at 1, 7, 14, 28, and 56 days;
- 7-, 14-, and 28-day rolling sales averages;
- previous-week sales and a four-week average for the same target weekday;
- forecast horizon from 1 to 7;
- target-date promotion, calendar, holiday, and event information;
- store, family, and origin-date transactions.

All sales rolling features are shifted by at least one day. Targets are joined by exact date, store, and family rather than generated through an unchecked row shift. Oil price is excluded because its relationship with daily store-family demand is difficult to justify clearly.

## Repository structure

```text
configs/                 Experiment settings
data/raw/                Kaggle source files, not tracked
models/                  Trained models, not tracked
outputs/                 Metrics and validation results, not tracked
scripts/run_experiment.py
src/retail_forecast/     Data, features, models, and pipeline
notebooks/01_eda.ipynb
notebooks/03_baseline_model.ipynb
notebooks/04_feature_engineering.ipynb
notebooks/05_model_comparison.ipynb
notebooks/06_error_analysis.ipynb
tests/                   Leakage and metric tests
```

## Run

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
python scripts/run_experiment.py --config configs/baseline.toml
python scripts/run_experiment.py --config configs/catboost.toml
python scripts/run_permutation_importance.py --config configs/baseline.toml
```

Rebuild the Parquet feature cache only after changing feature definitions or the training window:

```bash
python scripts/run_experiment.py --config configs/baseline.toml --rebuild-features
```

Use `configs/catboost.toml` for local CPU training and
`configs/catboost_gpu.toml` for NVIDIA GPU training.

## Colab GPU run

Place the following untracked feature cache in the cloned repository:

```text
data/processed/features_h1-7_d365_l1-7-14-28-56_r7-14-28.parquet
```

It contains the multi-horizon training table generated for the latest 365
forecast-origin dates. When this file is present, the raw CSV files and
`--rebuild-features` are not required.

```bash
pip install -r requirements.txt
pip install -e .
python scripts/run_experiment.py --config configs/catboost_gpu.toml
```

Copy `models` and `outputs` to persistent storage before the Colab runtime is
released. GPU training can vary slightly between runs because floating-point
reduction order is non-deterministic.

Generated artifacts:

- `models/<model>_h1-h7.joblib`
- `outputs/metrics_<model>.json`
- `outputs/metrics_by_horizon_<model>.csv`
- `outputs/validation_predictions_<model>.csv`
- `outputs/decision_summary_<model>.csv` (daily `h1..h7` forecasts and 1–3/1–7 day sums)
- `outputs/feature_importance_<model>.csv`
- `outputs/permutation_importance.csv`

## Current multi-horizon result

The model was trained on the latest 365 forecast-origin dates and evaluated on
the latest 28 origins.

| Model | RMSLE | MAE | WAPE | Bias |
|---|---:|---:|---:|---:|
| Previous-week same weekday | 0.5430 | 85.09 | 17.96% | 0.91% |
| Four-week same-weekday mean | 0.4617 | 69.92 | 14.76% | 0.83% |
| LightGBM | **0.3943** | **62.84** | **13.26%** | **-0.86%** |

Negative Bias indicates aggregate underforecasting; positive Bias indicates
aggregate overforecasting.

## Previous single-horizon baseline result

The first full-data run used 2,831,598 training rows and 49,896 validation rows. Validation covered the latest 28 target dates beginning on July 19, 2017.

| Model | RMSLE | MAE |
|---|---:|---:|
| Seven-day seasonal naive | 0.5468 | 86.93 |
| LightGBM | **0.4079** | **66.58** |

LightGBM reduced RMSLE by 25.4% and MAE by 23.4% relative to the seasonal baseline. These figures belong to the previous single `t+7` experiment and are not directly comparable with the new 1-to-7-day global model. The multi-horizon run will report overall and horizon-specific metrics.

## Scope

The first version focuses on reproducible forecasting and leakage-safe validation. API serving, monitoring, orchestration, and TabFM are intentionally deferred. TabFM can later be evaluated as a zero-shot comparison on the same engineered table without changing the main experiment.
