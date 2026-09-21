# Retail Demand Forecasting

This project predicts sales for each Favorita store and product family seven days ahead. It treats demand forecasting as a decision-support problem for inventory and promotion planning rather than as a leaderboard-only exercise.

[한국어 README](README_KR.md)

## Problem definition

- **Prediction unit:** store × product family × target date
- **Target:** non-negative sales seven days ahead
- **Main model:** LightGBM
- **Comparison model:** CatBoost
- **Baseline:** sales from the same store and family seven days earlier
- **Validation:** the most recent 28 target dates, separated chronologically
- **Primary metric:** RMSLE, supported by MAE

The forecast horizon is explicit. Features observed on date `t` predict sales on `t + 7`, while only scheduled promotion and calendar information from the target date are used as future-known variables.

## Data

Download the files from Kaggle's [Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data) competition and place them in `data/raw`.

Required files:

```text
train.csv
stores.csv
oil.csv
holidays_events.csv
transactions.csv
```

The raw data is excluded from Git because it is distributed under the Kaggle competition rules.

## Method

The pipeline combines:

- sales lags at 1, 7, 14, 28, and 56 days;
- 7-, 14-, and 28-day rolling sales averages;
- promotion, store, and product-family information;
- transaction volume, oil price, holidays, events, and calendar variables.

All sales rolling features are shifted by at least one day. The target is joined by its exact future date instead of being generated through an unchecked row shift.

## Repository structure

```text
configs/                 Experiment settings
data/raw/                Kaggle source files, not tracked
models/                  Trained models, not tracked
outputs/                 Metrics and validation results, not tracked
scripts/run_experiment.py
src/retail_forecast/     Data, features, models, and pipeline
notebooks/                EDA and model error analysis
tests/                   Leakage and metric tests
```

## Run

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
python scripts/run_experiment.py --config configs/baseline.toml
```

To use CatBoost, change `model = "lightgbm"` to `model = "catboost"` in the configuration file.

Generated artifacts:

- `models/<model>_h7.joblib`
- `outputs/metrics.json`
- `outputs/validation_predictions.csv`
- `outputs/feature_importance.csv`

## First baseline result

The first full-data run used 2,831,598 training rows and 49,896 validation rows. Validation covered the latest 28 target dates beginning on July 19, 2017.

| Model | RMSLE | MAE |
|---|---:|---:|
| Seven-day seasonal naive | 0.5468 | 86.93 |
| LightGBM | **0.4079** | **66.58** |

LightGBM reduced RMSLE by 25.4% and MAE by 23.4% relative to the seasonal baseline. These figures establish the initial benchmark; family-level errors and temporal stability still require further analysis.

## Scope

The first version focuses on reproducible forecasting and leakage-safe validation. API serving, monitoring, orchestration, and TabFM are intentionally deferred. TabFM can later be evaluated as a zero-shot comparison on the same engineered table without changing the main experiment.
