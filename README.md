# Retail Demand Forecasting

This project forecasts daily sales for each Favorita store and product family from one to seven days ahead. Its primary use case is company-wide, store-level demand planning to inform replenishment and allocation; store teams can review their own forecasts. Promotion analysis is a related use case.

[한국어 README](README_KR.md)

## Problem definition

- **Prediction unit:** store × product family × target date
- **Target:** non-negative daily sales from one to seven days ahead
- **Main model:** LightGBM
- **Comparison model:** CatBoost
- **Model structure:** one global model with `forecast_horizon=1..7`
- **Baselines:** previous-week daily profile and four-week same-weekday average
- **Current Colab evaluation:** seven validation origins, a seven-day gap, and seven final test origins
- **Metrics:** RMSLE, MAE, RMSE, WAPE, and Bias

Features available at forecast origin `t` are shared across seven rows, while `forecast_horizon` changes from 1 to 7 to predict `t+1` through `t+7`. Target-date inputs are limited to calendar attributes known in advance; future observed promotion counts are excluded from model inputs. Daily forecasts can be aggregated over the replenishment window of each product group.

Lag and rolling features are calculated from the full history. The historical baseline expands the latest 365 forecast-origin dates across horizons 1–7; the full-history Colab comparison uses all eligible origins. Training tables are cached as Parquet under `data/processed` and reused across model runs.

## Business use and SCM boundary

One global model learns from all stores, but produces a separate daily forecast for each **store × product family**. The seven daily forecasts from origin `t` can be summed into projected sales for `t+1` through `t+7`. Forecasts and aggregate metrics are implemented; the role-specific views below are the intended interpretation work.

| User | Decision supported | View to develop |
|---|---|---|
| Central demand and replenishment team | Compare expected demand across stores and families; prioritize review and allocation planning | Store-family daily forecasts, seven-day totals, and error by store or demand segment |
| Store team | Review the local forecast and flag unusual demand before replenishment decisions | Forecasts and exceptions filtered to that store |
| Marketing team | Examine how demand patterns and forecast errors vary around planned promotions | Descriptive store-family and promotion comparisons; no causal lift estimate |

This is an **input to supply-chain planning**, not an automated purchase-order system. The data is at product-family level and does not provide SKU-level stock on hand, incoming inventory, supplier lead times, pack sizes, or service-level rules needed to calculate order quantities. Observed sales may also fall below unconstrained demand when items are out of stock. Therefore, the project reports forecast accuracy and decision-relevant demand patterns, not measured savings, prevented stockouts, or optimal orders. These boundaries also guide the planned store and demand-segment error analysis.

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
- known calendar and holiday information; current or past promotion counts;
- store, family, and origin-date transactions.

All sales rolling features are shifted by at least one day. Targets are joined by exact date, store, and family rather than generated through an unchecked row shift. Oil price is excluded because its relationship with daily store-family demand is difficult to justify clearly.

**Selection rationale:** The seven-day pattern observed in EDA motivates the [previous-week and four-week same-weekday baselines](notebooks/03_baseline_model.ipynb). A table of [feature groups, reference dates, and availability conditions](notebooks/04_feature_engineering.ipynb) documents the model inputs.

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
notebooks/05_colab_model_comparison.ipynb
notebooks/06_error_analysis.ipynb
notebooks/07_dl_colab_validation.ipynb
requirements-neural.txt    Separate NeuralForecast dependencies
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

### Chronological Train / Valid / Test protocol (current notebooks)

For the available Favorita history, the last eligible seven-day forecast origin is **2017-08-08**. Notebook 05 and notebook 07 now use the same complete store-family-origin-horizon rows:

| Stage | Origin dates | Labels used for fitting | Purpose |
|---|---|---|---|
| Initial fit | Before validation | Target dates through 2017-07-18 | Select ML tree count and compare DL settings on validation |
| Validation | 2017-07-19 to 2017-07-25 | No validation targets enter initial fit | Seven rolling daily origins, each predicting `t+1` to `t+7` |
| Gap | 2017-07-26 to 2017-08-01 | No weight updates; observed sales may enter each origin's history | Ensures the last validation target is known before the first test origin |
| Test | 2017-08-02 to 2017-08-08 | The validation-stage model stays fixed | Seven rolling daily origins, each predicting `t+1` to `t+7` |

For ML, validation selects the best iteration by early stopping, and **that same fitted model** predicts test rows. For DL, each model trains on labels through July 18 using its chosen `MODEL_STEPS`. Its seven-day internal validation uses July 19–25 **for monitoring MSE only**, with no gradient updates on those labels; the rolling-origin external validation also scores horizons reaching August 1. Internal MSE is printed every `VAL_CHECK_STEPS` and saved as `validation_loss_<model>.csv`. The saved model is loaded for test inference without another fit. At each forecast origin, lag/rolling features and DL history may use observed sales **through that origin only**. Thus later test origins have more observed history, while model weights remain fixed throughout validation and test. Both model families use the same seven-day output window and exact evaluation keys. Only series with all 49 validation and 49 test targets are kept in the common cohort. This does not require that every store-family has 56 nonzero sales days; a zero sale is an observed value, and globally missing dates are marked in the DL panel.

The earlier 28-origin results included these proposed test dates and were repeatedly inspected. The new test week is therefore a **retrospective holdout**, not a previously untouched estimate of future generalization. Do not tune again on its score. Full training can be expensive; notebook 05 then 07 should be run only when ready for a fresh experiment. The ML run is `full_history_time_split_fixed_i600`; the DL run with internal loss monitoring is `neural_time_split_val_loss`, preserving the older results. If a DL step setting changes after reviewing validation, rerun that model's validation cell before test so the saved checkpoint matches the selected setting.

Prepare the repository in My Drive with the separate 00 clone notebook, then open [`notebooks/05_colab_model_comparison.ipynb`](notebooks/05_colab_model_comparison.ipynb) in Colab. Notebook 05 mounts Drive and installs packages but does not clone or pull. It builds Parquet features in date chunks and trains LightGBM with L2 `regression` and CatBoost with `RMSE` on the same temporal split. LightGBM uses the prebuilt OpenCL GPU package (`device_type="gpu"`), so no CUDA compiler or source build is required. `MODEL_OVERRIDES` caps both models at 600 trees and sets CatBoost `gpu_ram_part=0.8`; both full-history and 365-day CatBoost GPU configs also use 0.8. `RUN_NAME` initially selects `full_history_time_split_fixed_i600`, keeping earlier results separate; change it when rerunning to preserve earlier outputs. Effective parameters and tree counts are saved with the metrics. Feature chunks and model iterations show progress bars, with stage timings and validation scores retained. Place the untracked Kaggle CSV files in `data/raw` after cloning. A high-RAM runtime is recommended; future observed promotion counts are excluded from model features. Unexpected future event and earthquake features are excluded from this comparison.

Every notebook prepares Seaborn and the bundled Nanum Gothic font near the top, then applies the Korean chart theme. Library cells contain imports, with utilities defined separately. Experiment parameters and execution follow.

### Neural GPU validation

[`07_dl_colab_validation.ipynb`](notebooks/07_dl_colab_validation.ipynb) prepares NeuralForecast N-HiTS (MLP), TCN (CNN), and TFT (LSTM with attention) in a separate Colab notebook. It prepares the full daily input and provides full training cells for the three models in order. Run the model cells you need. The notebook installs the separate `requirements-neural.txt`.

The notebook reads the ML validation and test predictions from `outputs/<ML_RUN_NAME>` and verifies seven consecutive validation origins and seven consecutive test origins. Common prediction columns are `date` (origin), `target_date`, `forecast_horizon`, `store_nbr`, `family`, `target_sales`, and `prediction`. Full neural validation and test use the exact corresponding ML evaluation rows. Training stops before the first validation origin; the saved model is loaded for test, and inference at each origin uses observed sales only through that date, plus known calendar fields. Missing raw dates have `available_mask=0`, distinct from observed zero sales. The neural models currently use fewer exogenous variables than ML, so performance differences also reflect different inputs.

Neural predictions, metrics, loss curves, and seven-day totals are saved under `outputs/neural_time_split_val_loss`; the test common-row comparison is `comparison_ml_dl_test.csv` there. Model files are saved under the matching `models` folder. Full daily input is cached as Parquet under `data/processed`.

### Previous 365-day CatBoost GPU run

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

This historical LightGBM L1 result was trained on the latest 365 forecast-origin
dates and evaluated on the latest 28 origins. The 365-day configs retain their
original objectives for reproducibility; notebook 05 uses the full-history L2 configs.

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
