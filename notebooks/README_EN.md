# Notebook guide

[Overview](../docs/en/README.md) · [한국어](README.md) · [Method](../docs/method/README_EN.md)

Core data and training logic lives in `src/retail_forecast/`. Notebooks support exploration, experiment setup, execution and interpretation. The following files **currently exist** in this repository.

| Notebook | Purpose | Status |
|---|---|---|
| [01_eda.ipynb](01_eda.ipynb) | Sales scale, zero sales, weekly patterns and descriptive promotions | Existing |
| [02_demand_segments.ipynb](02_demand_segments.ipynb) | Store–family demand patterns from the 180 days before Valid | Existing; CPU analysis |
| [03_baseline_model.ipynb](03_baseline_model.ipynb) | Previous-week and four-week same-weekday baselines | Existing |
| [04_feature_engineering.ipynb](04_feature_engineering.ipynb) | Feature construction and availability at the origin | Existing |
| [05_colab_model_comparison.ipynb](05_colab_model_comparison.ipynb) | Full-history LightGBM/CatBoost GPU fit and Valid/Test outputs | Existing |
| [06_error_analysis.ipynb](06_error_analysis.ipynb) | Store, family and horizon error inspection | Existing |
| [07_dl_colab_validation.ipynb](07_dl_colab_validation.ipynb) | N-HiTS/TCN/TFT training, internal loss and common-row comparison | Existing |
| [08_model_evaluation_and_decision.ipynb](08_model_evaluation_and_decision.ipynb) | Common-row model comparison, LightGBM errors by segment, and one-origin review cases | Existing; no training |
| [09_lightgbm_historical_backtest.ipynb](09_lightgbm_historical_backtest.ipynb) | LightGBM, baseline and validation-selected rule at two earlier dates | Existing; optional GPU retraining |

The full-history EDA in 01 is descriptive. Notebook 02 defines segments using history **through July 18, 2017**, before Valid begins, and 08 joins those segments to saved Test predictions. Segment labels are not model features.

## Colab run order

1. Use the separate `00` clone notebook to prepare the repository at `/content/drive/MyDrive/retail-demand-forecasting`, then put the Kaggle `train.csv`, `stores.csv`, `holidays_events.csv` and `transactions.csv` files in `data/raw`. The `00` notebook is not part of this repository.
2. Use **01** for sales and weekly-pattern exploration, then **02** to summarize 180 days of history and save segment assignments at `outputs/demand_segments/segment_features.parquet`. Notebook 02 needs `train.csv` but no GPU training.
3. Run **05** with GPU and sufficient CPU RAM. It does not clone or pull. It builds the full-history Parquet feature cache and saves ML models, Valid/Test forecasts and metrics under `RUN_NAME=full_history_time_split_fixed_i600`.
4. Match **07**'s `ML_RUN_NAME` to 05's `RUN_NAME`, then run 07. It uses ML prediction files to form aligned evaluation rows. Set `MODEL_STEPS` and `VAL_CHECK_STEPS` near the top; model-specific loss and predictions are saved under `neural_time_split_val_loss`.
5. Match the two run names in **08** and use saved Valid/Test predictions to compare all five models on the same rows. When the 02 segment file exists, 08 also shows seven-day WAPE and bias by segment for LightGBM and the four-week baseline. Since Test has already been examined, treat fallback results as exploratory. 08 does not train or rebuild features.
6. To check whether the gain repeats at earlier dates, set `RUN_BACKTEST=True` in **09** and run only the dates needed. It reuses the feature cache but trains LightGBM once per date. The default `False` prevents an accidental full run.
7. Change the run name when rerunning with different settings to preserve earlier outputs. Full training is expensive; inspect parameters and required artifacts first.

The Git ignore rules exclude raw data, processed caches, models and outputs. Local CPU experiments can use `configs/baseline.toml` or `configs/catboost.toml` with `scripts/run_experiment.py`. The current seven-day Valid/gap/Test comparison is described by Colab notebooks 05 and 07.
