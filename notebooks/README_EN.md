# Notebook guide

[Overview](../docs/en/README.md) · [한국어](README.md) · [Method](../docs/method/README_EN.md)

Core data and training logic lives in `src/retail_forecast/`. Notebooks support exploration, experiment setup, execution and interpretation. The following files **currently exist** in this repository.

| Notebook | Purpose | Status |
|---|---|---|
| [01_eda.ipynb](01_eda.ipynb) | Sales scale, zero sales, weekly patterns and descriptive promotions | Existing |
| [03_baseline_model.ipynb](03_baseline_model.ipynb) | Previous-week and four-week same-weekday baselines | Existing |
| [04_feature_engineering.ipynb](04_feature_engineering.ipynb) | Feature construction and availability at the origin | Existing |
| [05_colab_model_comparison.ipynb](05_colab_model_comparison.ipynb) | Full-history LightGBM/CatBoost GPU fit and Valid/Test outputs | Existing |
| [06_error_analysis.ipynb](06_error_analysis.ipynb) | Store, family and horizon error inspection | Existing; segment analysis planned |
| [07_dl_colab_validation.ipynb](07_dl_colab_validation.ipynb) | N-HiTS/TCN/TFT training, internal loss and common-row comparison | Existing |

Demand clustering and operational decision visualizations are planned, so no nonexistent notebook is linked.

## Colab run order

1. Use the separate `00` clone notebook to prepare the repository at `/content/drive/MyDrive/retail-demand-forecasting`, then put the Kaggle `train.csv`, `stores.csv`, `holidays_events.csv` and `transactions.csv` files in `data/raw`. The `00` notebook is not part of this repository.
2. Run **05** with GPU and sufficient CPU RAM. It does not clone or pull. It builds the full-history Parquet feature cache and saves ML models, Valid/Test forecasts and metrics under `RUN_NAME=full_history_time_split_fixed_i600`.
3. Match **07**'s `ML_RUN_NAME` to 05's `RUN_NAME`, then run 07. It uses ML prediction files to form aligned evaluation rows. Set `MODEL_STEPS` and `VAL_CHECK_STEPS` near the top; model-specific loss and predictions are saved under `neural_time_split_val_loss`.
4. Change the run name when rerunning with different settings to preserve earlier outputs. Full training is expensive; inspect parameters and required artifacts first.

The Git ignore rules exclude raw data, processed caches, models and outputs. Local CPU experiments can use `configs/baseline.toml` or `configs/catboost.toml` with `scripts/run_experiment.py`. The current seven-day Valid/gap/Test comparison is described by Colab notebooks 05 and 07.
