# Forecasting method and validation design

[Overview](../en/README.md) · [한국어](README.md)

## Forecast origin, horizon and inputs

The forecast origin `t` is **after the close of business**, when that day's sales and transactions are available. `forecast_horizon=h` means the target is day `t+h`, for `h=1..7`. If `t` is July 19 and `h=3`, the target is July 22 sales; lag 7 is sales on July 12. Sales on the target day are labels, never inputs.

| Approach | Input | Output | Global training |
|---|---|---|---|
| LightGBM and CatBoost | Tabular features at `t`, repeated for seven horizon rows | One daily forecast per row | Learns across stores and families |
| N-HiTS, TCN and TFT | Previous 56 days of each store-family series and known weekday/month | Seven future days together | Learns from windows across series |

The 56-day input **does not require 56 positive-sales days**. Observed zeros are distinguished from dates missing in the source. The neural models currently receive fewer external variables than ML, so their comparison is not an isolated architecture test.

## Baselines and feature choices

Weekly repetition observed in [EDA](../../notebooks/01_eda.ipynb) motivates two baselines for target day `d=t+h`: sales on `d-7` and the mean of `d-7, d-14, d-21, d-28`. Since `h<=7`, even `d-7` is no later than `t`. These baselines ask whether a complex model improves on a simple weekly rule.

| ML feature group | Current configuration | Availability at `t` |
|---|---|---|
| Recent sales | Sales at the origin; lags 1, 7, 14, 28, 56 | Observed by `t` |
| Demand level | Rolling means over 7, 14, 28 days | Shifted at least one day |
| Same weekday | Previous week and four-week mean for the target weekday | Reference dates do not exceed `t` |
| Calendar and store | Target weekday/month, store/family attributes, holiday indicators | Calendar information assumed known in advance |
| Transactions | Transactions at the origin | Forecast made after the origin day's close |

Oil prices were excluded because their relation to daily store-family demand was difficult to explain. The **recorded promotion count on the target day (`target_onpromotion`) is excluded**: it is not a verified advance promotion plan. Unexpected future events and earthquake indicators are also excluded from the current comparison. Whether a holiday transfer or substitute day was known at the actual forecast origin requires further operational verification. The [feature notebook](../../notebooks/04_feature_engineering.ipynb) documents construction.

## Why these models?

| Model | Basic mechanism | Role |
|---|---|---|
| LightGBM | Gradient-boosted trees reduce the previous ensemble's loss step by step | Strong tabular reference |
| CatBoost | Gradient-boosted trees with categorical-data handling | Comparison on store/family attributes |
| N-HiTS | MLP-based multi-scale summaries | Direct multi-day forecasting |
| TCN | Past-facing temporal convolutions | Repeated local time patterns |
| TFT | Recurrent layers and attention | Changing importance across time and inputs |

LightGBM uses the L2 `regression` objective and CatBoost uses `RMSE` for the full-history ML experiment. The three neural models train with MSE; internal validation MSE monitors training. **Training objectives and reported metrics are different roles.** Final forecasts are assessed with MAE, RMSE, RMSLE, WAPE and Bias on aligned evaluation rows. No extra Transformer or hand-built PyTorch architecture is in the current scope.

## Chronological split and leakage controls

| Stage | Forecast origins | Role |
|---|---|---|
| Train | Before validation | Only labels through July 18, 2017 update weights |
| Valid | July 19–25 | ML early stopping and DL setting selection |
| Gap | July 26–August 1 | No evaluation origins |
| Test | August 2–8 | Evaluate the selected, unchanged model |

The last validation origin, July 25, has a day-seven target on **August 1**. Test begins on August 2, after that label becomes observable. The gap does not ban already observed sales from later forecast histories: each origin may use observations available **through that origin**. Weights are not refit for Test.

ML selects its tree count on Valid and uses the same fitted model on Test. DL updates weights with labels through July 18, monitors an internal seven-day MSE, and separately scores seven rolling validation origins. The saved DL model is then used on Test. Both families match store, family, origin, horizon and target values. A series enters the common cohort only if all seven origins × seven horizons are present in both Valid and Test; zero sales count as observed, not missing. Targets are joined on exact dates and keys, and random splits are not used.

The Test week had been inspected in earlier experiments. It is therefore a **retrospective holdout**, not a pristine first-look estimate of future generalization.
