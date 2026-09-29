# Metrics and interpretation

[Overview](../en/README.md) · [한국어](README.md)

## Why use several metrics?

Store-family series differ in scale. A small average error can hide rare large misses or persistent underforecasting. We therefore report **typical quantity error, large misses, error relative to total volume and direction of error**. In the formulas, `y` is actual sales, `ŷ` is predicted sales and `n` is the number of rows. Negative forecasts are clipped to zero before scoring.

| Metric | Formula | Main question | Caveat |
|---|---|---|---|
| MAE | `mean(|y−ŷ|)` | How many units are missed on average? | Same unit as sales; no squared penalty |
| MSE | `mean((y−ŷ)²)` | How strongly are large misses detected? | **Sensitive** to large errors and measured in squared units; not separately saved in final metric tables |
| RMSE | `√MSE` | How large are the misses when larger errors matter more? | Same unit as sales; outliers can dominate |
| RMSLE | `√mean((log(1+y)−log(1+ŷ))²)` | How different are forecasts on a log scale? | Defined for zero sales but not measured in sales units |
| WAPE | `Σ|y−ŷ| / Σy × 100` | What share of total actual volume is the absolute error? | Undefined when total actual sales are zero; high-volume errors contribute more to the sum |
| Bias | `Σ(ŷ−y) / Σy × 100` | Is the aggregate prediction high or low? | Positive means overforecast; negative means underforecast; errors can cancel |

MAE is an accessible quantity error; RMSE detects large misses; WAPE scales aggregate error by actual volume; Bias shows the direction relevant to supply decisions. RMSLE adds a log-scale view. WAPE avoids row-level division by zero found in ordinary MAPE, but still requires **positive total actual sales**.

## Training loss versus final metrics

The full-history LightGBM run uses the L2 `regression` objective, CatBoost uses `RMSE`, and the three neural models train with MSE. Neural internal validation MSE monitors training on its own validation window and scale. It is **not the same quantity** as final common-row Test WAPE or MAE. Final comparison recomputes metrics on aligned target and prediction rows.

## Recently submitted common Test comparison

The WAPE figures below are **draft documentation of results supplied by the user in this conversation**. The origin dates are August 2–8, 2017, with store, family, origin and horizon aligned. Exact run artifacts and settings should accompany the table before it is treated as the final published result.

| Model | Daily WAPE ↓ | Seven-day sum WAPE ↓ | Current reading |
|---|---:|---:|---|
| LightGBM | **14.88%** | **10.70%** | Lowest in both scopes in the submitted table |
| CatBoost | 16.77% | 11.96% | Comparison using the same tabular inputs |
| N-HiTS | 20.87% | 14.20% | Lowest WAPE among the three neural models |
| TFT | 25.21% | 18.58% | More training does not guarantee better final scores |
| TCN | 28.59% | 22.08% | Larger weekly error under the current settings |

Daily scoring compares each `t+h` target separately. Seven-day scoring **first sums the seven actuals and seven predictions within each store-family-origin**, then compares those totals. Errors on different days may cancel in the sum. Daily, horizon-level and Bias results remain necessary. ML and DL also use different input features, so the ranking cannot be attributed to architecture alone.

The Test week was inspected in earlier experiments. The table is a **retrospective comparison**, and further tuning against it would weaken the separation between model selection and evaluation. Store/family error inspection exists in the [06 notebook](../../notebooks/06_error_analysis.ipynb); demand-segment error analysis and operational views remain planned.

## Historical result, kept separate

An earlier **LightGBM L1 run** trained on the latest 365 forecast origins and evaluated on 28 origins. It obtained WAPE 13.26%, MAE 62.84 and RMSLE 0.3943. In that run, previous-week and four-week same-weekday baseline WAPEs were 17.96% and 14.76%. Training history, loss and evaluation split differ from the current full-history L2 experiment, so these figures are **not a direct ranking against the current Test table**.

## Decision boundary

Underforecasting may flag stockout risk and overforecasting may flag inventory burden, but this dataset cannot measure actual stockouts, waste or cost savings. Later analysis can show errors by volume, volatility, store and horizon. Item-level order recommendations require separately validated SKU forecasts plus inventory and lead-time data.
