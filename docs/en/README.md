# Favorita store-family demand forecasting

**How much will each store and product family sell over the next week?** This project forecasts daily sales for `t+1` through `t+7` from forecast origin `t`. Their seven-day sum supports central demand and allocation review. Item-level order recommendations would require additional SKU-level sales and inventory data.

[한국어 메인](../../README.md) · [Domain and data](../domain/README_EN.md) · [Method](../method/README_EN.md) · [Metrics and results](../evaluation/README_EN.md) · [Notebook guide](../../notebooks/README_EN.md)

## Forecast level and business use

The project assumes a **weekly replenishment review scenario**; the dataset does not record the retailer's actual ordering cycle. The forecast unit is **store × product family × target date**, not an individual SKU.

### Why not forecast individual items?

The current `train.csv` contains sales by date, store and **product family**, but no SKU identifier or item-level target. Item-level forecast accuracy therefore cannot be measured from this dataset. Family totals show weekly demand by store while leaving the mix and order quantity of individual SKUs unknown. New and frequently replaced items also need a separate approach to short sales histories. See the [domain guide](../domain/README_EN.md#why-forecast-a-product-family).

### Inputs to business decisions

| Practitioner | Question | Information from the current forecast or analysis | Decision it can inform |
|---|---|---|---|
| Central demand planner | Which store-family combinations may need more next week? | Daily forecasts for days 1–7 and their weekly sum | Prioritize cases for weekly demand review |
| Allocation planner | How does expected demand for the same family differ across stores? | Seven-day demand comparison across stores | Prioritize allocation review before checking supply and inventory constraints |
| Store operator | What is the next week's demand pattern at my store? | Family-level daily forecasts and seven-day totals | Flag unusual demand to headquarters; calculate SKU orders separately |
| Marketing analyst | How did demand patterns differ around a promotion? | Sales/promotion EDA; forecast-error joins are **planned analysis** | Select cases for review, without claiming causal promotion lift |
| Demand analyst | Where does the forecast repeatedly miss? | Overall and horizon-level errors, seven-day scores and Bias | Investigate large errors and persistent underforecasting |

These outputs are **decision-support inputs**. Actual allocation or SKU order quantities need inventory, inbound stock, available supply, lead times and order multiples. Joined promotion/error analysis and demand-segment visualizations remain planned work.

## Design choices

| Question | Choice | Reason |
|---|---|---|
| How should next week's demand be represented? | Seven daily predictions and their seven-day sum | Preserve weekday variation and weekly volume |
| What simple reference should a model beat? | Previous-week same weekday and recent four-week same-weekday mean | Weekly repetition observed in [EDA](../../notebooks/01_eda.ipynb) |
| How can data across stores be used? | Global LightGBM/CatBoost and N-HiTS/TCN/TFT models | Compare tabular features with sequential inputs on aligned evaluation rows |
| How is future information excluded? | Origin-available observations and advance-known calendar inputs | Match information available when a real forecast is made |
| How are settings chosen? | Chronological Train, Valid, gap and Test | Separate model selection from Test scoring |

The ML models create seven horizon rows for each origin; the neural models take 56 historical days and output seven future days together. **The target day's actual sales are never an input.** ML and DL do not use identical exogenous features, so the result is not a pure architecture comparison. The [method guide](../method/README_EN.md) explains the inputs, model mechanisms and leakage controls.

## Recently submitted results

The following WAPEs are **draft figures supplied in this conversation** for the common Test comparison. Test origins are August 2–8, 2017; store, family, origin, horizon and targets are aligned. Lower is better.

| Model | Daily WAPE ↓ | Seven-day sum WAPE ↓ |
|---|---:|---:|
| LightGBM | **14.88%** | **10.70%** |
| CatBoost | 16.77% | 11.96% |
| Recent four-week same-weekday mean | 16.98% | 12.16% |
| N-HiTS | 20.87% | 14.20% |
| Previous-week same weekday | 23.21% | 16.99% |
| TFT | 25.21% | 18.58% |
| TCN | 28.59% | 22.08% |

Against the strongest simple baseline, LightGBM reduces WAPE by about 12.4% daily and 12.0% on seven-day sums. Its Bias is still +6.44%, indicating aggregate overforecasting. Exact run artifacts and settings should accompany the final table. Daily over- and underforecasting can cancel in a seven-day sum. The earlier 365-origin LightGBM result used another training window, loss and split, and is documented separately in [metrics and results](../evaluation/README_EN.md).

## Validation and limits

| Stage | Forecast origins | Role |
|---|---|---|
| Train | Labels through July 18, 2017 | Update model weights |
| Valid | July 19–25 | Select stopping point and settings |
| Gap | July 26–August 1 | Let the last day-seven Valid target become observable |
| Test | August 2–8 | Evaluate the selected model without refitting |

The Test week was examined in earlier experiments, so this is a **retrospective holdout**, not a pristine first-look estimate. Observed sales can miss unmet demand during stockouts. The dataset lacks inventory and supply constraints needed for optimal orders or measured savings.

Future work can validate a separate SKU-level forecast using item sales data before connecting inventory and lead times to order recommendations. Weather and known-in-advance local events or promotions are possible inputs. Demand clustering, segment errors and operational visualizations are **planned**, not current findings.

## Documentation and execution

| Guide | Contents |
|---|---|
| [Domain and data](../domain/README_EN.md) | Retail use case, dataset, EDA and limits |
| [Method](../method/README_EN.md) | Baselines, features, model principles and chronological split |
| [Metrics and results](../evaluation/README_EN.md) | Metric formulas, interpretation and separate experiment results |
| [Notebook guide](../../notebooks/README_EN.md) | Existing notebooks and Colab run order |

Download the original data from Kaggle's [Store Sales competition](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data) into `data/raw`. Source data and large generated artifacts are excluded from Git. Full GPU training should be run only when a fresh experiment is needed.
