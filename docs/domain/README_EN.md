# Data and retail context

[Overview](../en/README.md) · [한국어](README.md)

## Why forecast seven days?

The project uses a **hypothetical weekly planning cycle**: headquarters reviews store allocation and stores review their next order. The dataset does not document the retailer's actual ordering cadence. Daily forecasts for `t+1` through `t+7` reveal weekday differences; their sum estimates demand over that planning window.

| User | What the current forecast supports | Information still needed |
|---|---|---|
| Central demand and supply team | Compare demand and prioritize allocation review across stores and families | Available supply and inventory for actual allocation |
| Store team | Review next-week demand at product-family level | SKU sales history, stock, inbound units, lead time and order units for item-level orders |
| Marketing team | Describe sales patterns and forecast errors around promotions | A causal design before claiming incremental promotion lift |

## What the data can support

The [Favorita Store Sales dataset](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data) provides date, store, product family and observed sales in `train.csv`. The ML pipeline also uses store attributes, calendar information and transactions observed at the forecast origin. The target is **product-family sales**, not individual SKU demand.

**A SKU is not simply a product title.** A title is a customer-facing name; a stock keeping unit identifies a specific sellable and inventory-tracked variant, such as a particular size or package. A 500 mL and a 1 L version may have separate SKUs even when their names are similar. [Oracle Retail's SKU definition](https://docs.oracle.com/en/industries/retail/retail-oms-suite-cloud/24.2.401.0/romtg/c_introducing_sku_setup.htm) distinguishes base items from their variants.

## Why forecast a product family?

Family-level totals help a central team see **which types of goods need more attention at each store** during next week's supply and allocation review. A family series can continue across individual item replacements. Yet an unchanged family total may hide a changed mix of SKUs. It does not determine how many units of a specific item a store should order or a distribution center should ship.

| Level | Question | Current project |
|---|---|---|
| Store × family | How many units might this family sell next week? | Forecasted and evaluated |
| Store × SKU | How many units of this specific variant might sell? | No item identifier or SKU-level target in the current dataset |

## New items and fast assortment turnover

New items have little history for 28- or 56-day windows and lags. Slow-selling existing items can also have many zero-sales days. **A short history is a forecasting challenge, not evidence that demand is zero.** Similar-item history, product attributes, parent-family demand and launch/end-of-life dates can help; [AWS's cold-start guidance](https://docs.aws.amazon.com/forecast/latest/dg/howitworks-forecast.html) describes using item metadata and similar items for new products.

Requiring a complete 56-day SKU history would systematically exclude many new items. Family-level history may remain available while SKUs change, but major changes in item mix, prices or promotions can also change family demand. The current 56-day neural input applies to **store-family series**; it has not been validated as an appropriate SKU window.

A separate [Favorita Grocery Sales Forecasting dataset with `item_nbr`](https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting/data) could support an item-level extension. That dataset omits store-item-date rows with zero recorded sales and does not report whether an item was stocked. Missing rows cannot automatically be labeled zero demand. Item-level data preparation and evaluation would therefore be a separate phase.

Observed sales can be lower than unconstrained demand during stockouts. SKU sales history, stock on hand, inbound inventory, supplier lead time and order multiples are unavailable. The project therefore does not claim item-level order quantities, avoided stockouts or measured savings. With those inputs, an SKU-level forecast would need separate validation before combining it with inventory and supply constraints. A family forecast cannot simply be divided into verified SKU forecasts.

## EDA as the basis for modeling

The existing [EDA notebook](../../notebooks/01_eda.ipynb) examines sales distributions, zero sales, family-level scale, descriptive promotion comparisons and weekly autocorrelation. Aggregate seven-day autocorrelation exceeding one-day autocorrelation motivates the previous-week and four-week same-weekday [baselines](../../notebooks/03_baseline_model.ipynb). This aggregate observation does not imply identical seasonality for every store and family.

| Question | Evidence to inspect | Planning relevance |
|---|---|---|
| Where is demand concentrated? | Sales volume by store and family | Size of allocation decisions |
| When does demand change? | Weekday, month and holiday patterns | Weekly planning and calendar inputs |
| Which series are unstable? | Zero-sales share and volatility | Errors hidden by overall averages |
| What differs during promotions? | Descriptive sales comparisons | Cases for marketing review, not causal lift |

## Planned analysis

Demand segments based on volume, zero-sales share and volatility, followed by segment-level forecast errors, are **not yet completed**. This analysis could identify cases for central and store review. Weather and known-in-advance local events or promotions may later improve seasonal coverage; information that becomes known only after the forecast origin must not enter model inputs.
