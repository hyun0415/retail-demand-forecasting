import numpy as np
import pandas as pd
import pytest

from retail_forecast.comparison import compare_on_common_rows
from retail_forecast.neural_data import (
    KEYS,
    format_neural_predictions,
    future_calendar,
    history_through_origin,
    load_neural_panel,
    load_validation_template,
    training_before_origin,
)


def sample_files(tmp_path):
    dates = pd.date_range("2026-01-01", periods=100)
    sales = pd.MultiIndex.from_product([[1, 2], ["A"], dates], names=["store_nbr", "family", "date"]).to_frame(index=False)
    sales["sales"] = np.arange(len(sales), dtype="float32")
    sales = sales.loc[~((sales["store_nbr"] == 1) & (sales["date"] == dates[10]))]
    raw = tmp_path / "raw"
    raw.mkdir()
    sales.to_csv(raw / "train.csv", index=False)
    origins = dates[-35:-7]
    template = pd.MultiIndex.from_product([origins, [1, 2], ["A"], range(1, 8)], names=["date", "store_nbr", "family", "forecast_horizon"]).to_frame(index=False)
    template["target_date"] = template["date"] + pd.to_timedelta(template["forecast_horizon"], unit="D")
    template = template.merge(sales.rename(columns={"date": "target_date", "sales": "target_sales"}), on=["store_nbr", "family", "target_date"])
    path = tmp_path / "ml.csv"
    template.to_csv(path, index=False)
    return raw, path


def test_panel_preserves_missing_dates_and_ml_origins(tmp_path):
    raw, path = sample_files(tmp_path)
    template = load_validation_template(path)
    panel, series = load_neural_panel(raw, template)
    missing_id = series.loc[series["store_nbr"].eq(1), "unique_id"].iloc[0]
    missing = panel.loc[panel["unique_id"].eq(missing_id) & panel["ds"].eq(pd.Timestamp("2026-01-11"))].iloc[0]
    assert missing["available_mask"] == 0
    assert missing["y"] == 0
    assert template["date"].nunique() == 28
    assert panel.groupby("unique_id")["ds"].nunique().nunique() == 1

    origin = template["date"].min()
    train = training_before_origin(panel, origin)
    history = history_through_origin(panel, origin, input_size=56)
    assert train["ds"].max() == origin - pd.Timedelta(days=1)
    assert history["ds"].max() == origin
    assert history["ds"].min() == origin - pd.Timedelta(days=55)
    future = future_calendar(series, origin)
    assert future["ds"].min() == origin + pd.Timedelta(days=1)
    assert future["ds"].max() == origin + pd.Timedelta(days=7)
    forecast = future[["unique_id", "ds"]].copy()
    forecast["NHITS"] = 4.0
    result = format_neural_predictions(forecast, template, series, origin, "NHITS")
    assert len(result) == len(template.loc[template["date"].eq(origin)])
    assert result[KEYS].equals(template.loc[template["date"].eq(origin)].sort_values(KEYS).reset_index(drop=True)[KEYS])


def test_common_comparison_rejects_different_validation_rows(tmp_path):
    _, path = sample_files(tmp_path)
    template = load_validation_template(path)
    a = template.assign(prediction=template["target_sales"])
    b = a.copy()
    a.to_parquet(tmp_path / "a.parquet", index=False)
    b.to_parquet(tmp_path / "b.parquet", index=False)
    comparison = compare_on_common_rows({"A": tmp_path / "a.parquet", "B": tmp_path / "b.parquet"})
    assert set(comparison["model"]) == {"A", "B"}
    a.to_csv(tmp_path / "ml.csv", index=False)
    mixed = compare_on_common_rows({"ML": tmp_path / "ml.csv", "DL": tmp_path / "b.parquet"})
    assert set(mixed["model"]) == {"ML", "DL"}
    b.loc[0, "target_sales"] += 1
    b.to_parquet(tmp_path / "b.parquet", index=False)
    with pytest.raises(ValueError, match="실제 판매량"):
        compare_on_common_rows({"A": tmp_path / "a.parquet", "B": tmp_path / "b.parquet"})
    b = b.iloc[1:]
    b.to_parquet(tmp_path / "b.parquet", index=False)
    with pytest.raises(ValueError, match="검증 행"):
        compare_on_common_rows({"A": tmp_path / "a.parquet", "B": tmp_path / "b.parquet"})
