import numpy as np

from retail_forecast.metrics import regression_metrics


def test_regression_metrics_clip_negative_predictions() -> None:
    metrics = regression_metrics(np.array([0.0, 3.0]), np.array([-1.0, 5.0]))

    assert metrics["rmsle"] >= 0
    assert metrics["mae"] == 1.0
    assert np.isclose(metrics["rmse"], np.sqrt(2))
    assert np.isclose(metrics["wape"], 200 / 3)
    assert np.isclose(metrics["bias"], 200 / 3)


def test_regression_metrics_report_underforecast_bias() -> None:
    metrics = regression_metrics(np.array([40.0, 60.0]), np.array([30.0, 50.0]))

    assert metrics["wape"] == 20.0
    assert metrics["bias"] == -20.0


def test_regression_metrics_handle_zero_actual_sum() -> None:
    metrics = regression_metrics(np.zeros(2), np.ones(2))

    assert np.isnan(metrics["wape"])
    assert np.isnan(metrics["bias"])
