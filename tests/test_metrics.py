import numpy as np

from retail_forecast.metrics import regression_metrics


def test_regression_metrics_clip_negative_predictions() -> None:
    metrics = regression_metrics(np.array([0.0, 3.0]), np.array([-1.0, 5.0]))

    assert metrics["rmsle"] >= 0
    assert metrics["mae"] == 1.0

