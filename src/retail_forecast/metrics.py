import numpy as np


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual = np.clip(np.asarray(actual), 0, None)
    predicted = np.clip(np.asarray(predicted), 0, None)
    actual_sum = actual.sum()
    absolute_error = np.abs(actual - predicted)

    return {
        "rmsle": float(np.sqrt(np.mean(np.square(np.log1p(actual) - np.log1p(predicted))))),
        "mae": float(np.mean(absolute_error)),
        "rmse": float(np.sqrt(np.mean(np.square(actual - predicted)))),
        "wape": float(absolute_error.sum() / actual_sum * 100) if actual_sum else float("nan"),
        "bias": float((predicted - actual).sum() / actual_sum * 100) if actual_sum else float("nan"),
    }
