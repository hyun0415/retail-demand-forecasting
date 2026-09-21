import numpy as np


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual = np.clip(np.asarray(actual), 0, None)
    predicted = np.clip(np.asarray(predicted), 0, None)

    return {
        "rmsle": float(np.sqrt(np.mean(np.square(np.log1p(actual) - np.log1p(predicted))))),
        "mae": float(np.mean(np.abs(actual - predicted))),
    }

