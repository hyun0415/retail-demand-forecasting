from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class ExperimentConfig:
    model: str
    forecast_horizons: tuple[int, ...]
    training_origin_days: int
    validation_days: int
    lags: tuple[int, ...]
    rolling_windows: tuple[int, ...]
    random_state: int
    raw_path: Path
    processed_path: Path
    model_path: Path
    output_path: Path
    model_params: dict
    excluded_features: tuple[str, ...] = ()


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path).resolve()
    project_root = path.parents[1]

    with path.open("rb") as file:
        values = tomllib.load(file)

    model_name = values["model"]
    paths = values["paths"]

    return ExperimentConfig(
        model=model_name,
        forecast_horizons=tuple(values["forecast_horizons"]),
        training_origin_days=values["training_origin_days"],
        validation_days=values["validation_days"],
        lags=tuple(values["lags"]),
        rolling_windows=tuple(values["rolling_windows"]),
        random_state=values["random_state"],
        raw_path=project_root / paths["raw"],
        processed_path=project_root / paths["processed"],
        model_path=project_root / paths["model"],
        output_path=project_root / paths["output"],
        model_params=values[model_name],
        excluded_features=tuple(values.get("excluded_features", ())),
    )
