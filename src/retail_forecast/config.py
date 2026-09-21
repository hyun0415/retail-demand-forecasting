from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class ExperimentConfig:
    model: str
    forecast_horizon: int
    validation_days: int
    lags: tuple[int, ...]
    rolling_windows: tuple[int, ...]
    random_state: int
    raw_path: Path
    model_path: Path
    output_path: Path
    model_params: dict


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path).resolve()
    project_root = path.parents[1]

    with path.open("rb") as file:
        values = tomllib.load(file)

    model_name = values["model"]
    paths = values["paths"]

    return ExperimentConfig(
        model=model_name,
        forecast_horizon=values["forecast_horizon"],
        validation_days=values["validation_days"],
        lags=tuple(values["lags"]),
        rolling_windows=tuple(values["rolling_windows"]),
        random_state=values["random_state"],
        raw_path=project_root / paths["raw"],
        model_path=project_root / paths["model"],
        output_path=project_root / paths["output"],
        model_params=values[model_name],
    )

