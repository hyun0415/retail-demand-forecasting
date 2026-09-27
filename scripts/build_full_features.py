import argparse
import gc
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from retail_forecast.config import load_config
from retail_forecast.data import load_favorita_data
from retail_forecast.features import add_sales_history, build_feature_table
from retail_forecast.pipeline import feature_cache_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build full-history features in date chunks")
    parser.add_argument("--config", default="configs/full_lightgbm_gpu.toml")
    parser.add_argument("--chunk-days", type=int, default=28)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.chunk_days < 1:
        raise ValueError("chunk-days must be positive")
    config = load_config(args.config)
    if config.training_origin_days != 0:
        raise ValueError("This builder requires training_origin_days = 0")

    cache_path = feature_cache_path(config)
    if cache_path.exists():
        print(f"Feature cache already exists: {cache_path}")
        return
    temporary_path = cache_path.with_name(cache_path.name + ".building")
    if temporary_path.exists():
        raise FileExistsError(f"Incomplete build found: {temporary_path}")
    temporary_path.mkdir(parents=True)

    data = load_favorita_data(config.raw_path)
    history = add_sales_history(data.sales, config.lags, config.rolling_windows)
    first_origin = history["date"].min()
    last_origin = history["date"].max() - pd.Timedelta(days=max(config.forecast_horizons))

    part = 0
    for start in pd.date_range(first_origin, last_origin, freq=f"{args.chunk_days}D"):
        end = min(start + pd.Timedelta(days=args.chunk_days - 1), last_origin)
        frame = build_feature_table(
            data,
            config.forecast_horizons,
            config.lags,
            config.rolling_windows,
            history_frame=history,
            origin_start=start,
            origin_end=end,
        )
        if not frame.empty:
            frame.to_parquet(temporary_path / f"part-{part:04d}.parquet", index=False)
            print(f"{start.date()} to {end.date()}: {len(frame):,} rows")
            part += 1
        del frame
        gc.collect()

    if part == 0:
        raise ValueError("No feature rows were generated")
    temporary_path.rename(cache_path)
    print(f"Saved {part} Parquet parts to {cache_path}")


if __name__ == "__main__":
    main()
