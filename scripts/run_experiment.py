import argparse
import json

from retail_forecast.config import load_config
from retail_forecast.pipeline import run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Favorita demand forecast experiment")
    parser.add_argument("--config", default="configs/baseline.toml")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = run_experiment(load_config(args.config))
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

