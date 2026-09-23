from dataclasses import dataclass
from pathlib import Path

import pandas as pd


REQUIRED_FILES = (
    "train.csv",
    "stores.csv",
    "holidays_events.csv",
    "transactions.csv",
)


@dataclass(frozen=True)
class FavoritaData:
    sales: pd.DataFrame
    stores: pd.DataFrame
    holidays: pd.DataFrame
    transactions: pd.DataFrame


def validate_data_files(raw_path: str | Path) -> None:
    raw_path = Path(raw_path)
    missing = [name for name in REQUIRED_FILES if not (raw_path / name).exists()]

    if missing:
        files = ", ".join(missing)
        raise FileNotFoundError(f"Missing Favorita files in {raw_path}: {files}")


def load_favorita_data(raw_path: str | Path) -> FavoritaData:
    raw_path = Path(raw_path)
    validate_data_files(raw_path)

    sales = pd.read_csv(
        raw_path / "train.csv",
        parse_dates=["date"],
        usecols=["date", "store_nbr", "family", "sales", "onpromotion"],
        dtype={
            "store_nbr": "int16",
            "family": "category",
            "sales": "float32",
            "onpromotion": "int16",
        },
    )
    stores = pd.read_csv(
        raw_path / "stores.csv",
        dtype={
            "store_nbr": "int16",
            "city": "category",
            "state": "category",
            "type": "category",
            "cluster": "int8",
        },
    )
    holidays = pd.read_csv(
        raw_path / "holidays_events.csv",
        parse_dates=["date"],
        dtype={"type": "category", "locale": "category", "transferred": "bool"},
    )
    transactions = pd.read_csv(
        raw_path / "transactions.csv",
        parse_dates=["date"],
        dtype={"store_nbr": "int16", "transactions": "int32"},
    )

    return FavoritaData(sales, stores, holidays, transactions)
