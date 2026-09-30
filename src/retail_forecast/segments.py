from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


SERIES_KEYS = ["store_nbr", "family"]
CLUSTER_COLUMNS = [
    "log_mean_sales", "zero_share", "log_cv", "log_weekday_cv",
    "trend_log",
]


def load_segment_history(raw_path: str | Path, cutoff: pd.Timestamp, days: int = 180) -> pd.DataFrame:
    cutoff = pd.Timestamp(cutoff)
    start = cutoff - pd.Timedelta(days=days - 1)
    chunks = []
    for chunk in pd.read_csv(
        Path(raw_path) / "train.csv",
        usecols=["date", "store_nbr", "family", "sales", "onpromotion"],
        parse_dates=["date"],
        dtype={"store_nbr": "int16", "family": "category", "sales": "float32", "onpromotion": "int16"},
        chunksize=1_000_000,
    ):
        selected = chunk.loc[chunk["date"].between(start, cutoff)]
        if not selected.empty:
            chunks.append(selected)
    if not chunks:
        raise ValueError("군집 분석 기간에 해당하는 판매 이력이 없습니다.")
    history = pd.concat(chunks, ignore_index=True)
    if history.duplicated(["date", *SERIES_KEYS]).any():
        raise ValueError("매장·상품군·날짜가 중복된 원본 판매 기록이 있습니다.")
    return history


def build_segment_features(history: pd.DataFrame, cutoff: pd.Timestamp, days: int = 180) -> pd.DataFrame:
    cutoff = pd.Timestamp(cutoff)
    start = cutoff - pd.Timedelta(days=days - 1)
    history = history.loc[history["date"].between(start, cutoff)].copy()
    history["sales"] = history["sales"].clip(lower=0)
    history["zero_sales"] = history["sales"].eq(0)
    history["promoted"] = history["onpromotion"].gt(0)
    history["weekday"] = history["date"].dt.dayofweek

    features = history.groupby(SERIES_KEYS, observed=True).agg(
        observed_days=("date", "nunique"),
        mean_sales=("sales", "mean"),
        std_sales=("sales", "std"),
        zero_share=("zero_sales", "mean"),
        promo_day_share=("promoted", "mean"),
    )
    weekdays = history.groupby([*SERIES_KEYS, "weekday"], observed=True)["sales"].mean().unstack()
    features["weekday_cv"] = (weekdays.std(axis=1) / weekdays.mean(axis=1).replace(0, np.nan)).fillna(0)

    first_end = cutoff - pd.Timedelta(days=days - 28)
    last_start = cutoff - pd.Timedelta(days=27)
    first = history.loc[history["date"].le(first_end)].groupby(SERIES_KEYS, observed=True)["sales"].mean()
    last = history.loc[history["date"].ge(last_start)].groupby(SERIES_KEYS, observed=True)["sales"].mean()
    features["trend_log"] = np.log1p(last) - np.log1p(first)

    features["coverage_ratio"] = features["observed_days"] / days
    features["log_mean_sales"] = np.log1p(features["mean_sales"])
    cv = features["std_sales"].fillna(0) / features["mean_sales"].replace(0, np.nan)
    features["log_cv"] = np.log1p(cv.fillna(0).clip(upper=20))
    features["log_weekday_cv"] = np.log1p(features["weekday_cv"].clip(upper=20))
    features["trend_log"] = features["trend_log"].fillna(0).clip(-2, 2)
    return features.reset_index()


def fit_demand_segments(
    features: pd.DataFrame, clusters: int = 4, min_coverage: float = 0.9,
) -> tuple[pd.DataFrame, float]:
    result = features.copy()
    result["segment"] = "이력 부족"
    eligible = result["coverage_ratio"].ge(min_coverage)
    inputs = result.loc[eligible, CLUSTER_COLUMNS]
    if len(inputs) <= clusters or inputs.isna().any().any():
        raise ValueError("군집에 사용할 이력이 부족하거나 요약 Feature에 결측값이 있습니다.")

    scaled = StandardScaler().fit_transform(inputs)
    labels = KMeans(n_clusters=clusters, n_init=10, random_state=42).fit_predict(scaled)
    result.loc[eligible, "segment"] = [str(label) for label in labels]
    score = float(silhouette_score(scaled, labels))
    return result, score


def summarize_segments(features: pd.DataFrame) -> pd.DataFrame:
    return features.groupby("segment", observed=True).agg(
        series=("segment", "size"),
        median_daily_sales=("mean_sales", "median"),
        median_zero_share=("zero_share", "median"),
        median_log_cv=("log_cv", "median"),
        median_log_weekday_cv=("log_weekday_cv", "median"),
        median_trend_log=("trend_log", "median"),
        median_promo_share=("promo_day_share", "median"),
        median_coverage=("coverage_ratio", "median"),
    ).reset_index()
