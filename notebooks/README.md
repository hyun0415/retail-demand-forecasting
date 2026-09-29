# 노트북 안내

[메인](../README.md) · [English](README_EN.md) · [예측 방법](../docs/method/README.md)

핵심 데이터 처리와 모델 학습 코드는 `src/retail_forecast/`에 두고, 노트북은 탐색·설정·실행·결과 해석에 사용합니다. 아래는 **현재 저장소에 있는 파일**입니다.

| 노트북 | 역할 | 상태 |
|---|---|---|
| [01_eda.ipynb](01_eda.ipynb) | 판매 규모·0 판매·요일성·판촉 기술 통계 | 현재 파일 있음 |
| [03_baseline_model.ipynb](03_baseline_model.ipynb) | 전주 동일 요일·최근 4주 동일 요일 평균 | 현재 파일 있음 |
| [04_feature_engineering.ipynb](04_feature_engineering.ipynb) | Feature 생성과 사용 가능 시점 점검 | 현재 파일 있음 |
| [05_colab_model_comparison.ipynb](05_colab_model_comparison.ipynb) | 전체 이력 LightGBM·CatBoost GPU 학습과 Valid/Test 결과 저장 | 현재 파일 있음 |
| [06_error_analysis.ipynb](06_error_analysis.ipynb) | 매장·상품군·horizon 오류 탐색 | 현재 파일 있음; 군집별 분석은 후속 작업 |
| [07_dl_colab_validation.ipynb](07_dl_colab_validation.ipynb) | N-HiTS·TCN·TFT 학습, 내부 손실 및 공통 행 비교 | 현재 파일 있음 |
| [08_model_evaluation_and_decision.ipynb](08_model_evaluation_and_decision.ipynb) | 저장된 ML·DL 예측과 기준선 비교, 낮은 수요·매장별 실패 및 한 기준일의 검토 사례 | 현재 파일 있음; 학습하지 않음 |
| [09_lightgbm_historical_backtest.ipynb](09_lightgbm_historical_backtest.ipynb) | 앞선 두 시점의 LightGBM·기준선·Valid 선택 규칙 재평가 | 현재 파일 있음; GPU 재학습은 선택 실행 |

수요 군집 노트북과 운영 의사결정 시각화 노트북은 계획 단계여서 실행 순서에 넣지 않았습니다.

## Colab 실행 순서

1. 별도로 사용하는 `00` clone 노트북으로 저장소를 `/content/drive/MyDrive/retail-demand-forecasting`에 준비하고, [Kaggle 원본 데이터](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data)의 `train.csv`, `stores.csv`, `holidays_events.csv`, `transactions.csv`를 `data/raw`에 배치합니다. `00` 파일은 현재 저장소에 포함되지 않습니다.
2. GPU와 충분한 CPU RAM을 갖춘 Colab에서 **05**를 실행합니다. 05는 저장소를 clone·pull하지 않습니다. 전체 이력 Feature를 Parquet으로 만들고, `RUN_NAME=full_history_time_split_fixed_i600` 아래에 ML 모델·Valid/Test 예측·지표를 저장합니다.
3. **07**의 `ML_RUN_NAME`이 05의 `RUN_NAME`과 같은지 확인하고 실행합니다. 07은 05의 ML 예측 파일로 공통 평가 행을 구성합니다. DL의 `MODEL_STEPS`·`VAL_CHECK_STEPS`는 노트북 상단에서 설정하며 모델별 검증 손실과 예측 결과를 `neural_time_split_val_loss` 실행 폴더에 저장합니다.
4. **08**에서 두 실행 이름을 맞춘 뒤 저장된 Valid/Test 예측으로 모델·기준선 비교, 매장·상품군·horizon·수요 규모별 오류를 확인합니다. 낮은 수요 대체 규칙은 Valid에서 정하고, 마지막 Test 기준일 하나의 검토 사례를 시각화합니다. Test는 이미 확인한 구간이므로 이 결과는 탐색적으로 해석합니다.
5. 다른 과거 시점의 반복 개선을 확인하려면 **09**에서 `RUN_BACKTEST=True`로 바꾸고 필요한 날짜만 실행합니다. 09는 기존 Feature 캐시를 재사용하지만 시점별 LightGBM GPU 학습이 필요합니다. 기본값은 `False`이므로 전체 실행만으로 재학습하지 않습니다.
6. 모델 설정을 바꿔 새로 실행할 때는 실행 이름을 바꿔 이전 결과를 보존합니다. 전체 학습은 시간이 길므로 노트북의 설정과 필요한 산출물을 먼저 확인합니다.

`data/raw`, `data/processed`, `models`, `outputs`의 원본·대용량 산출물은 Git에서 제외됩니다. 로컬 CPU 재현은 `configs/baseline.toml` 또는 `configs/catboost.toml`과 `scripts/run_experiment.py`를 사용할 수 있으나, 현재 7일 Valid/간격/Test 공통 비교는 Colab의 05·07 노트북을 기준으로 설명합니다.
