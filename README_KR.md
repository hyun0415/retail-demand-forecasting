# Favorita 유통 수요예측

Favorita의 점포·상품군별 향후 1~7일 일별 판매량을 예측하는 프로젝트입니다. 단순한 대회 점수보다 상품군별 보충·프로모션 계획의 의사결정을 지원하는 데 목적을 두었습니다.

[English README](README.md)

## 문제 정의

- **예측 단위:** 점포 × 상품군 × 목표일
- **예측값:** 기준일 다음 날부터 7일 뒤까지의 0 이상 일별 판매량
- **주 모델:** LightGBM
- **비교 모델:** CatBoost
- **모델 구조:** `forecast_horizon=1~7`을 입력받는 통합 글로벌 모델 1개
- **기준 모델:** 직전 주 일별 패턴과 동일 요일 최근 4주 평균
- **검증:** 최근 28개 예측 기준일을 분리하고, 검증 시작일 이후의 정답은 학습에서 제외
- **평가지표:** RMSLE, MAE, RMSE, WAPE, Bias

기준일 `t`까지 확인 가능한 판매 이력을 공통으로 사용하고, `forecast_horizon`을 1~7로 바꿔 `t+1`부터 `t+7`까지 각각 예측합니다. 목표일 정보 중에서는 사전에 정해지는 프로모션·요일·공휴일만 사용합니다. 결과는 일별로 확인하거나 상품군의 보충 주기에 맞춰 1~3일 또는 1~7일 합계로 활용할 수 있습니다.

전체 원본으로 Lag와 이동평균을 계산한 뒤 최근 365개 예측 기준일만 1~7일 Horizon으로 확장합니다. 생성된 학습 Feature는 `data/processed`에 Parquet으로 저장하여 모델 비교와 재실행에서 재사용합니다.

## 데이터 준비

Kaggle의 [Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data)에서 데이터를 내려받아 `data/raw`에 배치합니다.

```text
train.csv
stores.csv
holidays_events.csv
transactions.csv
```

원본 데이터는 Kaggle 대회 규칙의 적용을 받으므로 Git에 포함하지 않습니다.

## 분석 방식

- **최근 수요:** 기준일 판매량, 1·7·14·28·56일 전 판매량
- **수요 수준:** 과거 7·14·28일 이동평균
- **주간 주기:** 목표일과 같은 요일의 직전 주 판매량 및 최근 4주 평균
- **예측 거리:** 목표일까지 남은 `forecast_horizon` 1~7
- **미래 확정 정보:** 목표일의 프로모션·요일·월·공휴일·이벤트
- **기준일 정보:** 점포·상품군·기준일 거래량

이동평균은 반드시 하루 이상 이전의 판매량으로 계산합니다. 목표값도 단순 행 이동이 아니라 실제 날짜와 점포·상품군을 기준으로 결합해 누락된 날짜로 인한 오류를 방지했습니다. 유가는 수요와의 관계를 명확히 설명하기 어려워 모델에서 제외했습니다.

## 폴더 구성

```text
configs/                 실험 설정
data/raw/                Kaggle 원본 데이터, Git 제외
models/                  학습 모델, Git 제외
outputs/                 지표와 검증 결과, Git 제외
scripts/run_experiment.py
src/retail_forecast/     데이터·변수·모델·실험 코드
notebooks/01_eda.ipynb
notebooks/03_baseline_model.ipynb
notebooks/04_feature_engineering.ipynb
notebooks/05_colab_model_comparison.ipynb
notebooks/06_error_analysis.ipynb
tests/                   데이터 누수와 지표 검증
```

## 실행 방법

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
python scripts/run_experiment.py --config configs/baseline.toml
python scripts/run_experiment.py --config configs/catboost.toml
python scripts/run_permutation_importance.py --config configs/baseline.toml
```

변수 정의나 학습기간을 변경해 Parquet을 새로 만들 때만 다음 옵션을 사용합니다.

```bash
python scripts/run_experiment.py --config configs/baseline.toml --rebuild-features
```

로컬 CPU에서는 `configs/catboost.toml`, NVIDIA GPU에서는
`configs/catboost_gpu.toml`을 사용합니다.

## Colab GPU 실행

### 전체 이력 LightGBM·CatBoost 비교

[`notebooks/05_colab_model_comparison.ipynb`](notebooks/05_colab_model_comparison.ipynb)을 Colab에서 열고 GPU 및 가능한 경우 높은 RAM 런타임을 선택하세요. 노트북은 `/content/drive/MyDrive/retail-demand-forecasting`에 이 저장소를 clone합니다. 원본 CSV는 Git에 없으므로 clone 후 `data/raw`에 별도로 배치해야 합니다.

노트북은 전체 원본에서 기준일별 Feature를 28일 단위 Parquet으로 생성하고, 동일한 최근 28개 기준일을 검증 구간으로 사용해 LightGBM `regression`(L2), CatBoost `RMSE`를 GPU로 순서대로 학습합니다. Feature 청크와 학습 반복 횟수는 진행 막대로, 단계별 경과 시간과 검증 점수는 로그로 표시합니다. 생성 데이터는 `data/processed/features_h1-7_dall_l1-7-14-28-56_r7-14-28.parquet` 디렉터리에, 모델은 `models/full_history_l2`, 결과는 `outputs/full_history_l2`에 저장됩니다. `comparison_full_history.csv`는 전체 일별·7일 합계 지표를, `comparison_by_horizon_full_history.csv`는 예측 거리별 지표를 담습니다. 이전 L1 전체 이력 결과와 365일 실험 결과는 유지됩니다.

딥러닝 비교 후보는 NeuralForecast의 N-HiTS(MLP), TCN(CNN), TFT(순환 계층과 attention)입니다. 현재 노트북에는 선정 계획만 정리되어 있으며, 딥러닝 학습·평가 코드는 아직 추가하지 않았습니다.

### CatBoost 단기 진단

전체 이력 CatBoost가 높은 판매량을 거의 예측하지 못한 원인을 확인하려면 05 노트북의 **CatBoost 단기 진단** 셀을 실행합니다. 기존 Feature 캐시에서 검증 직전 180개 학습 기준일만 읽어 `MAE`와 `RMSE`를 각각 300회 GPU 학습합니다. 진단 결과는 `outputs/diagnostics/catboost_t180_i300`에 별도로 저장되며 기존 전체 실험 결과는 유지됩니다. 이 경로에서는 LightGBM CUDA 빌드, 전체 Feature 생성, 전체 ML 학습 셀을 건너뜁니다.

```bash
python scripts/diagnose_catboost.py --config configs/full_catboost_gpu.toml --train-origin-days 180 --iterations 300
```

LightGBM CUDA 버전은 Colab에서 소스 빌드가 필요합니다. 노트북이 `nvcc` 존재 여부와 두 모델의 GPU 시험 학습을 확인합니다. 전체 Feature와 모델 입력 표는 CPU RAM도 많이 사용하므로 런타임 메모리에 따라 실행이 실패할 수 있습니다. 목표일 프로모션은 7일 앞까지 확정 계획이 있을 때만 사용할 수 있으며, 기준일 판매량·거래량은 영업 종료 후 예측한다는 전제입니다. 전체 이력 실험에서는 미래의 지진·돌발 이벤트 변수를 모델 입력에서 제외합니다. 이 변경으로 아래 365일 결과와 수치를 직접 비교할 수 없습니다.

### 기존 365일 CatBoost GPU 실행

Colab에는 Git으로 추적하지 않는 다음 Feature 파일을 별도로 배치해야 합니다.

```text
data/processed/features_h1-7_d365_l1-7-14-28-56_r7-14-28.parquet
```

이 파일은 원본 데이터에서 생성한 최근 365개 예측 기준일의 1~7일 통합 학습
테이블입니다. 파일이 있으면 원본 CSV와 `--rebuild-features` 옵션은 필요하지
않습니다.

```bash
pip install -r requirements.txt
pip install -e .
python scripts/run_experiment.py --config configs/catboost_gpu.toml
```

Colab 런타임은 종료될 수 있으므로 완료 후 `models`와 `outputs`의 결과를
Google Drive 또는 로컬 컴퓨터로 복사합니다. GPU 학습은 부동소수점 연산
순서로 인해 같은 시드에서도 결과가 소폭 달라질 수 있습니다.

실행 결과는 다음과 같이 저장됩니다.

- `models/<모델명>_h1-h7.joblib`
- `outputs/metrics_<모델명>.json`
- `outputs/metrics_by_horizon_<모델명>.csv`
- `outputs/validation_predictions_<모델명>.csv`
- `outputs/decision_summary_<모델명>.csv` (`h1~h7` 일별 예측과 1~3일·1~7일 합계)
- `outputs/feature_importance_<모델명>.csv`
- `outputs/permutation_importance.csv`

## 현재 1~7일 통합 모델 결과

이 표는 이전 LightGBM L1 실험 결과입니다. 최근 365개 예측 기준일로 학습하고 최근 28개 기준일을 검증했습니다. 기존 365일 설정은 재현을 위해 유지하며, 05 노트북의 전체 이력 설정만 L2를 사용합니다.

| 모델 | RMSLE | MAE | WAPE | Bias |
|---|---:|---:|---:|---:|
| 직전 주 동일 요일 | 0.5430 | 85.09 | 17.96% | 0.91% |
| 동일 요일 최근 4주 평균 | 0.4617 | 69.92 | 14.76% | 0.83% |
| LightGBM | **0.3943** | **62.84** | **13.26%** | **-0.86%** |

Bias가 음수이면 전체적으로 과소 예측하고, 양수이면 과대 예측한다는 뜻입니다.

## 이전 단일 7일 예측 기준 실험

전체 데이터 중 2,831,598행을 학습하고 최근 28일에 해당하는 49,896행을 검증했습니다. 검증 목표일은 2017년 7월 19일부터 시작합니다.

| 모델 | RMSLE | MAE |
|---|---:|---:|
| 7일 계절 기준선 | 0.5468 | 86.93 |
| LightGBM | **0.4079** | **66.58** |

LightGBM은 계절 기준선보다 RMSLE를 25.4%, MAE를 23.4% 낮췄습니다. 이 수치는 이전 `t+7` 단일 목표 실험 결과이므로 새 1~7일 통합 모델의 성능과 직접 비교하지 않습니다. 통합 모델 실행 후 전체 지표와 예측 거리별 지표를 새로 기록합니다.

## 현재 범위

첫 버전은 재현 가능한 예측 실험과 시간 순서 기반 검증에 집중했습니다. API 배포, 모니터링, 자동 학습과 TabFM 비교는 의도적으로 제외했습니다. 이후에는 동일한 변수 표본을 사용해 TabFM의 제로샷 성능만 별도 비교할 수 있습니다.
