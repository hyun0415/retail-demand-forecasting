# Favorita 유통 수요예측

Favorita의 점포·상품군별 판매량을 7일 전에 예측하는 프로젝트입니다. 단순한 대회 점수보다 재고와 프로모션 계획에 활용할 수 있는 예측 절차를 만드는 데 목적을 두었습니다.

[English README](README.md)

## 문제 정의

- **예측 단위:** 점포 × 상품군 × 목표일
- **예측값:** 7일 뒤의 0 이상 판매량
- **주 모델:** LightGBM
- **비교 모델:** CatBoost
- **기준 모델:** 같은 점포·상품군의 7일 전 판매량
- **검증:** 가장 최근 28개 목표일을 시간순으로 분리
- **평가지표:** RMSLE와 MAE

기준일 `t`까지 확인 가능한 정보로 `t + 7`의 판매량을 예측합니다. 목표일의 정보 중에서는 사전에 정해지는 프로모션과 달력 정보만 사용합니다.

## 데이터 준비

Kaggle의 [Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data)에서 데이터를 내려받아 `data/raw`에 배치합니다.

```text
train.csv
stores.csv
oil.csv
holidays_events.csv
transactions.csv
```

원본 데이터는 Kaggle 대회 규칙의 적용을 받으므로 Git에 포함하지 않습니다.

## 분석 방식

- 1·7·14·28·56일 전 판매량
- 과거 7·14·28일 평균 판매량
- 목표일의 프로모션과 달력 정보
- 점포·상품군·거래량·유가·공휴일·이벤트 정보

이동평균은 반드시 하루 이상 이전의 판매량으로 계산합니다. 7일 뒤의 정답도 단순 행 이동이 아니라 실제 날짜와 점포·상품군을 기준으로 결합해 누락된 날짜로 인한 오류를 방지했습니다.

## 폴더 구성

```text
configs/                 실험 설정
data/raw/                Kaggle 원본 데이터, Git 제외
models/                  학습 모델, Git 제외
outputs/                 지표와 검증 결과, Git 제외
scripts/run_experiment.py
src/retail_forecast/     데이터·변수·모델·실험 코드
notebooks/                EDA와 모델 오차 분석
tests/                   데이터 누수와 지표 검증
```

## 실행 방법

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
python scripts/run_experiment.py --config configs/baseline.toml
```

CatBoost를 사용하려면 `configs/baseline.toml`의 `model`을 `catboost`로 바꿉니다.

실행 결과는 다음과 같이 저장됩니다.

- `models/<모델명>_h7.joblib`
- `outputs/metrics.json`
- `outputs/validation_predictions.csv`
- `outputs/feature_importance.csv`

## 첫 기준 실험 결과

전체 데이터 중 2,831,598행을 학습하고 최근 28일에 해당하는 49,896행을 검증했습니다. 검증 목표일은 2017년 7월 19일부터 시작합니다.

| 모델 | RMSLE | MAE |
|---|---:|---:|
| 7일 계절 기준선 | 0.5468 | 86.93 |
| LightGBM | **0.4079** | **66.58** |

LightGBM은 계절 기준선보다 RMSLE를 25.4%, MAE를 23.4% 낮췄습니다. 이는 첫 기준 성능이며, 상품군별 오차와 기간에 따른 안정성은 후속 분석에서 추가로 확인합니다.

## 현재 범위

첫 버전은 재현 가능한 예측 실험과 시간 순서 기반 검증에 집중했습니다. API 배포, 모니터링, 자동 학습과 TabFM 비교는 의도적으로 제외했습니다. 이후에는 동일한 변수 표본을 사용해 TabFM의 제로샷 성능만 별도 비교할 수 있습니다.
