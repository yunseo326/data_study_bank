# 데이터 분석 프로젝트 템플릿

이 템플릿의 목적은 AI에게 매번 전체 분석을 다시 설명하게 하는 대신, **설정 파일과 고정된 Python 파이프라인으로 EDA·검증·실험 기록을 반복 실행**하는 것이다.

## 빠른 시작

### 1. 환경 준비

```powershell
python -m pip install -r requirements.txt
```

### 2. 데이터 배치

기본 구조는 다음과 같다.

```text
data/
├── train.csv
└── test.csv
```

원본 파일은 수정하지 않는다. 비싼 전처리 결과만 `data/processed/`에 별도로 저장한다.

### 3. 설정 수정

`configs/baseline.json`에서 다음만 먼저 확인한다.

- `train_path`, `test_path`
- `target`: 모르면 `null`로 두되 train에만 있는 열이 정확히 하나여야 한다.
- `id_columns`: 학습에서 제외할 식별자
- `max_train_rows`: 1분 제한을 지키기 위한 학습 표본 수
- `time_budget_seconds`: experiment 자체의 제한 시간

### 4. 테스트

```powershell
python -m unittest discover -s src/tests -v
```

### 5. EDA만 실행

```powershell
python -m src.EDA.run_eda --config configs/baseline.json
```

결과는 `reports/EDA.md`, `reports/eda_summary.json`에 저장된다.

숫자형 feature 히스토그램, Pearson 상관 heatmap과 target 분포 SVG를 함께 만들려면 다음처럼 실행한다.

```powershell
python -m src.EDA.run_eda --config configs/baseline.json --render-visuals
```

시각화는 EDA 표본을 사용하고 식별자·target을 feature 그래프에서 제외한다. 가독성과 실행 비용을 위해 숫자형 feature는 원래 열 순서의 최대 20개만 표시한다. Pearson 상관은 선형 관계만 나타내므로 비선형 관계나 인과관계의 근거로 해석하지 않는다.

HTML 결과물이나 GitHub Pages 게시가 요청된 경우 게시 위치는 <https://yunseo326.github.io/data_study_bank/>이다. 결과물 생성 원칙은 `reports/README.md`를 따른다.

### 6. Formal baseline 실행

```powershell
python -m src.run_experiment --config configs/baseline.json --experiment-id E000
```

다음 artifact가 생성된다.

```text
experiments/E000/
├── results/
│   ├── experiment.md
│   ├── metrics.json
│   └── oof_predictions.parquet
└── reproducibility/
    ├── config.json
    └── run_manifest.json
```

`results/`는 결과 분석과 Reviewer 판정에 사용하는 파일을 모은다. `reproducibility/`는 실제 적용된 config snapshot과 실행 환경, 각 artifact의 SHA-256을 담아 재실행 및 무결성 확인에 사용한다. Formal experiment 단계는 submission을 생성하지 않는다.

`experiments/registry.csv`에도 검색용 한 행이 추가된다. 이미 사용한 experiment ID는 재사용할 수 없다.

### 7. EDA와 baseline을 한 번에 실행

```powershell
python -m src.run_pipeline --config configs/baseline.json --experiment-id E000
```

## 1분 제한 설계

- 전체 데이터를 읽어 스키마를 확인하되 학습은 `max_train_rows`의 고정 표본으로 수행한다.
- 분류 표본은 target 비율을 보존한다.
- 기본 모델은 전처리 비용이 낮은 histogram gradient boosting이다.
- 기본 검증은 seed가 고정된 3-fold CV다.
- experiment의 CV와 artifact 준비가 `time_budget_seconds`를 넘으면 실패 처리한다.
- 데이터가 더 크거나 열이 많으면 먼저 `max_train_rows`, `max_iter`, `n_splits` 순으로 낮춘다.

EDA와 experiment를 함께 실행한 전체 시간은 experiment 제한보다 길 수 있다. 엄격히 한 번의 1분 제한이 필요하면 EDA를 먼저 한 차례 수행하고 이후에는 `--skip-eda`를 사용한다.

### 검증 기본값의 의미

`stratified 3-fold`는 모든 데이터에 적용되는 절대 표준이 아니다. 이 템플릿에서는 **일반적인 IID 분류 문제를 1분 안에 안정적으로 비교하기 위한 기본값**으로 선택했다.

- Target 비율 보존: 분류 데이터의 각 fold에 class 비율을 비슷하게 유지하는 널리 쓰이는 기본 방법이다. 특히 불균형 분류에서 우연히 특정 class가 fold에 부족해지는 문제를 줄인다.
- 3-fold: 5-fold보다 계산량이 작고 단일 holdout보다 fold 운에 덜 민감한 절충안이다. 1분 제한이 없다면 5-fold가 더 적합할 수 있다.
- 고정 seed: 같은 데이터와 config에서 동일한 split을 재현하고 experiment 간 비교 가능성을 유지하기 위한 규칙이다.

다음 경우에는 기본 CV를 그대로 사용하지 않는다.

| 데이터 구조 | 권장 검증 |
|---|---|
| 일반 IID 분류 | `StratifiedKFold` |
| 일반 IID 회귀 | `KFold` |
| 시간 순서가 있는 데이터 | 시간 순서를 보존하는 split 또는 `TimeSeriesSplit` |
| 같은 사용자·장비·지역이 반복됨 | `GroupKFold` 또는 stratified group split |
| 희소 class | 최소 class 수보다 작은 `n_splits`, 필요 시 repeated holdout |
| 데이터가 매우 작음 | 예산이 허용되면 5-fold 이상 또는 repeated CV |
| 데이터가 매우 큼 | 2~3 fold 또는 충분히 큰 고정 holdout |

Validation 방식을 바꾸면 기존 score와 직접 비교하지 말고 `validation_version`을 새로 부여한다. 데이터 크기만 보고 CV를 자동 변경하면 experiment 간 비교 기준이 흔들리므로, 최초 baseline에서 정한 Primary Validation은 같은 실험 계열에서 고정한다.

## 새 프로젝트에 적용하는 순서

1. 이 템플릿을 복사한다.
2. `data/`에 원본 CSV를 배치한다.
3. `configs/baseline.json`의 경로, target, ID를 수정한다.
4. 테스트와 EDA를 실행한다.
5. EDA의 warning과 schema를 확인한다.
6. 첫 실험 ID `E000`으로 baseline을 실행한다.
7. `reviewer_rule.md`에 따라 결과를 검토한다.
8. 새 아이디어는 `IDEAS.md`에 먼저 등록한다.
9. 하나의 가설만 바꾼 새 config로 다음 experiment ID를 실행한다.

## AI에게 요청하는 권장 형식

좋은 요청은 이미 존재하는 코드와 artifact를 가리킨다.

```text
I001 아이디어를 E000을 parent로 검증해줘.
configs/e001_i001.json을 만들고 src/run_experiment.py로 E001을 실행해줘.
Primary Validation은 바꾸지 말고 reviewer_rule.md에 따라 판정해줘.
```

```text
E001의 OOF에서 관심 있는 열별 error pattern을 분석하는
src/EDA/analyze_subgroups.py를 만들고 실행해줘.
새 모델 아이디어는 Reviewer 판정 뒤 IDEAS.md에만 기록해줘.
```

피해야 할 요청:

```text
이 데이터 분석해줘.
성능 올려줘.
알아서 좋은 모델 찾아줘.
```

## 코드 책임 경계

- `src/EDA/`: 빠른 데이터 이해와 오류 분석
- `src/features/`: feature schema와 재사용 가능한 전처리
- `src/models/`: 모델 정의
- `src/validation.py`: 고정 CV
- `src/metrics.py`: 공식 metric 계산
- `src/train.py`: 전처리와 모델 연결
- `src/run_experiment.py`: formal experiment 실행과 `results/`·`reproducibility/` artifact 생성
- `src/run_pipeline.py`: EDA와 baseline orchestration

공통 동작을 바꿀 때는 `src/tests/`와 `reports/SYSTEM_CHANGES.md`를 함께 갱신한다.

## 데이터별로 결정할 사항

- 첫 실험 전에 target과 식별자 열, 평가 metric, Primary Validation을 확인한다.
- 기본 설정의 `target: null`은 train에만 있는 열이 정확히 하나일 때만 동작한다.
- 숫자형 열은 median imputation, 범주형 열은 unknown-safe ordinal encoding을 사용한다. 데이터의 시간·그룹 구조가 있으면 기본 CV를 그대로 적용하지 않는다.
