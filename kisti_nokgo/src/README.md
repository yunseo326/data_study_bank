# Source Code

## 3줄 요약

- `src/`는 재사용되는 canonical code를 저장한다.
- 일반 experiment 실행에서는 공통 구현을 직접 우회하거나 복제하지 않는다.
- 공통 계산이나 pipeline을 수정할 때 아래 검증 규칙을 적용한다.

## 파일 역할

| 경로 | 역할 |
|---|---|
| `EDA/` | 재사용 가능한 EDA 분석 코드 |
| `features/` | feature 생성 코드 |
| `models/` | 모델 정의와 학습 구성 |
| `validation.py` | 공통 validation 방식 |
| `metrics.py` | 공통 metric 계산 |
| `train.py` | feature → model → validation → prediction 연결 |
| `run_experiment.py` | formal experiment 실행과 artifact 기록 |
| `run_pipeline.py` | EDA와 baseline 실행 연결 |
| `tests/` | feature, validation, metric, pipeline 검증 |

## 실행 계약

- 실행 조건은 `configs/*.json`에 둔다.
- 공통 진입점은 `python -m src.run_experiment`이다.
- 각 experiment ID는 한 번만 사용하며 기존 artifact를 덮어쓰지 않는다.
- 분석·판정용 artifact는 `<experiment_id>/results/`, config snapshot과 실행 환경·hash 검증 정보는 `<experiment_id>/reproducibility/`에 저장한다.
- 기본 시간 제한은 config의 `runtime.time_budget_seconds`로 관리한다.
- 빠른 baseline은 전체 데이터의 대표 표본으로 검증하고, 전체 test에는 최종 모델만 추론한다.
- 실제 명령과 사용 순서는 프로젝트 루트 `README.md`를 따른다.

## 구현 원칙

### 1. Smoke Test

작은 데이터로 최소 실행 여부를 확인한다.

### 2. Known-answer Test

사람이 직접 계산할 수 있는 작은 예제와 구현 결과를 비교한다.

### 3. Split / Leakage Check

target 또는 미래 정보가 feature 생성 과정에 들어갈 가능성이 있으면 leakage 검사를 우선한다. 데이터 분할 뒤에 학습되어야 하는 변환은 fold 내부에서 fit한다.

### 4. End-to-End Check

공통 pipeline을 수정하면 작은 데이터로 다음 연결을 확인한다.

```text
feature → model → validation → prediction → metric → experiment artifact
```

### 5. 변경 기록

공통 계산, validation, metric, artifact 규격 또는 파일 구조가 바뀌면 관련 tests를 갱신하고 `reports/SYSTEM_CHANGES.md`에 영향과 비교 가능성을 기록한다.

## 현재 기본 구현의 한계

- 범주형 feature는 빠른 범용 실행을 위해 ordinal encoding을 사용한다.
- baseline 모델은 `hist_gradient_boosting` 하나만 제공한다.
- 시간 예산은 각 단계 사이에서 확인하므로 단일 fit을 강제로 중단하지는 못한다.
- CSV 전체 적재가 메모리를 초과하는 데이터에는 별도 chunked loader가 필요하다.
