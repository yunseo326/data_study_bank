# Experiments

## 목적

- Formal experiment의 구조와 기록 규격을 정의한다.
- artifact 생성과 저장은 가능한 한 `src/run_experiment.py`에서 자동화한다.
- 이 문서와 pipeline code를 experiment 세부 규격의 기준으로 사용한다.

## 기본 구조

```text
experiments/
├── registry.csv
└── E***/
    ├── results/
    │   ├── experiment.md
    │   ├── metrics.json
    │   └── oof_predictions.parquet
    └── reproducibility/
        ├── config.json
        └── run_manifest.json
```

실제 생성 파일은 `run_experiment.py`의 canonical output을 따른다.

### `results/`

결과 분석과 Reviewer 판정에 직접 사용하는 canonical artifact를 둔다.

- `experiment.md`: 가설, 관찰, 해석과 판단 근거
- `metrics.json`: 전체·fold metric, 실행 시간과 feature schema
- `oof_predictions.parquet`: 오류 분석과 metric 재계산에 사용하는 OOF 예측

Formal experiment는 가설의 검증과 결과 분석까지만 담당하며 submission을 생성하지 않는다. 제출이 필요해지면 검증 완료 후보를 입력으로 받는 별도 output 단계로 구현한다.

OOF는 `pyarrow`의 Zstandard 압축을 사용하는 Parquet으로 저장한다. CSV보다 dtype 보존과 읽기 속도, 저장 효율이 좋다.

### `reproducibility/`

동일 run의 재실행 조건과 artifact 검증 정보를 둔다.

- `config.json`: 실행 시 실제 적용된 config snapshot
- `run_manifest.json`: 진입점과 인자, Python·핵심 package 버전, config 및 `results/` artifact의 SHA-256과 크기

분석 파일과 재현 파일을 중복 저장하지 않는다. `run_manifest.json`이 상대 경로와 hash로 canonical artifact를 참조한다. 모델 바이너리는 환경 간 직렬화 호환성과 저장 비용 때문에 기본 artifact에 포함하지 않으며, 필요하면 별도 규격으로 추가한다.


## Formal Experiment 원칙

하나의 experiment는 하나의 핵심 hypothesis를 검증한다.

```text
Idea → Parent Experiment → Hypothesis → Experiment → Result → Reviewer
```

- experiment ID가 부여된 뒤 조건을 덮어쓰지 않는다.
- 조건을 바꾸면 새 experiment ID를 만든다.
- 실패한 experiment도 삭제하지 않는다.
- notebook 결과를 공식 결과로 승격하지 않고 pipeline으로 재실행한다.

## `registry.csv`

Registry는 전체 experiment의 검색용 index다. 상세 config나 metric을 복제하지 않는다.

최소 권장 column:

```text
experiment_id,parent_id,created_at,status,decision,hypothesis,model,
validation_version,primary_metric,primary_score,delta_vs_parent,
data_version,git_commit,experiment_path
```

- `status`: `RUNNING`, `COMPLETED`, `FAILED`
- `decision`: `KEEP`, `REJECT`, `ROBUSTNESS`, `INCONCLUSIVE`, `INVALID`

## 재현성

가능한 경우 다음 상태를 다시 확인할 수 있어야 한다.

- data version
- code version 또는 git commit
- config
- validation version
- seed
- model과 feature
- metric

## `results/experiment.md`

숫자를 반복 저장하는 문서가 아니라 가설, 관찰, 해석과 판단 근거를 남기는 문서다.

```markdown
# E***

## Parent

## Source Idea

## Hypothesis

## Reason

## Change

## Expected Result

## Result

## Reviewer

## Interpretation

## Verdict

## Conclusion
```

핵심 질문은 “왜 했는가”, “무엇을 예상했는가”, “실제로 무엇이 일어났는가”, “무엇을 배웠는가”이다.
