# Reviewer Rules

Reviewer의 우선 역할은 **새 아이디어 생성이 아니라 현재 experiment의 유효성을 검증하는 것**이다.

## 검토 순서

```text
registry.csv
↓
필요 시 experiment.md
↓
metrics / OOF 등 필요한 artifact
↓
Parent Experiment
↓
Reviewer
↓
Verdict
↓
필요 시 Idea Generation 및 IDEAS.md 기록
```

## 확인 항목

- 가설을 반증하는 근거
- 실패 가능성과 대안 설명
- leakage
- validation 오류
- 특정 fold 의존성
- error pattern과 subgroup 편향
- Parent 대비 실제 변경과 성능 차이
- hypothesis와 결과의 일치 여부
- validation overfitting 가능성
- 관찰된 차이가 노이즈나 variance로 설명될 가능성

Reviewer는 결과를 정당화하기 위한 설명을 만드는 역할이 아니다. 판단이 끝나기 전에 새로운 feature나 model 아이디어로 결과를 합리화하지 않는다.

## Verdict

### KEEP

Hypothesis를 지지할 근거가 있고 다음 실험의 Parent 또는 후보로 사용할 가치가 있다.

### REJECT

가설을 지지할 근거가 부족하거나 성능·안정성이 악화되었다.

### ROBUSTNESS

Primary Validation에서 유망하지만 seed, split, subgroup 등 추가 강건성 검증이 필요하다.

### INCONCLUSIVE

차이가 너무 작거나 variance가 커서 판단하기 어렵다.

### INVALID

Leakage, 구현 오류, validation 문제 등으로 experiment 자체가 유효하지 않다. 수정한 동일 ID로 재사용하지 말고 원인을 기록한 뒤 새 experiment를 실행한다.

