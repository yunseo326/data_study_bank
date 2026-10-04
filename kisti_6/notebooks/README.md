# Notebooks

## 3줄 요약

- Notebook은 formal experiment 이전의 탐색 공간이다.
- EDA, 빠른 아이디어 검토와 error analysis에 사용한다.
- Notebook 결과 자체를 공식 experiment 성과로 취급하지 않는다.

## 역할

### EDA

프로젝트 초기에 데이터를 이해한다.

- 데이터 구조와 column 의미
- target
- 결측과 이상치
- 분포와 category
- train/test 차이
- 초기 관계 분석
- 잠재적 leakage

### Sandbox

확인할 가치는 있지만 formal experiment로 만들기에는 이른 작업을 빠르게 검토한다.

- 새 feature 관계 확인
- 특정 subgroup 분석
- error pattern 확인
- 간단한 transformation
- model disagreement 확인

## 흐름

```text
Notebook 탐색
↓
가치 있음?
├─ NO → 종료하고 필요한 관찰만 기록
└─ YES
   ↓
IDEAS.md
   ↓
Formal Hypothesis
   ↓
src + Experiment Pipeline
```

## 규칙

- 재사용할 코드는 `src/`로 옮긴다.
- notebook score만으로 experiment를 `KEEP`하지 않는다.
- 중요한 아이디어는 `IDEAS.md`에 저장한다.
- formal 비교에 사용할 결과는 공통 validation과 experiment pipeline으로 다시 실행한다.
- notebook에 raw data를 덮어쓰는 코드를 두지 않는다.

