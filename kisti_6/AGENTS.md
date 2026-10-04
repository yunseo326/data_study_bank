# 프로젝트 작업 규칙

## 프로젝트 목적

이 프로젝트는 AI를 활용해 데이터 분석 실험을 빠르게 반복하면서도 다음을 유지하는 것을 목표로 한다.

- 실험 간 비교 가능성
- 재현성
- 데이터 누수(leakage) 방지
- 검증(validation) 일관성
- AI와 모델의 과적합 방지
- 가설과 결과의 추적 가능성
- 사람이 필요한 지점에서 개입할 수 있는 구조

프롬프트 하나의 품질보다 **고정된 분석 파이프라인과 검증 구조**를 우선한다.

## 사용자에게 답변할 때

1. 수식보다 개념과 원리를 먼저 설명한다.
2. 정확한 정의에 필요하면 수식을 함께 제시한다.
3. 선택지를 제시할 때 우선순위와 근거를 함께 설명한다.
4. 규칙에 없는 상황에서는 결과에 중요한 결정은 사용자에게 확인하고, 영향이 작은 구현 세부사항은 일반적으로 권장되는 방법을 따른다.
5. 사용자가 지정한 결과물 형식을 우선한다.
6. HTML 또는 GitHub Pages 게시가 명시적으로 요청된 경우에만 해당 형식으로 제공한다.

## HTML 결과물 스타일 규칙

- HTML 결과물은 `reports/styles.css`를 기본 스타일 기준으로 사용한다. 색상·글꼴 변수, 기본 타이포그래피, 키보드 포커스, 화면 크기별 배치와 동작 감소 설정을 우선 적용한다.
- 기본 글꼴 크기는 본문 16px, 표 14.4px, 보조 텍스트·링크 약 13px, H3 16.8px, H2 23.2px, H1 최대 56px의 밀도를 따른다. 제목 크기를 화면 너비에 따라 키우더라도 이 상한을 넘기지 않는다.
- `.hero`, `.topbar`, `.projects` 등은 랜딩 페이지용 선택자다. 해당 구성과 콘텐츠가 필요한 페이지에서만 사용하고, 분석 보고서의 구조와 가독성은 콘텐츠에 맞게 확장한다.
- HTML에서 CSS를 외부 파일로 연결한다. `reports/` 안의 HTML은 `<link rel="stylesheet" href="styles.css">`를 사용하며, 다른 위치나 GitHub Pages에 게시할 때는 CSS도 함께 배치하고 게시 경로에 맞게 상대 경로를 확인한다.
- CSS의 `.hero::before`는 `assets/hero-seoul-workspace.webp`를 참조한다. hero를 사용하는 경우 실제 이미지 파일을 배치하거나 해당 페이지에서 배경을 재정의한다. 누락된 이미지를 전제로 페이지를 완성하지 않는다.
- 사용자 지정 디자인이나 템플릿이 있으면 그 요청을 우선한다. HTML 생성·게시 자체는 명시적 요청이 있을 때만 수행한다.

## 시스템 변경 기록

데이터 분석 과정, 코드, 파일 구조 또는 핵심 개념을 변경하면 `reports/SYSTEM_CHANGES.md`에 다음을 기록한다.

- 변경 내용과 이유
- 영향을 받는 실험·데이터·문서
- 이전 결과와의 비교 가능성
- 사용자 확인이 필요한 판단

답변에도 변경 사실을 명확히 알린다. 필요하면 사용자와 함께 이 문서나 파일별 설명을 갱신한다.

## 금지 사항

- 사용자의 명시적 요청 없이 `AGENTS.md` 또는 `CLAUDE.md` 수정
- 원본(raw) 데이터 수정
- Primary Validation 임의 변경
- 같은 experiment ID의 조건 임의 변경
- notebook 결과를 formal experiment처럼 취급
- Public Leaderboard만 보고 실험 채택
- 실패한 experiment 삭제
- 기존 실험을 확인하지 않고 같은 아이디어 반복
- Reviewer 검증 전에 결과를 설명하기 위한 새 아이디어 생성
- 공통 파이프라인 기능을 임시 코드로 중복 구현

## 기본 구조

```text
project/
├── AGENTS.md
├── CURRENT_STATE.md
├── IDEAS.md
├── progress.md
├── reviewer_rule.md
├── notebooks/
│   └── README.md
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── README.md
│   ├── EDA/
│   ├── tests/
│   ├── features/
│   ├── models/
│   ├── validation.py
│   ├── metrics.py
│   ├── train.py
│   └── run_experiment.py
├── experiments/
│   ├── README.md
│   ├── registry.csv
│   └── <experiment_id>/
│       ├── results/
│       │   ├── experiment.md
│       │   ├── metrics.json
│       │   └── oof_predictions.parquet
│       └── reproducibility/
│           ├── config.json
│           └── run_manifest.json
├── reports/
│   ├── README.md
│   ├── EDA.html
│   ├── SYSTEM_CHANGES.md
│   ├── styles.css
│   └── EXPERIMENTS.md
└── references/
    ├── README.md
    └── competitions/
        └── <competition>/
            ├── AI_PLANNING_REFERENCE.md
            └── sources/
```

## 파일별 역할

- `CURRENT_STATE.md`: 현재 best, 후보, 진행 중 작업, 알려진 문제와 다음 행동만 기록한다.
- `IDEAS.md`: formal experiment로 확정되지 않은 아이디어와 실행된 실험의 연결을 관리한다.
- `notebooks/`: EDA, 빠른 검토, 오류 분석을 위한 정식 실험 이전의 탐색 공간이다.
- `progress.md`: 중요한 작업, 방향 변경, 결정, 문제와 milestone을 시간순으로 기록한다.
- `src/`: 공통 분석 파이프라인과 재사용 코드를 보관한다.
- `experiments/`: formal experiment를 run 단위로 보관한다. 분석·판정에 사용하는 산출물은 `results/`, 재현 조건과 무결성 검증 정보는 `reproducibility/`에 둔다.
- `reports/`: 사람이 읽기 위한 파생 결과물을 보관한다.
- `references/`: 대회 규정, 과거 웹 스냅샷, 초기 샘플처럼 조건부로만 필요한 참고자료를 보관한다. 일반 작업 재개나 formal experiment에서는 자동으로 읽지 않는다.

## Source of Truth

| 정보 | 기준 파일 |
|---|---|
| 프로젝트 최상위 규칙 | `AGENTS.md` |
| 현재 진행 상태 | `CURRENT_STATE.md` |
| 미실험 아이디어 | `IDEAS.md` |
| 실험 전체 인덱스 | `experiments/registry.csv` |
| 실험 실행 조건 | 각 experiment의 `reproducibility/config.json` |
| 실행 환경·산출물 검증 정보 | 각 experiment의 `reproducibility/run_manifest.json` |
| 실험 수치 결과 | 각 experiment의 `results/metrics.json` |
| 실험 가설·해석·결론 | 각 experiment의 `results/experiment.md` |
| 공통 구현 | `src/` |
| 구현 규칙 | `src/README.md` |
| experiment 구조 | `experiments/README.md` |
| notebook 규칙 | `notebooks/README.md` |
| reviewer 규칙 | `reviewer_rule.md` |
| 보고서 규칙 | `reports/README.md` |
| 프로젝트 진행 이력 | `progress.md` |
| 사람용 요약 결과 | `reports/` |
| 대회 계획용 조건부 참고 | `references/competitions/<competition>/AI_PLANNING_REFERENCE.md` |

`reports/`는 파생 결과이며 원본 experiment 기록을 대체하지 않는다.
`references/`는 저장 시점의 보조 근거이며 최신 공식 공지, 현재 상태 또는 experiment artifact를 대체하지 않는다.

## 기본 분석 흐름

```text
DATA → EDA → Baseline → 아이디어 생성 → 필요 시 Notebook 탐색
→ Formal Hypothesis → 공통 Experiment Pipeline 실행 → 결과 분석
→ Reviewer → KEEP / REJECT / ROBUSTNESS / INCONCLUSIVE / INVALID
→ 유망 후보만 Robustness Check → Final Candidate
→ 필요 시 Ensemble → Final Report
```

## 작업 시작 시 탐색 순서

모든 파일을 매번 읽지 말고 작업에 필요한 파일만 확인한다.

- 일반 작업 재개: `AGENTS.md` → `CURRENT_STATE.md`
- 새 아이디어 검토: `CURRENT_STATE.md` → `experiments/registry.csv` → `IDEAS.md` → 필요 시 과거 experiment·notebook
- Formal Experiment: `CURRENT_STATE.md` → 관련 idea 또는 parent → `experiments/README.md` → `src/run_experiment.py`
- 기존 Experiment 분석: `reviewer_rule.md` → 필요한 experiment artifact
- 코드 수정: `src/README.md` → 관련 source → 관련 tests
- Notebook 작업: `notebooks/README.md`
- 보고서 작성: `reports/README.md`
- 과거 흐름 확인: `experiments/registry.csv` → `progress.md` → 필요한 과거 experiment
- 대회 참여 여부·전체 전략·일정·제출 규정 검토가 명시된 경우에만: `CURRENT_STATE.md` → 해당 `references/competitions/<competition>/AI_PLANNING_REFERENCE.md` → 최신 공식 페이지 재확인

`references/` 전체를 관성적으로 탐색하지 않는다. 일반 코드 수정, EDA, 개별 formal experiment, reviewer 판정에는 현재 작업과 직접 관련된 Source of Truth만 읽는다.

공통 experiment 파이프라인을 임의로 우회하지 않는다.

## Primary Validation과 Robustness

Primary Validation은 experiment 간 공식 비교 기준이다.

1. 실험 간 비교를 위해 고정한다.
2. 일반 experiment 도중 임의로 변경하지 않는다.
3. Validation 방식이 바뀌면 이전 experiment와 score를 직접 비교하지 않는다.
4. Validation 변경 자체가 연구 대상이면 별도 experiment 또는 validation version으로 관리한다.
5. test/Public Leaderboard는 Primary Validation을 대체하지 않는다.

Robustness Check는 모든 experiment가 아니라 Reviewer가 유망하다고 판단한 후보에만 수행한다. Robustness 결과는 Primary CV와 구분해 기록하며 Primary Validation 결과를 덮어쓰지 않는다.
