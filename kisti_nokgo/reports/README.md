# Reports

## 3줄 요약

- `reports/`는 사람이 프로젝트를 이해하기 위한 결과물 공간이다.
- 원본 experiment 기록을 복제하기보다 중요한 결과만 요약한다.
- 사용자가 요청한 경우에만 HTML 또는 GitHub Pages 결과물을 생성한다.

## 기본 파일

```text
EDA.md (EDA 실행 후 생성)
EXPERIMENTS.md
SYSTEM_CHANGES.md
```

## `EDA.md`

사람이 빠르게 확인할 수 있도록 다음 내용을 정리한다.

- Dataset Overview
- Target
- Missing Values
- Distribution
- Train/Test Comparison
- Potential Leakage
- Important Relationships
- Initial Findings

`--render-visuals`를 사용하면 `reports/assets/`에 target 분포, 숫자형 feature 히스토그램과 Pearson 상관 heatmap을 SVG로 생성한다. 이는 일반 표형 데이터용 선택 기능이다. 시간축·그룹·도메인별 그래프는 해당 구조가 확인된 뒤 별도 EDA로 추가한다.

## `EXPERIMENTS.md`

다음을 중심으로 요약한다.

- 현재 best
- 중요한 Parent → Child 변화
- 유효했던 방향과 실패한 방향
- 현재 후보
- 다음 질문
- 기타 중요한 맥락

실험의 상세 조건과 원본 수치는 각 experiment의 config와 metrics를 참조한다.

## `SYSTEM_CHANGES.md`

코드, 데이터 처리, validation, metric, pipeline, artifact 또는 파일 구조를 바꾸면 다음을 기록한다.

- 날짜
- 변경 내용과 이유
- 영향 범위
- 이전 결과와의 호환·비교 가능성
- 필요한 migration 또는 재실행
- 사용자 판단이 필요한 항목

## 결과물 생성 원칙

- HTML 또는 GitHub Pages는 사용자 요청에 포함되었을 때만 생성한다.
- 요청받은 HTML 결과물의 GitHub Pages 게시 위치는 <https://yunseo326.github.io/data_study_bank/>이다.
- HTML을 만들 때 `styles.css`를 기본 스타일로 연결한다. `reports/` 내 HTML에서는 `href="styles.css"`를 사용하고, 게시할 때는 HTML과 CSS의 상대 경로가 유지되도록 함께 배치한다.
- 타이포그래피는 본문 16px, 표 14.4px, 보조 텍스트 약 13px, H3 16.8px, H2 23.2px, H1 최대 56px을 기본으로 한다. 모바일 H1은 32px을 기준으로 한다.
- 이 CSS의 hero 및 project 선택자는 랜딩 페이지용이다. 필요한 구성만 사용하고, hero 배경 이미지 `assets/hero-seoul-workspace.webp`가 없으면 실제 자산을 추가하거나 배경을 재정의한다.
- 모든 experiment마다 별도 보고서를 만들 필요는 없다.
- 보고서는 사람이 읽기 위한 파생물이며 Source of Truth인 원본 experiment 기록을 대체하지 않는다.
