# System Changes

## 2026-09-24 — 하천수위 문제 참가신청서 초안 작성

### 변경 내용과 이유

- 공식 참가신청서의 필수 항목 순서와 대회 분석 자료를 바탕으로 `reports/2026_하천수위_참가신청서_초안.docx`를 작성했다.
- 개인정보와 서명은 제출자가 직접 입력하도록 비워 두고, 문제해결 제안의 데이터 활용, 검증, 모델링, 추진일정, 결과물과 위험 대응을 핵심 중심으로 작성했다.
- 초기 296행 샘플은 target tail과 데이터 형식의 사전 점검에만 사용했으며 공식 성능 근거로 사용하지 않았다.
- 유역 반응 지문, 상승량·위험구간 다중과제 학습, 호우사상 난이도 균형 앙상블, 사상 단위 불확실성을 추가 아이디어로 제시했다.

### 영향을 받는 실험·데이터·문서

- `reports/2026_하천수위_참가신청서_초안.docx`, `CURRENT_STATE.md`, `progress.md`.
- 원본 HWP와 CSV는 수정하지 않았고 formal experiment도 실행하지 않았다.

### 이전 결과와의 비교 가능성

- 분석 pipeline, Primary Validation, metric과 실험 artifact를 변경하지 않아 기존 결과 비교 가능성에 영향이 없다.
- 문서의 일정과 방법론은 최종 데이터 확보 및 공식 Q&A 갱신 시 수정될 수 있다.

### 사용자 확인이 필요한 판단

- 팀명, 참가자 개인정보, 역할 배정과 서명을 직접 입력해야 한다.
- 최종 제출 전 공식 HWP 원본으로 문안을 옮기고 최신 공지·Q&A를 다시 확인해야 한다.

## 2026-09-23 — 대회 참고자료를 조건부 참조 영역으로 분리

### 변경 내용과 이유

- 6시간 최대 하천수위 상승량 예측 대회의 공식 페이지, 참가신청서와 초기 샘플을 `references/competitions/2026_river_level/` 아래로 옮겼다.
- 대회 전체 전략을 세울 때만 읽는 `AI_PLANNING_REFERENCE.md`와 조건부 읽기 규칙을 추가했다.
- 일반 작업 재개와 formal experiment가 오래된 웹 스냅샷이나 최종 데이터가 아닌 296행 초기 샘플을 자동으로 읽지 않도록 `AGENTS.md`의 탐색 순서를 구분했다.

### 영향을 받는 실험·데이터·문서

- `AGENTS.md`, `CURRENT_STATE.md`, `references/README.md`, `references/competitions/2026_river_level/`, `progress.md`.
- 루트의 `2026_참가신청서.hwp`, `6시간뒤 하천수위.md`와 `data/6시간뒤 수위.csv`는 참고자료 폴더로 이동했다.
- 원본 바이트를 수정하지 않았고 등록된 formal experiment는 없다.

### 이전 결과와의 비교 가능성

- 모델, metric, validation과 experiment pipeline은 변경하지 않아 실험 비교 가능성에는 영향이 없다.
- 296행 CSV는 최종 배포 데이터가 아니므로 이전에도 공식 score 비교 근거가 될 수 없었다.

### 사용자 확인이 필요한 판단

- 최종 Train/Test/Data Dictionary를 확보하면 `data/raw/` 배치와 storm-event group 복원 방법을 확정해야 한다.
- 공개 Q&A의 Test 미래행 누수 관련 공식 답변 또는 데이터 재배포 여부를 baseline 전에 확인해야 한다.

## 2026-09-23 — Formal experiment에서 submission 단계 제거

### 변경 내용과 이유

- `submission.csv`는 결과 분석이나 Primary Validation의 근거가 아니라 외부 제출용 output이므로 formal experiment artifact에서 제거했다.
- 제출 전용 config 항목과 생성 함수, 제출에만 필요했던 전체 학습 데이터 최종 fit 및 test 추론을 공통 experiment pipeline에서 제거했다.
- 향후 제출이 필요하면 Reviewer 검증을 통과한 후보를 입력으로 받는 별도 output 단계로 구현한다.

### 영향을 받는 실험·데이터·문서

- `src/run_experiment.py`, `src/config.py`, `src/train.py`, `configs/baseline.json`, `src/tests/test_experiment_layout.py`, `AGENTS.md`, `README.md`, `experiments/README.md`, `CURRENT_STATE.md`.
- 원본 데이터와 현재 등록된 experiment는 변경하지 않았다.

### 이전 결과와의 비교 가능성

- CV, OOF와 metric 계산은 바뀌지 않아 실험 점수의 비교 가능성에는 영향이 없다.
- 최종 fit과 test 추론이 제거되어 새 run의 전체 실행 시간은 과거 submission 생성형 run과 직접 비교하지 않는다.

### 사용자 확인이 필요한 판단

- 실제 제출 단계가 필요해질 때 output 위치, 전체 데이터 재학습 조건과 제출 형식을 별도로 확정한다.

## 2026-09-23 — Run artifact의 결과·재현 영역 분리

### 변경 내용과 이유

- 각 experiment run의 분석·판정용 artifact를 `results/`에, config snapshot과 재현·검증 정보를 `reproducibility/`에 분리했다.
- `run_manifest.json`에 실행 진입점과 인자, Python·핵심 package 버전, config 및 결과 artifact의 SHA-256과 파일 크기를 기록하도록 했다.
- 책임이 다른 파일을 한 디렉터리에 섞지 않고, 결과 파일을 복제하지 않은 채 manifest가 canonical artifact를 참조하도록 했다.

### 영향을 받는 실험·데이터·문서

- `src/run_experiment.py`, `src/tests/test_experiment_layout.py`, `AGENTS.md`, `README.md`, `experiments/README.md`, `src/README.md`.
- 현재 등록된 experiment가 없어 기존 artifact 이동은 발생하지 않았다. 원본 데이터와 Primary Validation은 변경하지 않았다.

### 이전 결과와의 비교 가능성

- metric 계산, OOF 생성, validation과 모델 학습은 바뀌지 않아 수치 비교 가능성에는 영향이 없다.
- artifact 경로 규격은 layout version 2로 바뀌었다. 과거 layout을 가진 외부 복사본이 있다면 새 경로를 가정하는 도구와 바로 호환되지 않는다.

### 사용자 확인이 필요한 판단

- 없음. 모델 바이너리는 환경 간 호환성과 저장 비용 때문에 기본 재현 artifact에서 제외했다.

## 2026-09-22 — 범용 표형 EDA 시각화 추가

### 변경 내용과 이유

- 선택 옵션 `--render-visuals`로 target 분포, 숫자형 feature 히스토그램, Pearson 상관 heatmap을 의존성 없는 SVG로 생성하도록 했다.
- 식별자와 target은 feature 그래프에서 제외하고, 표본 및 최대 20개 숫자형 열만 사용해 실행 비용과 출력 크기를 제한했다.
- 회귀 target은 개별 값 빈도표 대신 기술통계를 출력해 연속값 target에서 보고서가 비정상적으로 커지는 문제를 막았다.
- 첨부안의 시계열 개요는 timestamp와 값 열의 의미를 사용자가 지정해야 하므로 범용 기본 기능에 포함하지 않았다.

### 영향을 받는 실험·데이터·문서

- `src/EDA/run_eda.py`, `src/EDA/visuals.py`, `src/tests/test_eda.py`, `README.md`, `reports/README.md`.
- 원본 데이터, Primary Validation, 모델 학습과 기존 experiment artifact는 변경하지 않았다.

### 이전 결과와의 비교 가능성

- EDA 출력 형식만 확장되므로 실험 점수의 비교 가능성에는 영향이 없다.
- 회귀 EDA의 target 표현은 빈도표에서 기술통계로 바뀌므로 과거 `EDA.md`와 문서 구조는 다를 수 있다.

### 사용자 확인이 필요한 판단

- 시간·그룹 구조가 있는 실제 데이터가 들어오면 timestamp, group 열과 해당 구조 전용 EDA 범위를 별도로 확정한다.

## 2026-09-21 — HTML 타이포그래피 크기 조정

### 변경 내용과 이유

- 사용자가 지정한 참고 페이지의 계산된 글꼴 크기를 기준으로 공통 타이포그래피 변수를 추가했다.
- 본문 16px, 표 14.4px, 보조 텍스트 약 13px, H3 16.8px, H2 23.2px, H1 최대 56px로 정리했다.
- 기존 hero 제목 최대 116px과 섹션 제목 최대 76.8px을 줄이고, 모바일 H1을 32px로 조정했다. 큰 제목에 맞춰 과도했던 일부 섹션 여백도 줄였다.

### 영향을 받는 실험·데이터·문서

- `reports/styles.css`, `AGENTS.md`, `reports/README.md`. 실험, 원본 데이터, 분석 코드는 변경하지 않았다.

### 이전 결과와의 비교 가능성

- 표현 스타일만 바뀌며 실험 수치와 비교 가능성에는 영향이 없다.

### 사용자 확인이 필요한 판단

- 없음.

## 2026-09-21 — HTML 공통 스타일 기준 추가

### 변경 내용과 이유

- 사용자가 제공한 `styles.css`를 `reports/styles.css`에 보관하고, 요청받은 HTML 및 GitHub Pages 결과물에서 사용할 기준으로 지정했다.
- 공통 색상·글꼴·접근성 설정과 특정 랜딩 페이지 전용 구성을 구분하고, 게시 시 CSS와 hero 이미지의 상대 경로를 확인하도록 규칙을 추가했다.

### 영향을 받는 실험·데이터·문서

- `AGENTS.md`, `reports/README.md`, `reports/styles.css`. 실험, 원본 데이터, 분석 코드는 변경하지 않았다.

### 이전 결과와의 비교 가능성

- 분석 파이프라인과 Primary Validation은 바뀌지 않았다. 기존 실험 결과의 비교 가능성에 영향이 없다.

### 사용자 확인이 필요한 판단

- 없음. 실제 HTML 제작 시 페이지별 콘텐츠와 이미지 자산은 해당 요청에 맞춰 결정한다.

## 2026-09-21 — HTML 결과물 게시 위치 지정

### 변경 내용과 이유

- 사용자가 지정한 GitHub Pages 주소 <https://yunseo326.github.io/data_study_bank/>를 HTML 결과물의 게시 위치로 문서화했다.
- HTML 생성과 게시는 명시적으로 요청된 경우에만 수행한다는 기존 원칙을 유지한다.

### 영향을 받는 실험·데이터·문서

- `README.md`, `reports/README.md`. 실험과 데이터 변경은 없다.

### 이전 결과와의 비교 가능성

- 분석 파이프라인과 검증 기준은 바뀌지 않아 기존 결과의 비교 가능성에 영향이 없다.

### 사용자 확인이 필요한 판단

- 없음. 게시 주소는 사용자가 직접 지정했다.

## 2026-09-18 — 설정 기반 분석 파이프라인 구축

### 변경 내용과 이유

- JSON 설정 기반 EDA, feature schema, baseline model, 고정 CV, metric, experiment artifact와 submission 생성을 추가했다.
- 데이터 교체 시 공통 코드를 다시 작성하지 않고 경로, target, ID와 시간 제한을 설정으로 바꾸도록 했다.
- OOF artifact는 dtype 보존과 저장 효율을 위해 Zstandard 압축 Parquet으로 저장한다. `pyarrow`가 필요하다.

### 영향을 받는 실험·데이터·문서

- 공통 코드: `src/`, 설정: `configs/`, 실행 안내: `README.md`, 실험 규격: `experiments/README.md`.
- 현재 보관된 데이터와 실험 결과는 없다.

### 이전 결과와의 비교 가능성

- 새 데이터의 첫 baseline에서 Primary Validation과 metric을 정한다. 검증 방식, sampling 또는 metric이 다른 실험의 점수는 직접 비교하지 않는다.

### 사용자 확인이 필요한 판단

- 데이터 특성에 맞는 target, metric, split 방식과 제출 형식.
- 최종 후보에 전체 데이터 학습이 필요하면 별도 시간 예산을 허용할지 여부.

## 2026-09-21 — 재사용 템플릿 초기화

### 변경 내용과 이유

- 이전 예시 데이터의 분석 산출물과 실험 참조를 정리했다.
- `configs/baseline.json`에서 target 자동 추론과 단일 예측 제출을 기본값으로 설정했다.
- 현재 상태, 진행 기록, 실험 요약과 registry를 새 프로젝트 시작 상태로 갱신했다.
- 공통 파이프라인 구현과 테스트는 유지했다.
- 복사본에 불필요한 로컬 의존성 폴더 `vendor/`와 `.runtime_pkgs/`를 사용자 승인 후 제거했다. 의존성은 `requirements.txt`로 다시 설치한다.

### 영향을 받는 실험·데이터·문서

- 이전 예시 실험의 재현용 artifact는 템플릿에 남아 있지 않다.
- `CURRENT_STATE.md`, `progress.md`, `README.md`, `configs/baseline.json`, `experiments/registry.csv`, `reports/`.

### 이전 결과와의 비교 가능성

- 새 데이터의 실험은 기존 예시 데이터 결과와 비교하지 않는다. 새 프로젝트에서 별도의 Primary Validation을 확정한다.

### 사용자 확인이 필요한 판단

- 새 데이터가 들어오면 target, ID, metric, validation과 submission 형식을 확인해야 한다.
