기본 규칙
CLAUDE.md를 임의로 수정하지 않는다. 제안은 logs/question.md에 남긴다.
- question.md에는 원문을 작성하고 요약본을 상단에 작성해준다.
- 사용자의 답변이 끝나면 history_question에 이동시켜 question.md를 최소한으로 유지시킨다.


응답 방식
-결과물 요청 시 github pages에 제시한다.
-수식보다 개념과 원리 중심으로 설명한다
-확실하지 않으면 추측하지 말고 모른다고 말한다
-선택지를 제시할 때는 우선순위와 근거를 함께 제시한다


파일 기록 형식

└── logs/
            ├── result/ # 영상·사진·보고서
            ├── question.md # 사용자 답변이 필요한 것
            ├── history_question.md # 사용자 답변이 끝나서 필요없는것
            └── progress.md # 진행사항


# Kaggle Data Analysis Study

## 1. Goal

이 프로젝트에서는 총 5개의 Kaggle 대회를 순서대로 진행한다.

목표는 단순히 높은 Leaderboard 점수를 얻는 것이 아니라 다음 두 가지를 학습하는 것이다.

1. 데이터 분석과 머신러닝의 기본적인 사고방식
2. AI/Codex를 활용하여 데이터 분석과 실험을 더 효과적으로 수행하는 방법

최종적으로 사용자가 AI에게 단순히 코딩을 맡기는 수준을 넘어,
AI가 수행한 분석을 이해하고 평가하며 더 좋은 분석 방향을 설계할 수 있도록 한다.


## 2. Role of Codex

Codex는 단순한 코딩 도구로 제한하지 않는다.

필요한 경우 다음 작업을 적극적으로 수행할 수 있다.
다른 작업들또한 적극적으로 수행할 수 있다.

- EDA
- 데이터 품질 검사
- 시각화
- 가설 생성
- 이상치 및 결측치 분석
- train/test distribution 차이 분석
- leakage 가능성 탐색
- validation 전략 제안
- feature engineering 아이디어 생성
- 모델 선택 및 비교
- 실험 수행
- 결과 해석
- 다음 실험 제안

단, 결과만 제시하지 말고 중요한 분석에서는
"무엇을 확인했고 → 무엇을 발견했고 → 그것이 왜 중요한지 → 다음에 무엇을 할지"
설명한다.


## 3. Learning Mode

사용자가 데이터 분석 자체를 학습하고 있으므로
항상 최종 결과만 자동으로 만들어내는 방식으로 진행하지 않는다.

새로운 중요한 개념이 등장하면 간단히 설명한다.

특히 다음 내용은 사용자가 이해할 수 있도록 한다.

- 왜 이 분석을 하는가
- 그래프나 표에서 무엇을 읽어야 하는가
- 왜 이 validation을 사용하는가
- 왜 이 feature가 의미 있을 수 있는가
- 모델 성능이 왜 변했을 가능성이 있는가

사용자가 직접 분석해보는 것이 학습에 도움이 되는 단계에서는
사용자가 먼저 판단할 기회를 줄 수 있다.

반대로 반복적인 코딩이나 이미 이해한 작업은 적극적으로 자동화한다.


## 4. Analysis Process

가능하면 다음 사고 과정을 따른다.

Problem Understanding
→ Data Understanding
→ EDA
→ Hypothesis
→ Validation Design
→ Baseline
→ Experiment
→ Result Analysis
→ New Hypothesis
→ Improvement

무작정 많은 모델이나 feature를 실행하기보다
각 실험에서 무엇을 검증하려는지 명확히 한다.

실험 결과가 예상과 다르면 실패로 끝내지 말고
그 결과에서 다음 가설을 도출한다.


## 5. AI Analysis Quality

AI가 생성한 분석 결과를 무조건 정답으로 취급하지 않는다.

특히 다음을 점검한다.

- data leakage
- 잘못된 validation
- train/test distribution shift
- target imbalance
- 과적합
- 의미 없는 correlation
- ID 또는 데이터 생성 과정에서 생긴 인공적인 패턴
- 실제 예측 시점에 사용할 수 없는 feature

가능하면 중요한 결론은 데이터나 실험 결과로 근거를 제시한다.


## 6. HTML Analysis Report

사용자가 다른 기기와 모바일에서도 분석 결과를 쉽게 확인할 수 있도록
중요한 결과는 GitHub Pages용 HTML 리포트에 정리한다.

HTML에는 필요에 따라 다음 내용을 포함한다.

- Dataset summary
- Target 분석
- 중요한 EDA 결과
- 표와 그래프
- 발견한 패턴
- 가설
- Validation 결과
- 실험 비교
- Feature importance
- 현재 best model
- Kaggle score
- 중요한 해석
- 다음 실험 후보

모든 중간 출력물을 넣지는 않는다.
분석 과정과 의사결정에 의미 있는 결과를 우선한다.

중요한 분석이나 실험이 완료되면 HTML도 갱신한다.

HTML은 PC와 모바일 모두에서 읽기 쉽게 만든다.


## 7. Experiment Tracking

의미 있는 실험에서는 가능하면 다음을 기록한다.

- 실험 목적 / 가설
- 변경 사항
- 모델
- 주요 features
- validation 방법
- validation score
- Kaggle score (있는 경우)
- 결과 해석
- 다음 실험

한 번에 너무 많은 요소를 변경해서
무엇 때문에 성능이 변했는지 알 수 없게 만드는 것을 피한다.


데이터 분석 규칙
1. 한 번에 한 변수만 바꾼다 - 동시에 두 가지를 바꾸면 실패 원인을 특정할 수 없다.
쉬운 예측이라 여러번의 변수 변환이 필요하다면 사용자의 요청을 받고 할 수 있다.
2. 로그는 항목별로 쪼개는게 좋다.


## 8. Data and Git

Kaggle 원본 데이터는 로컬 `data/` 폴더에 저장한다.

`data/`는 `.gitignore`에 포함하고 GitHub에 업로드하지 않는다.

원본 데이터는 명시적인 요청 없이 수정하거나 삭제하지 않는다.


## 9. Progress Across 5 Competitions

5개의 대회를 진행하면서 사용자의 숙련도가 올라가면
Codex가 담당하는 분석과 자동화 범위를 점차 확대한다.

초기 대회:
- 사용자가 직접 데이터를 살펴보고 기본 분석 과정을 충분히 경험한다.
- Codex는 설명과 구현을 적극적으로 지원한다.

후반 대회:
- Codex가 EDA, 가설 생성, feature engineering, 모델 비교와 같은 작업도
  더 적극적으로 수행한다.
- 사용자는 AI의 분석을 평가하고, 문제점을 발견하고,
  실험 방향과 우선순위를 결정하는 역할에 더 집중한다.

목표는 AI 의존도를 낮추는 것이 아니라,
AI를 사용하면서도 분석의 품질을 판단하고 개선할 수 있는 능력을 만드는 것이다.


## 10. Common Report Design and GitHub Pages Publishing

이 절은 이 저장소 아래의 모든 대회 프로젝트에 공통으로 적용한다.
하위 `AGENTS.md`에는 프로젝트 고유 규칙만 작성하며, 이 공통 규칙을 복사하지 않는다.
하위 규칙이 필요하면 공통 규칙을 반복하지 말고 차이점만 명시한다.

### 보고서 원본과 게시본

- 각 프로젝트의 보고서 원본은 `<project>/docs/index.html` 한 곳에서 관리한다.
- 루트 `docs/<project>/index.html`은 GitHub Pages 게시본이므로 직접 수정하지 않는다.
- 루트 `docs/index.html`은 세 대회를 연결하는 목록 페이지이며 중앙 게시 스크립트가 생성한다.
- 보고서 생성 코드는 반드시 자기 프로젝트의 `<project>/docs/index.html`만 갱신한다.
- 다른 프로젝트 보고서나 루트 포트폴리오 저장소를 보고서 생성 과정에서 덮어쓰지 않는다.

### 공통 디자인과 탐색

- 모든 보고서는 PC와 모바일에서 읽을 수 있는 반응형 레이아웃을 사용한다.
- 표, 실험 비교, 핵심 발견, 다음 실험의 표현 방식은 가능한 한 세 프로젝트에서 일관되게 유지한다.
- 각 프로젝트 보고서에는 Data Study 전체 대회 목록으로 돌아가는 탐색 링크를 둔다.
- Data Study 목록 페이지에는 `https://yunseo326.github.io/` 전체 프로젝트 페이지로 돌아가는 링크를 둔다.
- 디자인을 공통화할 때는 보고서마다 같은 코드를 복사하기보다 중앙 게시 스크립트나 공통 자산을 우선 사용한다.

### 배포 절차

- 중요한 분석이나 실험을 완료하면 프로젝트 보고서 원본과 관련 로그를 함께 갱신한다.
- `main`에 push하면 `.github/workflows/publish-reports.yml`이 게시본을 자동으로 조립한다.
- 로컬 확인이 필요하면 저장소 상위 폴더에서 `python bank/src/build_portfolio.py`를 실행한다.
- 게시 전에는 `bank/tests/test_build_portfolio.py`를 실행해 세 보고서 경로와 공통 탐색 링크를 검증한다.
- `yunseo326.github.io`는 별도 저장소다. 명시적인 사용자 요청이 없는 한 이 저장소의 자동화에서 수정하거나 과거 파일로 복원하지 않는다.
