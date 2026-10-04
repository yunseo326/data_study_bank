# Progress

## 2026-09-23

### Summary

- experiment run 안에서 분석·판정용 `results/`와 재현·검증용 `reproducibility/`를 분리했다.
- 실행 환경과 artifact SHA-256을 담는 `run_manifest.json`을 추가하고 새 layout을 end-to-end test로 검증했다.
- 외부 제출용 `submission.csv`와 관련 설정·최종 fit·test 추론을 formal experiment 단계에서 제거했다.
- 2026 DATA·AI 분석 경진대회 녹조 문제의 공식 페이지, Q&A, 공개 API와 샘플 엑셀을 검토해 AI용 참가·분석 브리프를 작성했다.
- 상시 탐색을 피하기 위해 대회 원본과 브리프를 `references/competitions/2026_aida_algal_bloom/`로 이동했다.

### Experiments

- 등록된 실험 없음. 이후 생성되는 experiment부터 artifact layout version 2를 사용한다.

### Handoff

- 결과 분석은 `results/`, 재실행과 artifact 무결성 확인은 `reproducibility/`를 기준으로 한다.
- 제출이 필요하면 Reviewer 검증을 마친 후보를 대상으로 별도 output 단계를 설계한다.
- 일반 작업에서는 `references/`를 읽지 않는다. 대회 참가·규칙·전략 판단 때만 해당 브리프를 연다.
- 회귀+분류 설명과 회귀 단일 제출형식의 충돌을 공식 Q&A로 해소한 뒤 baseline을 시작한다.

## 2026-09-22

### Summary

- 첨부된 EDA 재사용안을 검토해 표형 데이터에 공통 적용 가능한 target 분포, 숫자형 histogram, Pearson 상관 heatmap을 선택 옵션으로 추가했다.
- 시계열 개요는 timestamp와 신호 열에 대한 도메인 판단이 필요하므로 공통 기본 기능에는 포함하지 않았다.
- 회귀 target은 고유값별 빈도 대신 기술통계로 요약하도록 수정하고, 신규 EDA 테스트를 포함한 전체 7개 테스트를 통과했다.

### Experiments

- 등록된 실험 없음. EDA 표현과 코드만 변경했다.

### Handoff

- 실제 데이터가 들어오면 기본 EDA와 `--render-visuals` 결과를 확인하고, 시간·그룹 구조가 있으면 해당 구조 전용 EDA 범위를 결정한다.

## 2026-09-21

### Summary

- 이전 예시 데이터의 산출물과 실험 참조를 제거하고 새 데이터용 템플릿으로 초기화했다.
- 공통 분석 파이프라인과 검증 규칙은 유지했다.
- HTML 결과물을 요청받을 때 사용할 GitHub Pages 게시 위치를 `https://yunseo326.github.io/data_study_bank/`로 지정했다.

### Experiments

- 등록된 실험 없음.

### Handoff

- 새 데이터를 배치하고 `configs/baseline.json`을 확인한 뒤 첫 실험을 시작한다.

## 작성 원칙

중요한 실험, 방향 변경, 결정, 문제와 milestone을 시간순으로 기록한다. 상세 조건과 수치는 각 experiment artifact에 둔다.

## 2026-09-24

### Summary

- 녹조 경진대회용 참가신청서 및 공모계획서 초안을 `outputs/competition_application/`에 작성했다.
- 제공 샘플의 변동성·불규칙 목표일 간격을 명시하고, 실제 선행연구의 시간 일반화·장기 예측·관측 이식성 한계를 누수 방지형 검증과 수문 메모리 피처 설계로 보완하는 계획을 담았다.
- 개인·팀 신원 정보는 임의로 만들지 않고 `[작성 필요]` 입력란으로 남겼다.

### Experiments

- 등록된 실험 없음. 문서 내 성능 수치는 주장하지 않았으며, 모델 학습·평가는 시작하지 않았다.

### Handoff

- 대표자·소속·팀원·연락처를 채운 뒤 공고의 HWP 양식에 맞춰 옮겨 적는다.
- 회귀·분류 제출 규칙은 공식 Q&A로 확인한 뒤 formal baseline을 시작한다.
