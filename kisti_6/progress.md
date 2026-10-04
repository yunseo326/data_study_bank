# Progress

## 2026-09-24

### Summary

- 하천수위 예측 문제의 참가신청서와 문제해결 제안서 초안을 DOCX로 작성했다.
- 초기 샘플은 target tail과 형식 확인에만 활용하고 최종 성능 주장은 제외했다.
- 호우사상 Group CV, 미래행 누수 차단, 극단 상승 대응과 재현성 중심의 방법론을 제안했다.
- 유역 반응 지문, 다중과제 학습, 난이도 균형 앙상블과 사상 단위 불확실성을 차별화 아이디어로 포함했다.
- Microsoft Word 렌더링으로 6개 페이지를 모두 시각 검증하고 표 분할과 제목 스타일을 수정했다.

### Experiments

- 등록·실행된 formal experiment 없음.

### Handoff

- 팀명, 참가자 개인정보, 역할과 서명은 사용자가 입력한다.
- 최종 데이터 확보와 공식 Q&A 갱신 후 일정·수치·방법론을 다시 확인하고 공식 HWP 원본으로 옮긴다.

## 2026-09-23

### Competition Analysis

- 공식 대회 페이지, 문제 설명, 일정·시상·발표평가 기준, 문제별 Q&A와 참가신청서를 교차 확인했다.
- 문제는 학습 신호가 확인되어 해결 가능하지만 122개 학습 호우사상에 대한 group validation, 극단 사상 RMSE, Test 미래행 누수 차단이 핵심 난점이라고 판단했다.
- 로컬 296행 CSV가 최종 배포 Train이 아님을 확인하고, 대회 자료와 AI 계획용 분석을 일반 실험 경로 밖의 조건부 참고 폴더로 이동했다.
- 최종 데이터가 없으므로 formal experiment는 시작하지 않았다.

### Competition Handoff

- 최종 Train/Test/Sample Submission/Data Dictionary 확보 후 공식 Q&A의 미래행 누수 답변을 재확인한다.
- Primary Validation은 공식 예비모델과 정렬한 storm-event group 4-fold를 우선 검토하며 확정 전 random CV를 실행하지 않는다.

### Summary

- experiment run 안에서 분석·판정용 `results/`와 재현·검증용 `reproducibility/`를 분리했다.
- 실행 환경과 artifact SHA-256을 담는 `run_manifest.json`을 추가하고 새 layout을 end-to-end test로 검증했다.
- 외부 제출용 `submission.csv`와 관련 설정·최종 fit·test 추론을 formal experiment 단계에서 제거했다.

### Experiments

- 등록된 실험 없음. 이후 생성되는 experiment부터 artifact layout version 2를 사용한다.

### Handoff

- 결과 분석은 `results/`, 재실행과 artifact 무결성 확인은 `reproducibility/`를 기준으로 한다.
- 제출이 필요하면 Reviewer 검증을 마친 후보를 대상으로 별도 output 단계를 설계한다.

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
