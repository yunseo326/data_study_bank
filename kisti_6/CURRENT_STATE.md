# Current State

Last Updated: 2026-09-24

## Objective

- 2026 DATA·AI 분석 경진대회의 6시간 최대 하천수위 상승량 예측 문제를 누수 없이 해결할 준비.
- 최종 배포 데이터가 들어오면 Global RMSE와 storm-event group 4-fold CV를 확인한 뒤 Primary Validation을 고정한다.

## Current Best

- 없음. 등록된 experiment가 없다.

## Active Work

- 대회 분석과 조건부 계획 참고자료 작성 완료.
- 초기 문제 설명, 참가신청서와 296행 샘플은 일반 실험 경로 밖의 `references/competitions/2026_river_level/`로 보존했다.
- 참가신청서 문제해결 제안 초안을 `reports/2026_하천수위_참가신청서_초안.docx`로 작성했다. 개인정보·팀명·서명은 사용자 입력 대기 상태다.

## Current Candidates

- 없음.

## Known Issues

- 기본 시간 예산 검사는 단계 사이에서 수행되므로 단일 model fit을 강제로 중단하지 못한다.
- OOF Parquet 저장에는 `pyarrow`가 필요하다.
- 현재 로컬 CSV는 최종 Train이 아닌 296행 초기 샘플이다. formal experiment에 사용하지 않는다.
- Test의 후속 시점 행을 이용하면 일부 target을 재구성할 수 있다는 공개 Q&A가 있으나 공식 규정상 금지된다. Test 추론은 행별 시점 `t`까지의 정보만 사용한다.

## Next Actions

1. 최종 Train, Test, Sample Submission, Data Dictionary를 `data/raw/`에 확보한다.
2. 공식 Q&A의 미래행 누수 답변과 데이터 재배포 여부를 확인한다.
3. storm-event group을 확인·복원하고 4-fold Primary Validation을 고정한다.
4. 0 예측 RMSE 약 0.410과 공식 전체-feature baseline 약 0.244의 재현 가능성을 확인한다.
5. 참가자 정보와 역할을 확정하고 초안 문안을 공식 HWP 양식으로 옮긴다.

## Blockers

- 최종 배포 데이터가 아직 로컬에 없다.

## 작성 원칙

현재 상태만 기록한다. 전체 실험 이력, 상세 config와 metric은 각각 `progress.md`와 experiment artifact에 기록한다.
