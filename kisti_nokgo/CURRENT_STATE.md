# Current State

Last Updated: 2026-09-24

## Objective

- 2026 DATA·AI 분석 경진대회 녹조 7일 예측 문제의 참가 가능성과 분석 방향 검토.
- 전체 참가자 데이터 입수 후 Target, Primary Metric, Primary Validation을 확정한다.

## Current Best

- 없음. 등록된 experiment가 없다.

## Active Work

- 대회 분석 브리프와 참가신청서·공모계획서 초안 작성 완료. formal experiment는 시작하지 않았다.

## Current Candidates

- 없음.

## Known Issues

- 기본 시간 예산 검사는 단계 사이에서 수행되므로 단일 model fit을 강제로 중단하지 못한다.
- OOF Parquet 저장에는 `pyarrow`가 필요하다.
- 기본 CV는 이 문제에 부적합하다. 공식 설명의 학습(~2023)·검증(2024) 시간 분할을 우선 검토한다.
- 공식 문제 페이지의 회귀+분류 설명과 샘플 제출형식의 회귀 단일 과제 설명이 충돌한다.
- 전체 참가자 데이터와 최종 채점 산식이 아직 확정되지 않았다.

## Next Actions

1. 공식 Q&A로 제출 열, 회귀·분류 가중치, metric 집계 방식을 확인한다.
2. 전체 참가자 데이터를 검증해 `data/raw/`에 배치하고 데이터 manifest를 만든다.
3. `references/competitions/2026_aida_algal_bloom/AI_COMPETITION_BRIEF.md`의 누수 방지 조건을 반영해 Primary Validation과 baseline을 설계한다.

## Blockers

- 전체 참가자 데이터가 아직 `data/raw/`에 없다.
- 최종 채점 규칙의 공개 설명이 서로 일치하지 않는다.

## 작성 원칙

현재 상태만 기록한다. 전체 실험 이력, 상세 config와 metric은 각각 `progress.md`와 experiment artifact에 기록한다.
