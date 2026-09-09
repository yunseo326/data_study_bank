# 진행사항

## Bank 01 — 데이터 이해와 1차 EDA (2026-09-09)

- 목적: 모델링 전에 데이터 구조, 품질, 타깃 분포, train/test 차이와 누수 위험을 확인한다.
- 입력: `bank/data/train.csv`, `bank/data/test.csv`, `bank/data/sample_submission.csv`
- 확인: train 750,000행, test 250,000행, 예측 변수 16개, 타깃 양성률 12.07%
- 품질: 실제 NaN 없음. 범주형 `unknown`은 별도 결측 표현으로 존재한다.
- 품질: ID/타깃을 제외한 완전 중복과 train/test 동일 피처 행은 발견되지 않았다.
- 분포: 수치형 KS와 범주형 총변동거리 기준으로 큰 train/test 이동은 발견되지 않았다.
- 패턴: `duration`의 단변량 연관성이 가장 강하고, `poutcome`, `balance`, `month`가 뒤를 이었다.
- 위험: `duration`이 통화 종료 후 생성되는 값이면 사전 타기팅에서는 누수다.
- 해석: `pdays=-1`은 이전 연락 없음 상태로 보이지만 `previous=0`, `poutcome=unknown`과 완전히 일치하지 않는다.
- 산출물: `docs/index.html`, `logs/result/bank_eda_summary.json`, `bank/src/eda_bank.py`
- 다음 우선순위: Stratified 5-Fold 기준선을 만든 뒤 `duration` 포함/제외만 바꾸어 비교한다.

## Bank 02 — 대회 원문 대조 (2026-09-09)

- 확인: 대회는 Kaggle Playground Series Season 5 Episode 8이다.
- 문제 정의: 고객의 은행 정기예금 가입 확률을 예측한다.
- 평가 지표: ROC AUC로 확정했다.
- 데이터 생성: 원본 Bank Marketing Dataset을 학습한 딥러닝 모델이 현재 train/test를 생성했다.
- 새 실험 후보: 대회에서 원본 Bank Marketing Dataset 사용을 허용하므로, 기준선 이후 원본 데이터 결합 효과를 한 변수만 바꾸어 검증할 수 있다.
- 산출물: `bank/README.md`와 HTML 보고서에 원문에서 확인한 정보를 반영했다.
