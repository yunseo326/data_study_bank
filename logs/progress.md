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

## Bank 03 — 공개 모델 벤치마크 구성 (2026-09-09)

- 목적: 우리의 OOF ROC AUC가 어느 수준인지 공개 모델과 공정하게 비교한다.
- 비교 기준: 단일 모델과 대규모 앙상블을 분리하고, 모델 복잡도와 출처를 함께 기록한다.
- 공개 기준: MLP 0.974, 강한 단일 XGBoost 0.975782, 공개 상위 앙상블 약 0.9772~0.9774 CV.
- 현재 평가: EDA는 완료했지만 기준 모델이 없어 성능 점수는 아직 평가하지 않는다.
- 산출물: `bank/benchmarks/public_models.csv`, `bank/benchmarks/our_experiments.csv`, `bank/benchmarks/README.md`와 HTML 비교 대시보드.
- 다음 통과 기준: 고정된 Stratified 5-Fold에서 첫 OOF ROC AUC를 생성한다.

## Bank 04 — 공식 변수 정의와 해석 보완 (2026-09-10)

- 목적: Kaggle 설명, UCI 변수 사전, 원 연구 논문을 대조해 변수 의미와 모델링 해석을 바로잡는다.
- 수정: `unknown`을 일반 결측값이 아닌 ‘정보 없음’이라는 명시적 범주로 정리했다.
- 수정: `pdays=-1`은 공식적으로 이전 연락 없음이며, `day`는 요일이 아닌 월중 일자임을 명시했다.
- 수정: `balance`는 유로 단위 연평균 잔액이고, `campaign`은 마지막 연락을 포함한 현 캠페인 연락 횟수임을 반영했다.
- 해석: `duration`은 마지막 통화시간이므로 Kaggle 점수 모델에는 포함하되, 통화 전 사전 타기팅 모델에서는 제외하는 두 평가 트랙을 제안했다.
- 해석: 월별 차이는 계절성과 캠페인 운영이 섞인 연관성이며, 연도 정보가 없어 `month`만으로 시간순 검증을 만들지 않는다.
- 원본 데이터 실험: UCI 원본은 학습 폴드에만 추가하고 Kaggle train 검증 폴드는 유지해 합성 테스트 분포에 대한 개선 여부를 평가한다.
- 산출물: `bank/VARIABLES.md`, 갱신된 GitHub Pages HTML과 재현 스크립트.
- 다음 우선순위: 동일한 Stratified 5-Fold에서 `duration` 포함 기준 모델을 먼저 만들고, 제외 실험은 그다음 한 변수만 바꿔 비교한다.
