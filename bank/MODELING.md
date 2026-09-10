# Bank 모델링 프레임워크

## 판단 순서

문제 이해 → 데이터 품질 → 누수와 예측 시점 → 고정 검증 → 기준 모델 → 한 변수 실험 → 오류 분석 → 모델 비교 → 외부 데이터 → 앙상블 순서로 진행합니다.

주지표는 seed 326의 Stratified 5-Fold OOF ROC AUC입니다. Public LB는 OOF 개선이 Kaggle test에도 이어지는지 확인하는 보조 지표로만 사용합니다.

## 두 평가 트랙

- Kaggle 점수 트랙: 대회가 제공한 16개 피처를 모두 사용합니다.
- 통화 전 현실 트랙: 통화 종료 후에 확정되는 `duration`을 제외합니다.

두 트랙의 점수는 목적이 다르므로 우열로 해석하지 않습니다. `duration` 제외 시 하락 폭은 사전에는 사용할 수 없는 정보가 점수에 얼마나 기여하는지를 보여줍니다.

## 실험 규칙

1. 한 실험에서는 한 요소만 변경합니다.
2. 같은 fold와 seed를 유지합니다.
3. 인코더와 모델은 각 학습 fold에서만 학습합니다.
4. 전체 OOF뿐 아니라 fold 편차와 고객군별 AUC를 확인합니다.
5. 실패한 실험도 원인과 다음 가설을 남깁니다.
6. UCI 원본 데이터는 Kaggle 검증 fold가 아니라 학습 fold에만 추가합니다.

## 실행 순서

```powershell
python bank/src/model_bank.py --experiment E001
python bank/src/model_bank.py --experiment E002
python bank/src/model_bank.py --experiment E003
python bank/src/model_bank.py --experiment E004
python bank/src/model_bank.py --experiment E005
python bank/src/model_bank.py --experiment E006
python bank/src/model_bank.py --experiment E007
python bank/src/model_bank.py --experiment E008
python bank/src/analyze_models.py
```

- E001: 16개 피처를 사용하는 CatBoost Kaggle 기준선
- E002: E001에서 `duration`만 제외한 현실 트랙 기준선
- E003: E001에 `previous_contacted`만 추가한 첫 피처 가설
- E004: LightGBM 범주형 피처 명시
- E005: E004에서 최대 트리 수만 3,000으로 확대
- E006: E005에서 `num_leaves`만 63으로 확대
- E007: E006에서 `min_child_samples`만 50으로 강화
- E008: E006에서 `subsample_freq=1`만 적용해 행 샘플링 활성화

`--model lightgbm` 또는 `--model xgboost`를 지정하면 같은 실험 정의와 fold로 모델 종류만 비교할 수 있습니다. 모델 비교 실험에는 충돌하지 않는 새 experiment ID를 추가한 뒤 실행합니다.

## 산출물

각 실험은 `bank/outputs/<experiment_id>/`에 OOF·test 예측, 제출 파일, fold·세그먼트·피처 중요도 진단을 저장합니다. 이 폴더는 Git에 올리지 않습니다. `analyze_models.py`는 완료된 OOF의 상관과 고정 혼합을 비교하며, 요약 지표만 `benchmarks/our_experiments.csv`, `logs/result/`와 GitHub Pages에 반영합니다.
