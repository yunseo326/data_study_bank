# Bank competition workspace

이 폴더는 Bank 관련 Kaggle 학습 작업 공간입니다.

## 대회 원문

- 대회: [Binary Classification with a Bank Dataset](https://www.kaggle.com/competitions/playground-series-s5e8)
- 시리즈: Playground Series - Season 5, Episode 8
- 목표: 고객이 은행 정기예금에 가입할 확률 `y`를 예측
- 평가 지표: ROC AUC
- 데이터 생성: 원본 Bank Marketing Dataset을 학습한 딥러닝 모델로 train/test를 생성
- 외부 데이터: Kaggle 설명에 따르면 원본 Bank Marketing Dataset을 탐색하거나 학습에 결합할 수 있음
- 변수 정의와 해석: [VARIABLES.md](VARIABLES.md)
- 모델링 프레임워크: [MODELING.md](MODELING.md)

이 정보는 로컬 CSV만으로 확정할 수 없으므로 대회 원문을 분석 기준으로 사용합니다.

- 원본 CSV는 `data/`에 저장하며 Git에는 포함하지 않습니다.
- 노트북은 `notebooks/`, 재사용 코드는 `src/`에 정리합니다.
- 공유할 분석 결과는 저장소 루트의 `docs/` HTML 리포트에 반영합니다.

## 기준 모델 실행

프로젝트 전용 의존성을 설치한 뒤 아래 순서로 실행합니다.

```powershell
python -m pip install --target bank/.runtime_pkgs -r bank/requirements.txt
python bank/src/model_bank.py --experiment E001
python bank/src/model_bank.py --experiment E002
python bank/src/model_bank.py --experiment E008
python bank/src/analyze_models.py
python bank/src/eda_bank.py
python bank/src/build_portfolio.py
```

OOF·test 예측과 제출 파일은 `bank/outputs/`에만 저장되며 Git에는 포함되지 않습니다.
`eda_bank.py`는 Bank 전용 하위 보고서 `docs/bank/index.html`만 갱신하고, `build_portfolio.py`가 세 대회 보고서와 상위 Data Study 홈을 저장소 루트 `docs/`에 게시합니다.

## 모델링 프레임워크

- 고정 검증: seed 326의 Stratified 5-Fold OOF ROC AUC
- 실험 원칙: 한 번에 한 요소만 변경하고 `benchmarks/our_experiments.csv`에 결과 기록
- 파생 산출물: `outputs/<experiment_id>/`에 OOF, test 예측, 제출 파일, 세그먼트 진단 저장
- E001: `duration` 포함 CatBoost Kaggle 점수 기준선
- E002: E001에서 `duration`만 제외한 통화 전 현실 기준선
- E003: E001에서 `previous_contacted`만 추가하는 첫 피처 가설
- E004~E008: LightGBM에서 범주형 처리, 학습 상한, 리프 수, 일반화 제약, 행 샘플링을 한 요소씩 비교
- 현재 최고: E008 OOF ROC AUC 0.969283

실행 예시: `python bank/src/model_bank.py --experiment E001`
