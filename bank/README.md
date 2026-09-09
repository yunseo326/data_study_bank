# Bank competition workspace

이 폴더는 Bank 관련 Kaggle 학습 작업 공간입니다.

## 대회 원문

- 대회: [Binary Classification with a Bank Dataset](https://www.kaggle.com/competitions/playground-series-s5e8)
- 시리즈: Playground Series - Season 5, Episode 8
- 목표: 고객이 은행 정기예금에 가입할 확률 `y`를 예측
- 평가 지표: ROC AUC
- 데이터 생성: 원본 Bank Marketing Dataset을 학습한 딥러닝 모델로 train/test를 생성
- 외부 데이터: Kaggle 설명에 따르면 원본 Bank Marketing Dataset을 탐색하거나 학습에 결합할 수 있음

이 정보는 로컬 CSV만으로 확정할 수 없으므로 대회 원문을 분석 기준으로 사용합니다.

- 원본 CSV는 `data/`에 저장하며 Git에는 포함하지 않습니다.
- 노트북은 `notebooks/`, 재사용 코드는 `src/`에 정리합니다.
- 공유할 분석 결과는 저장소 루트의 `docs/` HTML 리포트에 반영합니다.
