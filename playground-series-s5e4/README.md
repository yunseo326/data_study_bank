# Predict Podcast Listening Time

Kaggle Playground Series Season 5 Episode 4 학습 프로젝트입니다.

목표는 팟캐스트 메타데이터로 `Listening_Time_minutes`를 예측하는 회귀 문제를 통해 데이터 품질 진단, 고정 교차검증, 단일 변수 실험과 오류 분석을 익히는 것입니다. 평가지표는 RMSE이며 낮을수록 좋습니다.

## 데이터

- `data/train.csv`: 750,000행, 타깃 포함
- `data/test.csv`: 250,000행, 예측 대상
- `data/sample_submission.csv`: 제출 형식
- 원본 CSV는 읽기 전용으로 사용하고 Git에 포함하지 않습니다.

## 실행

프로젝트 전용 패키지를 설치한 뒤 EDA와 기준 모델을 실행합니다.

```powershell
python -m pip install --target .runtime_pkgs -r requirements.txt
python src/eda_podcast.py
python src/model_podcast.py --experiment B000
python src/model_podcast.py --experiment B001
python src/model_podcast.py --experiment B002
```

현재 PC에서는 저장소의 공용 로컬 패키지 폴더가 있으면 자동으로 재사용합니다. 모든 모델 실험은 seed 326의 동일한 shuffled 5-Fold를 사용합니다.

방향 탐색은 정식 검증과 분리된 5분 목표의 빠른 프로토콜을 먼저 사용합니다.

```powershell
python src/quick_experiments.py --experiment B000
python src/quick_experiments.py --experiment B001
python src/quick_experiments.py --experiment B002
```

`quick-v1`은 타깃 10분위와 주요 결측 조합을 보존한 고정 7.5만 행에서 동일한 5-fold를 실행합니다. CatBoost는 최대 200회로 제한하고 방향 탐색에 불필요한 test 예측은 생략합니다. 결과는 `benchmarks/quick_experiments.csv`와 `outputs/quick/`에만 기록되며 정식 OOF 결과를 대체하지 않습니다.

## 현재 결과

- B000 평균 기준: OOF RMSE 27.138370
- B001 길이 선형 기준: OOF RMSE 13.543573
- B002 CatBoost 기준: OOF RMSE 13.049999
- P001 0~120분 제한: OOF RMSE 13.049970, 현재 최선
- E101 길이 이상값 결측 처리: OOF RMSE 13.052363, 채택하지 않음

에피소드 길이가 있는 행의 B002 RMSE는 10.416569이지만, 길이가 결측인 행은 25.311535입니다. 다음 핵심 개선 대상은 이상값보다 길이 결측 구간입니다.

## 주요 산출물

- `docs/index.html`: 모바일 대응 분석 보고서
- `logs/result/podcast_eda_summary.json`: 기계 판독 가능한 EDA 요약
- `benchmarks/experiments.csv`: 실험 목적, 변경점, OOF RMSE와 해석
- `outputs/<experiment_id>/`: OOF·test 예측, 제출 파일, 세그먼트 진단

분석 원칙과 실험 순서는 [MODELING.md](MODELING.md)에 정리되어 있습니다.
