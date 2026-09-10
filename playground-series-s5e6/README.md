# Predicting Optimal Fertilizers workspace

Kaggle Playground Series Season 5, Episode 6을 학습하는 작업 공간입니다.

- 목표: 날씨, 토양, 작물, 영양소 조건에서 적합한 비료 3개를 순서대로 예측
- 평가: MAP@3
- 데이터: `data/train.csv`, `data/test.csv`, `data/sample_submission.csv` (Git 제외)
- 분석 리포트: `docs/index.html`
- 모델링 원칙: 고정된 Stratified 5-Fold에서 한 번에 한 요소만 변경

## 실행 순서

```powershell
python -m pip install --target .runtime_pkgs -r requirements.txt
python src/eda_fertilizer.py
python src/model_fertilizer.py --experiment E000
python src/model_fertilizer.py --experiment E001
python src/eda_fertilizer.py
```

빠른 구조 검증에는 `--smoke`를 사용합니다. 파생 예측과 제출 파일은
`outputs/<experiment_id>/`에 저장되며 Git에 포함하지 않습니다.

## 문서

- [분석 및 실험 설계](MODELING.md)
- [실험 기록](benchmarks/our_experiments.csv)
- [진행 기록](logs/progress.md)
