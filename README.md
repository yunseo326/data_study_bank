# Data Study Bank

Kaggle 대회를 순서대로 학습하며 EDA, 검증, 피처 엔지니어링, 모델링 과정을 기록하는 저장소입니다.

## Repository structure

- `<project>/data/`: Kaggle 원본 데이터(로컬 전용, Git 제외)
- `<project>/src/`: 재사용 가능한 분석·모델링 코드
- `<project>/docs/index.html`: 프로젝트별 보고서 원본
- `docs/`: 자동 수집된 GitHub Pages 게시본

공개 리포트: <https://yunseo326.github.io/data_study_bank/>

GitHub Pages는 `main` 브랜치의 `/docs` 폴더에서 배포합니다. 각 프로젝트의
`docs/index.html`을 수정해 `main`에 push하면 GitHub Actions가 게시본을 자동으로
수집하고 커밋합니다.

로컬에서 게시 결과를 미리 만들려면 저장소 상위 폴더에서 다음을 실행합니다.

```powershell
python bank/src/build_portfolio.py
```

새 대회를 추가할 때는 `<project>/docs/index.html`을 만든 뒤
`bank/src/build_portfolio.py`의 `REPORTS`와 홈페이지 카드에 한 번만 등록합니다.
