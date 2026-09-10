# Experiment registry

`our_experiments.csv`는 동일한 seed와 fold를 사용한 실험만 비교합니다. 한 행은 하나의
가설을 나타내며, `changed_element`에는 직전 기준에서 바꾼 한 요소만 기록합니다.

OOF 확률, test 확률, fold 할당, 제출 파일과 상세 지표는 Git에서 제외되는
`outputs/<experiment_id>/`에 저장합니다.
