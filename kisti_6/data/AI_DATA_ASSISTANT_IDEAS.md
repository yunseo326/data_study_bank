# AI Data Assistant Ideas

이 문서는 AI가 데이터 분석을 보조할 수 있는 작업의 예시 목록이다. 실제 적용 전에는 데이터 접근 권한, 개인정보, leakage와 검증 영향을 확인하고, 채택할 아이디어는 `IDEAS.md`에 등록한다.

## Data Understanding

- column 의미 추론
- feature group 분류
- data dictionary 초안
- 단위, 범위와 category 의미 확인

## Data Quality

- 이상치 후보 탐색
- 결측 pattern 분석
- category 표기 오류 후보
- 중복·키 충돌·타입 불일치 확인

## Feature

- interaction 후보
- aggregation 후보
- domain-based feature 후보
- 시간·그룹 단위 변환 후보

## Annotation

- text labeling
- categorization
- metadata 생성
- reasoning 초안 생성

AI 생성 annotation은 표본 검수 기준과 품질 지표를 정한 뒤 사용한다.

## Error Analysis

- OOF failure pattern
- subgroup 탐색
- model disagreement
- confidence와 residual 분석

## Research

- 관련 논문 탐색
- Kaggle/DACON write-up 조사
- 유사 dataset과 문제 조사
- 적용 조건과 재현 가능성 정리

## Review

- leakage 가능성
- validation 오류
- 과적합 가능성
- Parent 대비 실제 변경 확인

## 아이디어 승격 체크리스트

`IDEAS.md`에 옮길 때 다음을 함께 기록한다.

- 무엇을 제안하는가
- 왜 현재 문제에 유효할 수 있는가
- 아이디어의 출처는 무엇인가
- 예상되는 leakage·비용·리스크는 무엇인가
- notebook 탐색이 먼저 필요한가
- 기존 experiment와 중복되는가

