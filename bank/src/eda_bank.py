"""Reproducible first-pass EDA for the Bank Kaggle dataset.

Reads the untouched CSV files in bank/data and writes derived analysis only.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "bank" / "data"
BENCHMARK_DIR = ROOT / "bank" / "benchmarks"
DOCS_PATH = ROOT / "bank" / "docs" / "index.html"
SUMMARY_PATH = ROOT / "bank" / "logs" / "result" / "bank_eda_summary.json"
MODEL_ANALYSIS_PATH = ROOT / "bank" / "logs" / "result" / "model_analysis.json"
OUTPUT_DIR = ROOT / "bank" / "outputs"
RNG = np.random.default_rng(326)

DATA_DICTIONARY = [
    {"변수": "age", "구분": "고객", "공식 의미": "고객 나이", "해석 주의": "비선형 효과와 연령대별 표본 수를 함께 확인"},
    {"변수": "job", "구분": "고객", "공식 의미": "직업 유형", "해석 주의": "unknown은 직업 정보가 없는 상태"},
    {"변수": "marital", "구분": "고객", "공식 의미": "결혼 상태", "해석 주의": "divorced는 이혼과 사별을 함께 포함"},
    {"변수": "education", "구분": "고객", "공식 의미": "교육 수준", "해석 주의": "선형 순서를 가정하지 않고 unknown을 유지"},
    {"변수": "default", "구분": "고객", "공식 의미": "신용 채무불이행 여부", "해석 주의": "희소 범주의 표본 수와 양성률을 함께 확인"},
    {"변수": "balance", "구분": "고객", "공식 의미": "연평균 잔액(유로)", "해석 주의": "음수와 큰 양수 값이 있어 일반 로그 변환에 부적합"},
    {"변수": "housing", "구분": "고객", "공식 의미": "주택담보대출 보유 여부", "해석 주의": "자산 수준과 동일한 의미가 아님"},
    {"변수": "loan", "구분": "고객", "공식 의미": "개인대출 보유 여부", "해석 주의": "housing과 다른 대출 정보"},
    {"변수": "contact", "구분": "현재 연락", "공식 의미": "연락 통신 방식", "해석 주의": "unknown은 연락 방식 정보가 없는 상태"},
    {"변수": "day", "구분": "현재 연락", "공식 의미": "마지막 연락의 월중 일자", "해석 주의": "요일이나 경과일이 아니며 month와 함께 해석"},
    {"변수": "month", "구분": "현재 연락", "공식 의미": "마지막 연락 월", "해석 주의": "계절성과 캠페인 대상 선택이 섞이며 연도 정보가 없음"},
    {"변수": "duration", "구분": "현재 연락", "공식 의미": "마지막 통화시간(초)", "해석 주의": "사전 고객 선별 시점에는 알 수 없는 결과 이후 정보"},
    {"변수": "campaign", "구분": "캠페인", "공식 의미": "현 캠페인 연락 횟수(마지막 연락 포함)", "해석 주의": "예측 시점에 따라 사용할 수 있는 값이 달라짐"},
    {"변수": "pdays", "구분": "이전 캠페인", "공식 의미": "이전 마지막 연락 후 경과일", "해석 주의": "-1은 이전에 연락한 적이 없다는 명시적 상태"},
    {"변수": "previous", "구분": "이전 캠페인", "공식 의미": "현 캠페인 이전 연락 횟수", "해석 주의": "합성 데이터에서는 pdays와 완전히 일치하지 않음"},
    {"변수": "poutcome", "구분": "이전 캠페인", "공식 의미": "이전 캠페인 결과", "해석 주의": "unknown을 pdays=-1로 기계적으로 치환하지 않음"},
    {"변수": "y", "구분": "타깃", "공식 의미": "정기예금 가입 여부", "해석 주의": "y=1 확률을 예측하고 ROC AUC로 평가"},
]

FEATURE_GUIDE = [
    {"형태": "수치", "변수": "age", "모델이 보는 정보": "18~95세의 고객 나이", "예상 영향과 읽는 법": "가입 성향이 나이에 비례하기보다 학생·은퇴 연령대 등에서 달라질 수 있어 비선형 효과를 예상", "다음 표현/가설": "원값 유지 → 필요하면 연령 구간을 한 변수 실험", "우선순위": "중"},
    {"형태": "수치", "변수": "balance", "모델이 보는 정보": "음수도 가능한 연평균 잔액(유로)", "예상 영향과 읽는 법": "잔액 수준이 가입 여력과 연결될 수 있지만 큰 이상치와 음수 때문에 단순 비례 관계는 예상하지 않음", "다음 표현/가설": "원값 유지 → 부호 보존 로그 또는 분위 구간", "우선순위": "상"},
    {"형태": "수치", "변수": "duration", "모델이 보는 정보": "마지막 통화시간(초)", "예상 영향과 읽는 법": "길수록 가입률이 크게 높아지는 가장 강한 변수지만 통화 종료 전에는 알 수 없음", "다음 표현/가설": "Kaggle 트랙만 사용, 현실 사전 타기팅 트랙에서는 제외", "우선순위": "필수 분리"},
    {"형태": "수치", "변수": "day, campaign", "모델이 보는 정보": "월중 연락일, 현재 캠페인 연락 횟수", "예상 영향과 읽는 법": "캠페인 운영과 고객 피로도가 섞일 수 있어 값이 커질수록 좋다고 단정하지 않음", "다음 표현/가설": "원값 기준선 → month와 조합 또는 연락 횟수 구간", "우선순위": "중"},
    {"형태": "상태+수치", "변수": "pdays, previous", "모델이 보는 정보": "이전 연락 여부·경과일·과거 연락 횟수", "예상 영향과 읽는 법": "처음 연락한 고객과 재접촉 고객의 구조가 다를 수 있으며 -1은 숫자가 아니라 '이전 연락 없음' 상태", "다음 표현/가설": "previous_contacted 상태 분리(검증 완료: 효과 거의 없음) 또는 재접촉 강도", "우선순위": "검증됨"},
    {"형태": "범주", "변수": "job, marital, education", "모델이 보는 정보": "고객의 직업·결혼·교육 집단", "예상 영향과 읽는 법": "생활 단계와 소득 안정성의 간접 신호일 수 있으나 인과효과나 사람의 가치로 해석하면 안 됨", "다음 표현/가설": "순서를 강제하지 않는 범주 처리, 희소 집단 안정성 확인", "우선순위": "중"},
    {"형태": "범주", "변수": "default, housing, loan", "모델이 보는 정보": "채무불이행·주택담보·개인대출 보유 상태", "예상 영향과 읽는 법": "자금 여력 또는 금융상품 보유 상황과 연결될 수 있지만 각 변수는 서로 다른 의미", "다음 표현/가설": "개별 범주 유지 → 대출 보유 조합은 후순위 상호작용", "우선순위": "중"},
    {"형태": "범주", "변수": "contact, month", "모델이 보는 정보": "연락 방식과 마지막 연락 월", "예상 영향과 읽는 법": "채널 효과·계절성뿐 아니라 은행이 누구에게 언제 연락했는지가 함께 반영됨", "다음 표현/가설": "native categorical 유지 → contact × month", "우선순위": "상"},
    {"형태": "범주", "변수": "poutcome", "모델이 보는 정보": "이전 캠페인 결과", "예상 영향과 읽는 법": "과거 성공 고객의 가입 가능성이 높아 강한 신호를 예상하며 unknown은 정보 없음으로 유지", "다음 표현/가설": "범주 유지 → 이전 접촉 상태와 조합", "우선순위": "상"},
    {"형태": "표현/변환", "변수": "balance, campaign, duration", "모델이 보는 정보": "원값을 구간·순위·부호 보존 로그로 다시 표현", "예상 영향과 읽는 법": "극단값의 영향을 줄이거나 임계점을 쉽게 찾게 할 수 있지만 트리 모델이 이미 비선형 분할을 학습하므로 개선은 검증이 필요", "다음 표현/가설": "한 번에 한 변환만 E008과 비교", "우선순위": "상"},
    {"형태": "상호작용", "변수": "duration × contact", "모델이 보는 정보": "연락 채널에 따라 같은 통화시간의 의미가 다른지", "예상 영향과 읽는 법": "채널별 통화 패턴이 다르면 고객 순위 구분을 보완할 수 있음", "다음 표현/가설": "공개 강한 모델 대조 후 첫 후보로 검토", "우선순위": "상"},
    {"형태": "상호작용", "변수": "poutcome × pdays/previous", "모델이 보는 정보": "과거 결과와 그 결과의 시점·접촉 횟수 조합", "예상 영향과 읽는 법": "같은 과거 성공이라도 최근성과 접촉 강도에 따라 재가입 가능성이 달라질 수 있음", "다음 표현/가설": "희소 조합의 표본 수를 확인한 뒤 실험", "우선순위": "상"},
]


def ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample KS statistic without a SciPy dependency."""
    a = np.sort(np.asarray(a))
    b = np.sort(np.asarray(b))
    points = np.unique(np.concatenate([a, b]))
    cdf_a = np.searchsorted(a, points, side="right") / len(a)
    cdf_b = np.searchsorted(b, points, side="right") / len(b)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def discrete_mutual_information(x: pd.Series, y: pd.Series) -> float:
    """Empirical mutual information for two discrete variables."""
    counts = pd.crosstab(x, y, dropna=False).to_numpy(dtype=float)
    total = counts.sum()
    joint = counts / total
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    expected = px @ py
    mask = joint > 0
    return float(np.sum(joint[mask] * np.log(joint[mask] / expected[mask])))


def fmt_int(value: int | float) -> str:
    return f"{int(value):,}"


def fmt_pct(value: float, digits: int = 1) -> str:
    return f"{100 * value:.{digits}f}%"


def esc(value: object) -> str:
    return html.escape(str(value))


def dataframe_table(frame: pd.DataFrame, percent_cols: set[str] | None = None) -> str:
    percent_cols = percent_cols or set()
    headings = "".join(f"<th>{esc(col)}</th>" for col in frame.columns)
    rows: list[str] = []
    for _, row in frame.iterrows():
        cells = []
        for col in frame.columns:
            value = row[col]
            if col in percent_cols and pd.notna(value):
                shown = fmt_pct(float(value))
            elif isinstance(value, (float, np.floating)):
                digits = 6 if "AUC" in col or "상관" in col else 3
                shown = f"{value:,.{digits}f}"
            elif isinstance(value, (int, np.integer)):
                shown = fmt_int(value)
            else:
                shown = str(value)
            cells.append(f"<td>{esc(shown)}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f'<div class="table-wrap"><table><thead><tr>{headings}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def bar_list(items: list[tuple[str, float]], percent: bool = False) -> str:
    max_value = max((value for _, value in items), default=1.0) or 1.0
    blocks = []
    for label, value in items:
        width = max(1.0, 100 * value / max_value)
        shown = fmt_pct(value) if percent else f"{value:.3f}"
        blocks.append(
            f'<div class="bar-row"><div class="bar-label">{esc(label)}</div>'
            f'<div class="bar-track"><span style="width:{width:.1f}%"></span></div>'
            f'<div class="bar-value">{esc(shown)}</div></div>'
        )
    return "".join(blocks)


def analyze() -> dict:
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    submission = pd.read_csv(DATA_DIR / "sample_submission.csv")

    target = "y"
    id_col = "id"
    features = [col for col in test.columns if col != id_col]
    numeric = [col for col in features if pd.api.types.is_numeric_dtype(train[col])]
    categorical = [col for col in features if col not in numeric]

    train_hash = pd.util.hash_pandas_object(train[features], index=False)
    test_hash = pd.util.hash_pandas_object(test[features], index=False)
    train_dup_rows = int(train_hash.duplicated(keep=False).sum())
    test_dup_rows = int(test_hash.duplicated(keep=False).sum())
    hash_target = pd.DataFrame({"h": train_hash, "y": train[target]})
    conflicting_groups = int((hash_target.groupby("h")["y"].nunique() > 1).sum())
    cross_split_matches = int(test_hash.isin(set(train_hash)).sum())

    missing = pd.DataFrame({
        "변수": features,
        "train 실제 결측": [int(train[col].isna().sum()) for col in features],
        "test 실제 결측": [int(test[col].isna().sum()) for col in features],
        "train unknown": [int(train[col].eq("unknown").sum()) if col in categorical else 0 for col in features],
        "test unknown": [int(test[col].eq("unknown").sum()) if col in categorical else 0 for col in features],
    })
    missing = missing[(missing.iloc[:, 1:] != 0).any(axis=1)].reset_index(drop=True)

    drift_rows = []
    for col in numeric:
        a = train[col].dropna().to_numpy()
        b = test[col].dropna().to_numpy()
        if len(a) > 100_000:
            a = RNG.choice(a, 100_000, replace=False)
        if len(b) > 100_000:
            b = RNG.choice(b, 100_000, replace=False)
        drift_rows.append((col, ks_statistic(a, b), "KS"))
    unseen_levels: dict[str, list[str]] = {}
    for col in categorical:
        tr_share = train[col].fillna("(NA)").value_counts(normalize=True)
        te_share = test[col].fillna("(NA)").value_counts(normalize=True)
        levels = tr_share.index.union(te_share.index)
        tv = 0.5 * float((tr_share.reindex(levels, fill_value=0) - te_share.reindex(levels, fill_value=0)).abs().sum())
        drift_rows.append((col, tv, "TV"))
        unseen = sorted(set(te_share.index) - set(tr_share.index))
        if unseen:
            unseen_levels[col] = [str(v) for v in unseen]
    drift = pd.DataFrame(drift_rows, columns=["변수", "차이 통계량", "기준"]).sort_values("차이 통계량", ascending=False)

    sample_n = min(150_000, len(train))
    sample_idx = RNG.choice(len(train), sample_n, replace=False)
    sampled = train.iloc[sample_idx]
    encoded = pd.DataFrame(index=sampled.index)
    for col in numeric:
        encoded[col] = pd.qcut(sampled[col].rank(method="first"), q=20, labels=False, duplicates="drop")
    for col in categorical:
        encoded[col] = pd.factorize(sampled[col].fillna("(NA)"), sort=True)[0]
    mi = [discrete_mutual_information(encoded[col], sampled[target]) for col in features]
    importance = pd.DataFrame({"변수": features, "상호정보량": mi}).sort_values("상호정보량", ascending=False)

    target_rate = float(train[target].mean())
    id_decile = pd.qcut(train[id_col], 10, labels=False)
    id_rates = train.groupby(id_decile, observed=True)[target].agg(["size", "mean"]).reset_index()
    id_rates.columns = ["ID 구간", "행 수", "양성률"]
    id_rates["ID 구간"] = id_rates["ID 구간"].map(lambda x: f"{int(x)+1}/10")

    num_stats = train[numeric].describe(percentiles=[.01, .25, .5, .75, .99]).T.reset_index()
    num_stats = num_stats.rename(columns={"index": "변수", "count": "행 수", "mean": "평균", "std": "표준편차", "min": "최소", "1%": "1%", "25%": "25%", "50%": "중앙값", "75%": "75%", "99%": "99%", "max": "최대"})
    num_stats = num_stats[["변수", "평균", "표준편차", "최소", "1%", "중앙값", "99%", "최대"]]

    category_cards = {}
    for col in categorical:
        grouped = train.groupby(col, dropna=False)[target].agg(["size", "mean"]).reset_index()
        grouped.columns = [col, "행 수", "양성률"]
        grouped = grouped.sort_values("행 수", ascending=False)
        category_cards[col] = grouped.head(12)

    numeric_patterns = {}
    for col in numeric:
        try:
            bins = pd.qcut(train[col], 10, duplicates="drop")
        except ValueError:
            bins = pd.cut(train[col], 10, duplicates="drop")
        grouped = train.groupby(bins, observed=True)[target].agg(["size", "mean"]).reset_index()
        first_col = grouped.columns[0]
        grouped[first_col] = grouped[first_col].astype(object).map(str)
        grouped.columns = ["구간", "행 수", "양성률"]
        numeric_patterns[col] = grouped

    return {
        "train": train,
        "test": test,
        "submission": submission,
        "features": features,
        "numeric": numeric,
        "categorical": categorical,
        "target_rate": target_rate,
        "train_dup_rows": train_dup_rows,
        "test_dup_rows": test_dup_rows,
        "conflicting_groups": conflicting_groups,
        "cross_split_matches": cross_split_matches,
        "missing": missing,
        "drift": drift,
        "unseen_levels": unseen_levels,
        "importance": importance,
        "id_rates": id_rates,
        "num_stats": num_stats,
        "category_cards": category_cards,
        "numeric_patterns": numeric_patterns,
    }


def make_html(a: dict) -> str:
    train, test = a["train"], a["test"]
    top_mi = a["importance"].head(10)
    drift_top = a["drift"].head(10)
    duration_pattern = a["numeric_patterns"].get("duration")
    poutcome = a["category_cards"].get("poutcome")
    month = a["category_cards"].get("month")
    job = a["category_cards"].get("job")
    public_models = pd.read_csv(BENCHMARK_DIR / "public_models.csv")
    our_experiments = pd.read_csv(BENCHMARK_DIR / "our_experiments.csv")
    our_experiments["cv_auc_num"] = pd.to_numeric(our_experiments["cv_auc"], errors="coerce")
    completed = our_experiments.dropna(subset=["cv_auc_num"]).copy()

    benchmark_rows = []
    for _, row in public_models.iterrows():
        def score(name: str) -> str:
            return "—" if pd.isna(row[name]) else f"{float(row[name]):.6f}"
        benchmark_rows.append(
            "<tr>"
            f"<td><a href=\"{esc(row['source_url'])}\" target=\"_blank\" rel=\"noopener noreferrer\">{esc(row['benchmark'])}</a></td>"
            f"<td>{esc(row['model'])}</td><td>{score('cv_auc')}</td><td>{score('public_lb')}</td>"
            f"<td>{score('private_lb')}</td><td>{esc(row['complexity'])}</td></tr>"
        )
    benchmark_table = (
        '<div class="table-wrap"><table><thead><tr><th>비교 기준</th><th>모델</th><th>CV AUC</th>'
        '<th>Public LB</th><th>Private LB</th><th>복잡도</th></tr></thead><tbody>'
        + "".join(benchmark_rows) + "</tbody></table></div>"
    )
    if len(completed):
        best = completed.loc[completed["cv_auc_num"].idxmax()]
        current_score = f"{best['cv_auc_num']:.6f}"
        current_title = f"현재 최고 · {esc(best['experiment_id'])}"
        current_status = "1차 목표 통과" if best["cv_auc_num"] >= 0.970 else "1차 목표까지 분석 필요"
        current_class = "good" if best["cv_auc_num"] >= 0.970 else "warning"
        current_note = esc(best["interpretation"])
    else:
        best = our_experiments.iloc[-1]
        current_score = "점수 없음"
        current_title = "현재 단계"
        current_status = "기준 모델 대기"
        current_class = "warning"
        current_note = esc(best["interpretation"])

    score_by_id = dict(zip(completed["experiment_id"], completed["cv_auc_num"]))
    kaggle_track = score_by_id.get("E001")
    realistic_track = score_by_id.get("E002")
    kaggle_track_text = "대기" if kaggle_track is None else f"{kaggle_track:.6f}"
    realistic_track_text = "대기" if realistic_track is None else f"{realistic_track:.6f}"
    duration_gap_text = "두 실험 완료 후 계산" if kaggle_track is None or realistic_track is None else f"{kaggle_track - realistic_track:+.6f} AUC"

    best_id = str(best["experiment_id"])
    best_score = float(best["cv_auc_num"]) if not pd.isna(best["cv_auc_num"]) else None
    initial_lgbm = score_by_id.get("E001-LIGHTGBM")
    goal_gap = None if best_score is None else max(0.0, 0.970 - best_score)
    catboost_gain = None if best_score is None or kaggle_track is None else best_score - kaggle_track
    tuning_gain = None if best_score is None or initial_lgbm is None else best_score - initial_lgbm

    fold_scores = [float(value) for value in str(best.get("fold_scores", "")).split("|") if value]
    best_iterations: list[int | str] = ["—"] * len(fold_scores)
    metrics_path = OUTPUT_DIR / best_id / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        best_iterations = metrics.get("best_iterations", best_iterations)
    fold_frame = pd.DataFrame({
        "Fold": [f"Fold {index}" for index in range(1, len(fold_scores) + 1)],
        "OOF AUC": fold_scores,
        "최적 반복": best_iterations,
    })
    fold_html = dataframe_table(fold_frame) if len(fold_frame) else "<p>Fold별 기록이 없습니다.</p>"

    model_config = pd.DataFrame([
        {"설정": "모델", "값": "LightGBM", "의미": "비선형 관계와 범주 간 상호작용을 트리 분할로 학습"},
        {"설정": "입력 변수", "값": "16개 전체", "의미": "Kaggle 점수 트랙이므로 duration 포함"},
        {"설정": "범주형 처리", "값": "native categorical", "의미": "임의 숫자 순서 없이 범주 정보를 직접 사용"},
        {"설정": "학습률 / 최대 트리", "값": "0.04 / 3,000", "의미": "작은 보폭으로 충분히 학습하되 각 fold에서 조기 종료"},
        {"설정": "리프 / 최소 표본", "값": "63 / 20", "의미": "복잡한 패턴을 허용하면서 지나치게 작은 잎은 제한"},
        {"설정": "행 / 열 샘플링", "값": "0.8 / 0.8", "의미": "트리마다 일부 행과 변수를 사용해 과적합을 완화"},
        {"설정": "L2 규제", "값": "2.0", "의미": "분할의 극단적인 예측을 완화"},
    ])
    model_config_html = dataframe_table(model_config)
    experiment_rows = []
    for _, row in our_experiments.iloc[::-1].iterrows():
        score_value = row["cv_auc_num"]
        score = "—" if pd.isna(score_value) else f"{float(score_value):.6f}"
        baseline_id = row.get("baseline_id", "")
        baseline_score = score_by_id.get(baseline_id)
        delta = "—" if pd.isna(score_value) or baseline_score is None else f"{float(score_value) - baseline_score:+.6f}"
        fold_std_value = row.get("fold_std", "")
        fold_std = "—" if pd.isna(fold_std_value) or fold_std_value == "" else esc(fold_std_value)
        experiment_rows.append(
            "<tr>"
            f"<td>{esc(row['experiment_id'])}</td><td>{esc(row['purpose'])}</td>"
            f"<td>{esc(row['model'])}</td><td>{score}</td><td>{delta}</td>"
            f"<td>{fold_std}</td><td>{esc(row['status'])}</td></tr>"
        )
    experiment_table = (
        '<div class="table-wrap"><table><thead><tr><th>실험</th><th>목적</th><th>모델</th>'
        '<th>OOF AUC</th><th>기준 대비</th><th>Fold 표준편차</th><th>상태</th></tr></thead><tbody>'
        + "".join(experiment_rows) + "</tbody></table></div>"
    )
    model_diagnostics_html = ""
    if MODEL_ANALYSIS_PATH.exists():
        model_analysis = json.loads(MODEL_ANALYSIS_PATH.read_text(encoding="utf-8"))
        importance = model_analysis.get("feature_importance", [])[:10]
        duration_gain_share = next(
            (float(row["gain_share"]) for row in importance if row["feature"] == "duration"), 0.0
        )
        importance_html = bar_list([
            (row["feature"], float(row["gain_share"])) for row in importance
        ]) if importance else "<p>새 실험의 피처 중요도 기록이 없습니다.</p>"

        weak = pd.DataFrame(model_analysis.get("weakest_segments", []))
        if len(weak):
            weak = weak.rename(columns={
                "segment": "구분", "value": "값", "rows": "행 수",
                "positive_rate": "양성률", "auc": "OOF AUC",
            })[["구분", "값", "행 수", "양성률", "OOF AUC"]]
            weak_html = dataframe_table(weak, {"양성률"})
        else:
            weak_html = "<p>세그먼트 진단이 없습니다.</p>"

        blends = pd.DataFrame(model_analysis.get("fixed_half_blends", [])[:6])
        if len(blends):
            blends = blends.rename(columns={
                "left": "모델 A", "right": "모델 B", "weight": "가중치", "oof_auc": "OOF AUC",
            })
            blend_html = dataframe_table(blends)
            best_blend_score = float(blends["OOF AUC"].max())
            blend_gain = best_blend_score - float(model_analysis["best_oof_auc"])
        else:
            blend_html = "<p>비교할 OOF 조합이 없습니다.</p>"
            blend_gain = 0.0

        correlations = pd.DataFrame(model_analysis.get("prediction_correlations", []))
        if len(correlations):
            correlations = correlations.rename(columns={
                "left": "모델 A", "right": "모델 B", "correlation": "예측 상관",
            })
            correlation_html = dataframe_table(correlations)
        else:
            correlation_html = "<p>예측 상관 진단이 없습니다.</p>"

        model_diagnostics_html = f"""
        <h2>최고 모델 진단</h2><section class="grid two">
          <article><h3>무엇을 보고 예측했나</h3><p><code>duration</code>이 전체 gain의 <strong>{duration_gain_share:.1%}</strong>를 차지했습니다. 현재 점수의 상당 부분이 통화가 끝난 뒤 알게 되는 시간 정보에서 옵니다.</p><p class="note">{esc(model_analysis['best_experiment'])}의 5개 fold 평균입니다. gain은 모델이 분할에서 얻은 이득이며 인과효과가 아닙니다.</p>{importance_html}</article>
          <article><h3>어디에서 순위 구분이 어려웠나</h3><p>12월, 3월, 10월처럼 표본이 작거나 가입률이 높은 월과 이전 캠페인 성공 고객군에서 그룹 내부 AUC가 낮았습니다. 이미 가입 가능성이 전반적으로 높은 집단에서는 고객 간 미세한 순서를 구분하기가 더 어렵다는 뜻입니다.</p><p class="note">표본 1,000개 이상인 그룹만 비교했습니다. 그룹 AUC가 낮다고 전체 모델이 그 그룹을 낮게 평가한다는 뜻은 아닙니다.</p>{weak_html}</article>
          <article><h3>왜 단순 앙상블 효과가 작았나</h3><p>기준 모델들의 OOF 예측 상관이 매우 높아 같은 고객을 비슷한 순서로 평가했습니다. 모델 이름은 달라도 오류가 충분히 다르지 않으면 평균을 내도 새 정보가 거의 생기지 않습니다.</p>{correlation_html}</article>
          <article><h3>고정 50:50 혼합 결과</h3><p>가장 좋은 고정 혼합도 최고 단일 모델보다 <strong>{blend_gain:+.6f}</strong> AUC 개선에 그쳤습니다. 현재 병목은 혼합 비율보다 새로운 정보나 다른 오류 구조를 만드는 데 있습니다.</p><p class="note">가중치를 탐색하지 않은 진단용 비교이며 단일 모델 0.970 목표와는 별도로 봅니다.</p>{blend_html}</article>
        </section>"""
    dictionary_html = dataframe_table(
        pd.DataFrame(DATA_DICTIONARY).rename(columns={"공식 의미": "의미"})
    ).replace(
        "<table>", '<table class="dictionary">'
    )
    feature_guide_html = dataframe_table(pd.DataFrame(FEATURE_GUIDE)).replace(
        "<table>", '<table class="dictionary feature-guide">'
    )

    duration_html = dataframe_table(duration_pattern, {"양성률"}) if duration_pattern is not None else ""
    cat_sections = []
    for title, frame in [("이전 캠페인 결과(poutcome)", poutcome), ("연락 월(month)", month), ("직업(job)", job)]:
        if frame is not None:
            cat_sections.append(f"<article><h3>{title}</h3>{dataframe_table(frame, {'양성률'})}</article>")

    unknown_total = int(a["missing"]["train unknown"].sum()) if len(a["missing"]) else 0
    pdays_minus_one = int(train["pdays"].eq(-1).sum()) if "pdays" in train else 0
    pdays_minus_one_rate = pdays_minus_one / len(train)
    state_disagreement = int((train["pdays"].eq(-1) != train["previous"].eq(0)).sum())
    outcome_disagreement = int((train["pdays"].eq(-1) != train["poutcome"].eq("unknown")).sum())
    max_id_delta = float((a["id_rates"]["양성률"] - a["target_rate"]).abs().max())

    css = """
    :root{--bg:#f5f7fb;--card:#fff;--ink:#152238;--muted:#5f6f86;--line:#dfe6f0;--blue:#246bfd;--teal:#0b9f8a;--amber:#e28a16;--red:#cb3a4a}
    *{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:Inter,"Noto Sans KR",system-ui,sans-serif;line-height:1.6}
    main{width:min(1180px,calc(100% - 32px));margin:auto;padding:48px 0 72px}.eyebrow{color:var(--blue);font-weight:800;font-size:12px;letter-spacing:.12em;text-transform:uppercase}
    h1{font-size:clamp(36px,6vw,64px);line-height:1.08;letter-spacing:-.045em;margin:10px 0 14px}.lead{font-size:18px;color:var(--muted);max-width:800px}
    h2{font-size:28px;letter-spacing:-.025em;margin:56px 0 18px}h3{margin:0 0 14px;font-size:18px}.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}.two{grid-template-columns:repeat(2,1fr)}
    .card,article{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:20px;box-shadow:0 8px 28px rgba(33,56,93,.05)}.metric{font-size:30px;font-weight:800;letter-spacing:-.04em}.label,.note{color:var(--muted);font-size:13px}
    .finding{border-left:4px solid var(--blue)}.warning{border-left-color:var(--amber)}.risk{border-left-color:var(--red)}.good{border-left-color:var(--teal)}
    .status-line{display:flex;align-items:center;gap:10px;margin-top:12px}.pill{display:inline-flex;padding:5px 9px;border-radius:999px;background:#e9f8f4;color:#087764;font-size:12px;font-weight:800}.pending{background:#fff3df;color:#9a5a05}.score-empty{font-size:36px;font-weight:800;letter-spacing:-.04em}.rule-list{margin:0;padding-left:20px}.rule-list li{margin:7px 0}
    .table-wrap{overflow-x:auto}table{width:100%;border-collapse:collapse;font-size:13px}th,td{padding:9px 11px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}th{color:var(--muted);font-weight:700;background:#f8faff}th:first-child,td:first-child{text-align:left}.dictionary th,.dictionary td{text-align:left}.dictionary td:nth-child(3),.dictionary td:nth-child(4){white-space:normal;min-width:220px}
    .feature-guide td:nth-child(4),.feature-guide td:nth-child(5){white-space:normal;min-width:260px}.roadmap{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.road-step{border:1px solid var(--line);border-radius:12px;padding:16px;background:#fbfcff}.road-step span{display:inline-flex;margin-bottom:10px;padding:3px 8px;border-radius:999px;font-size:12px;font-weight:800}.road-step h3{margin-bottom:8px}.road-step p{margin:0;font-size:14px}.road-step.done span{background:#e9f8f4;color:#087764}.road-step.now{border:2px solid var(--blue);background:#f5f8ff}.road-step.now span{background:#e8efff;color:#174fc4}.road-step.next span{background:#eef2f8;color:var(--muted)}
    .bar-row{display:grid;grid-template-columns:110px 1fr 64px;gap:10px;align-items:center;margin:10px 0;font-size:13px}.bar-track{height:9px;background:#e9eef7;border-radius:9px;overflow:hidden}.bar-track span{display:block;height:100%;background:linear-gradient(90deg,var(--blue),#63a0ff);border-radius:9px}.bar-value{text-align:right;font-variant-numeric:tabular-nums}
    code{background:#eef2f8;padding:2px 5px;border-radius:5px}ol li{margin:8px 0}.small{font-size:12px;color:var(--muted)}footer{margin-top:56px;padding-top:20px;border-top:1px solid var(--line);color:var(--muted);font-size:12px}
    @media(max-width:820px){.grid,.two{grid-template-columns:1fr 1fr}.roadmap{grid-template-columns:repeat(2,1fr)}}@media(max-width:560px){main{padding-top:28px}.grid,.two,.roadmap{grid-template-columns:1fr}.bar-row{grid-template-columns:88px 1fr 58px}h2{margin-top:42px}}
    """
    return f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="description" content="Bank Kaggle 데이터 분석과 모델링 실험"><title>Bank 데이터 분석과 모델링</title><style>{css}</style></head>
    <body><main><div class="eyebrow">Kaggle Data Analysis Study · Competition 1</div><h1>Bank 데이터 분석과 모델링</h1>
    <p class="lead">데이터 품질과 누수 위험을 확인하고, 고정된 OOF 검증에서 한 요소씩 바꾼 실험 결과를 정리했습니다. 원본 CSV는 변경하지 않았습니다.</p>
    <section class="grid"><div class="card"><div class="metric">{fmt_int(len(train))}</div><div class="label">학습 행</div></div><div class="card"><div class="metric">{fmt_int(len(test))}</div><div class="label">테스트 행</div></div><div class="card"><div class="metric">{len(a['features'])}</div><div class="label">예측 변수</div></div><div class="card"><div class="metric">{fmt_pct(a['target_rate'])}</div><div class="label">타깃 y=1 비율</div></div></section>

    <h2>무엇을 예측하고, 무엇이 좋은 점수인가</h2><section class="grid two">
      <article class="finding good"><h3>정답 y와 예측값</h3><p><code>y=1</code>은 고객이 정기예금에 <strong>가입</strong>했다는 뜻이고, <code>y=0</code>은 가입하지 않았다는 뜻입니다. 모델은 각 고객에게 <strong>y=1일 확률</strong>을 0~1 사이로 냅니다. 1에 가까울수록 가입 가능성이 높다고 판단한 고객입니다.</p><p class="note">제출값 자체를 모두 1에 가깝게 만드는 것이 목표는 아닙니다. 실제 가입 고객에게 비가입 고객보다 더 높은 점수를 주는 것이 핵심입니다.</p></article>
      <article><h3>평가지표 ROC AUC</h3><p>ROC AUC는 무작위로 고른 가입 고객 한 명과 비가입 고객 한 명을 비교할 때, 가입 고객에게 더 높은 예측값을 줄 확률처럼 읽을 수 있습니다. <strong>1.0에 가까울수록 좋고</strong>, 0.5는 무작위 순위 수준입니다. 현재 E008의 <strong>0.969283</strong>은 약 96.9%의 쌍을 올바른 순서로 놓는 수준이며 1차 목표 0.970까지 0.000717 남았습니다.</p><p class="note">ROC AUC는 확률이 정확히 보정됐는지보다 고객의 상대적 순서를 평가합니다. 따라서 임계값 0.5에서의 정확도와는 다른 지표입니다.</p></article>
    </section>
    <section class="grid two" style="margin-top:14px">
      <article><h3>Kaggle 점수 트랙</h3><p>대회 규칙 안에서 <code>duration</code>을 포함한 16개 변수를 사용해 ROC AUC를 최대화합니다. <strong>클수록 좋으며 단일 모델 0.970</strong>을 현재 1차 통과선으로 둡니다.</p></article>
      <article class="finding warning"><h3>현실 사전 타기팅 트랙</h3><p>전화하기 전에 가입 가능성이 높은 고객을 고르는 목적이라면 통화 후 알게 되는 <code>duration</code>을 제외합니다. 이 점수도 클수록 좋지만, 정보 조건이 다르므로 Kaggle 트랙보다 낮다고 실패가 아니며 두 점수를 직접 우열 비교하지 않습니다.</p></article>
    </section>

    <h2>전체 진행 로드맵</h2><article><div class="roadmap">
      <div class="road-step done"><span>1 · 완료</span><h3>문제·데이터 이해</h3><p>타깃, 변수 의미, 결측·중복, train/test 차이와 <code>duration</code>의 사용 시점 위험을 확인했습니다.</p></div>
      <div class="road-step done"><span>2 · 완료</span><h3>검증·기준선</h3><p>고정 Stratified 5-Fold를 만들고 CatBoost, LightGBM, XGBoost를 같은 조건에서 비교했습니다.</p></div>
      <div class="road-step done"><span>3 · 완료</span><h3>단일 설정 개선</h3><p>한 번에 한 설정만 바꿔 E008 LightGBM 0.969283까지 개선하고 기각한 변경도 기록했습니다.</p></div>
      <div class="road-step now"><span>4 · 현재</span><h3>피처 가설 검증</h3><p>공개 강한 단일 모델과 차이를 대조한 뒤 아래 후보 중 근거가 가장 강한 피처 하나만 E008에 추가해 검증합니다.</p></div>
      <div class="road-step next"><span>5 · 다음</span><h3>외부 데이터·오류 다양화</h3><p>UCI 원본 데이터는 학습 fold에만 추가해 보고, 서로 다른 오류를 만드는 모델이 확인될 때만 앙상블을 재검토합니다.</p></div>
      <div class="road-step next"><span>6 · 마무리</span><h3>제출·해석·회고</h3><p>OOF 개선이 재현되면 Kaggle 제출로 확인하고, 점수·현실 적용 한계·다음 대회에 가져갈 교훈을 정리합니다.</p></div>
    </div><p class="note">현재 초점은 4단계입니다. 설정 후보를 무작정 늘리기보다 새로운 정보를 줄 가능성이 있는 피처 가설을 한 번에 하나씩 검증합니다.</p></article>

    <h2>Feature를 어떻게 읽고 만들 것인가</h2><article><p>Feature는 모델이 고객을 구분할 때 보는 단서입니다. <strong>수치형</strong>은 크기와 임계점을, <strong>범주형</strong>은 집단별 차이를, <strong>표현/변환(representation)</strong>은 같은 원정보를 모델이 더 쉽게 읽는 형태로 바꾼 것을, <strong>상호작용(interaction)</strong>은 두 조건이 함께 있을 때 생기는 차이를 뜻합니다.</p><p class="note">아래의 “예상 영향”은 인과관계나 성능 향상을 확정한 결과가 아니라 EDA와 업무 의미에서 세운 가설입니다. 실제 채택 여부는 동일한 fold에서 한 후보씩 비교해 결정합니다.</p>{feature_guide_html}</article>

    <h2>모델링 결과 한눈에 보기</h2><section class="grid two">
      <article class="finding {current_class}"><h3>{current_title}</h3><div class="score-empty">{current_score}</div><div class="status-line"><span class="pill">EDA 완료</span><span class="pill">{current_status}</span></div><p class="note">{current_note}</p></article>
      <article><h3>현재 결론</h3><p>초기 CatBoost보다 LightGBM이 크게 앞섰고, LightGBM 안에서는 학습량·리프 수·행 샘플링을 차례로 바꿔 점수를 올렸습니다. 현재 최고는 <strong>E008 LightGBM 0.969283</strong>이며 목표 0.970까지 <strong>{goal_gap:.6f}</strong> 남았습니다.</p><p class="note">Kaggle 제출 점수는 아직 없으므로 보고하지 않습니다. 현재 비교는 모두 동일한 로컬 OOF 기준입니다.</p></article>
    </section>
    <section class="grid" style="margin-top:14px"><div class="card"><div class="metric">{kaggle_track_text}</div><div class="label">초기 CatBoost · E001</div></div><div class="card"><div class="metric">{initial_lgbm:.6f}</div><div class="label">초기 LightGBM</div></div><div class="card"><div class="metric">{catboost_gain:+.6f}</div><div class="label">CatBoost 대비 현재 개선</div></div><div class="card"><div class="metric">{tuning_gain:+.6f}</div><div class="label">LightGBM 내부 개선</div></div></section>

    <h2>모델 개선 과정</h2><section class="grid two">
      <article class="finding good"><h3>가장 큰 선택 · 모델 계열</h3><p>같은 16개 변수와 같은 fold에서 초기 LightGBM은 <strong>0.968502</strong>로 CatBoost <strong>0.963737</strong>보다 <strong>+0.004765</strong> 높았습니다. 이번 데이터에서는 범주형 전용 모델의 기본 설정 이점보다 LightGBM의 트리 성장 방식이 더 잘 맞았습니다.</p><p class="note">모델 계열 변경의 효과가 이후 개별 하이퍼파라미터 조정보다 훨씬 컸습니다.</p></article>
      <article class="finding"><h3>가장 큰 조정 · 학습 상한</h3><p>E004에서 최대 트리를 1,500개에서 3,000개로 늘린 E005가 <strong>+0.000444</strong> 개선됐습니다. 조기 종료가 각 fold의 적정 시점을 고르므로, 기존 상한이 학습을 너무 일찍 막았던 것으로 해석됩니다.</p></article>
      <article class="finding"><h3>복잡도와 샘플링</h3><p>리프 수를 31에서 63으로 늘린 E006은 <strong>+0.000135</strong>, 매 트리마다 80%의 행을 다시 뽑게 한 E008은 E006 대비 <strong>+0.000115</strong> 개선됐습니다. 더 세밀한 상호작용과 약한 무작위성이 모두 도움이 됐습니다.</p></article>
      <article class="finding risk"><h3>채택하지 않은 변경</h3><p><code>min_child_samples</code>를 20에서 50으로 높인 E007은 E006보다 <strong>-0.000040</strong> 낮았습니다. 잎을 지나치게 크게 제한해 소수 패턴을 놓친 것으로 보고 되돌렸습니다.</p></article>
    </section>

    <h2>검증 설계와 신뢰도</h2><section class="grid two">
      <article><h3>어떻게 비교했나</h3><ol class="rule-list"><li>타깃 비율을 유지하는 shuffled Stratified 5-fold, seed 326</li><li>각 fold 약 60만 행으로 학습하고 약 15만 행으로 검증</li><li>범주 인코딩과 모델 학습은 검증 데이터를 보지 않고 fold 안에서 수행</li><li>모든 실험을 같은 fold의 OOF ROC AUC로 비교</li><li>평균뿐 아니라 fold 표준편차로 흔들림 확인</li></ol><p class="note">E008 fold 표준편차는 0.000199로 작아 특정 fold 하나가 평균을 끌어올린 결과는 아닙니다.</p></article>
      <article><h3>{best_id} fold별 결과</h3>{fold_html}<p class="note">전체 학습 시간은 약 {float(best['runtime_minutes']):.2f}분입니다. 최적 반복 수가 fold마다 다른 것은 조기 종료가 각 학습/검증 조합에 맞는 지점을 선택했기 때문입니다.</p></article>
    </section>
    <article style="margin-top:14px"><h3>최고 모델 구성</h3>{model_config_html}<p class="note">설정값은 성능의 원인이 아니라 검증으로 선택된 현재 상태입니다. 이후 실험에서도 한 항목만 바꿔 영향의 원인을 분리합니다.</p></article>

    <h2>전체 실험 기록</h2><article>{experiment_table}<p class="note">기준 대비 값은 같은 고정 fold에서 선언된 기준 실험과 비교합니다. 한 번에 한 요소만 바꾼 경우에만 원인을 해석합니다.</p></article>
    {model_diagnostics_html}
    <article style="margin-top:14px"><h3>공개 모델 벤치마크</h3>{benchmark_table}<p class="note">선택한 공개 자료의 보고값입니다. 서로 다른 검증 분할에서 나온 CV는 완전히 같은 조건의 순위표가 아니므로, 절대 순위보다 도달 가능한 수준을 판단하는 기준으로 사용합니다.</p></article>
    <section class="grid two" style="margin-top:14px"><article><h3>단일 모델 목표</h3><p><strong>1차 통과:</strong> CV 0.970<br><strong>강한 기준:</strong> CV 0.974<br><strong>상위 단일 모델권:</strong> CV 0.976 전후</p><p class="note">첫 목표는 복잡한 앙상블이 아니라 재현 가능한 단일 모델입니다.</p></article><article><h3>앙상블 목표</h3><p><strong>경쟁력 있는 수준:</strong> CV 0.9765 이상<br><strong>공개 상위권 사례:</strong> CV 0.9773 전후</p><p class="note">단일 모델의 가설 실험이 끝난 뒤에만 비교합니다. 수십~수백 모델 앙상블과 첫 기준선을 직접 비교하지 않습니다.</p></article></section>

    <h2>먼저 알아야 할 결론</h2><section class="grid two">
      <article class="finding risk"><h3>duration은 점수용과 현실용을 분리합니다</h3><p><code>duration</code>은 마지막 연락의 통화시간이며 타깃과의 단변량 연관성이 가장 큽니다. 하지만 통화가 끝난 뒤에야 확정되므로 Kaggle 점수 모델에는 포함하고, 통화 전 고객 선별 모델에서는 제외해 별도 평가합니다.</p></article>
      <article class="finding warning"><h3>unknown은 정보 없음 범주입니다</h3><p>실제 NaN은 없지만 범주형 변수에 <code>unknown</code>이 {fmt_int(unknown_total)}건 있습니다. 이를 문자 그대로의 결측치로 바꾸거나 최빈값으로 덮지 않고, ‘정보를 알 수 없음’이라는 명시적 범주로 유지합니다.</p></article>
      <article class="finding warning"><h3>pdays=-1은 이전 연락 없음입니다</h3><p><code>pdays=-1</code>은 ‘이전에 연락한 적 없음’을 뜻합니다. 해당 상태는 {fmt_int(pdays_minus_one)}행({fmt_pct(pdays_minus_one_rate)})이며, 경과일 수치와 상태 표시를 분리해 실험할 가치가 있습니다.</p></article>
      <article class="finding good"><h3>Kaggle train/test 이동은 작습니다</h3><p>각 수치형의 KS 통계량과 범주형의 총변동거리를 비교했으며 가장 큰 값은 {a['drift'].iloc[0]['변수']} {a['drift'].iloc[0]['차이 통계량']:.3f}입니다. 따라서 무작위 층화 검증을 시작점으로 사용할 수 있습니다. 다만 외부 데이터를 추가할 때는 별도로 분포 차이를 검증해야 합니다.</p></article>
    </section>

    <h2>변수 사전</h2><article><p class="note">이 데이터는 은행의 전화 마케팅을 통해 고객이 정기예금에 가입할 가능성을 예측합니다. 변수의 업무상 의미와 모델링할 때 주의할 점을 함께 정리했습니다.</p>{dictionary_html}</article>

    <h2>변수 해석과 모델링 주의</h2><section class="grid two">
      <article class="finding warning"><h3>day와 month는 마지막 연락 시점입니다</h3><p><code>day</code>는 요일이 아니라 월중 일자이고, <code>month</code>는 마지막 연락 월입니다. 월별 양성률 차이는 고객 선호뿐 아니라 계절성과 은행의 캠페인 대상 선정이 섞인 연관성입니다. 연도 정보가 없어 두 변수만으로 신뢰할 만한 시간순 검증을 만들기 어렵습니다.</p></article>
      <article class="finding warning"><h3>campaign은 마지막 연락을 포함합니다</h3><p><code>campaign</code>은 현 캠페인에서 해당 고객에게 연락한 횟수이며 마지막 연락도 포함합니다. ‘첫 전화 전에 예측’하는지 ‘현재 통화 후 예측’하는지에 따라 사용할 수 있는 값이 달라지므로 모델 목적을 먼저 고정해야 합니다.</p></article>
      <article class="finding"><h3>balance는 유로 단위 연평균 잔액입니다</h3><p>단순 현재 잔액이 아닙니다. 음수와 큰 양수가 함께 있으므로 일반 로그 변환은 바로 적용할 수 없습니다. 트리 모델의 원값 기준선을 먼저 만들고, 필요하면 부호를 보존하는 변환만 한 변수 실험으로 비교합니다.</p></article>
      <article class="finding risk"><h3>외부 데이터는 학습 폴드에만 추가합니다</h3><p>Kaggle train/test는 합성 데이터이므로 외부의 실제 은행 마케팅 데이터를 섞더라도 검증 폴드는 Kaggle train으로 유지합니다. 외부 행은 각 학습 폴드에만 추가하고, 데이터 집단별 분포와 성능을 함께 확인해야 실제 Kaggle 테스트에 도움이 되는지 판단할 수 있습니다.</p></article>
    </section>

    <h2>데이터 품질 점검</h2><section class="grid two"><article><h3>중복과 분할 겹침</h3>
      <p>학습 피처 기준 중복에 속한 행: <strong>{fmt_int(a['train_dup_rows'])}</strong><br>테스트 피처 기준 중복에 속한 행: <strong>{fmt_int(a['test_dup_rows'])}</strong><br>동일 피처인데 y가 다른 해시 그룹: <strong>{fmt_int(a['conflicting_groups'])}</strong><br>train과 완전히 같은 피처를 가진 test 행: <strong>{fmt_int(a['cross_split_matches'])}</strong></p>
      <p class="note">ID와 타깃은 중복 판단에서 제외했습니다. 해시 충돌 가능성은 이론적으로 남지만 매우 작습니다.</p></article>
      <article><h3>ID 안정성</h3><p>ID 10분위별 양성률의 전체 평균 대비 최대 차이는 <strong>{fmt_pct(max_id_delta,2)}</strong>입니다. ID가 생성 순서나 숨은 시간축을 담는지 계속 감시해야 합니다.</p>{dataframe_table(a['id_rates'], {'양성률'})}</article></section>
    <article style="margin-top:14px"><h3>결측·unknown 현황</h3>{dataframe_table(a['missing']) if len(a['missing']) else '<p>실제 결측과 unknown이 없습니다.</p>'}<p class="note"><code>pdays=-1</code>과 <code>previous=0</code>의 상태가 다른 행은 {fmt_int(state_disagreement)}개, <code>pdays=-1</code>과 <code>poutcome=unknown</code>의 상태가 다른 행은 {fmt_int(outcome_disagreement)}개입니다. 거의 같은 개념처럼 보여도 완전히 동일한 규칙으로 치환하면 정보가 손실됩니다.</p></article>

    <h2>어떤 변수가 타깃과 연결되는가</h2><section class="grid two"><article><h3>단변량 상호정보량 상위 10개</h3><p class="note">15만 행 표본에서 각 변수를 구간/범주로 바꿔 계산한 탐색용 지표입니다. 값이 높다고 인과관계이거나 모델 중요도가 확정되는 것은 아닙니다.</p>{bar_list([(r['변수'],float(r['상호정보량'])) for _,r in top_mi.iterrows()])}</article>
    <article><h3>train/test 차이 상위 10개</h3><p class="note">수치형은 KS, 범주형은 총변동거리(TV)를 사용했습니다. 서로 단위는 비슷하지만 같은 검정통계량은 아닙니다.</p>{bar_list([(r['변수'],float(r['차이 통계량'])) for _,r in drift_top.iterrows()])}</article></section>

    <h2>핵심 패턴</h2><section class="grid two"><article><h3>통화시간(duration) 10분위</h3><p class="note">각 구간의 양성률을 비교합니다. 강한 패턴이 검증 점수를 지배할 수 있어 사용 가능 시점이 핵심입니다.</p>{duration_html}</article>{''.join(cat_sections)}</section>
    <p class="note"><code>month</code>는 마지막 연락 월입니다. 월별 양성률 차이를 ‘그 달 고객이 가입을 선호한다’는 인과관계로 읽지 말고, 계절성과 캠페인 운영이 섞인 탐색 결과로 해석합니다.</p>

    <h2>수치형 변수의 범위</h2><article>{dataframe_table(a['num_stats'])}</article>

    <h2>현재 결론과 다음 실험</h2><article><ol>
      <li><strong>확인:</strong> LightGBM 계열이 CatBoost와 XGBoost 기준선보다 높았고, E008이 현재 최고입니다.</li>
      <li><strong>발견:</strong> 범주형 명시, 학습 상한, 리프 수, 행 샘플링은 개선됐지만 <code>min_child_samples=50</code>은 평균 AUC를 낮췄습니다.</li>
      <li><strong>중요성:</strong> 단순 설정 조정만으로 얻은 개선은 제한적이며, 예측 상관도 높아 고정 혼합 이득도 작습니다.</li>
      <li><strong>다음 1순위:</strong> 공개 강한 단일 모델과의 피처·검증 차이를 대조한 뒤, 근거가 있는 피처 가설 하나를 고정 fold에서 검증합니다.</li>
      <li><strong>다음 2순위:</strong> 외부 은행 마케팅 데이터를 쓸 경우 Kaggle 검증 fold는 그대로 두고 학습 fold에만 추가해 데이터 집단 차이를 확인합니다.</li>
      <li><strong>현실 트랙:</strong> 실제 사전 타기팅 목적이라면 <code>duration</code> 없는 E002를 별도 기준선으로 개선해야 합니다.</li>
    </ol></article>
    <footer>분석 기준: 로컬 train.csv, test.csv, sample_submission.csv · 참고 자료: <a href="https://www.kaggle.com/competitions/playground-series-s5e8/data" target="_blank" rel="noopener noreferrer">Kaggle 데이터 설명</a> · <a href="https://archive.ics.uci.edu/dataset/222/bank" target="_blank" rel="noopener noreferrer">변수 설명</a> · <a href="https://repositorio.biblioteca.iscte-iul.pt/bitstream/10071/9499/5/dss_v3.pdf" target="_blank" rel="noopener noreferrer">연구 배경</a> · 생성 스크립트: bank/src/eda_bank.py · 재현용 랜덤 시드: 326</footer></main></body></html>"""


def main() -> None:
    analysis = analyze()
    summary = {
        "train_shape": list(analysis["train"].shape),
        "test_shape": list(analysis["test"].shape),
        "target_rate": analysis["target_rate"],
        "numeric_features": analysis["numeric"],
        "categorical_features": analysis["categorical"],
        "duplicate_feature_rows_train": analysis["train_dup_rows"],
        "duplicate_feature_rows_test": analysis["test_dup_rows"],
        "conflicting_duplicate_groups_train": analysis["conflicting_groups"],
        "cross_split_feature_matches": analysis["cross_split_matches"],
        "unseen_test_levels": analysis["unseen_levels"],
        "top_mutual_information": analysis["importance"].head(16).to_dict(orient="records"),
        "train_test_drift": analysis["drift"].to_dict(orient="records"),
    }
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOCS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    rendered = "\n".join(line.rstrip() for line in make_html(analysis).splitlines()) + "\n"
    DOCS_PATH.write_text(rendered, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
