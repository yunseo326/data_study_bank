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
DOCS_PATH = ROOT / "docs" / "index.html"
SUMMARY_PATH = ROOT / "bank" / "logs" / "result" / "bank_eda_summary.json"
MODEL_ANALYSIS_PATH = ROOT / "bank" / "logs" / "result" / "model_analysis.json"
RNG = np.random.default_rng(326)

DATA_DICTIONARY = [
    {"변수": "age", "구분": "고객", "공식 의미": "고객 나이", "해석 주의": "비선형 효과와 연령대별 표본 수를 함께 확인"},
    {"변수": "job", "구분": "고객", "공식 의미": "직업 유형", "해석 주의": "unknown은 직업 정보가 없는 상태"},
    {"변수": "marital", "구분": "고객", "공식 의미": "결혼 상태", "해석 주의": "원본의 divorced는 이혼과 사별을 함께 포함"},
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
        else:
            blend_html = "<p>비교할 OOF 조합이 없습니다.</p>"

        correlations = pd.DataFrame(model_analysis.get("prediction_correlations", []))
        if len(correlations):
            correlations = correlations.rename(columns={
                "left": "모델 A", "right": "모델 B", "correlation": "예측 상관",
            })
            correlation_html = dataframe_table(correlations)
        else:
            correlation_html = "<p>예측 상관 진단이 없습니다.</p>"

        model_diagnostics_html = f"""
        <h2>최고 모델을 어떻게 해석할까</h2><section class="grid two">
          <article><h3>피처 중요도 · gain 비중</h3><p class="note">{esc(model_analysis['best_experiment'])}의 5개 fold 평균입니다. 모델이 분할에서 얻은 이득이지 인과효과는 아닙니다.</p>{importance_html}</article>
          <article><h3>성능이 낮은 고객군</h3><p class="note">표본 1,000개 이상인 그룹 중 AUC가 낮은 순서입니다. 양성률이 매우 높은 작은 월은 전체보다 순위 구분이 어렵습니다.</p>{weak_html}</article>
          <article><h3>모델 예측 상관</h3><p class="note">상관이 1에 가까울수록 같은 고객을 비슷하게 평가합니다. 높은 상관은 혼합의 추가 이득이 작을 수 있음을 뜻합니다.</p>{correlation_html}</article>
          <article><h3>고정 50:50 혼합</h3><p class="note">OOF에서 가중치를 탐색하지 않은 진단용 비교입니다. 단일 모델 0.970 목표와는 별도로 봅니다.</p>{blend_html}</article>
        </section>"""
    dictionary_html = dataframe_table(pd.DataFrame(DATA_DICTIONARY)).replace(
        "<table>", '<table class="dictionary">'
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
    .bar-row{display:grid;grid-template-columns:110px 1fr 64px;gap:10px;align-items:center;margin:10px 0;font-size:13px}.bar-track{height:9px;background:#e9eef7;border-radius:9px;overflow:hidden}.bar-track span{display:block;height:100%;background:linear-gradient(90deg,var(--blue),#63a0ff);border-radius:9px}.bar-value{text-align:right;font-variant-numeric:tabular-nums}
    code{background:#eef2f8;padding:2px 5px;border-radius:5px}ol li{margin:8px 0}.small{font-size:12px;color:var(--muted)}footer{margin-top:56px;padding-top:20px;border-top:1px solid var(--line);color:var(--muted);font-size:12px}
    @media(max-width:820px){.grid,.two{grid-template-columns:1fr 1fr}}@media(max-width:560px){main{padding-top:28px}.grid,.two{grid-template-columns:1fr}.bar-row{grid-template-columns:88px 1fr 58px}h2{margin-top:42px}}
    """
    return f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="description" content="Bank Kaggle 데이터 분석과 모델링 실험"><title>Bank 데이터 분석과 모델링</title><style>{css}</style></head>
    <body><main><div class="eyebrow">Kaggle Data Analysis Study · Competition 1</div><h1>Bank 데이터 분석과 모델링</h1>
    <p class="lead">데이터 품질과 누수 위험을 확인하고, 고정된 OOF 검증에서 한 요소씩 바꾼 실험 결과를 정리했습니다. 원본 CSV는 변경하지 않았습니다.</p>
    <section class="grid"><div class="card"><div class="metric">{fmt_int(len(train))}</div><div class="label">학습 행</div></div><div class="card"><div class="metric">{fmt_int(len(test))}</div><div class="label">테스트 행</div></div><div class="card"><div class="metric">{len(a['features'])}</div><div class="label">예측 변수</div></div><div class="card"><div class="metric">{fmt_pct(a['target_rate'])}</div><div class="label">타깃 y=1 비율</div></div></section>

    <h2>우리 모델은 어느 수준인가</h2><section class="grid two">
      <article class="finding {current_class}"><h3>{current_title}</h3><div class="score-empty">{current_score}</div><div class="status-line"><span class="pill">EDA 완료</span><span class="pill">{current_status}</span></div><p class="note">{current_note}</p></article>
      <article><h3>평가 원칙</h3><ol class="rule-list"><li>주지표는 같은 분할에서 계산한 OOF ROC AUC</li><li>Public LB는 검증이 실제 테스트에도 이어지는지 확인하는 보조 지표</li><li>단일 모델은 단일 모델끼리 비교</li><li>점수와 함께 모델 수·외부 데이터·학습 비용을 기록</li></ol></article>
    </section>
    <section class="grid" style="margin-top:14px"><div class="card"><div class="metric">{kaggle_track_text}</div><div class="label">Kaggle 점수 트랙 · E001</div></div><div class="card"><div class="metric">{realistic_track_text}</div><div class="label">통화 전 현실 트랙 · E002</div></div><div class="card"><div class="metric">{duration_gap_text}</div><div class="label">duration이 만든 검증 격차</div></div><div class="card"><div class="metric">0.970</div><div class="label">현재 1차 통과 목표</div></div></section>
    <article style="margin-top:14px"><h3>우리 실험 기록</h3>{experiment_table}<p class="note">기준 대비 값은 같은 고정 폴드에서 선언된 기준 실험과 비교합니다. 한 번에 한 요소만 바꾼 경우에만 원인을 해석합니다.</p></article>
    {model_diagnostics_html}
    <article style="margin-top:14px"><h3>공개 모델 벤치마크</h3>{benchmark_table}<p class="note">선택한 공개 자료의 보고값입니다. 서로 다른 검증 분할에서 나온 CV는 완전히 같은 조건의 순위표가 아니므로, 절대 순위보다 도달 가능한 수준을 판단하는 기준으로 사용합니다.</p></article>
    <section class="grid two" style="margin-top:14px"><article><h3>단일 모델 목표</h3><p><strong>1차 통과:</strong> CV 0.970<br><strong>강한 기준:</strong> CV 0.974<br><strong>상위 단일 모델권:</strong> CV 0.976 전후</p><p class="note">첫 목표는 복잡한 앙상블이 아니라 재현 가능한 단일 모델입니다.</p></article><article><h3>앙상블 목표</h3><p><strong>경쟁력 있는 수준:</strong> CV 0.9765 이상<br><strong>공개 상위권 사례:</strong> CV 0.9773 전후</p><p class="note">단일 모델의 가설 실험이 끝난 뒤에만 비교합니다. 수십~수백 모델 앙상블과 첫 기준선을 직접 비교하지 않습니다.</p></article></section>

    <h2>먼저 알아야 할 결론</h2><section class="grid two">
      <article class="finding risk"><h3>duration은 점수용과 현실용을 분리합니다</h3><p>공식 정의는 ‘마지막 연락 통화시간(초)’입니다. <code>duration</code>은 타깃과의 단변량 연관성이 가장 크지만 통화가 끝난 뒤에야 확정됩니다. 따라서 Kaggle 점수 모델에는 포함하고, 통화 전 고객 선별 모델에서는 제외해 별도 평가합니다. 사전 예측에서 제외해야 한다는 판단은 공식 정의에 근거한 예측 시점 해석입니다.</p></article>
      <article class="finding warning"><h3>unknown은 정보 없음 범주입니다</h3><p>실제 NaN은 없지만 범주형 변수에 <code>unknown</code>이 {fmt_int(unknown_total)}건 있습니다. 이를 문자 그대로의 결측치로 바꾸거나 최빈값으로 덮지 않고, ‘정보를 알 수 없음’이라는 명시적 범주로 유지합니다.</p></article>
      <article class="finding warning"><h3>pdays=-1은 이전 연락 없음입니다</h3><p>UCI 공식 정의에 따르면 <code>pdays=-1</code>은 ‘이전에 연락한 적 없음’입니다. 해당 상태는 {fmt_int(pdays_minus_one)}행({fmt_pct(pdays_minus_one_rate)})이며, 경과일 수치와 상태 표시를 분리해 실험할 가치가 있습니다.</p></article>
      <article class="finding good"><h3>Kaggle train/test 이동은 작습니다</h3><p>각 수치형의 KS 통계량과 범주형의 총변동거리를 비교했으며 가장 큰 값은 {a['drift'].iloc[0]['변수']} {a['drift'].iloc[0]['차이 통계량']:.3f}입니다. 무작위 층화 검증을 시작점으로 쓸 수 있지만, 이 결과가 별도 출처인 UCI 원본 데이터와의 호환성까지 보장하지는 않습니다.</p></article>
    </section>

    <h2>공식 변수 사전</h2><article><p class="note">Kaggle은 UCI Bank Marketing 원본을 학습한 모델로 합성 데이터를 만들었습니다. 아래는 UCI의 변수 정의를 기준으로 하되, 합성 데이터에서 그대로 단정하면 안 되는 해석을 함께 적었습니다.</p>{dictionary_html}</article>

    <h2>원문으로 보완된 해석</h2><section class="grid two">
      <article class="finding warning"><h3>day와 month는 마지막 연락 시점입니다</h3><p><code>day</code>는 요일이 아니라 월중 일자이고, <code>month</code>는 마지막 연락 월입니다. 월별 양성률 차이는 고객 선호뿐 아니라 계절성과 은행의 캠페인 대상 선정이 섞인 연관성입니다. 연도 정보가 없어 두 변수만으로 신뢰할 만한 시간순 검증을 만들기 어렵습니다.</p></article>
      <article class="finding warning"><h3>campaign은 마지막 연락을 포함합니다</h3><p><code>campaign</code>은 현 캠페인에서 해당 고객에게 연락한 횟수이며 마지막 연락도 포함합니다. ‘첫 전화 전에 예측’하는지 ‘현재 통화 후 예측’하는지에 따라 사용할 수 있는 값이 달라지므로 모델 목적을 먼저 고정해야 합니다.</p></article>
      <article class="finding"><h3>balance는 유로 단위 연평균 잔액입니다</h3><p>단순 현재 잔액이 아닙니다. 음수와 큰 양수가 함께 있으므로 일반 로그 변환은 바로 적용할 수 없습니다. 트리 모델의 원값 기준선을 먼저 만들고, 필요하면 부호를 보존하는 변환만 한 변수 실험으로 비교합니다.</p></article>
      <article class="finding risk"><h3>원본 데이터는 학습 폴드에만 추가합니다</h3><p>Kaggle train/test는 합성 데이터이므로 UCI 원본을 섞을 때도 검증 폴드는 Kaggle train으로 유지합니다. 원본 행은 각 학습 폴드에만 추가하고, 출처별 분포와 성능을 함께 확인해야 실제 Kaggle 테스트에 도움이 되는지 판단할 수 있습니다.</p></article>
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
      <li><strong>다음 2순위:</strong> UCI 원본을 쓸 경우 Kaggle 검증 fold는 그대로 두고 학습 fold에만 추가해 출처 차이를 확인합니다.</li>
      <li><strong>현실 트랙:</strong> 실제 사전 타기팅 목적이라면 <code>duration</code> 없는 E002를 별도 기준선으로 개선해야 합니다.</li>
    </ol></article>
    <footer>분석 기준: 로컬 train.csv, test.csv, sample_submission.csv · 원문: <a href="https://www.kaggle.com/competitions/playground-series-s5e8/data" target="_blank" rel="noopener noreferrer">Kaggle 데이터 설명</a> · <a href="https://archive.ics.uci.edu/dataset/222/bank" target="_blank" rel="noopener noreferrer">UCI 변수 정의</a> · <a href="https://repositorio.biblioteca.iscte-iul.pt/bitstream/10071/9499/5/dss_v3.pdf" target="_blank" rel="noopener noreferrer">원 연구 논문</a> · 생성 스크립트: bank/src/eda_bank.py · 재현용 랜덤 시드: 326</footer></main></body></html>"""


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
