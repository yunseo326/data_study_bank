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
DOCS_PATH = ROOT / "docs" / "index.html"
SUMMARY_PATH = ROOT / "logs" / "result" / "bank_eda_summary.json"
RNG = np.random.default_rng(326)


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
                shown = f"{value:,.3f}"
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
    .table-wrap{overflow-x:auto}table{width:100%;border-collapse:collapse;font-size:13px}th,td{padding:9px 11px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}th{color:var(--muted);font-weight:700;background:#f8faff}th:first-child,td:first-child{text-align:left}
    .bar-row{display:grid;grid-template-columns:110px 1fr 64px;gap:10px;align-items:center;margin:10px 0;font-size:13px}.bar-track{height:9px;background:#e9eef7;border-radius:9px;overflow:hidden}.bar-track span{display:block;height:100%;background:linear-gradient(90deg,var(--blue),#63a0ff);border-radius:9px}.bar-value{text-align:right;font-variant-numeric:tabular-nums}
    code{background:#eef2f8;padding:2px 5px;border-radius:5px}ol li{margin:8px 0}.small{font-size:12px;color:var(--muted)}footer{margin-top:56px;padding-top:20px;border-top:1px solid var(--line);color:var(--muted);font-size:12px}
    @media(max-width:820px){.grid,.two{grid-template-columns:1fr 1fr}}@media(max-width:560px){main{padding-top:28px}.grid,.two{grid-template-columns:1fr}.bar-row{grid-template-columns:88px 1fr 58px}h2{margin-top:42px}}
    """
    return f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="description" content="Bank Kaggle 데이터 1차 EDA"><title>Bank 데이터 1차 분석</title><style>{css}</style></head>
    <body><main><div class="eyebrow">Kaggle Data Analysis Study · Competition 1</div><h1>Bank 데이터 1차 분석</h1>
    <p class="lead">모델을 돌리기 전에 데이터가 무엇을 말하는지, 검증에서 무엇을 조심해야 하는지 확인했습니다. 원본 CSV는 변경하지 않았습니다.</p>
    <section class="grid"><div class="card"><div class="metric">{fmt_int(len(train))}</div><div class="label">학습 행</div></div><div class="card"><div class="metric">{fmt_int(len(test))}</div><div class="label">테스트 행</div></div><div class="card"><div class="metric">{len(a['features'])}</div><div class="label">예측 변수</div></div><div class="card"><div class="metric">{fmt_pct(a['target_rate'])}</div><div class="label">타깃 y=1 비율</div></div></section>

    <h2>먼저 알아야 할 결론</h2><section class="grid two">
      <article class="finding risk"><h3>duration은 가장 강하지만 사용 시점을 확인해야 합니다</h3><p><code>duration</code>은 타깃과의 단변량 연관성이 가장 큽니다. 통화가 끝난 뒤에만 알 수 있는 통화시간이라면 사전 고객 선별 모델에서는 누수입니다. Kaggle 제출 모델과 실제 영업 모델의 허용 피처가 달라질 수 있습니다.</p></article>
      <article class="finding warning"><h3>unknown은 결측값으로 해석해야 합니다</h3><p>실제 NaN과 별도로 범주형 변수에 <code>unknown</code>이 {fmt_int(unknown_total)}건 있습니다. 이를 평범한 고객 특성으로만 보면 결측의 의미를 놓칠 수 있습니다.</p></article>
      <article class="finding warning"><h3>pdays=-1은 특별한 상태입니다</h3><p><code>pdays=-1</code>은 {fmt_int(pdays_minus_one)}행({fmt_pct(pdays_minus_one_rate)})입니다. 연속 숫자가 아니라 ‘이전 연락 없음’ 상태일 가능성이 높아 별도 표시 피처가 필요합니다.</p></article>
      <article class="finding good"><h3>큰 train/test 분포 이동은 보이지 않습니다</h3><p>각 수치형의 KS 통계량과 범주형의 총변동거리를 비교했으며 가장 큰 값은 {a['drift'].iloc[0]['변수']} {a['drift'].iloc[0]['차이 통계량']:.3f}입니다. 현 단계에서는 무작위 층화 검증을 시작점으로 삼을 수 있습니다.</p></article>
    </section>

    <h2>데이터 품질 점검</h2><section class="grid two"><article><h3>중복과 분할 겹침</h3>
      <p>학습 피처 기준 중복에 속한 행: <strong>{fmt_int(a['train_dup_rows'])}</strong><br>테스트 피처 기준 중복에 속한 행: <strong>{fmt_int(a['test_dup_rows'])}</strong><br>동일 피처인데 y가 다른 해시 그룹: <strong>{fmt_int(a['conflicting_groups'])}</strong><br>train과 완전히 같은 피처를 가진 test 행: <strong>{fmt_int(a['cross_split_matches'])}</strong></p>
      <p class="note">ID와 타깃은 중복 판단에서 제외했습니다. 해시 충돌 가능성은 이론적으로 남지만 매우 작습니다.</p></article>
      <article><h3>ID 안정성</h3><p>ID 10분위별 양성률의 전체 평균 대비 최대 차이는 <strong>{fmt_pct(max_id_delta,2)}</strong>입니다. ID가 생성 순서나 숨은 시간축을 담는지 계속 감시해야 합니다.</p>{dataframe_table(a['id_rates'], {'양성률'})}</article></section>
    <article style="margin-top:14px"><h3>결측·unknown 현황</h3>{dataframe_table(a['missing']) if len(a['missing']) else '<p>실제 결측과 unknown이 없습니다.</p>'}<p class="note"><code>pdays=-1</code>과 <code>previous=0</code>의 상태가 다른 행은 {fmt_int(state_disagreement)}개, <code>pdays=-1</code>과 <code>poutcome=unknown</code>의 상태가 다른 행은 {fmt_int(outcome_disagreement)}개입니다. 거의 같은 개념처럼 보여도 완전히 동일한 규칙으로 치환하면 정보가 손실됩니다.</p></article>

    <h2>어떤 변수가 타깃과 연결되는가</h2><section class="grid two"><article><h3>단변량 상호정보량 상위 10개</h3><p class="note">15만 행 표본에서 각 변수를 구간/범주로 바꿔 계산한 탐색용 지표입니다. 값이 높다고 인과관계이거나 모델 중요도가 확정되는 것은 아닙니다.</p>{bar_list([(r['변수'],float(r['상호정보량'])) for _,r in top_mi.iterrows()])}</article>
    <article><h3>train/test 차이 상위 10개</h3><p class="note">수치형은 KS, 범주형은 총변동거리(TV)를 사용했습니다. 서로 단위는 비슷하지만 같은 검정통계량은 아닙니다.</p>{bar_list([(r['변수'],float(r['차이 통계량'])) for _,r in drift_top.iterrows()])}</article></section>

    <h2>핵심 패턴</h2><section class="grid two"><article><h3>통화시간(duration) 10분위</h3><p class="note">각 구간의 양성률을 비교합니다. 강한 패턴이 검증 점수를 지배할 수 있어 사용 가능 시점이 핵심입니다.</p>{duration_html}</article>{''.join(cat_sections)}</section>

    <h2>수치형 변수의 범위</h2><article>{dataframe_table(a['num_stats'])}</article>

    <h2>권장 검증과 다음 실험</h2><article><ol>
      <li><strong>1순위: Stratified 5-Fold 기준선.</strong> 양성 비율을 폴드마다 유지하고, 대회의 평가 지표가 ROC AUC라면 OOF AUC로 비교합니다. 평가지표는 대회 페이지에서 별도 확인해야 합니다.</li>
      <li><strong>2순위: duration 포함/제외를 한 변수만 바꿔 비교.</strong> Kaggle 점수용 모델과 실제 사전 타기팅 모델의 차이를 학습할 수 있습니다.</li>
      <li><strong>3순위: pdays의 상태 분리.</strong> <code>previous_contacted = (pdays != -1)</code>를 추가하고 원래 pdays는 유지한 채 성능 변화를 봅니다.</li>
      <li><strong>4순위: 범주형 모델 기준선.</strong> CatBoost 또는 적절한 인코딩을 사용한 LightGBM 계열이 자연스럽습니다. 먼저 단순 기준선을 고정한 뒤 하나씩 바꿉니다.</li>
      <li><strong>5순위: 검증 안정성 확인.</strong> ID 구간별·월별 OOF 성능과 양성률을 확인해 무작위 폴드가 숨은 생성 순서를 놓치지 않는지 점검합니다.</li>
    </ol></article>
    <footer>분석 기준: 로컬 train.csv, test.csv, sample_submission.csv · 생성 스크립트: bank/src/eda_bank.py · 재현용 랜덤 시드: 326</footer></main></body></html>"""


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
    DOCS_PATH.write_text(make_html(analysis) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
