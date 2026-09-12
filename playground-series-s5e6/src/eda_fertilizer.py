"""Read-only EDA and mobile-friendly HTML report for fertilizer prediction."""

from __future__ import annotations

import hashlib
import html
import json
import sys
from itertools import combinations
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCAL_PACKAGES = ROOT / ".runtime_pkgs"
if LOCAL_PACKAGES.exists():
    sys.path.insert(0, str(LOCAL_PACKAGES))

import numpy as np
import pandas as pd


DATA_DIR = ROOT / "data"
SUMMARY_PATH = ROOT / "logs" / "result" / "fertilizer_eda_summary.json"
DOCS_PATH = ROOT / "docs" / "index.html"
REGISTRY_PATH = ROOT / "benchmarks" / "our_experiments.csv"
TARGET = "Fertilizer Name"
ID_COLUMN = "id"
FEATURES = [
    "Temparature", "Humidity", "Moisture", "Soil Type", "Crop Type",
    "Nitrogen", "Potassium", "Phosphorous",
]
NUMERIC = ["Temparature", "Humidity", "Moisture", "Nitrogen", "Potassium", "Phosphorous"]
CATEGORICAL = ["Soil Type", "Crop Type"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def total_variation(train: pd.Series, test: pd.Series) -> float:
    left = train.value_counts(normalize=True)
    right = test.value_counts(normalize=True)
    values = left.index.union(right.index)
    return float(0.5 * (left.reindex(values, fill_value=0) - right.reindex(values, fill_value=0)).abs().sum())


def grouped_top1_accuracy(frame: pd.DataFrame, columns: list[str]) -> float:
    counts = frame.groupby(columns + [TARGET], observed=True).size().unstack(fill_value=0)
    return float(counts.max(axis=1).sum() / len(frame))


def analyze() -> dict:
    paths = {name: DATA_DIR / name for name in ["train.csv", "test.csv", "sample_submission.csv"]}
    train = pd.read_csv(paths["train.csv"])
    test = pd.read_csv(paths["test.csv"])
    sample = pd.read_csv(paths["sample_submission.csv"])

    expected_train = [ID_COLUMN, *FEATURES, TARGET]
    expected_test = [ID_COLUMN, *FEATURES]
    if list(train.columns) != expected_train or list(test.columns) != expected_test:
        raise ValueError("Competition CSV schema differs from the expected schema")
    if len(sample) != len(test) or not sample[ID_COLUMN].equals(test[ID_COLUMN]):
        raise ValueError("sample_submission IDs do not match test IDs")

    target_counts = train[TARGET].value_counts()
    target_rates = target_counts / len(train)
    prior_map3 = float(target_rates.iloc[0] + target_rates.iloc[1] / 2 + target_rates.iloc[2] / 3)
    missing = pd.DataFrame({
        "feature": expected_train,
        "train_missing": [int(train[column].isna().sum()) for column in expected_train],
        "test_missing": [int(test[column].isna().sum()) if column in test else None for column in expected_train],
    })
    cardinality = pd.DataFrame({
        "feature": FEATURES,
        "train_unique": [int(train[column].nunique()) for column in FEATURES],
        "test_unique": [int(test[column].nunique()) for column in FEATURES],
    })
    drift = pd.DataFrame([
        {"feature": column, "total_variation": total_variation(train[column], test[column])}
        for column in FEATURES
    ]).sort_values("total_variation", ascending=False)
    unseen = {
        column: sorted(map(str, set(test[column].dropna()) - set(train[column].dropna())))
        for column in CATEGORICAL
    }

    single_strength = pd.DataFrame([
        {"features": column, "order": 1, "in_sample_top1": grouped_top1_accuracy(train, [column])}
        for column in FEATURES
    ]).sort_values("in_sample_top1", ascending=False)
    pair_rows = []
    for left, right in combinations(FEATURES, 2):
        pair_rows.append({
            "features": f"{left} + {right}", "order": 2,
            "in_sample_top1": grouped_top1_accuracy(train, [left, right]),
        })
    pair_strength = pd.DataFrame(pair_rows).sort_values("in_sample_top1", ascending=False)

    id_bucket = pd.qcut(train[ID_COLUMN], 10, labels=False)
    id_rates = pd.crosstab(id_bucket, train[TARGET], normalize="index")
    max_id_delta = float((id_rates.max(axis=0) - id_rates.min(axis=0)).max())
    class_means = train.groupby(TARGET)[NUMERIC].mean().round(3)

    train_hashes = pd.util.hash_pandas_object(train[FEATURES], index=False)
    test_hashes = pd.util.hash_pandas_object(test[FEATURES], index=False)
    exact_cross_matches = int(test_hashes.isin(set(train_hashes)).sum())
    feature_duplicates = int(train.duplicated(FEATURES).sum())
    full_duplicates = int(train.drop(columns=ID_COLUMN).duplicated().sum())

    return {
        "train": train,
        "test": test,
        "sample": sample,
        "target_counts": target_counts,
        "target_rates": target_rates,
        "prior_map3": prior_map3,
        "missing": missing,
        "cardinality": cardinality,
        "drift": drift,
        "unseen": unseen,
        "single_strength": single_strength,
        "pair_strength": pair_strength,
        "max_id_delta": max_id_delta,
        "class_means": class_means,
        "feature_duplicates": feature_duplicates,
        "full_duplicates": full_duplicates,
        "exact_cross_matches": exact_cross_matches,
        "hashes": {name: sha256_file(path) for name, path in paths.items()},
    }


def esc(value) -> str:
    return html.escape(str(value))


def pct(value, digits=2) -> str:
    return f"{float(value) * 100:.{digits}f}%"


def num(value) -> str:
    return f"{int(value):,}"


def bars(rows, label_key, value_key, scale=None) -> str:
    rows = list(rows)
    maximum = scale or max(float(row[value_key]) for row in rows) or 1
    return "".join(
        f'<div class="bar-row"><span>{esc(row[label_key])}</span>'
        f'<span class="bar-track"><i style="width:{min(100, float(row[value_key]) / maximum * 100):.2f}%"></i></span>'
        f'<strong>{float(row[value_key]):.4f}</strong></div>'
        for row in rows
    )


def make_table(frame: pd.DataFrame, percent_columns=()) -> str:
    if frame.empty:
        return "<p>표시할 데이터가 없습니다.</p>"
    parts = ["<div class=\"table-wrap\"><table><thead><tr>"]
    parts.extend(f"<th>{esc(column)}</th>" for column in frame.columns)
    parts.append("</tr></thead><tbody>")
    for _, row in frame.iterrows():
        parts.append("<tr>")
        for column in frame.columns:
            value = row[column]
            if column in percent_columns and pd.notna(value):
                value = pct(value)
            elif isinstance(value, (float, np.floating)):
                value = f"{value:.4f}"
            parts.append(f"<td>{esc(value)}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table></div>")
    return "".join(parts)


def experiment_table() -> tuple[str, str, str]:
    if not REGISTRY_PATH.exists():
        return "<p>실험 기록이 없습니다.</p>", "기준 모델 대기", "—"
    frame = pd.read_csv(REGISTRY_PATH, keep_default_na=False)
    completed = frame[frame["status"].isin(["완료", "스모크 완료"])].copy()
    completed["score"] = pd.to_numeric(completed["cv_map3"], errors="coerce")
    full = completed[completed["status"].eq("완료") & completed["score"].notna()]
    if len(full):
        best = full.sort_values("score", ascending=False).iloc[0]
        best_text, best_score = f"현재 최고 · {best['experiment_id']}", f"{best['score']:.6f}"
    elif len(completed):
        best = completed.sort_values("score", ascending=False).iloc[0]
        best_text, best_score = f"스모크 · {best['experiment_id']}", f"{best['score']:.6f}"
    else:
        best_text, best_score = "기준 모델 대기", "—"
    display = frame[[
        "experiment_id", "purpose", "changed_element", "model", "cv_map3",
        "fold_std", "log_loss", "status",
    ]].rename(columns={
        "experiment_id": "실험", "purpose": "목적", "changed_element": "한 가지 변경",
        "model": "모델", "cv_map3": "OOF MAP@3", "fold_std": "Fold 편차",
        "log_loss": "Log loss", "status": "상태",
    })
    return make_table(display), best_text, best_score


def class_diagnostic_table() -> str:
    metrics_path = ROOT / "outputs" / "E001" / "metrics.json"
    if not metrics_path.exists():
        return "<p>E001 완료 후 클래스별 결과가 표시됩니다.</p>"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if metrics.get("smoke") or not metrics.get("per_class"):
        return "<p>E001 전체 실행 완료 후 클래스별 결과가 표시됩니다.</p>"
    frame = pd.DataFrame(metrics["per_class"]).rename(columns={
        "label": "비료", "rows": "행 수", "top1_accuracy": "1순위 정답률",
        "top3_recall": "상위 3개 포함률",
    })
    return make_table(frame, {"1순위 정답률", "상위 3개 포함률"})


def interaction_experiment_section(train: pd.DataFrame) -> str:
    baseline_path = ROOT / "outputs" / "E001" / "metrics.json"
    interaction_path = ROOT / "outputs" / "E004" / "metrics.json"
    partial_te_path = ROOT / "outputs" / "E005" / "partial_run.json"
    if not baseline_path.exists() or not interaction_path.exists():
        return "<p>E004 전체 실행 완료 후 상호작용 비교가 표시됩니다.</p>"

    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    interaction = json.loads(interaction_path.read_text(encoding="utf-8"))
    selected_pairs = [
        ("Nitrogen", "Phosphorous"),
        ("Moisture", "Phosphorous"),
        ("Nitrogen", "Potassium"),
        ("Potassium", "Phosphorous"),
        ("Moisture", "Nitrogen"),
    ]
    pair_rows = []
    for left, right in selected_pairs:
        groups = train.groupby([left, right], observed=True).size()
        pair_rows.append({
            "조합": f"{left} × {right}",
            "고유 조합": int(len(groups)),
            "조합당 평균 행": float(groups.mean()),
            "최소 행": int(groups.min()),
        })
    pair_table = make_table(pd.DataFrame(pair_rows))

    base_classes = {row["label"]: row for row in baseline["per_class"]}
    class_rows = []
    for row in interaction["per_class"]:
        old = base_classes[row["label"]]
        class_rows.append({
            "비료": row["label"],
            "E001 Top-3": old["top3_recall"],
            "E004 Top-3": row["top3_recall"],
            "변화": row["top3_recall"] - old["top3_recall"],
        })
    class_table = make_table(
        pd.DataFrame(class_rows), {"E001 Top-3", "E004 Top-3", "변화"},
    )

    score_gain = interaction["cv_map3"] - baseline["cv_map3"]
    loss_gain = interaction["log_loss"] - baseline["log_loss"]
    runtime_ratio = interaction["runtime_minutes"] / baseline["runtime_minutes"]
    te_html = "<p>E005 부분 실행 기록이 없습니다.</p>"
    if partial_te_path.exists():
        partial = json.loads(partial_te_path.read_text(encoding="utf-8"))
        fold_text = " · ".join(f"{value:.6f}" for value in partial["fold_scores"])
        te_html = f"""
        <div class="card warning"><h3>E005 · Target encoding은 중단</h3>
        <div class="metric">{partial['observed_mean_map3']:.6f}</div>
        <p class="note">완료된 4개 fold의 관측 평균이며 정식 OOF 점수가 아닙니다.</p>
        <p>Fold 점수: <code>{fold_text}</code></p>
        <p>원 변수 8개와 쌍 5개를 클래스별 확률로 바꾼 91개 특성을 추가했지만,
        best iteration이 1·1·3·1에서 멈췄습니다. CatBoost의 자체 범주 통계와 정보가
        중복되고 강하게 상관된 확률 특성이 한꺼번에 들어가 학습을 방해했을 가능성이 큽니다.</p>
        <p class="note">사용자 요청으로 마지막 fold를 중단했습니다. 따라서 E004와 같은
        완성된 5-fold 결과로 비교하거나 모델 선택에 사용하지 않습니다.</p></div>"""

    return f"""
    <section class="grid two"><article class="card finding"><h3>E004 · 쌍 상호작용 채택 후보</h3>
    <div class="metric">{interaction['cv_map3']:.6f}</div>
    <p>E001 대비 MAP@3 <strong>{score_gain:+.6f}</strong>, log loss
    <strong>{loss_gain:+.6f}</strong>입니다. 다섯 fold 모두 0.318754~0.320777로
    같은 개선 방향을 보였습니다.</p><p class="note">실행 시간 {interaction['runtime_minutes']:.2f}분 ·
    E001의 {runtime_ratio:.1f}배 · 모든 fold가 180 iteration 한도 도달</p></article>
    {te_html}</section>
    <h3 style="margin-top:18px">선택한 조합의 표본 밀도</h3>{pair_table}
    <p class="note">각 조합에 평균 수백 행이 있어 쌍 수준 통계는 충분히 안정적입니다.
    다만 이를 7개 클래스 확률 91개로 동시에 확장하는 것은 별개의 문제입니다.</p>
    <h3 style="margin-top:18px">클래스별 Top-3 recall 변화</h3>{class_table}
    <p class="callout">DAP는 18.59%→26.76%, Urea는 9.72%→19.36%로 개선됐습니다.
    반면 10-26-26과 14-35-14는 하락했습니다. 전체 점수 상승은 모든 클래스를 똑같이
    개선한 결과가 아니라, 기존에 거의 잡지 못했던 소수 클래스로 확률 순위가 재배치된 결과입니다.</p>
    """


def make_html(a: dict) -> str:
    experiment_html, best_text, best_score = experiment_table()
    class_html = class_diagnostic_table()
    interaction_html = interaction_experiment_section(a["train"])
    target = pd.DataFrame({
        "비료": a["target_counts"].index,
        "행 수": a["target_counts"].values,
        "비율": a["target_rates"].values,
    })
    target_bars = bars(
        [{"label": row["비료"], "value": row["비율"]} for _, row in target.iterrows()],
        "label", "value", scale=float(target["비율"].max()),
    )
    drift_rows = a["drift"].head(8).to_dict(orient="records")
    single_rows = a["single_strength"].head(8).to_dict(orient="records")
    pair_rows = a["pair_strength"].head(8).to_dict(orient="records")
    numeric_summary = a["train"][NUMERIC].describe().T.reset_index().rename(columns={
        "index": "특성", "count": "행 수", "mean": "평균", "std": "표준편차",
        "min": "최소", "25%": "25%", "50%": "중앙", "75%": "75%", "max": "최대",
    })
    css = """
    :root{--navy:#10273f;--blue:#176b9c;--cyan:#27a8a1;--paper:#f3f7fa;--card:#fff;--line:#d7e2e9;--muted:#566b7c;--amber:#b86912}
    *{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--paper);color:var(--navy);font:16px/1.65 Inter,"Noto Sans KR",system-ui,sans-serif}
    main{width:min(1160px,calc(100% - 32px));margin:auto;padding:34px 0 72px}.top{display:flex;justify-content:space-between;gap:18px;align-items:center;margin-bottom:26px}
    .crumb{font-size:.86rem;color:var(--muted)}a{color:var(--blue)}h1{font-size:clamp(2.25rem,6vw,4.7rem);line-height:1.02;letter-spacing:-.055em;margin:.35rem 0 1rem;max-width:920px}
    .lead{font-size:1.12rem;color:var(--muted);max-width:820px}h2{font-size:1.75rem;letter-spacing:-.03em;margin:3.5rem 0 1rem}h3{margin:0 0 .6rem;font-size:1.05rem}
    .grid{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}.two{grid-template-columns:repeat(2,1fr)}.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:19px;box-shadow:0 8px 26px rgba(16,39,63,.05)}
    .metric{font-size:1.85rem;font-weight:800;letter-spacing:-.04em}.label,.note{font-size:.84rem;color:var(--muted)}.finding{border-top:4px solid var(--cyan)}.warning{border-top-color:var(--amber)}
    .flow{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}.step{background:var(--navy);color:white;border-radius:12px;padding:16px}.step small{display:block;color:#a8c7d9;margin-bottom:5px}
    .quick-nav{display:flex;flex-wrap:wrap;gap:8px;margin:22px 0 4px}.quick-nav a{padding:7px 11px;border:1px solid var(--line);border-radius:999px;background:white;text-decoration:none;font-size:.82rem;font-weight:700}
    .feature-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:12px}.feature-card{background:white;border:1px solid var(--line);border-radius:12px;padding:16px}.feature-card h3{display:flex;justify-content:space-between;gap:12px}.type{font-size:.72rem;color:var(--blue);background:#e8f3f8;border-radius:999px;padding:3px 8px;white-space:nowrap}.feature-card p{margin:.45rem 0}.feature-card .expected{color:var(--muted);font-size:.88rem}
    .metric-guide td:nth-child(2),.metric-guide td:nth-child(3){text-align:left;white-space:normal}.metric-guide th:nth-child(2),.metric-guide th:nth-child(3){text-align:left}.roadmap{display:grid;gap:10px}.roadmap-item{display:grid;grid-template-columns:94px 1fr;gap:14px;align-items:start;background:white;border:1px solid var(--line);border-radius:12px;padding:15px}.status{display:inline-block;text-align:center;border-radius:999px;padding:4px 8px;font-size:.74rem;font-weight:800}.done{background:#dff4ef;color:#08796f}.now{background:#dfeefa;color:#176b9c}.next{background:#f4eadc;color:#92520c}.later{background:#edf0f2;color:#60717d}.roadmap-item p{margin:3px 0;color:var(--muted)}
    .bar-row{display:grid;grid-template-columns:150px 1fr 62px;gap:10px;align-items:center;margin:10px 0;font-size:.82rem}.bar-track{height:9px;background:#e7eef3;border-radius:8px;overflow:hidden}.bar-track i{display:block;height:100%;background:linear-gradient(90deg,var(--blue),var(--cyan));border-radius:8px}.bar-row strong{text-align:right;font-variant-numeric:tabular-nums}
    .table-wrap{overflow:auto;border:1px solid var(--line);border-radius:12px}table{border-collapse:collapse;width:100%;background:white;font-size:.82rem}th,td{padding:9px 11px;border-bottom:1px solid var(--line);white-space:nowrap;text-align:right}th{background:#eaf1f5;color:var(--muted)}th:first-child,td:first-child{text-align:left}
    ol,ul{padding-left:1.25rem}li{margin:.45rem 0}code{background:#e6eef4;padding:.12rem .35rem;border-radius:5px}.callout{border-left:4px solid var(--blue);padding:3px 0 3px 16px;margin:18px 0}
    footer{margin-top:52px;padding-top:20px;border-top:1px solid var(--line);font-size:.8rem;color:var(--muted)}
    @media(max-width:820px){.grid,.flow,.two,.feature-grid{grid-template-columns:1fr 1fr}.top{align-items:flex-start}}
    @media(max-width:560px){main{padding-top:22px}.grid,.flow,.two,.feature-grid{grid-template-columns:1fr}.bar-row{grid-template-columns:105px 1fr 55px}h2{margin-top:2.7rem}.roadmap-item{grid-template-columns:78px 1fr}}
    """
    return f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="description" content="Kaggle 비료 추천 데이터 분석과 실험 계획"><title>비료 추천 데이터 분석</title><style>{css}</style></head><body><main>
    <div class="top"><div class="crumb">Kaggle Data Analysis Study · Playground S5E6</div><a href="../">전체 대회</a></div>
    <h1>비료 추천 데이터에서 무엇을 먼저 배워야 할까</h1>
    <p class="lead">개별 변수의 평균만 보면 클래스 차이가 거의 보이지 않습니다. 이 데이터의 핵심은 영양소·수분·작물 조건의 조합과 그 조합을 누수 없이 검증하는 방법입니다.</p>
    <section class="grid"><div class="card"><div class="metric">{num(len(a['train']))}</div><div class="label">학습 행</div></div><div class="card"><div class="metric">{num(len(a['test']))}</div><div class="label">테스트 행</div></div><div class="card"><div class="metric">8 → 7</div><div class="label">예측 특성 → 클래스</div></div><div class="card"><div class="metric">{a['prior_map3']:.6f}</div><div class="label">빈도 기준선 MAP@3</div></div></section>
    <nav class="quick-nav" aria-label="보고서 바로가기"><a href="#features">Feature 설명</a><a href="#objective">목표와 점수</a><a href="#roadmap">전체 로드맵</a><a href="#experiments">실험 결과</a></nav>

    <h2 id="features">Feature를 어떻게 읽고 모델에 넣는가</h2>
    <p class="callout"><strong>Feature</strong>는 비료를 고를 때 모델이 참고하는 입력 정보입니다. 이 데이터에는 환경 3개, 재배 조건 2개, 영양소 3개가 있습니다. 아래의 “예상 영향”은 실험 전 가설이며, 높고 낮음 자체가 좋다는 뜻은 아닙니다. 어떤 작물·토양·다른 영양소와 함께 나타나는지가 핵심입니다.</p>
    <section class="feature-grid">
      <article class="feature-card"><h3>Temparature <span class="type">수치 · 14값</span></h3><p>재배 환경의 온도 조건입니다. 원본 열 이름의 철자를 그대로 사용합니다.</p><p class="expected"><strong>예상:</strong> 작물과 습도에 따라 적합한 비료 조건이 달라질 수 있어 <code>온도 × 작물</code>, <code>온도 × 습도</code>를 후보로 봅니다.</p></article>
      <article class="feature-card"><h3>Humidity <span class="type">수치 · 23값</span></h3><p>공기 중 습도 조건을 나타내는 값입니다.</p><p class="expected"><strong>예상:</strong> 온도 및 작물 조건과 함께 비료 선택에 간접 영향을 줄 수 있습니다. 단독 신호는 실제 탐색에서 약했습니다.</p></article>
      <article class="feature-card"><h3>Moisture <span class="type">수치 · 41값</span></h3><p>토양의 수분 상태를 나타내는 값입니다.</p><p class="expected"><strong>예상:</strong> 영양소 상태와 결합할 때 비료 요구량을 더 잘 구분할 수 있습니다. 실제로 단일 feature 중 탐색 구분력이 가장 컸고, 인·질소와의 쌍도 강했습니다.</p></article>
      <article class="feature-card"><h3>Soil Type <span class="type">범주 · 5종</span></h3><p>토양 종류입니다. 숫자 크기로 비교하지 않고 이름이 다른 그룹으로 처리합니다.</p><p class="expected"><strong>예상:</strong> 같은 영양소 값도 토양 종류에 따라 의미가 달라질 수 있어 <code>토양 × 영양소</code> 상호작용 후보입니다.</p></article>
      <article class="feature-card"><h3>Crop Type <span class="type">범주 · 11종</span></h3><p>재배 작물 종류입니다. 순서가 없는 범주형 feature입니다.</p><p class="expected"><strong>예상:</strong> 작물별 필요한 영양 조건이 다를 수 있으므로 환경·토양·영양소 효과의 기준을 바꾸는 역할을 기대합니다.</p></article>
      <article class="feature-card"><h3>Nitrogen <span class="type">수치 · 39값</span></h3><p>질소 상태를 나타내는 값입니다.</p><p class="expected"><strong>예상:</strong> 인·칼륨과의 균형이 비료 종류 구분에 중요합니다. 실제 <code>질소 × 인</code> 조합이 두-feature 탐색에서 가장 강했습니다.</p></article>
      <article class="feature-card"><h3>Potassium <span class="type">수치 · 20값</span></h3><p>칼륨 상태를 나타내는 값입니다.</p><p class="expected"><strong>예상:</strong> 질소·인과 함께 N-P-K 구성 차이를 표현합니다. 단독보다 <code>칼륨 × 인</code>, <code>질소 × 칼륨</code> 조합을 우선 검증합니다.</p></article>
      <article class="feature-card"><h3>Phosphorous <span class="type">수치 · 43값</span></h3><p>인 상태를 나타내는 값입니다. 원본 열 이름의 철자를 그대로 사용합니다.</p><p class="expected"><strong>예상:</strong> 질소·수분과 결합한 구분력이 강했습니다. E004에서 이 조합들을 별도 범주로 표현해 성능이 개선됐습니다.</p></article>
    </section>
    <article class="card" style="margin-top:14px"><h3>같은 feature도 여러 방식으로 표현할 수 있습니다</h3><ul><li><strong>원 수치(raw numeric):</strong> 값의 크기와 가까운 정도를 그대로 사용합니다. 현재 기준선의 기본 표현입니다.</li><li><strong>범주(category):</strong> 값 사이의 거리보다 “서로 다른 상태”로 봅니다. 고유값이 14~43개로 적어 별도 실험 후보지만, 아직 전체 범주화 결과는 확정되지 않았습니다.</li><li><strong>상호작용(interaction):</strong> 두 값을 하나의 조건으로 묶습니다. E004는 영양소·수분 쌍 5개를 추가해 MAP@3을 0.309568에서 0.319455로 높였으므로 현재 채택한 표현입니다.</li><li><strong>Target encoding:</strong> 각 조건에서 비료별 빈도를 확률 feature로 바꿉니다. 누수 방지가 필수이며, 91개를 한꺼번에 추가한 E005는 성능이 하락해 보류했습니다.</li><li><strong>ID:</strong> 행 식별자일 뿐 실제 예측 시점의 원인 정보가 아니므로 모델 입력에서 제외합니다.</li></ul><p class="note">원본 데이터에 단위 설명이 없어 온도·습도·영양소 값에 임의 단위를 붙이지 않았습니다. 또한 이 분석은 연관성을 검증하며 인과효과를 주장하지 않습니다.</p></article>

    <h2 id="objective">무엇을 예측하며, 어느 방향이 좋은가</h2>
    <section class="grid two"><article class="card"><h3>Target: Fertilizer Name</h3><p>각 행에서 정답은 7종 비료 중 <strong>하나</strong>입니다. 그러나 제출할 때는 가능성이 높은 비료 <strong>3개를 서로 다르게, 확률 순서대로</strong> 적습니다. 클래스 이름에는 크고 작음의 순서가 없습니다.</p></article><article class="card finding"><h3>최종 목표</h3><div class="metric">MAP@3 ↑ 1.0</div><p><strong>클수록 좋고 이론적 최고는 1</strong>입니다. 모든 행의 정답을 첫 번째 후보로 맞히면 1점입니다. 정답이 2위면 0.5점, 3위면 약 0.333점, 상위 3개 밖이면 0점입니다.</p></article></section>
    <div class="table-wrap metric-guide" style="margin-top:14px"><table><thead><tr><th>지표</th><th>좋은 방향</th><th>이 보고서에서 읽는 법</th></tr></thead><tbody><tr><td><strong>OOF MAP@3</strong></td><td><strong>클수록 좋음 · 최고 1</strong></td><td>모델 선택의 주지표입니다. 현재 최고 E004는 0.319455이며 빈도 기준선 0.278485보다 높습니다.</td></tr><tr><td><strong>Log loss</strong></td><td><strong>작을수록 좋음 · 최저 0</strong></td><td>정답에 높은 확률을 주는지 봅니다. 틀린 답을 지나치게 확신하면 큰 벌점을 받습니다.</td></tr><tr><td><strong>Fold 표준편차</strong></td><td><strong>작을수록 안정적 · 최저 0</strong></td><td>데이터를 다섯 번 나눴을 때 점수가 얼마나 흔들리는지 봅니다. 낮으면 우연한 분할에 덜 민감합니다.</td></tr><tr><td><strong>Top-3 recall</strong></td><td><strong>클수록 좋음 · 최고 100%</strong></td><td>특정 비료가 정답일 때 상위 3개 후보 안에 들어간 비율입니다. 취약한 클래스를 찾는 진단값입니다.</td></tr><tr><td><strong>Train/test 총변동거리</strong></td><td><strong>0에 가까울수록 유사</strong></td><td>두 데이터의 값 분포 차이입니다. 예측 점수가 아니라 분포 이동 위험을 보는 진단값입니다.</td></tr></tbody></table></div>

    <h2 id="roadmap">전체 진행 로드맵</h2>
    <p class="callout"><strong>현재 위치:</strong> 기본 검증과 기준선을 확정했고, feature 표현 실험에서 E004 쌍 상호작용을 채택한 상태입니다. 다음부터도 한 번에 한 요소만 바꿔 개선 원인을 분리합니다.</p>
    <section class="roadmap"><article class="roadmap-item"><span class="status done">완료</span><div><strong>1. 문제·데이터 이해</strong><p>타깃, MAP@3 제출 형식, 결측·중복·범주와 train/test 분포를 확인했습니다.</p></div></article><article class="roadmap-item"><span class="status done">완료</span><div><strong>2. 검증과 기준선 고정</strong><p>seed 326 Stratified 5-Fold를 고정하고 빈도 기준선 E000, CatBoost 기준선 E001을 만들었습니다.</p></div></article><article class="roadmap-item"><span class="status done">채택</span><div><strong>3. Feature 표현·상호작용</strong><p>E004의 영양소·수분 쌍이 성능을 개선했습니다. 대규모 target encoding E005는 악화되어 중단했습니다.</p></div></article><article class="roadmap-item"><span class="status now">현재</span><div><strong>4. 오류 분석과 학습량 점검</strong><p>DAP·Urea 개선과 다른 클래스 하락의 원인을 보고, E004 iteration 상한만 바꾸는 실험을 우선 후보로 둡니다.</p></div></article><article class="roadmap-item"><span class="status next">다음</span><div><strong>5. 독립 모델 비교</strong><p>같은 fold에서 LightGBM 또는 제한된 범주 표현을 하나씩 비교해 CatBoost와 다른 오류를 만드는지 확인합니다.</p></div></article><article class="roadmap-item"><span class="status later">이후</span><div><strong>6. 앙상블·제출 검증</strong><p>완료 모델의 OOF 확률 평균이 실제로 개선될 때만 앙상블하고, 제출 형식과 Kaggle 점수를 확인합니다.</p></div></article><article class="roadmap-item"><span class="status later">반복</span><div><strong>7. 결과 해석 → 새 가설</strong><p>OOF와 Kaggle 결과가 다르면 분포 차이와 과적합을 점검하고, 근거가 있는 다음 feature 실험으로 돌아갑니다.</p></div></article></section>

    <h2>확인 → 발견 → 의미 → 다음 행동</h2><section class="flow"><div class="step"><small>01 · 확인</small>결측, 중복, 범주, train/test 분포와 ID 순서를 검사했습니다.</div><div class="step"><small>02 · 발견</small>결측과 미등록 범주는 없고 단변량 분포 이동도 매우 작습니다.</div><div class="step"><small>03 · 의미</small>정제보다 변수 조합과 확률 순위 품질이 성능을 좌우합니다.</div><div class="step"><small>04 · 다음</small>고정된 5-Fold에서 기준선부터 한 요소씩 바꿉니다.</div></section>

    <h2 id="experiments">현재 실험 상태</h2><section class="grid two"><article class="card finding"><h3>{esc(best_text)}</h3><div class="metric">{best_score}</div><p class="note">주지표: seed 326 Stratified 5-Fold OOF MAP@3 · 클수록 좋음 · 최고 1</p></article><article class="card"><h3>현재 점수의 의미</h3><p>E004의 0.319455는 “31.95% 정확도”가 아닙니다. 각 행에서 정답이 몇 번째 후보에 놓였는지를 1, 0.5, 약 0.333, 0점으로 환산한 평균입니다.</p></article></section><div style="margin-top:14px">{experiment_html}</div>

    <h2>타깃 분포</h2><section class="grid two"><article class="card"><h3>7개 비료의 비율</h3>{target_bars}</article><article class="card warning"><h3>심한 불균형은 아닙니다</h3><p>가장 많은 클래스는 {pct(target['비율'].max())}, 가장 적은 클래스는 {pct(target['비율'].min())}입니다. 단순 정확도 왜곡보다 각 클래스가 상위 3개 안에서 어느 순위에 놓이는지 확인하는 편이 중요합니다.</p><p class="note">빈도 상위 3개만 고정 예측하면 MAP@3은 {a['prior_map3']:.6f}입니다.</p></article></section>

    <h2>데이터 품질과 분포 이동</h2><section class="grid two"><article class="card"><h3>품질 검사</h3><ul><li>결측값: {num(int(a['missing']['train_missing'].sum()))}건</li><li>학습 특성 중복 행: {num(a['feature_duplicates'])}건</li><li>ID 제외 완전 중복 행: {num(a['full_duplicates'])}건</li><li>test에만 있는 범주: {sum(len(v) for v in a['unseen'].values())}개</li><li>train과 완전히 같은 test 특성 행: {num(a['exact_cross_matches'])}건</li></ul></article><article class="card"><h3>train/test 총변동거리</h3><p class="note">0에 가까울수록 두 데이터의 값 비율이 비슷합니다.</p>{bars(drift_rows,'feature','total_variation',scale=max(0.01,float(a['drift']['total_variation'].max())))}</article></section>
    <p class="callout">ID 10분위 사이 클래스 비율의 최대 변화는 {pct(a['max_id_delta'])}입니다. 강한 순서 패턴은 보이지 않아 ID는 기본 모델에서 제외하고 안정성 진단에만 사용합니다.</p>

    <h2>왜 상호작용을 봐야 하는가</h2><section class="grid two"><article class="card"><h3>단일 특성의 최적 분류 정확도</h3><p class="note">각 값에서 가장 흔한 클래스를 고른 탐색용 상한입니다. 교차검증 점수가 아닙니다.</p>{bars(single_rows,'features','in_sample_top1',scale=.20)}</article><article class="card"><h3>두 특성 조합 상위</h3><p class="note">조합값에서 가장 흔한 클래스를 고른 학습 내 수치입니다. 개선 여부는 OOF에서 다시 확인해야 합니다.</p>{bars(pair_rows,'features','in_sample_top1',scale=.22)}</article></section>
    <p class="callout">단일 특성 최고치는 {a['single_strength'].iloc[0]['in_sample_top1']:.4f}지만 두 특성 조합은 {a['pair_strength'].iloc[0]['in_sample_top1']:.4f}까지 올라갑니다. 그래서 첫 피처 실험은 영양소와 수분의 쌍 조합이며, 타깃 인코딩은 반드시 fold 안에서만 계산합니다.</p>

    <h2>상호작용과 Target encoding 실험 결과</h2>{interaction_html}

    <h2>수치형 특성의 실제 형태</h2><article class="card">{make_table(numeric_summary)}<p class="note">이름은 수치형이지만 고유값 수가 14~43개뿐입니다. 원값을 연속량으로 처리한 기준선과 모든 값을 범주로 처리한 실험을 분리해 비교합니다.</p></article>

    <h2>E001은 어떤 클래스를 어려워하는가</h2><article class="card">{class_html}<p class="note">전체 정확도가 아니라 각 정답 클래스가 1순위 또는 상위 3개에 들어간 비율입니다. DAP와 Urea의 낮은 포함률은 다음 오류 분석의 최우선 대상입니다.</p></article>

    <h2>실험 판단</h2><article class="card"><ol><li><strong>E004 쌍 상호작용:</strong> 5-fold MAP@3이 0.319455로 개선되어 현재 최고 모델로 유지합니다.</li><li><strong>E005 target encoding:</strong> 누수 없이 구현했지만 4개 fold 모두 E004보다 낮아 중단했습니다. 정식 OOF 결과로 취급하지 않습니다.</li><li><strong>후속 후보:</strong> target encoding을 다시 볼 때는 강한 쌍 하나만 추가하거나 CatBoost가 아닌 별도 모델에서 검증합니다.</li><li><strong>학습량 후보:</strong> E004의 모든 fold가 180회 한도에 도달했으므로 iteration 상한 변경은 별도 실험으로 분리합니다.</li></ol></article>

    <h2>현재 결론</h2><section class="grid two"><article class="card finding"><h3>채택 · 쌍 상호작용</h3><p>명시적 영양소·수분 조합은 MAP@3과 log loss를 함께 개선했고 fold 편차도 작았습니다. 현재 최고 모델은 E004입니다.</p></article><article class="card warning"><h3>보류 · 일괄 target encoding</h3><p>누수를 막아도 91개 확률 특성을 동시에 추가하면 성능이 크게 악화됐습니다. 기법 이름보다 모델과 표현의 궁합이 중요합니다.</p></article><article class="card"><h3>읽어야 할 변화</h3><p>E004는 DAP와 Urea를 크게 개선했지만 일부 다수 클래스의 top-3 recall은 낮췄습니다. 평균 점수만 보지 않고 클래스별 순위 이동을 함께 봐야 합니다.</p></article><article class="card"><h3>다음 우선순위</h3><p>현재 요청 범위에서는 E004를 유지합니다. 후속 실험은 한 번에 강한 쌍 하나의 target encoding 또는 iteration 상한 중 하나만 바꾸는 순서가 적절합니다.</p></article></section>

    <footer>데이터: 로컬 competition CSV · <a href="https://www.kaggle.com/competitions/playground-series-s5e6" target="_blank" rel="noopener noreferrer">Kaggle 대회 설명</a> · <a href="https://www.kaggle.com/competitions/playground-series-s5e6/writeups/chris-deotte-1st-place-fast-gpu-experimentation-wi" target="_blank" rel="noopener noreferrer">공개 1위 해법</a> · 생성 스크립트: src/eda_fertilizer.py · 원본 CSV는 변경하지 않음</footer>
    </main></body></html>"""


def serializable_summary(a: dict) -> dict:
    return {
        "train_shape": list(a["train"].shape),
        "test_shape": list(a["test"].shape),
        "target_counts": {str(k): int(v) for k, v in a["target_counts"].items()},
        "target_rates": {str(k): float(v) for k, v in a["target_rates"].items()},
        "frequency_prior_map3": a["prior_map3"],
        "missing_train": int(a["missing"]["train_missing"].sum()),
        "cardinality": a["cardinality"].to_dict(orient="records"),
        "train_test_total_variation": a["drift"].to_dict(orient="records"),
        "unseen_test_categories": a["unseen"],
        "single_feature_strength": a["single_strength"].to_dict(orient="records"),
        "pair_feature_strength_top10": a["pair_strength"].head(10).to_dict(orient="records"),
        "max_target_rate_delta_across_id_deciles": a["max_id_delta"],
        "duplicate_feature_rows_train": a["feature_duplicates"],
        "duplicate_rows_excluding_id_train": a["full_duplicates"],
        "exact_feature_matches_in_test": a["exact_cross_matches"],
        "source_sha256": a["hashes"],
    }


def main() -> None:
    analysis = analyze()
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOCS_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary = serializable_summary(analysis)
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report = "\n".join(line.rstrip() for line in make_html(analysis).splitlines()) + "\n"
    DOCS_PATH.write_text(report, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
