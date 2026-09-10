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


def make_html(a: dict) -> str:
    experiment_html, best_text, best_score = experiment_table()
    class_html = class_diagnostic_table()
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
    .bar-row{display:grid;grid-template-columns:150px 1fr 62px;gap:10px;align-items:center;margin:10px 0;font-size:.82rem}.bar-track{height:9px;background:#e7eef3;border-radius:8px;overflow:hidden}.bar-track i{display:block;height:100%;background:linear-gradient(90deg,var(--blue),var(--cyan));border-radius:8px}.bar-row strong{text-align:right;font-variant-numeric:tabular-nums}
    .table-wrap{overflow:auto;border:1px solid var(--line);border-radius:12px}table{border-collapse:collapse;width:100%;background:white;font-size:.82rem}th,td{padding:9px 11px;border-bottom:1px solid var(--line);white-space:nowrap;text-align:right}th{background:#eaf1f5;color:var(--muted)}th:first-child,td:first-child{text-align:left}
    ol,ul{padding-left:1.25rem}li{margin:.45rem 0}code{background:#e6eef4;padding:.12rem .35rem;border-radius:5px}.callout{border-left:4px solid var(--blue);padding:3px 0 3px 16px;margin:18px 0}
    footer{margin-top:52px;padding-top:20px;border-top:1px solid var(--line);font-size:.8rem;color:var(--muted)}
    @media(max-width:820px){.grid,.flow,.two{grid-template-columns:1fr 1fr}.top{align-items:flex-start}}
    @media(max-width:560px){main{padding-top:22px}.grid,.flow,.two{grid-template-columns:1fr}.bar-row{grid-template-columns:105px 1fr 55px}h2{margin-top:2.7rem}}
    """
    return f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="description" content="Kaggle 비료 추천 데이터 분석과 실험 계획"><title>비료 추천 데이터 분석</title><style>{css}</style></head><body><main>
    <div class="top"><div class="crumb">Kaggle Data Analysis Study · Playground S5E6</div><a href="../">전체 대회</a></div>
    <h1>비료 추천 데이터에서 무엇을 먼저 배워야 할까</h1>
    <p class="lead">개별 변수의 평균만 보면 클래스 차이가 거의 보이지 않습니다. 이 데이터의 핵심은 영양소·수분·작물 조건의 조합과 그 조합을 누수 없이 검증하는 방법입니다.</p>
    <section class="grid"><div class="card"><div class="metric">{num(len(a['train']))}</div><div class="label">학습 행</div></div><div class="card"><div class="metric">{num(len(a['test']))}</div><div class="label">테스트 행</div></div><div class="card"><div class="metric">8 → 7</div><div class="label">예측 특성 → 클래스</div></div><div class="card"><div class="metric">{a['prior_map3']:.6f}</div><div class="label">빈도 기준선 MAP@3</div></div></section>

    <h2>확인 → 발견 → 의미 → 다음 행동</h2><section class="flow"><div class="step"><small>01 · 확인</small>결측, 중복, 범주, train/test 분포와 ID 순서를 검사했습니다.</div><div class="step"><small>02 · 발견</small>결측과 미등록 범주는 없고 단변량 분포 이동도 매우 작습니다.</div><div class="step"><small>03 · 의미</small>정제보다 변수 조합과 확률 순위 품질이 성능을 좌우합니다.</div><div class="step"><small>04 · 다음</small>고정된 5-Fold에서 기준선부터 한 요소씩 바꿉니다.</div></section>

    <h2>현재 실험 상태</h2><section class="grid two"><article class="card finding"><h3>{esc(best_text)}</h3><div class="metric">{best_score}</div><p class="note">주지표: seed 326 Stratified 5-Fold OOF MAP@3</p></article><article class="card"><h3>왜 MAP@3인가</h3><p>정답이 1위면 1점, 2위면 0.5점, 3위면 약 0.333점입니다. 정답률만 높이는 것보다 올바른 클래스의 확률 순서를 만드는 것이 중요합니다.</p></article></section><div style="margin-top:14px">{experiment_html}</div>

    <h2>타깃 분포</h2><section class="grid two"><article class="card"><h3>7개 비료의 비율</h3>{target_bars}</article><article class="card warning"><h3>심한 불균형은 아닙니다</h3><p>가장 많은 클래스는 {pct(target['비율'].max())}, 가장 적은 클래스는 {pct(target['비율'].min())}입니다. 단순 정확도 왜곡보다 각 클래스가 상위 3개 안에서 어느 순위에 놓이는지 확인하는 편이 중요합니다.</p><p class="note">빈도 상위 3개만 고정 예측하면 MAP@3은 {a['prior_map3']:.6f}입니다.</p></article></section>

    <h2>데이터 품질과 분포 이동</h2><section class="grid two"><article class="card"><h3>품질 검사</h3><ul><li>결측값: {num(int(a['missing']['train_missing'].sum()))}건</li><li>학습 특성 중복 행: {num(a['feature_duplicates'])}건</li><li>ID 제외 완전 중복 행: {num(a['full_duplicates'])}건</li><li>test에만 있는 범주: {sum(len(v) for v in a['unseen'].values())}개</li><li>train과 완전히 같은 test 특성 행: {num(a['exact_cross_matches'])}건</li></ul></article><article class="card"><h3>train/test 총변동거리</h3><p class="note">0에 가까울수록 두 데이터의 값 비율이 비슷합니다.</p>{bars(drift_rows,'feature','total_variation',scale=max(0.01,float(a['drift']['total_variation'].max())))}</article></section>
    <p class="callout">ID 10분위 사이 클래스 비율의 최대 변화는 {pct(a['max_id_delta'])}입니다. 강한 순서 패턴은 보이지 않아 ID는 기본 모델에서 제외하고 안정성 진단에만 사용합니다.</p>

    <h2>왜 상호작용을 봐야 하는가</h2><section class="grid two"><article class="card"><h3>단일 특성의 최적 분류 정확도</h3><p class="note">각 값에서 가장 흔한 클래스를 고른 탐색용 상한입니다. 교차검증 점수가 아닙니다.</p>{bars(single_rows,'features','in_sample_top1',scale=.20)}</article><article class="card"><h3>두 특성 조합 상위</h3><p class="note">조합값에서 가장 흔한 클래스를 고른 학습 내 수치입니다. 개선 여부는 OOF에서 다시 확인해야 합니다.</p>{bars(pair_rows,'features','in_sample_top1',scale=.22)}</article></section>
    <p class="callout">단일 특성 최고치는 {a['single_strength'].iloc[0]['in_sample_top1']:.4f}지만 두 특성 조합은 {a['pair_strength'].iloc[0]['in_sample_top1']:.4f}까지 올라갑니다. 그래서 첫 피처 실험은 영양소와 수분의 쌍 조합이며, 타깃 인코딩은 반드시 fold 안에서만 계산합니다.</p>

    <h2>수치형 특성의 실제 형태</h2><article class="card">{make_table(numeric_summary)}<p class="note">이름은 수치형이지만 고유값 수가 14~43개뿐입니다. 원값을 연속량으로 처리한 기준선과 모든 값을 범주로 처리한 실험을 분리해 비교합니다.</p></article>

    <h2>E001은 어떤 클래스를 어려워하는가</h2><article class="card">{class_html}<p class="note">전체 정확도가 아니라 각 정답 클래스가 1순위 또는 상위 3개에 들어간 비율입니다. DAP와 Urea의 낮은 포함률은 다음 오류 분석의 최우선 대상입니다.</p></article>

    <h2>진행 순서</h2><article class="card"><ol><li><strong>E000 빈도 기준선:</strong> 지표와 제출 형식이 맞는지 확인합니다.</li><li><strong>E001 CatBoost:</strong> 원본 8개 특성으로 첫 OOF 기준선을 만듭니다.</li><li><strong>E002 모델 비교:</strong> 같은 fold에서 모델만 LightGBM으로 바꿉니다.</li><li><strong>E003 범주형 가설:</strong> 8개 특성을 모두 범주형으로 처리합니다.</li><li><strong>E004 상호작용:</strong> 영양소·수분 쌍을 추가합니다.</li><li><strong>E005 누수 없는 타깃 인코딩:</strong> 학습 fold 통계만 사용합니다.</li><li><strong>E006 앙상블:</strong> 서로 다른 오류를 가진 완료 모델만 OOF 확률 평균합니다.</li><li><strong>공개 해법 비교:</strong> 독립 실험을 고정한 뒤 고차 조합과 원본 데이터 사용을 별도 검증합니다.</li></ol></article>

    <h2>현재 결론</h2><section class="grid two"><article class="card finding"><h3>우선순위 1 · 검증 고정</h3><p>75만 행이라 작은 점수 차이도 보일 수 있지만 fold가 바뀌면 원인을 비교할 수 없습니다. 모든 실험에 같은 seed와 fold를 사용합니다.</p></article><article class="card finding"><h3>우선순위 2 · 조합 가설</h3><p>클래스별 수치 평균은 거의 비슷합니다. 단순 상관계수보다 조건 조합을 잘 표현하는 모델과 인코딩이 중요합니다.</p></article><article class="card warning"><h3>주의 · 타깃 인코딩 누수</h3><p>전체 학습 데이터에서 만든 타깃 평균을 검증 행에 넣으면 정답 정보를 미리 보게 됩니다. 각 fold의 학습 부분에서만 통계를 만듭니다.</p></article><article class="card"><h3>다음 판단</h3><p>E001과 E003의 차이로 수치값의 범주형 처리 효과를 확인한 뒤, E004 쌍 조합을 실행합니다.</p></article></section>

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
    DOCS_PATH.write_text(make_html(analysis) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
