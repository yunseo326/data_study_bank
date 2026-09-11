"""First-pass EDA and GitHub Pages report for Podcast Listening Time."""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for package_dir in [PROJECT_ROOT / ".runtime_pkgs", PROJECT_ROOT.parent / "bank" / ".runtime_pkgs"]:
    if package_dir.exists():
        sys.path.insert(0, str(package_dir))
        break

import numpy as np
import pandas as pd


DATA_DIR = PROJECT_ROOT / "data"
DOCS_PATH = PROJECT_ROOT / "docs" / "index.html"
SUMMARY_PATH = PROJECT_ROOT / "logs" / "result" / "podcast_eda_summary.json"
REGISTRY_PATH = PROJECT_ROOT / "benchmarks" / "experiments.csv"
TARGET = "Listening_Time_minutes"
ID_COLUMN = "id"
RNG = np.random.default_rng(326)


DATA_DICTIONARY = [
    ("Podcast_Name", "범주형", "팟캐스트 프로그램 이름", "프로그램별 고유 청취 성향과 장르 효과가 섞일 수 있음"),
    ("Episode_Title", "범주형", "Episode 1~100 형식의 회차 제목", "범주형과 숫자로 추출한 회차를 별도 비교"),
    ("Episode_Length_minutes", "수치형", "에피소드 전체 길이(분)", "가장 강한 타깃 관계, 결측과 극단값 별도 점검"),
    ("Genre", "범주형", "팟캐스트 장르", "장르 평균보다 길이와의 상호작용 가능성 확인"),
    ("Host_Popularity_percentage", "수치형", "호스트 인기도(%)", "100 초과 소수 값은 품질 가설로 분리"),
    ("Publication_Day", "범주형", "공개 요일", "시간순 인덱스가 아니므로 시계열 분할 근거로 사용하지 않음"),
    ("Publication_Time", "범주형", "공개 시간대", "시간대별 평균과 길이 조건부 효과 비교"),
    ("Guest_Popularity_percentage", "수치형", "게스트 인기도(%)", "결측 약 19.5%, 결측 상태 자체의 신호 확인"),
    ("Number_of_Ads", "수치형", "광고 수", "정상값은 0~3에 집중, 큰 값은 독립 정제 실험"),
    ("Episode_Sentiment", "범주형", "에피소드 감정", "Positive/Neutral/Negative의 작은 평균 차이 확인"),
    (TARGET, "타깃", "예측할 청취시간(분)", "RMSE 평가, 큰 오차의 영향이 큼"),
]


def esc(value: object) -> str:
    return html.escape(str(value))


def fmt(value: object) -> str:
    if pd.isna(value):
        return "-"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):,.4f}"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    return str(value)


def table(frame: pd.DataFrame) -> str:
    header = "".join(f"<th>{esc(column)}</th>" for column in frame.columns)
    body = []
    for _, row in frame.iterrows():
        body.append("<tr>" + "".join(f"<td>{esc(fmt(row[column]))}</td>" for column in frame.columns) + "</tr>")
    return f'<div class="table-wrap"><table><thead><tr>{header}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def horizontal_bars(labels: list[str], values: list[float], value_format: str = ".3f") -> str:
    maximum = max(values, default=1.0) or 1.0
    rows = []
    for label, value in zip(labels, values):
        width = max(1.0, 100 * value / maximum)
        rows.append(
            f'<div class="bar-row"><span class="bar-label">{esc(label)}</span>'
            f'<span class="bar-track"><i style="width:{width:.1f}%"></i></span>'
            f'<strong>{format(value, value_format)}</strong></div>'
        )
    return '<div class="bars">' + "".join(rows) + "</div>"


def ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    a = np.sort(np.asarray(a))
    b = np.sort(np.asarray(b))
    points = np.unique(np.concatenate([a, b]))
    return float(np.max(np.abs(
        np.searchsorted(a, points, side="right") / len(a)
        - np.searchsorted(b, points, side="right") / len(b)
    )))


def anomaly_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    length = frame["Episode_Length_minutes"]
    host = frame["Host_Popularity_percentage"]
    guest = frame["Guest_Popularity_percentage"]
    ads = frame["Number_of_Ads"]
    return {
        "에피소드 길이 범위 이탈": length.notna() & ~length.between(0, 120),
        "호스트 인기도 범위 이탈": host.notna() & ~host.between(0, 100),
        "게스트 인기도 범위 이탈": guest.notna() & ~guest.between(0, 100),
        "광고 수 범위·정수 이탈": ads.notna() & (~ads.between(0, 3) | ~np.isclose(ads, np.round(ads))),
    }


def analyze() -> dict:
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
    features = [column for column in test.columns if column != ID_COLUMN]
    numeric = [column for column in features if pd.api.types.is_numeric_dtype(train[column])]
    categorical = [column for column in features if column not in numeric]

    train_hash = pd.util.hash_pandas_object(train.drop(columns=[TARGET, ID_COLUMN]), index=False)
    test_hash = pd.util.hash_pandas_object(test.drop(columns=[ID_COLUMN]), index=False)
    train_duplicates = int(train_hash.duplicated().sum())
    test_duplicates = int(test_hash.duplicated().sum())
    cross_matches = int(test_hash.isin(set(train_hash)).sum())

    missing = pd.DataFrame({
        "변수": features,
        "train 결측": [int(train[column].isna().sum()) for column in features],
        "train 결측률": [float(train[column].isna().mean()) for column in features],
        "test 결측": [int(test[column].isna().sum()) for column in features],
        "test 결측률": [float(test[column].isna().mean()) for column in features],
    })
    missing = missing[(missing["train 결측"] > 0) | (missing["test 결측"] > 0)].copy()

    anomaly_rows = []
    train_masks, test_masks = anomaly_masks(train), anomaly_masks(test)
    for label in train_masks:
        anomaly_rows.append({
            "검사": label,
            "train": int(train_masks[label].sum()),
            "test": int(test_masks[label].sum()),
        })
    anomalies = pd.DataFrame(anomaly_rows)

    suspicious_test = test[pd.DataFrame(test_masks).any(axis=1)][[
        ID_COLUMN, "Episode_Length_minutes", "Host_Popularity_percentage",
        "Guest_Popularity_percentage", "Number_of_Ads",
    ]].copy()
    suspicious_test["최대 이상 정도"] = suspicious_test[[
        "Episode_Length_minutes", "Host_Popularity_percentage",
        "Guest_Popularity_percentage", "Number_of_Ads",
    ]].max(axis=1)
    suspicious_test = suspicious_test.sort_values("최대 이상 정도", ascending=False).head(12).drop(columns="최대 이상 정도")

    correlation = (
        train[numeric + [TARGET]].corr(method="pearson")[TARGET]
        .drop(TARGET).sort_values(key=abs, ascending=False)
    )

    length_valid = train.dropna(subset=["Episode_Length_minutes"]).copy()
    length_valid["길이 구간"] = pd.qcut(length_valid["Episode_Length_minutes"], 10, duplicates="drop")
    length_target = length_valid.groupby("길이 구간", observed=True).agg(
        행_수=(TARGET, "size"), 평균_길이=("Episode_Length_minutes", "mean"), 평균_청취시간=(TARGET, "mean"),
    ).reset_index()
    length_target["길이 구간"] = length_target["길이 구간"].astype(str)

    category_summary = []
    for column in categorical:
        means = train.groupby(column)[TARGET].mean()
        category_summary.append({
            "변수": column, "수준 수": int(train[column].nunique()),
            "그룹 평균 최소": float(means.min()), "그룹 평균 최대": float(means.max()),
            "그룹 평균 표준편차": float(means.std()),
            "test 신규 수준": len(set(test[column].dropna()) - set(train[column].dropna())),
        })
    category_summary = pd.DataFrame(category_summary).sort_values("그룹 평균 표준편차", ascending=False)

    drift_rows = []
    for column in numeric:
        a, b = train[column].dropna().to_numpy(), test[column].dropna().to_numpy()
        if len(a) > 100_000:
            a = RNG.choice(a, 100_000, replace=False)
        if len(b) > 100_000:
            b = RNG.choice(b, 100_000, replace=False)
        drift_rows.append({"변수": column, "기준": "KS", "차이": ks_statistic(a, b)})
    for column in categorical:
        a = train[column].fillna("(NA)").value_counts(normalize=True)
        b = test[column].fillna("(NA)").value_counts(normalize=True)
        levels = a.index.union(b.index)
        tv = 0.5 * float((a.reindex(levels, fill_value=0) - b.reindex(levels, fill_value=0)).abs().sum())
        drift_rows.append({"변수": column, "기준": "TV", "차이": tv})
    drift = pd.DataFrame(drift_rows).sort_values("차이", ascending=False)

    id_decile = pd.qcut(train[ID_COLUMN], 10, labels=False)
    id_stability = train.groupby(id_decile, observed=True).agg(
        행_수=(TARGET, "size"), 타깃_평균=(TARGET, "mean"), 타깃_표준편차=(TARGET, "std"),
    ).reset_index().rename(columns={ID_COLUMN: "ID 구간"})
    id_stability["ID 구간"] = id_stability["ID 구간"].map(lambda value: f"{int(value) + 1}/10")

    hist_counts, hist_edges = np.histogram(train[TARGET], bins=np.linspace(0, 120, 21))
    target_hist = pd.DataFrame({
        "구간": [f"{hist_edges[i]:.0f}–{hist_edges[i + 1]:.0f}" for i in range(len(hist_counts))],
        "행 수": hist_counts,
    })

    experiments = pd.DataFrame()
    if REGISTRY_PATH.exists():
        experiments = pd.read_csv(REGISTRY_PATH)
    model_metrics = {}
    for metrics_path in (PROJECT_ROOT / "outputs").glob("*/metrics.json"):
        try:
            model_metrics[metrics_path.parent.name] = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

    dictionary = pd.DataFrame(DATA_DICTIONARY, columns=["변수", "유형", "의미", "해석 주의"])
    target_stats = train[TARGET].describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]).to_dict()
    return {
        "train": train, "test": test, "sample": sample, "features": features,
        "numeric": numeric, "categorical": categorical, "missing": missing,
        "anomalies": anomalies, "suspicious_test": suspicious_test,
        "correlation": correlation, "length_target": length_target,
        "category_summary": category_summary, "drift": drift,
        "id_stability": id_stability, "target_hist": target_hist,
        "experiments": experiments, "dictionary": dictionary,
        "model_metrics": model_metrics,
        "train_duplicates": train_duplicates, "test_duplicates": test_duplicates,
        "cross_matches": cross_matches, "target_stats": target_stats,
        "schema_ok": list(test.columns) == [c for c in train.columns if c != TARGET],
        "submission_ok": list(sample.columns) == [ID_COLUMN, TARGET] and sample[ID_COLUMN].equals(test[ID_COLUMN]),
    }


def render_report(result: dict) -> str:
    train, test = result["train"], result["test"]
    target = result["target_stats"]
    missing_view = result["missing"].copy()
    for column in ["train 결측률", "test 결측률"]:
        missing_view[column] = missing_view[column].map(lambda value: f"{100 * value:.2f}%")
    corr = result["correlation"]
    experiment_html = '<p class="muted">아직 실행된 모델 실험이 없습니다.</p>'
    if not result["experiments"].empty:
        columns = [c for c in ["experiment_id", "purpose", "changed_element", "model", "cv_rmse", "fold_std", "runtime_minutes", "status"] if c in result["experiments"]]
        experiment_html = table(result["experiments"][columns])
    current_conclusion = (
        "에피소드 길이가 문제의 중심 신호입니다. 동일한 5-Fold에서 평균 → 길이 선형식 → CatBoost 순으로 기준선을 만든 뒤, 이상값 처리를 한 변수씩 검증합니다."
    )
    completed_summary = ""
    if "B002" in result["model_metrics"]:
        b002 = result["model_metrics"]["B002"]
        missing_segments = {
            row["value"]: row for row in b002.get("segments", [])
            if row["segment"] == "episode_length_missing"
        }
        best_id = min(
            result["model_metrics"],
            key=lambda key: result["model_metrics"][key].get("cv_rmse", float("inf")),
        )
        best_score = result["model_metrics"][best_id]["cv_rmse"]
        current_conclusion = (
            f"현재 최선은 {best_id}의 OOF RMSE {best_score:.6f}입니다. "
            f"B002 CatBoost는 길이 선형 기준보다 좋아졌지만, 길이 결측 행이 다음 병목입니다."
        )
        if "yes" in missing_segments and "no" in missing_segments:
            completed_summary = (
                f'<div class="card warning" style="margin-top:14px"><h3>현재 오류 병목</h3>'
                f'<p>길이가 있는 행 RMSE는 <b>{missing_segments["no"]["rmse"]:.6f}</b>, '
                f'결측 행은 <b>{missing_segments["yes"]["rmse"]:.6f}</b>입니다. '
                f'따라서 다음 실험은 전체 모델을 무작정 키우기보다 길이 결측 구간의 정보 복원에 집중합니다.</p></div>'
            )

    css = """
    :root{--ink:#17212b;--muted:#637083;--paper:#f4f7fa;--card:#fff;--blue:#1769aa;--teal:#16867a;--red:#b64545;--line:#dce4eb}
    *{box-sizing:border-box} body{margin:0;background:var(--paper);color:var(--ink);font-family:Inter,"Noto Sans KR",Arial,sans-serif;line-height:1.62}
    .hero{background:linear-gradient(135deg,#102c46,#1769aa);color:#fff;padding:64px 24px}.wrap{max-width:1120px;margin:auto}.hero h1{font-size:clamp(2rem,5vw,3.5rem);margin:0 0 12px}.hero p{max-width:760px;color:#d7e8f6;margin:0}
    main{max-width:1120px;margin:28px auto;padding:0 18px 64px}.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}.card{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:20px;box-shadow:0 5px 18px rgba(25,45,64,.05)}
    .metric strong{display:block;font-size:1.7rem;color:var(--blue)}.metric span,.muted{color:var(--muted)}section{margin-top:28px}h2{font-size:1.45rem;margin:0 0 12px}h3{font-size:1.05rem;margin:20px 0 8px}.callout{border-left:5px solid var(--teal)}.warning{border-left:5px solid var(--red)}
    .table-wrap{overflow:auto;border:1px solid var(--line);border-radius:12px}table{width:100%;border-collapse:collapse;background:#fff;font-size:.9rem}th,td{padding:10px 12px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}th{background:#eaf1f7;color:#27445d;position:sticky;top:0}tr:last-child td{border-bottom:0}
    .bars{display:grid;gap:9px}.bar-row{display:grid;grid-template-columns:minmax(140px,1.4fr) 4fr 80px;align-items:center;gap:10px}.bar-label{font-size:.88rem}.bar-track{height:12px;background:#e5edf3;border-radius:99px;overflow:hidden}.bar-track i{display:block;height:100%;background:linear-gradient(90deg,var(--blue),var(--teal));border-radius:99px}.bar-row strong{text-align:right;font-size:.82rem}
    .steps{counter-reset:s}.step{position:relative;padding-left:54px;margin:18px 0}.step:before{counter-increment:s;content:counter(s);position:absolute;left:0;top:0;width:36px;height:36px;border-radius:50%;display:grid;place-items:center;background:var(--blue);color:#fff;font-weight:700}.step b{display:block}.tag{display:inline-block;padding:3px 9px;border-radius:99px;background:#e7f4f2;color:#116b62;font-size:.78rem;font-weight:700}
    footer{color:var(--muted);font-size:.85rem;margin-top:36px}@media(max-width:800px){.grid{grid-template-columns:repeat(2,1fr)}.bar-row{grid-template-columns:110px 1fr 64px}.card{padding:16px}}@media(max-width:480px){.grid{grid-template-columns:1fr}.hero{padding:42px 20px}main{padding:0 12px 50px}.bar-label{font-size:.76rem}}
    """
    return f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Podcast Listening Time 분석</title><style>{css}</style></head>
    <body><header class="hero"><div class="wrap"><span class="tag">Kaggle Playground S5E4</span><h1>Podcast Listening Time</h1><p>무엇을 확인했고, 무엇을 발견했으며, 그것이 왜 중요한지와 다음 실험을 한 흐름으로 정리한 분석 보고서입니다.</p></div></header><main>
    <div class="grid">
      <div class="card metric"><span>train</span><strong>{len(train):,}</strong><span>{len(result['features'])}개 예측 변수</span></div>
      <div class="card metric"><span>test</span><strong>{len(test):,}</strong><span>제출 대상 행</span></div>
      <div class="card metric"><span>타깃 평균</span><strong>{target['mean']:.2f}분</strong><span>표준편차 {target['std']:.2f}</span></div>
      <div class="card metric"><span>길이 상관</span><strong>{corr['Episode_Length_minutes']:.3f}</strong><span>가장 강한 단일 관계</span></div>
    </div>

    <section class="card callout"><h2>현재 결론</h2><p><b>{current_conclusion}</b> 길이가 길수록 청취시간이 거의 선형적으로 증가하지만, 길이 결측과 test 극단값을 별도로 다뤄야 합니다.</p></section>

    <section><h2>1. 데이터 계약과 품질</h2><div class="grid">
      <div class="card metric"><span>train 피처 중복</span><strong>{result['train_duplicates']:,}</strong><span>ID·타깃 제외</span></div>
      <div class="card metric"><span>test 피처 중복</span><strong>{result['test_duplicates']:,}</strong><span>ID 제외</span></div>
      <div class="card metric"><span>train/test 동일 피처</span><strong>{result['cross_matches']:,}</strong><span>해시 기준</span></div>
      <div class="card metric"><span>제출 형식</span><strong>{'정상' if result['submission_ok'] else '오류'}</strong><span>ID 순서 포함</span></div>
    </div><h3>결측</h3>{table(missing_view)}<h3>범위 이탈</h3>{table(result['anomalies'])}</section>

    <section class="card warning"><h2>test 극단값 주의</h2><p>정상 분포를 크게 벗어난 값이 소수 존재합니다. 특히 길이 78,486,264분과 광고 2,063개는 선형 모델의 test 예측을 폭발시킬 수 있습니다. 원본 CSV는 수정하지 않고 실험용 사본에서만 결측 처리합니다.</p>{table(result['suspicious_test'])}</section>

    <section><h2>2. 타깃과 핵심 관계</h2><div class="card"><h3>타깃 분포</h3>{horizontal_bars(result['target_hist']['구간'].tolist(), result['target_hist']['행 수'].astype(float).tolist(), ',.0f')}</div>
    <div class="card" style="margin-top:14px"><h3>수치형 변수와 타깃의 Pearson 상관</h3>{horizontal_bars(corr.index.tolist(), corr.abs().tolist())}<p class="muted">막대 길이는 관계의 방향이 아닌 절대 크기입니다. 광고 수는 음의 관계이고 나머지 인기도 변수는 약한 관계입니다.</p></div>
    <h3>에피소드 길이 10분위</h3>{table(result['length_target'])}<h3>범주형 변수의 그룹 평균 차이</h3>{table(result['category_summary'])}</section>

    <section><h2>3. train/test와 검증 전략</h2><div class="card callout"><p>모든 범주형 수준이 양쪽 데이터에 존재하고 ID 10분위별 타깃 평균도 안정적입니다. 따라서 seed 326의 shuffled 5-Fold를 고정합니다. 수치형 KS는 극단값에 민감하므로 분위수와 이상값 표를 함께 해석합니다.</p></div><h3>분포 차이 통계량</h3>{table(result['drift'])}<h3>ID 구간 안정성</h3>{table(result['id_stability'])}</section>

    <section><h2>4. 실험 진행</h2>{experiment_html}{completed_summary}<div class="card" style="margin-top:14px">
      <div class="step"><b>B000 평균 기준</b><span>피처가 전혀 없을 때의 RMSE를 확인합니다.</span></div>
      <div class="step"><b>B001 길이 선형 기준</b><span>문제의 핵심 구조가 어느 정도를 설명하는지 확인합니다.</span></div>
      <div class="step"><b>B002 CatBoost 기준</b><span>범주형과 비선형 상호작용을 포함한 강한 단일 모델을 만듭니다.</span></div>
      <div class="step"><b>데이터 품질 실험</b><span>길이, 호스트, 게스트, 광고 이상값을 각각 하나씩 바꿉니다.</span></div>
      <div class="step"><b>구조·모델 확장</b><span>결측 플래그, 회차 번호, 잔차 모델, LightGBM과 XGBoost를 순서대로 비교합니다.</span></div>
    </div></section>

    <section><h2>변수 해석</h2>{table(result['dictionary'])}</section>
    <footer>생성 기준: 로컬 원본 CSV · seed 326 · 원본 데이터는 변경하지 않음</footer></main></body></html>"""


def write_outputs(result: dict) -> None:
    summary = {
        "dataset": {
            "train_rows": len(result["train"]), "test_rows": len(result["test"]),
            "feature_count": len(result["features"]), "numeric": result["numeric"],
            "categorical": result["categorical"], "schema_ok": result["schema_ok"],
            "submission_ok": result["submission_ok"],
        },
        "target": result["target_stats"],
        "missing": result["missing"].to_dict(orient="records"),
        "anomalies": result["anomalies"].to_dict(orient="records"),
        "correlations": {key: float(value) for key, value in result["correlation"].items()},
        "duplicates": {
            "train": result["train_duplicates"], "test": result["test_duplicates"],
            "cross_split_matches": result["cross_matches"],
        },
        "drift": result["drift"].to_dict(orient="records"),
    }
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOCS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    DOCS_PATH.write_text(render_report(result), encoding="utf-8")
    print(json.dumps({
        "summary": str(SUMMARY_PATH), "report": str(DOCS_PATH),
        "train_rows": len(result["train"]), "test_rows": len(result["test"]),
    }, ensure_ascii=False, indent=2))


def main() -> None:
    write_outputs(analyze())


if __name__ == "__main__":
    main()
