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
MISSING_RESIDUAL_PATH = PROJECT_ROOT / "logs" / "result" / "podcast_missing_residual_analysis.json"
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
    missing_residual = {}
    if MISSING_RESIDUAL_PATH.exists():
        try:
            missing_residual = json.loads(MISSING_RESIDUAL_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass

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
        "missing_residual": missing_residual,
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
    diagnostic = result["missing_residual"]
    missing_residual_html = ""
    if diagnostic:
        pattern_shift = pd.DataFrame(diagnostic["missing_pattern_shift"])
        pattern_shift = pattern_shift[pattern_shift["train_rows"] >= 20].rename(columns={
            "missing_pattern": "결측 패턴", "train_rows": "train 행",
            "train_rate": "train 비율", "test_rate": "test 비율",
            "gap_percentage_points": "test-train 차이(%p)",
        })
        for column in ["train 비율", "test 비율"]:
            pattern_shift[column] = pattern_shift[column].map(lambda value: f"{100 * value:.2f}%")
        missingness = pd.concat([
            pd.DataFrame(diagnostic["length_missingness_drivers"]).head(4),
            pd.DataFrame(diagnostic["guest_missingness_drivers"]).head(4),
        ], ignore_index=True).rename(columns={
            "missing_column": "결측 변수", "feature": "조건 변수", "groups": "그룹 수",
            "train_rate_min": "최소 결측률", "train_rate_max": "최대 결측률",
            "train_rate_range": "결측률 범위", "train_test_group_rate_correlation": "train/test 그룹 상관",
        })
        for column in ["최소 결측률", "최대 결측률", "결측률 범위"]:
            missingness[column] = missingness[column].map(lambda value: f"{100 * value:.2f}%")
        pattern_residual = pd.DataFrame(diagnostic["missing_pattern_residuals"])[[
            "missing_pattern", "rows", "target_mean", "prediction_mean", "bias", "mae", "rmse"
        ]].rename(columns={
            "missing_pattern": "결측 패턴", "rows": "행", "target_mean": "타깃 평균",
            "prediction_mean": "예측 평균", "bias": "평균 잔차", "mae": "MAE", "rmse": "RMSE",
        })
        length = diagnostic["group_length_imputation"]
        quintiles = pd.DataFrame(length["missing_quintiles"]).rename(columns={
            "quintile": "대체 길이 분위", "rows": "행",
            "imputed_length_mean": "대체 길이 평균", "target_mean": "타깃 평균",
        })
        drivers = pd.DataFrame(diagnostic["residual_drivers_length_missing"]).head(5).rename(columns={
            "feature": "변수", "eligible_groups": "비교 그룹 수",
            "mean_residual_range": "그룹 평균 잔차 범위", "lowest_group": "과대예측 그룹",
            "lowest_bias": "최저 평균 잔차", "highest_group": "과소예측 그룹",
            "highest_bias": "최고 평균 잔차",
        })
        deciles = pd.DataFrame(diagnostic["prediction_deciles"])[[
            "prediction_decile", "target_mean", "prediction_mean", "bias", "rmse"
        ]].rename(columns={
            "prediction_decile": "예측 10분위", "target_mean": "타깃 평균",
            "prediction_mean": "예측 평균", "bias": "평균 잔차", "rmse": "RMSE",
        })
        comparison = pd.DataFrame(diagnostic["experiment_comparison"]).rename(columns={
            "experiment_id": "실험", "overall_rmse": "전체 RMSE",
            "length_present_rmse": "길이 있음 RMSE", "length_missing_rmse": "길이 결측 RMSE",
            "overall_bias": "전체 평균 잔차", "length_missing_bias": "길이 결측 평균 잔차",
        })
        comparison_html = table(comparison) if not comparison.empty else '<p class="muted">비교 실험 실행 전입니다.</p>'
        partial = pd.DataFrame(diagnostic.get("partial_experiments", [])).rename(columns={
            "experiment_id": "실험", "status": "상태", "completed_folds": "완료 fold",
            "planned_folds": "계획 fold", "partial_rmse": "잠정 RMSE",
            "baseline_partial_rmse": "동일 fold B002", "partial_improvement": "잠정 개선",
            "interpretation": "해석",
        })
        partial_columns = [
            "실험", "상태", "완료 fold", "계획 fold", "잠정 RMSE",
            "동일 fold B002", "잠정 개선", "해석",
        ]
        partial_html = table(partial[partial_columns]) if not partial.empty else ""
        missing_residual_html = f"""
    <section><h2>4. 결측 구조 분석</h2>
      <div class="card callout"><p><b>결측은 train과 test에서 거의 같은 비율로 나타납니다.</b> 가장 큰 패턴 비율 차이도 0.13%p 미만이어서 결측 분포 이동 위험은 작습니다. 다만 길이 결측 87,093행은 전체의 11.61%인데 B002 전체 제곱오차의 약 {100 * diagnostic['length_missing_squared_error_share']:.1f}%를 차지합니다.</p></div>
      <h3>결측 패턴의 train/test 안정성</h3>{table(pattern_shift)}
      <h3>결측 발생을 설명하는 그룹 구조</h3>{table(missingness)}
      <p class="muted">결측률이 그룹마다 다르고 그 차이가 test에서도 같은 방향으로 반복됩니다. 따라서 완전한 무작위 결측이라기보다 데이터 생성 과정에 연결된 결측이며, CatBoost가 범주형 변수와 결측 상태의 상호작용을 학습할 여지가 있습니다.</p>
      <h3>결측 패턴별 OOF 오차</h3>{table(pattern_residual)}
      <p class="muted">L은 에피소드 길이, G는 게스트 인기도, A는 광고 수입니다. +는 값이 있고 -는 결측이라는 뜻입니다. 평균 잔차는 실제값−예측값이므로 양수는 평균적으로 과소예측했다는 뜻입니다.</p>
      <div class="card warning" style="margin-top:14px"><h3>그룹 중앙값 대체의 한계</h3><p>Podcast_Name×Episode_Title 중앙값으로 알려진 길이를 가렸다고 가정해 복원하면 길이 RMSE가 <b>{length['observed_length_rmse']:.2f}분</b>입니다. 전역 중앙값 기준 <b>{length['observed_length_baseline_rmse']:.2f}분</b>보다 오히려 나쁩니다. 즉 같은 프로그램·회차 번호라도 에피소드 길이가 고정되지 않아, 이 대체법을 정답 복원처럼 해석하면 안 됩니다.</p></div>
      <h3>실제 길이 결측 행의 대체 길이 분위</h3>{table(quintiles)}
      <p class="muted">복원 정확도는 낮지만 대체 길이와 청취시간 상관은 {length['missing_imputed_length_target_correlation']:.3f}이고, Q1→Q5 타깃 평균은 약 4.72분 증가합니다. 약한 순위 신호로는 쓸 수 있어 E203의 OOF 결과로 최종 판단합니다.</p>
    </section>

    <section><h2>5. 잔차 분석과 잔차 모델링</h2>
      <div class="card callout"><p><b>B002는 평균적으로는 잘 보정되어 있지만 오차 크기는 일정하지 않습니다.</b> 낮은 예측 구간의 RMSE는 약 4분인 반면 중간·상단 일부 구간은 16~18분입니다. 잔차 모델링은 이 남은 구조를 별도 타깃으로 학습해 직접 예측보다 안정적인지 검증하는 방법입니다.</p></div>
      <h3>예측값 10분위별 잔차</h3>{table(deciles)}
      <h3>길이 결측 행에서 남은 범주형 편향</h3>{table(drivers)}
      <p class="muted">Episode_Title의 평균 잔차 범위가 가장 크지만, 이는 그룹별 표본 잡음도 포함합니다. 따라서 표의 차이를 그대로 보정하지 않고 E401에서 fold 내부 길이 기준식의 잔차를 CatBoost로 학습합니다.</p>
      <h3>결측·잔차 실험 비교</h3>{comparison_html}
      <h3>중단된 잔차 실험의 잠정 결과</h3>{partial_html}
      <p class="muted">E401은 사용자 요청에 따라 2/5 fold에서 중단했습니다. 완료된 두 fold는 모두 개선됐지만 정식 OOF 점수, 제출 파일, 채택 모델로 사용하지 않습니다.</p>
    </section>"""
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
    .hero{background:linear-gradient(135deg,#102c46,#1769aa);color:#fff;padding:64px 24px}.wrap{max-width:1120px;margin:auto}.hero h1{font-size:clamp(2rem,5vw,3.5rem);margin:0 0 12px}.hero p{max-width:760px;color:#d7e8f6;margin:0}.back{display:inline-block;margin-bottom:18px;color:#d7e8f6;text-decoration:none}.back:hover{text-decoration:underline}
    main{max-width:1120px;margin:28px auto;padding:0 18px 64px}.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}.two{grid-template-columns:repeat(2,1fr)}.card{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:20px;box-shadow:0 5px 18px rgba(25,45,64,.05)}
    .metric strong{display:block;font-size:1.7rem;color:var(--blue)}.metric span,.muted{color:var(--muted)}section{margin-top:28px}h2{font-size:1.45rem;margin:0 0 12px}h3{font-size:1.05rem;margin:20px 0 8px}.callout{border-left:5px solid var(--teal)}.warning{border-left:5px solid var(--red)}
    .table-wrap{overflow:auto;border:1px solid var(--line);border-radius:12px}table{width:100%;border-collapse:collapse;background:#fff;font-size:.9rem}th,td{padding:10px 12px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}th{background:#eaf1f7;color:#27445d;position:sticky;top:0}tr:last-child td{border-bottom:0}
    .bars{display:grid;gap:9px}.bar-row{display:grid;grid-template-columns:minmax(140px,1.4fr) 4fr 80px;align-items:center;gap:10px}.bar-label{font-size:.88rem}.bar-track{height:12px;background:#e5edf3;border-radius:99px;overflow:hidden}.bar-track i{display:block;height:100%;background:linear-gradient(90deg,var(--blue),var(--teal));border-radius:99px}.bar-row strong{text-align:right;font-size:.82rem}
    .steps{counter-reset:s}.step{position:relative;padding-left:54px;margin:18px 0}.step:before{counter-increment:s;content:counter(s);position:absolute;left:0;top:0;width:36px;height:36px;border-radius:50%;display:grid;place-items:center;background:var(--blue);color:#fff;font-weight:700}.step b{display:block}.tag{display:inline-block;padding:3px 9px;border-radius:99px;background:#e7f4f2;color:#116b62;font-size:.78rem;font-weight:700}
    .quick-nav{display:flex;flex-wrap:wrap;gap:8px;margin:22px 0 4px}.quick-nav a{padding:7px 11px;border:1px solid var(--line);border-radius:8px;background:#fff;color:var(--blue);text-decoration:none;font-size:.82rem;font-weight:700}
    .guide td{white-space:normal;vertical-align:top;min-width:150px}.guide td:first-child{min-width:210px}.guide code{white-space:nowrap}.type{display:inline-block;margin-left:6px;padding:2px 7px;border-radius:8px;background:#e7f4f2;color:#116b62;font-size:.72rem;font-weight:700}.direction{font-size:1.35rem;font-weight:800;color:var(--teal)}
    .roadmap{display:grid;gap:10px}.roadmap-item{display:grid;grid-template-columns:86px 1fr;gap:14px;align-items:start;background:#fff;border:1px solid var(--line);border-radius:8px;padding:15px}.roadmap-item p{margin:3px 0;color:var(--muted)}.status{display:inline-block;text-align:center;border-radius:8px;padding:4px 8px;font-size:.74rem;font-weight:800}.done{background:#dff4ef;color:#08796f}.now{background:#dfeefa;color:#176b9c}.next{background:#f5eadc;color:#92520c}.later{background:#edf0f2;color:#60717d}
    footer{color:var(--muted);font-size:.85rem;margin-top:36px}@media(max-width:800px){.grid{grid-template-columns:repeat(2,1fr)}.bar-row{grid-template-columns:110px 1fr 64px}.card{padding:16px}}@media(max-width:480px){.grid,.two{grid-template-columns:1fr}.hero{padding:42px 20px}main{padding:0 12px 50px}.bar-label{font-size:.76rem}.roadmap-item{grid-template-columns:70px 1fr}}
    """
    return f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Podcast Listening Time 분석</title><style>{css}</style></head>
    <body><header class="hero"><div class="wrap"><a class="back" href="../">Data Study 전체 대회</a><br><span class="tag">Kaggle Playground S5E4</span><h1>Podcast Listening Time</h1><p>무엇을 확인했고, 무엇을 발견했으며, 그것이 왜 중요한지와 다음 실험을 한 흐름으로 정리한 분석 보고서입니다.</p></div></header><main>
    <div class="grid">
      <div class="card metric"><span>train</span><strong>{len(train):,}</strong><span>{len(result['features'])}개 예측 변수</span></div>
      <div class="card metric"><span>test</span><strong>{len(test):,}</strong><span>제출 대상 행</span></div>
      <div class="card metric"><span>타깃 평균</span><strong>{target['mean']:.2f}분</strong><span>표준편차 {target['std']:.2f}</span></div>
      <div class="card metric"><span>길이 상관</span><strong>{corr['Episode_Length_minutes']:.3f}</strong><span>가장 강한 단일 관계</span></div>
    </div>

    <nav class="quick-nav" aria-label="보고서 바로가기"><a href="#objective">목표와 점수</a><a href="#features">Feature 설명</a><a href="#roadmap">전체 로드맵</a><a href="#experiments">실험 결과</a></nav>

    <section id="objective"><h2>무엇을 예측하며, 어느 방향이 좋은가</h2><div class="grid two">
      <article class="card"><h3>Target: Listening_Time_minutes</h3><p>각 에피소드를 실제로 청취한 시간을 <b>분 단위 수치</b>로 예측하는 회귀 문제입니다. 학습 데이터의 실제값은 0분부터 약 120분까지이며, 1에 가까워지는 것이 목표인 분류 문제가 아닙니다.</p><p class="muted">예측값도 현실적인 청취시간이어야 하므로 현재 P001은 최종 예측을 0~120분으로 제한합니다.</p></article>
      <article class="card callout"><h3>평가 지표: RMSE</h3><div class="direction">작을수록 좋음 · 최저 0</div><p>예측과 실제 청취시간 차이를 분 단위로 평가합니다. 완벽히 맞히면 0이며, 큰 오차를 작은 오차보다 더 강하게 벌점 줍니다.</p><p class="muted">현재 OOF RMSE 13.049970은 평균 오차가 정확히 13.05분이라는 뜻은 아니지만, 같은 5-Fold에서 더 낮은 모델이 더 좋다는 뜻입니다.</p></article>
    </div><div class="table-wrap" style="margin-top:14px"><table class="guide"><thead><tr><th>함께 보는 값</th><th>좋은 방향</th><th>이 보고서에서 읽는 법</th></tr></thead><tbody><tr><td><b>OOF RMSE</b></td><td><b>작을수록 좋음 · 최저 0</b></td><td>모델 선택의 주지표입니다. 한 번도 학습에 쓰지 않은 fold의 예측을 합쳐 계산합니다.</td></tr><tr><td><b>Fold 표준편차</b></td><td><b>작을수록 안정적 · 최저 0</b></td><td>다섯 번의 검증 점수가 얼마나 흔들리는지 보여줍니다. 낮으면 데이터 분할 운에 덜 민감합니다.</td></tr><tr><td><b>세그먼트 RMSE</b></td><td><b>작을수록 좋음</b></td><td>길이 결측 같은 특정 구간의 약점을 찾는 진단값입니다. 전체 점수와 함께 판단합니다.</td></tr><tr><td><b>Pearson 상관</b></td><td><b>-1~1, 목표 점수 아님</b></td><td>양수는 함께 증가, 음수는 반대로 움직임을 뜻합니다. 0.917은 강한 관계이지만 인과관계를 증명하지는 않습니다.</td></tr></tbody></table></div></section>

    <section id="features"><h2>Feature를 어떻게 읽고 모델에 넣는가</h2><p class="card callout"><b>Feature</b>는 청취시간을 예측할 때 모델이 참고하는 입력 정보입니다. 아래의 예상 영향은 데이터에서 확인된 관계와 다음 실험 가설을 구분해 적었습니다. 값이 높다고 무조건 좋은 것이 아니라, 청취시간이 어떻게 달라질지를 예측하는 단서입니다.</p><div class="table-wrap"><table class="guide"><thead><tr><th>Feature</th><th>의미와 유형</th><th>현재 확인된 영향</th><th>사용할 표현과 interaction 후보</th></tr></thead><tbody>
      <tr><td><b>Episode_Length_minutes</b><span class="type">수치</span></td><td>에피소드 전체 길이(분)</td><td>타깃 상관 0.917로 가장 강합니다. 길이가 길수록 청취시간도 대체로 증가합니다. 결측 11.61%는 가장 큰 오류 병목입니다.</td><td>원 수치, 결측 플래그, 현실 범위 제한을 비교합니다. <code>길이 × 장르</code>, <code>길이 × 광고 수</code>는 같은 길이에서 청취 패턴이 달라지는지 보는 가설입니다.</td></tr>
      <tr><td><b>Number_of_Ads</b><span class="type">수치</span></td><td>에피소드의 광고 수</td><td>단순 상관은 약한 음의 관계이며, B002 중요도는 3.55%로 길이 다음입니다. 극단값 2,063개는 별도 품질 문제입니다.</td><td>원 수치와 이상값 정제를 한 요소씩 비교합니다. <code>광고 수 × 길이</code>로 같은 광고 수의 부담이 에피소드 길이에 따라 다른지 확인할 수 있습니다.</td></tr>
      <tr><td><b>Host_Popularity_percentage</b><span class="type">수치</span></td><td>호스트 인기도(%)</td><td>타깃과의 단순 상관은 0.051, B002 중요도는 2.06%로 단독 영향은 제한적입니다.</td><td>원 수치를 우선 사용합니다. <code>호스트 인기도 × Podcast_Name</code>은 프로그램별 호스트 효과 차이를 보는 후보입니다.</td></tr>
      <tr><td><b>Guest_Popularity_percentage</b><span class="type">수치</span></td><td>게스트 인기도(%)</td><td>단순 상관은 0.016으로 약하고, 19.47%가 결측입니다. 값 자체보다 게스트 정보의 존재 여부가 신호일 수 있습니다.</td><td>원 수치와 결측 플래그를 비교합니다. <code>게스트 인기도 × 장르</code>, <code>게스트 결측 × Podcast_Name</code>이 후보입니다.</td></tr>
      <tr><td><b>Podcast_Name</b><span class="type">범주</span></td><td>48개 팟캐스트 프로그램 이름</td><td>프로그램별 평균 청취시간은 41.80~48.11분입니다. 이 차이에는 장르와 에피소드 구성 효과가 함께 섞일 수 있습니다.</td><td>순서 없는 범주로 CatBoost에 전달합니다. <code>프로그램 × 회차</code>, <code>프로그램 × 결측 상태</code>를 후보로 봅니다.</td></tr>
      <tr><td><b>Episode_Title</b><span class="type">범주</span></td><td>Episode 1~100 형식의 회차 제목</td><td>범주별 평균은 40.62~51.22분으로 차이가 가장 크지만, 회차 자체 효과와 표본 변동을 구분해야 합니다.</td><td>원 범주와 숫자로 추출한 회차 번호를 각각 검증합니다. 회차 번호의 순서·구간 표현도 단일 변경 실험 후보입니다.</td></tr>
      <tr><td><b>Genre</b><span class="type">범주</span></td><td>10개 팟캐스트 장르</td><td>장르별 평균은 44.41~46.58분으로 단독 차이는 작습니다.</td><td>순서 없는 범주로 사용합니다. <code>장르 × 길이</code>, <code>장르 × 감정</code>은 조건부 효과 후보입니다.</td></tr>
      <tr><td><b>Publication_Day</b><span class="type">범주</span></td><td>공개 요일</td><td>요일별 평균 차이는 작고, 실제 시간순 인덱스가 아니므로 시계열 정보로 해석하지 않습니다.</td><td>순서 없는 범주로 사용합니다. <code>요일 × 공개 시간대</code> 조합을 생활 패턴 가설로 검증할 수 있습니다.</td></tr>
      <tr><td><b>Publication_Time</b><span class="type">범주</span></td><td>Morning 등 4개 공개 시간대</td><td>시간대별 평균은 44.76~46.46분으로 약한 차이가 있습니다.</td><td>순서 없는 범주로 사용하며, <code>요일 × 시간대</code>, <code>시간대 × 장르</code>를 후보로 둡니다.</td></tr>
      <tr><td><b>Episode_Sentiment</b><span class="type">범주</span></td><td>Positive, Neutral, Negative 감정</td><td>감정별 평균은 44.10~46.72분입니다. 단독 효과보다 콘텐츠 맥락과의 조합 가능성이 있습니다.</td><td>순서 없는 범주로 사용합니다. <code>감정 × 장르</code>를 후보로 검증합니다.</td></tr>
      <tr><td><b>id</b><span class="type">식별자</span></td><td>각 행을 구분하는 번호</td><td>ID 10분위별 타깃 평균이 안정적이며, 실제 청취 행동을 설명하는 정보가 아닙니다.</td><td>모델 입력에서는 제외하고 제출 행 순서를 맞추는 데만 사용합니다.</td></tr>
    </tbody></table></div><div class="card" style="margin-top:14px"><h3>표현 방식 요약</h3><p><b>수치형</b>은 크기와 거리 정보를 유지하고, <b>범주형</b>은 이름 사이에 임의의 순서를 만들지 않습니다. <b>결측 플래그</b>는 값이 없다는 상태를 별도 정보로 주며, <b>interaction</b>은 두 feature가 함께 있을 때만 나타나는 패턴을 표현합니다.</p><p class="muted">현재 B002 CatBoost는 원본 수치·범주와 비선형 상호작용을 자동 학습합니다. 명시적 interaction은 아직 성능이 확인된 결과가 아니라 우선순위가 있는 실험 후보이며, 한 번에 하나씩 추가해 OOF RMSE로 판단합니다.</p></div></section>

    <section id="roadmap"><h2>전체 진행 로드맵</h2><p class="card callout"><b>현재 위치:</b> 문제 이해, 데이터 품질 진단, 고정 검증과 기준 모델을 완료했습니다. 이제 가장 큰 병목인 길이 결측 구간을 설명할 feature 표현과 별도 모델링을 한 요소씩 검증하는 단계입니다.</p><div class="roadmap">
      <article class="roadmap-item"><span class="status done">완료</span><div><b>1. 문제와 목표 이해</b><p>청취시간 회귀 문제, RMSE의 방향, 제출 형식을 확인했습니다.</p></div></article>
      <article class="roadmap-item"><span class="status done">완료</span><div><b>2. 데이터 이해와 품질 진단</b><p>타깃 분포, 결측, 극단값, 중복과 train/test 분포 차이를 점검했습니다.</p></div></article>
      <article class="roadmap-item"><span class="status done">완료</span><div><b>3. 검증과 기준선 고정</b><p>seed 326 shuffled 5-Fold에서 평균, 길이 선형식, CatBoost 기준선을 비교했습니다.</p></div></article>
      <article class="roadmap-item"><span class="status done">채택</span><div><b>4. 예측 범위 안전장치</b><p>P001의 0~120분 제한을 채택했습니다. 점수 개선은 0.000030으로 작아 핵심 개선책은 아닙니다.</p></div></article>
      <article class="roadmap-item"><span class="status now">현재</span><div><b>5. 길이 결측 병목 개선</b><p>결측 플래그, 그룹 기반 길이 정보, 결측 전용 모델과 잔차 모델을 동일 fold에서 검증합니다.</p></div></article>
      <article class="roadmap-item"><span class="status next">다음</span><div><b>6. Feature 표현과 모델 확장</b><p>회차 번호와 우선 interaction을 하나씩 비교하고, 필요할 때 LightGBM·XGBoost의 다른 오류 패턴을 확인합니다.</p></div></article>
      <article class="roadmap-item"><span class="status later">이후</span><div><b>7. 앙상블과 Kaggle 제출</b><p>OOF에서 실제 개선된 모델만 결합하고 제출 형식, 공개 점수와 로컬 검증 차이를 확인합니다.</p></div></article>
      <article class="roadmap-item"><span class="status later">반복</span><div><b>8. 결과 해석과 새 가설</b><p>어느 구간이 개선·악화됐는지 확인하고, 근거가 생긴 다음 단일 변경 실험으로 돌아갑니다.</p></div></article>
    </div></section>

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

    {missing_residual_html.strip()}

    <section id="experiments"><h2>6. 실험 진행</h2>{experiment_html}{completed_summary}<div class="card" style="margin-top:14px">
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
        "missing_residual": result["missing_residual"],
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
