"""Missing-value and OOF residual diagnostics for the Podcast competition.

The script never changes the raw CSV files.  It reuses saved out-of-fold
predictions so every reported residual is from a row the base model did not
train on.
"""

from __future__ import annotations

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


TARGET = "Listening_Time_minutes"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
RESULT_DIR = PROJECT_ROOT / "logs" / "result"
SUMMARY_PATH = RESULT_DIR / "podcast_missing_residual_analysis.json"
PARTIAL_EXPERIMENTS_PATH = RESULT_DIR / "podcast_partial_experiments.json"


def rmse(values: pd.Series | np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(np.square(array))))


def summarize_groups(frame: pd.DataFrame, group: str, minimum_rows: int = 20) -> pd.DataFrame:
    result = (
        frame.groupby(group, dropna=False, observed=True)
        .agg(
            rows=(TARGET, "size"),
            target_mean=(TARGET, "mean"),
            prediction_mean=("prediction", "mean"),
            bias=("residual", "mean"),
            mae=("residual", lambda values: float(np.mean(np.abs(values)))),
            rmse=("residual", rmse),
            residual_std=("residual", "std"),
        )
        .reset_index()
    )
    result = result[result["rows"] >= minimum_rows].copy()
    result["share"] = result["rows"] / len(frame)
    return result.sort_values("rmse", ascending=False).reset_index(drop=True)


def add_missing_pattern(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    length = np.where(result["Episode_Length_minutes"].isna(), "L-", "L+")
    guest = np.where(result["Guest_Popularity_percentage"].isna(), "G-", "G+")
    ads = np.where(result["Number_of_Ads"].isna(), "A-", "A+")
    result["missing_pattern"] = length + " / " + guest + " / " + ads
    result["missing_count"] = result[
        ["Episode_Length_minutes", "Guest_Popularity_percentage", "Number_of_Ads"]
    ].isna().sum(axis=1)
    return result


def missing_pattern_shift(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    train_rate = train["missing_pattern"].value_counts(normalize=True)
    test_rate = test["missing_pattern"].value_counts(normalize=True)
    counts = train["missing_pattern"].value_counts()
    patterns = sorted(set(train_rate.index) | set(test_rate.index))
    return pd.DataFrame({
        "missing_pattern": patterns,
        "train_rows": [int(counts.get(key, 0)) for key in patterns],
        "train_rate": [float(train_rate.get(key, 0.0)) for key in patterns],
        "test_rate": [float(test_rate.get(key, 0.0)) for key in patterns],
        "gap_percentage_points": [
            float(100 * (test_rate.get(key, 0.0) - train_rate.get(key, 0.0))) for key in patterns
        ],
    }).sort_values("train_rows", ascending=False).reset_index(drop=True)


def missingness_drivers(train: pd.DataFrame, test: pd.DataFrame, missing_column: str) -> pd.DataFrame:
    rows = []
    categorical = [
        "Podcast_Name", "Episode_Title", "Genre", "Publication_Day",
        "Publication_Time", "Episode_Sentiment", "Number_of_Ads",
    ]
    for column in categorical:
        train_group = train.assign(_missing=train[missing_column].isna()).groupby(column, observed=True)["_missing"].agg(["size", "mean"])
        test_group = test.assign(_missing=test[missing_column].isna()).groupby(column, observed=True)["_missing"].agg(["size", "mean"])
        train_rates = train_group.loc[train_group["size"] >= 200, "mean"]
        test_rates = test_group.loc[test_group["size"] >= 100, "mean"]
        common = train_rates.index.intersection(test_rates.index)
        correlation = float(train_rates.loc[common].corr(test_rates.loc[common])) if len(common) > 1 else np.nan
        rows.append({
            "missing_column": missing_column,
            "feature": column,
            "groups": int(len(train_rates)),
            "train_rate_min": float(train_rates.min()),
            "train_rate_max": float(train_rates.max()),
            "train_rate_range": float(train_rates.max() - train_rates.min()),
            "train_test_group_rate_correlation": correlation,
        })
    return pd.DataFrame(rows).sort_values("train_rate_range", ascending=False).reset_index(drop=True)


def crossfit_group_length(train: pd.DataFrame, folds: np.ndarray) -> tuple[pd.Series, dict]:
    """Impute length with fold-train Podcast_Name x Episode_Title medians."""
    prediction = pd.Series(np.nan, index=train.index, dtype=float)
    columns = ["Podcast_Name", "Episode_Title"]
    for fold in sorted(np.unique(folds)):
        fit = train.loc[folds != fold]
        valid = train.loc[folds == fold]
        group_median = fit.groupby(columns, observed=True)["Episode_Length_minutes"].median()
        podcast_median = fit.groupby("Podcast_Name", observed=True)["Episode_Length_minutes"].median()
        keys = pd.MultiIndex.from_frame(valid[columns])
        values = pd.Series(group_median.reindex(keys).to_numpy(), index=valid.index)
        values = values.fillna(valid["Podcast_Name"].map(podcast_median))
        values = values.fillna(float(fit["Episode_Length_minutes"].median()))
        prediction.loc[valid.index] = values

    observed = train["Episode_Length_minutes"].notna()
    errors = train.loc[observed, "Episode_Length_minutes"] - prediction.loc[observed]
    missing = ~observed
    quintile = pd.qcut(prediction.loc[missing], 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"], duplicates="drop")
    missing_view = pd.DataFrame({TARGET: train.loc[missing, TARGET], "imputed_length": prediction.loc[missing], "quintile": quintile})
    quintile_summary = (
        missing_view.groupby("quintile", observed=True)
        .agg(rows=(TARGET, "size"), imputed_length_mean=("imputed_length", "mean"), target_mean=(TARGET, "mean"))
        .reset_index()
    )
    diagnostics = {
        "observed_rows": int(observed.sum()),
        "observed_length_mae": float(np.mean(np.abs(errors))),
        "observed_length_rmse": rmse(errors),
        "observed_length_baseline_rmse": rmse(
            train.loc[observed, "Episode_Length_minutes"] - train.loc[observed, "Episode_Length_minutes"].median()
        ),
        "missing_rows": int(missing.sum()),
        "missing_imputation_coverage": float(prediction.loc[missing].notna().mean()),
        "missing_imputed_length_target_correlation": float(
            np.corrcoef(prediction.loc[missing], train.loc[missing, TARGET])[0, 1]
        ),
        "missing_quintiles": quintile_summary.to_dict(orient="records"),
    }
    return prediction, diagnostics


def residual_driver_summary(frame: pd.DataFrame, length_missing_only: bool = False) -> pd.DataFrame:
    source = frame[frame["Episode_Length_minutes"].isna()].copy() if length_missing_only else frame.copy()
    rows = []
    categorical = [
        "Podcast_Name", "Episode_Title", "Genre", "Publication_Day",
        "Publication_Time", "Episode_Sentiment", "Number_of_Ads",
    ]
    for column in categorical:
        grouped = source.assign(_group=source[column].fillna("(missing)").astype(str)).groupby("_group")["residual"]
        stats = grouped.agg(["size", "mean"])
        stats = stats[stats["size"] >= 200]
        if stats.empty:
            continue
        low = stats["mean"].idxmin()
        high = stats["mean"].idxmax()
        rows.append({
            "feature": column,
            "eligible_groups": int(len(stats)),
            "mean_residual_range": float(stats.loc[high, "mean"] - stats.loc[low, "mean"]),
            "lowest_group": str(low),
            "lowest_bias": float(stats.loc[low, "mean"]),
            "highest_group": str(high),
            "highest_bias": float(stats.loc[high, "mean"]),
        })
    return pd.DataFrame(rows).sort_values("mean_residual_range", ascending=False).reset_index(drop=True)


def prediction_deciles(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["prediction_decile"] = pd.qcut(
        result["prediction"], 10, labels=[f"D{i}" for i in range(1, 11)], duplicates="drop"
    )
    return summarize_groups(result, "prediction_decile", minimum_rows=1).sort_values("prediction_decile")


def load_prediction_frame(train: pd.DataFrame, experiment_id: str) -> pd.DataFrame | None:
    path = OUTPUT_DIR / experiment_id / "predictions.npz"
    if not path.exists():
        return None
    saved = np.load(path)
    result = train.copy()
    result["prediction"] = saved["oof"]
    result["residual"] = result[TARGET] - result["prediction"]
    result["fold"] = saved["folds"]
    return add_missing_pattern(result)


def analyze(base_id: str = "B002") -> dict:
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = add_missing_pattern(pd.read_csv(DATA_DIR / "test.csv"))
    frame = load_prediction_frame(train, base_id)
    if frame is None:
        raise FileNotFoundError(f"Run {base_id} before residual analysis")

    _, length_diagnostics = crossfit_group_length(train, frame["fold"].to_numpy())
    experiment_comparison = []
    for experiment_id in [base_id, "E201", "E202", "E203", "E401"]:
        candidate = load_prediction_frame(train, experiment_id)
        if candidate is None:
            continue
        length_missing = candidate["Episode_Length_minutes"].isna()
        experiment_comparison.append({
            "experiment_id": experiment_id,
            "overall_rmse": rmse(candidate["residual"]),
            "length_present_rmse": rmse(candidate.loc[~length_missing, "residual"]),
            "length_missing_rmse": rmse(candidate.loc[length_missing, "residual"]),
            "overall_bias": float(candidate["residual"].mean()),
            "length_missing_bias": float(candidate.loc[length_missing, "residual"].mean()),
        })
    partial_experiments = []
    if PARTIAL_EXPERIMENTS_PATH.exists():
        partial_experiments = json.loads(PARTIAL_EXPERIMENTS_PATH.read_text(encoding="utf-8"))

    result = {
        "base_id": base_id,
        "missing_pattern_shift": missing_pattern_shift(frame, test).to_dict(orient="records"),
        "missing_pattern_residuals": summarize_groups(frame, "missing_pattern").to_dict(orient="records"),
        "missing_count_residuals": summarize_groups(frame, "missing_count").to_dict(orient="records"),
        "length_missingness_drivers": missingness_drivers(
            train, test, "Episode_Length_minutes"
        ).to_dict(orient="records"),
        "guest_missingness_drivers": missingness_drivers(
            train, test, "Guest_Popularity_percentage"
        ).to_dict(orient="records"),
        "length_missing_squared_error_share": float(
            np.square(frame.loc[frame["Episode_Length_minutes"].isna(), "residual"]).sum()
            / np.square(frame["residual"]).sum()
        ),
        "prediction_deciles": prediction_deciles(frame).to_dict(orient="records"),
        "residual_drivers_all": residual_driver_summary(frame).to_dict(orient="records"),
        "residual_drivers_length_missing": residual_driver_summary(frame, True).to_dict(orient="records"),
        "group_length_imputation": length_diagnostics,
        "experiment_comparison": experiment_comparison,
        "partial_experiments": partial_experiments,
    }
    return result


def write_outputs(result: dict) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    for key in [
        "missing_pattern_shift", "missing_pattern_residuals", "prediction_deciles",
        "residual_drivers_all", "residual_drivers_length_missing", "experiment_comparison",
        "length_missingness_drivers", "guest_missingness_drivers",
    ]:
        pd.DataFrame(result[key]).to_csv(RESULT_DIR / f"{key}.csv", index=False, encoding="utf-8-sig")
    print(json.dumps({
        "summary": str(SUMMARY_PATH),
        "patterns": len(result["missing_pattern_residuals"]),
        "experiments": [row["experiment_id"] for row in result["experiment_comparison"]],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    write_outputs(analyze())
