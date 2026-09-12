"""Reproducible OOF training framework for Podcast Listening Time.

Raw CSV files are read-only. Derived predictions are written under outputs/, and
compact experiment results are recorded in benchmarks/experiments.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_CANDIDATES = [
    PROJECT_ROOT / ".runtime_pkgs",
    PROJECT_ROOT.parent / "bank" / ".runtime_pkgs",
]
for package_dir in PACKAGE_CANDIDATES:
    if package_dir.exists():
        sys.path.insert(0, str(package_dir))
        break

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import OrdinalEncoder


DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
REGISTRY_PATH = PROJECT_ROOT / "benchmarks" / "experiments.csv"
TARGET = "Listening_Time_minutes"
ID_COLUMN = "id"
SEED = 326
FOLDS = 5
PREDICTION_LOWER = 0.0
PREDICTION_UPPER = 120.0

REGISTRY_FIELDS = [
    "experiment_id", "date", "purpose", "baseline_id", "changed_element",
    "model", "features", "validation", "seed", "cv_rmse", "fold_std",
    "fold_scores", "public_lb", "runtime_minutes", "status", "interpretation",
]


@dataclass(frozen=True)
class Experiment:
    experiment_id: str
    purpose: str
    model: str
    feature_set: str = "base"
    target_mode: str = "direct"
    baseline_id: str = ""
    changed_element: str = ""
    interpretation: str = ""


@dataclass(frozen=True)
class ModelBudget:
    """Iteration limits separated from experiment definitions for cheap screening."""

    catboost_iterations: int = 500
    catboost_early_stopping: int = 60
    lightgbm_estimators: int = 1600
    lightgbm_early_stopping: int = 100
    xgboost_estimators: int = 1200
    xgboost_early_stopping: int = 80


DEFAULT_MODEL_BUDGET = ModelBudget()


EXPERIMENTS = {
    "B000": Experiment(
        "B000", "평균 예측 최저 기준", "mean", changed_element="학습 fold 타깃 평균",
        interpretation="피처를 사용하지 않는 문제 난이도 기준",
    ),
    "B001": Experiment(
        "B001", "에피소드 길이 구조 기준", "linear_length", baseline_id="B000",
        changed_element="길이와 길이 결측 플래그 사용",
        interpretation="가장 강한 단일 변수 관계를 선형식으로 측정",
    ),
    "B002": Experiment(
        "B002", "전체 원본 피처 기준선", "catboost", baseline_id="B001",
        changed_element="ID를 제외한 원본 피처 전체",
        interpretation="범주형과 결측을 직접 처리하는 강한 단일 모델 기준선",
    ),
    "E101": Experiment(
        "E101", "에피소드 길이 이상값 처리", "catboost", "clean_length", baseline_id="B002",
        changed_element="Episode_Length_minutes 범위 이탈만 결측 처리",
        interpretation="극단 길이가 검증 오차에 미치는 영향 분리",
    ),
    "E102": Experiment(
        "E102", "호스트 인기도 이상값 처리", "catboost", "clean_host", baseline_id="B002",
        changed_element="Host_Popularity_percentage 범위 이탈만 결측 처리",
        interpretation="호스트 인기도 범위 오류의 영향 분리",
    ),
    "E103": Experiment(
        "E103", "게스트 인기도 이상값 처리", "catboost", "clean_guest", baseline_id="B002",
        changed_element="Guest_Popularity_percentage 범위 이탈만 결측 처리",
        interpretation="게스트 인기도 범위 오류의 영향 분리",
    ),
    "E104": Experiment(
        "E104", "광고 수 이상값 처리", "catboost", "clean_ads", baseline_id="B002",
        changed_element="Number_of_Ads 범위·정수 조건 이탈만 결측 처리",
        interpretation="광고 수 오류가 모델에 미치는 영향 분리",
    ),
    "E201": Experiment(
        "E201", "결측 상태 피처 검증", "catboost", "missing_indicators", baseline_id="B002",
        changed_element="수치형 결측 플래그 추가",
        interpretation="결측이라는 상태 자체가 추가 신호인지 확인",
    ),
    "E202": Experiment(
        "E202", "길이 전역 중앙값 대체", "catboost", "length_global_median", baseline_id="B002",
        changed_element="Episode_Length_minutes를 학습 fold 중앙값으로 대체",
        interpretation="길이 결측을 단일 대표값으로 채우는 효과 확인",
    ),
    "E203": Experiment(
        "E203", "길이 그룹 중앙값 대체", "catboost", "length_group_median", baseline_id="E202",
        changed_element="길이를 학습 fold의 Podcast_Name×Episode_Title 중앙값으로 대체",
        interpretation="팟캐스트와 회차 정보로 결측 길이를 복원하는 효과 확인",
    ),
    "E301": Experiment(
        "E301", "에피소드 번호 피처 검증", "catboost", "episode_number", baseline_id="B002",
        changed_element="Episode_Title에서 episode_number 추가",
        interpretation="제목에 포함된 순서 정보를 수치로 명시",
    ),
    "E401": Experiment(
        "E401", "길이 기준식 잔차 예측", "catboost", "base", "length_residual", "B002",
        "직접 타깃 대신 fold 내부 길이 선형식의 잔차 예측",
        "강한 길이 관계와 나머지 조절 요인을 분리",
    ),
    "M001": Experiment(
        "M001", "LightGBM 모델 비교", "lightgbm", baseline_id="B002",
        changed_element="모델만 CatBoost에서 LightGBM으로 변경",
        interpretation="같은 피처와 fold에서 모델 다양성 비교",
    ),
    "M002": Experiment(
        "M002", "XGBoost 모델 비교", "xgboost", baseline_id="B002",
        changed_element="모델만 CatBoost에서 XGBoost로 변경",
        interpretation="같은 피처와 fold에서 모델 다양성 비교",
    ),
}


def rmse(y_true: pd.Series | np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, prediction)))


def validate_data_contract(train: pd.DataFrame, test: pd.DataFrame, sample: pd.DataFrame) -> None:
    if TARGET not in train or TARGET in test:
        raise ValueError("Target must exist only in train.csv")
    expected_test = [column for column in train.columns if column != TARGET]
    if list(test.columns) != expected_test:
        raise ValueError("Train/test feature columns or order do not match")
    if list(sample.columns) != [ID_COLUMN, TARGET]:
        raise ValueError("sample_submission.csv must contain id and target columns")
    if len(sample) != len(test) or not sample[ID_COLUMN].equals(test[ID_COLUMN]):
        raise ValueError("Submission IDs must match test IDs in the same order")
    if not train[ID_COLUMN].is_unique or not test[ID_COLUMN].is_unique:
        raise ValueError("IDs must be unique within each split")
    if set(train[ID_COLUMN]).intersection(set(test[ID_COLUMN])):
        raise ValueError("Train and test IDs must not overlap")
    if train[TARGET].isna().any():
        raise ValueError("Target contains missing values")


def anomaly_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    length = frame["Episode_Length_minutes"]
    host = frame["Host_Popularity_percentage"]
    guest = frame["Guest_Popularity_percentage"]
    ads = frame["Number_of_Ads"]
    return {
        "length": length.notna() & ~length.between(0, 120),
        "host": host.notna() & ~host.between(0, 100),
        "guest": guest.notna() & ~guest.between(0, 100),
        "ads": ads.notna() & (~ads.between(0, 3) | ~np.isclose(ads, np.round(ads))),
    }


def apply_feature_set(frame: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    result = frame.drop(columns=[ID_COLUMN, TARGET], errors="ignore").copy()
    masks = anomaly_masks(frame)
    if feature_set == "clean_length":
        result.loc[masks["length"], "Episode_Length_minutes"] = np.nan
    elif feature_set == "clean_host":
        result.loc[masks["host"], "Host_Popularity_percentage"] = np.nan
    elif feature_set == "clean_guest":
        result.loc[masks["guest"], "Guest_Popularity_percentage"] = np.nan
    elif feature_set == "clean_ads":
        result.loc[masks["ads"], "Number_of_Ads"] = np.nan
    elif feature_set == "missing_indicators":
        for column in result.select_dtypes(include=np.number).columns:
            result[f"{column}_missing"] = result[column].isna().astype("int8")
    elif feature_set == "episode_number":
        result["episode_number"] = (
            result["Episode_Title"].str.extract(r"(\d+)", expand=False).astype(float)
        )
    elif feature_set not in {"base", "length_global_median", "length_group_median"}:
        raise ValueError(f"Unknown feature set: {feature_set}")
    return result


def prepare_fold_features(
    train_x: pd.DataFrame,
    valid_x: pd.DataFrame,
    test_x: pd.DataFrame,
    feature_set: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit feature-only imputations on one training fold and apply them elsewhere."""
    if feature_set not in {"length_global_median", "length_group_median"}:
        return train_x, valid_x, test_x
    column = "Episode_Length_minutes"
    global_median = float(train_x[column].median())
    frames = [train_x.copy(), valid_x.copy(), test_x.copy()]
    if feature_set == "length_global_median":
        for frame in frames:
            frame[column] = frame[column].fillna(global_median)
        return tuple(frames)

    group_columns = ["Podcast_Name", "Episode_Title"]
    group_medians = train_x.groupby(group_columns, dropna=False)[column].median()
    podcast_medians = train_x.groupby("Podcast_Name", dropna=False)[column].median()
    for frame in frames:
        missing = frame[column].isna()
        keys = pd.MultiIndex.from_frame(frame.loc[missing, group_columns])
        group_values = pd.Series(group_medians.reindex(keys).to_numpy(), index=frame.index[missing])
        podcast_values = frame.loc[missing, "Podcast_Name"].map(podcast_medians)
        frame.loc[missing, column] = group_values.fillna(podcast_values).fillna(global_median)
    return tuple(frames)


def make_folds(row_count: int, folds: int = FOLDS, seed: int = SEED) -> np.ndarray:
    assignments = np.full(row_count, -1, dtype=np.int8)
    splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold, (_, valid_idx) in enumerate(splitter.split(np.arange(row_count))):
        assignments[valid_idx] = fold
    validate_fold_assignments(assignments, row_count, folds)
    return assignments


def validate_fold_assignments(assignments: np.ndarray, row_count: int, folds: int) -> None:
    if len(assignments) != row_count or np.any(assignments < 0):
        raise ValueError("Every training row must belong to one validation fold")
    if set(np.unique(assignments)) != set(range(folds)):
        raise ValueError("Fold identifiers are incomplete")
    sizes = np.bincount(assignments, minlength=folds)
    if sizes.max() - sizes.min() > 1:
        raise ValueError("Fold sizes differ by more than one row")


def validate_predictions(values: np.ndarray, expected_length: int, name: str) -> None:
    if len(values) != expected_length:
        raise ValueError(f"{name} length mismatch: {len(values)} != {expected_length}")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains NaN or infinite values")


def linear_length_prediction(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    *frames: pd.DataFrame,
) -> tuple[list[np.ndarray], np.ndarray]:
    column = "Episode_Length_minutes"
    median = float(train_x[column].median())

    def design(frame: pd.DataFrame) -> np.ndarray:
        raw = frame[column]
        return np.column_stack([
            np.ones(len(frame)), raw.fillna(median).to_numpy(), raw.isna().astype(float).to_numpy(),
        ])

    train_design = design(train_x)
    coefficients = np.linalg.lstsq(train_design, train_y.to_numpy(), rcond=None)[0]
    return [design(frame) @ coefficients for frame in frames], coefficients


def ordinal_encode(
    train_x: pd.DataFrame, valid_x: pd.DataFrame, test_x: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    categorical = list(train_x.select_dtypes(exclude=np.number).columns)
    numeric = [column for column in train_x.columns if column not in categorical]
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    train_cat = encoder.fit_transform(train_x[categorical]) if categorical else np.empty((len(train_x), 0))
    valid_cat = encoder.transform(valid_x[categorical]) if categorical else np.empty((len(valid_x), 0))
    test_cat = encoder.transform(test_x[categorical]) if categorical else np.empty((len(test_x), 0))
    medians = train_x[numeric].median()
    return (
        np.column_stack([train_x[numeric].fillna(medians).to_numpy(), train_cat]),
        np.column_stack([valid_x[numeric].fillna(medians).to_numpy(), valid_cat]),
        np.column_stack([test_x[numeric].fillna(medians).to_numpy(), test_cat]),
    )


def fit_tree_model(
    model_name: str,
    train_x: pd.DataFrame,
    train_y: pd.Series,
    valid_x: pd.DataFrame,
    valid_y: pd.Series,
    test_x: pd.DataFrame | None,
    seed: int,
    budget: ModelBudget = DEFAULT_MODEL_BUDGET,
) -> tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    if model_name == "catboost":
        from catboost import CatBoostRegressor

        categorical = list(train_x.select_dtypes(exclude=np.number).columns)
        model = CatBoostRegressor(
            iterations=budget.catboost_iterations, depth=7, learning_rate=0.08, loss_function="RMSE",
            eval_metric="RMSE", l2_leaf_reg=5.0, random_strength=0.5,
            bootstrap_type="Bernoulli", subsample=0.8, random_seed=seed,
            thread_count=-1, allow_writing_files=False, verbose=False,
        )
        model.fit(
            train_x, train_y, cat_features=categorical, eval_set=(valid_x, valid_y),
            early_stopping_rounds=budget.catboost_early_stopping, use_best_model=True,
        )
        iteration = int(model.get_best_iteration() + 1)
        importance = np.asarray(model.get_feature_importance(), dtype=float)
        test_prediction = model.predict(test_x) if test_x is not None else np.empty(0)
        return model.predict(valid_x), test_prediction, iteration, importance

    encoding_test = test_x if test_x is not None else valid_x.iloc[:0]
    encoded_train, encoded_valid, encoded_test = ordinal_encode(train_x, valid_x, encoding_test)
    if model_name == "lightgbm":
        from lightgbm import LGBMRegressor, early_stopping, log_evaluation

        model = LGBMRegressor(
            n_estimators=budget.lightgbm_estimators, learning_rate=0.04, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
            random_state=seed, n_jobs=-1, verbosity=-1,
        )
        model.fit(
            encoded_train, train_y, eval_set=[(encoded_valid, valid_y)],
            eval_metric="rmse", callbacks=[early_stopping(budget.lightgbm_early_stopping, verbose=False), log_evaluation(0)],
        )
        iteration = int(model.best_iteration_)
    elif model_name == "xgboost":
        from xgboost import XGBRegressor

        model = XGBRegressor(
            n_estimators=budget.xgboost_estimators, learning_rate=0.04, max_depth=7, min_child_weight=5,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0,
            objective="reg:squarederror", eval_metric="rmse", tree_method="hist",
            random_state=seed, n_jobs=-1, early_stopping_rounds=budget.xgboost_early_stopping,
        )
        model.fit(encoded_train, train_y, eval_set=[(encoded_valid, valid_y)], verbose=False)
        iteration = int(model.best_iteration + 1)
    else:
        raise ValueError(f"Unknown tree model: {model_name}")
    importance = np.asarray(model.feature_importances_, dtype=float)
    test_prediction = model.predict(encoded_test) if test_x is not None else np.empty(0)
    return model.predict(encoded_valid), test_prediction, iteration, importance


def fit_fold(
    experiment: Experiment,
    train_x: pd.DataFrame,
    train_y: pd.Series,
    valid_x: pd.DataFrame,
    valid_y: pd.Series,
    test_x: pd.DataFrame | None,
    seed: int,
    budget: ModelBudget = DEFAULT_MODEL_BUDGET,
) -> tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    if experiment.model == "mean":
        value = float(train_y.mean())
        test_length = len(test_x) if test_x is not None else 0
        return np.full(len(valid_x), value), np.full(test_length, value), 1, np.zeros(train_x.shape[1])
    if experiment.model == "linear_length":
        frames = (valid_x, test_x) if test_x is not None else (valid_x,)
        predictions, _ = linear_length_prediction(train_x, train_y, *frames)
        test_prediction = predictions[1] if test_x is not None else np.empty(0)
        return predictions[0], test_prediction, 1, np.zeros(train_x.shape[1])
    if experiment.target_mode == "length_residual":
        frames = (train_x, valid_x, test_x) if test_x is not None else (train_x, valid_x)
        bases, _ = linear_length_prediction(train_x, train_y, *frames)
        residual_y = train_y.to_numpy() - bases[0]
        residual_valid = valid_y.to_numpy() - bases[1]
        valid_residual, test_residual, iteration, importance = fit_tree_model(
            experiment.model, train_x, pd.Series(residual_y, index=train_x.index),
            valid_x, pd.Series(residual_valid, index=valid_x.index), test_x, seed, budget,
        )
        test_prediction = bases[2] + test_residual if test_x is not None else np.empty(0)
        return bases[1] + valid_residual, test_prediction, iteration, importance
    return fit_tree_model(experiment.model, train_x, train_y, valid_x, valid_y, test_x, seed, budget)


def segment_rmse(train: pd.DataFrame, target: pd.Series, predictions: np.ndarray) -> list[dict]:
    length = train["Episode_Length_minutes"]
    any_anomaly = pd.DataFrame(anomaly_masks(train)).any(axis=1)
    length_bins = pd.qcut(length, 10, labels=False, duplicates="drop").astype("Int64").astype(str)
    segments: list[tuple[str, pd.Series]] = [
        ("episode_length_missing", length.isna().map({True: "yes", False: "no"})),
        ("episode_length_decile", length_bins),
        ("any_range_anomaly", any_anomaly.map({True: "yes", False: "no"})),
        ("number_of_ads", train["Number_of_Ads"].fillna(-1).astype(str)),
        ("episode_sentiment", train["Episode_Sentiment"].astype(str)),
    ]
    rows: list[dict] = []
    for variable, groups in segments:
        groups = groups.fillna("(NA)").astype(str)
        for value in sorted(groups.unique()):
            mask = groups.eq(value).to_numpy()
            if mask.sum() < 20:
                continue
            rows.append({
                "segment": variable, "value": value, "rows": int(mask.sum()),
                "target_mean": float(target[mask].mean()),
                "prediction_mean": float(predictions[mask].mean()),
                "rmse": rmse(target[mask], predictions[mask]),
            })
    return rows


def upsert_registry(row: dict[str, object]) -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict[str, str]] = []
    if REGISTRY_PATH.exists():
        with REGISTRY_PATH.open(encoding="utf-8-sig", newline="") as handle:
            existing = list(csv.DictReader(handle))
    normalized = [{field: item.get(field, "") for field in REGISTRY_FIELDS} for item in existing]
    normalized = [item for item in normalized if item["experiment_id"] != str(row["experiment_id"])]
    normalized.append({field: row.get(field, "") for field in REGISTRY_FIELDS})
    normalized.sort(key=lambda item: item["experiment_id"])
    with REGISTRY_PATH.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REGISTRY_FIELDS)
        writer.writeheader()
        writer.writerows(normalized)


def save_results(
    experiment: Experiment,
    train: pd.DataFrame,
    test: pd.DataFrame,
    sample: pd.DataFrame,
    assignments: np.ndarray,
    oof: np.ndarray,
    test_prediction: np.ndarray,
    metrics: dict,
    importance: pd.DataFrame | None = None,
) -> None:
    output = OUTPUT_DIR / experiment.experiment_id
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "predictions.npz", oof=oof, test=test_prediction, folds=assignments)
    pd.DataFrame({ID_COLUMN: train[ID_COLUMN], "fold": assignments, "prediction": oof}).to_csv(
        output / "oof.csv.gz", index=False, compression="gzip",
    )
    pd.DataFrame({ID_COLUMN: test[ID_COLUMN], "prediction": test_prediction}).to_csv(
        output / "test_predictions.csv.gz", index=False, compression="gzip",
    )
    submission = sample.copy()
    submission[TARGET] = test_prediction
    submission.to_csv(output / "submission.csv", index=False)
    (output / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    pd.DataFrame(metrics["segments"]).to_csv(output / "segment_rmse.csv", index=False, encoding="utf-8-sig")
    if importance is not None:
        importance.to_csv(output / "feature_importance.csv", index=False, encoding="utf-8-sig")


def train_experiment(experiment: Experiment, folds: int = FOLDS, seed: int = SEED) -> dict:
    start = time.perf_counter()
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
    validate_data_contract(train, test, sample)
    train_x = apply_feature_set(train, experiment.feature_set)
    test_x = apply_feature_set(test, experiment.feature_set)
    target = train[TARGET].astype(float)
    assignments = make_folds(len(train), folds=folds, seed=seed)
    oof = np.zeros(len(train), dtype=np.float64)
    test_prediction = np.zeros(len(test), dtype=np.float64)
    fold_scores: list[float] = []
    best_iterations: list[int] = []
    importances: list[np.ndarray] = []

    for fold in range(folds):
        valid_mask = assignments == fold
        train_idx, valid_idx = np.flatnonzero(~valid_mask), np.flatnonzero(valid_mask)
        fold_train_x, fold_valid_x, fold_test_x = prepare_fold_features(
            train_x.iloc[train_idx], train_x.iloc[valid_idx], test_x, experiment.feature_set,
        )
        valid_pred, fold_test_pred, best_iteration, importance = fit_fold(
            experiment, fold_train_x, target.iloc[train_idx],
            fold_valid_x, target.iloc[valid_idx], fold_test_x, seed + fold,
        )
        validate_predictions(valid_pred, len(valid_idx), f"fold {fold} prediction")
        validate_predictions(fold_test_pred, len(test), f"fold {fold} test prediction")
        oof[valid_idx] = valid_pred
        test_prediction += fold_test_pred / folds
        score = rmse(target.iloc[valid_idx], valid_pred)
        fold_scores.append(score)
        best_iterations.append(best_iteration)
        importances.append(importance)
        print(f"fold={fold} rmse={score:.6f} best_iteration={best_iteration}", flush=True)

    validate_predictions(oof, len(train), "OOF prediction")
    validate_predictions(test_prediction, len(test), "test prediction")
    cv_rmse = rmse(target, oof)
    runtime_minutes = (time.perf_counter() - start) / 60
    importance_frame = None
    if importances and np.asarray(importances).shape == (folds, train_x.shape[1]):
        importance_frame = pd.DataFrame({
            "feature": train_x.columns,
            "importance": np.mean(np.vstack(importances), axis=0),
        }).sort_values("importance", ascending=False)
    metrics = {
        **asdict(experiment), "seed": seed, "folds": folds, "rows": len(train),
        "feature_count": train_x.shape[1], "cv_rmse": cv_rmse,
        "fold_scores": fold_scores, "fold_std": float(np.std(fold_scores)),
        "best_iterations": best_iterations, "runtime_minutes": runtime_minutes,
        "prediction_min": float(test_prediction.min()),
        "prediction_max": float(test_prediction.max()),
        "segments": segment_rmse(train, target, oof),
    }
    save_results(
        experiment, train, test, sample, assignments, oof, test_prediction,
        metrics, importance_frame,
    )
    upsert_registry({
        "experiment_id": experiment.experiment_id, "date": date.today().isoformat(),
        "purpose": experiment.purpose, "baseline_id": experiment.baseline_id,
        "changed_element": experiment.changed_element, "model": experiment.model,
        "features": f"{experiment.feature_set} ({train_x.shape[1]})",
        "validation": f"shuffled {folds}-Fold OOF RMSE", "seed": seed,
        "cv_rmse": f"{cv_rmse:.6f}", "fold_std": f"{np.std(fold_scores):.6f}",
        "fold_scores": "|".join(f"{score:.6f}" for score in fold_scores),
        "public_lb": "", "runtime_minutes": f"{runtime_minutes:.2f}", "status": "완료",
        "interpretation": experiment.interpretation,
    })
    print(json.dumps({
        "experiment_id": experiment.experiment_id, "cv_rmse": cv_rmse,
        "fold_std": float(np.std(fold_scores)), "runtime_minutes": runtime_minutes,
    }, indent=2))
    return metrics


def run_clipping_experiment(base_id: str = "B002") -> dict:
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
    validate_data_contract(train, test, sample)
    source = OUTPUT_DIR / base_id / "predictions.npz"
    if not source.exists():
        raise FileNotFoundError(f"Run {base_id} before P001")
    values = np.load(source)
    raw_oof, raw_test, assignments = values["oof"], values["test"], values["folds"]
    oof = np.clip(raw_oof, PREDICTION_LOWER, PREDICTION_UPPER)
    test_prediction = np.clip(raw_test, PREDICTION_LOWER, PREDICTION_UPPER)
    target = train[TARGET].astype(float)
    fold_scores = [rmse(target[assignments == fold], oof[assignments == fold]) for fold in sorted(np.unique(assignments))]
    experiment = Experiment(
        "P001", "예측 범위 제한", "postprocess", baseline_id=base_id,
        changed_element="예측값을 0~120분으로 제한",
        interpretation="RMSE에서 불가능한 극단 예측이 주는 손실을 확인",
    )
    metrics = {
        **asdict(experiment), "seed": SEED, "folds": len(fold_scores), "rows": len(train),
        "feature_count": None, "cv_rmse": rmse(target, oof), "raw_cv_rmse": rmse(target, raw_oof),
        "fold_scores": fold_scores, "fold_std": float(np.std(fold_scores)), "best_iterations": [],
        "runtime_minutes": 0.0, "prediction_min": float(test_prediction.min()),
        "prediction_max": float(test_prediction.max()), "segments": segment_rmse(train, target, oof),
    }
    save_results(experiment, train, test, sample, assignments, oof, test_prediction, metrics)
    upsert_registry({
        "experiment_id": "P001", "date": date.today().isoformat(), "purpose": experiment.purpose,
        "baseline_id": base_id, "changed_element": experiment.changed_element, "model": "postprocess",
        "features": f"reuse {base_id}", "validation": "same OOF predictions", "seed": SEED,
        "cv_rmse": f"{metrics['cv_rmse']:.6f}", "fold_std": f"{metrics['fold_std']:.6f}",
        "fold_scores": "|".join(f"{score:.6f}" for score in fold_scores), "public_lb": "",
        "runtime_minutes": "0.00", "status": "완료", "interpretation": experiment.interpretation,
    })
    print(json.dumps({"experiment_id": "P001", "cv_rmse": metrics["cv_rmse"], "raw_cv_rmse": metrics["raw_cv_rmse"]}, indent=2))
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    choices = sorted([*EXPERIMENTS, "P001"])
    parser.add_argument("--experiment", choices=choices, required=True)
    parser.add_argument("--folds", type=int, default=FOLDS)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.experiment == "P001":
        run_clipping_experiment()
    else:
        train_experiment(EXPERIMENTS[args.experiment], folds=args.folds, seed=args.seed)


if __name__ == "__main__":
    main()
