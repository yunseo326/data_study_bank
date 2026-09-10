"""Reproducible OOF training framework for the Bank Kaggle competition.

Raw CSV files are read-only. Derived predictions and model diagnostics are written
under bank/outputs (gitignored), while the compact experiment registry is tracked.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LOCAL_PACKAGES = ROOT / "bank" / ".runtime_pkgs"
if LOCAL_PACKAGES.exists():
    sys.path.insert(0, str(LOCAL_PACKAGES))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder


DATA_DIR = ROOT / "bank" / "data"
OUTPUT_DIR = ROOT / "bank" / "outputs"
REGISTRY_PATH = ROOT / "bank" / "benchmarks" / "our_experiments.csv"
TARGET = "y"
ID_COLUMN = "id"
SEED = 326
FOLDS = 5

REGISTRY_FIELDS = [
    "experiment_id", "date", "purpose", "baseline_id", "changed_element",
    "model", "features", "validation", "seed", "cv_auc", "fold_std",
    "fold_scores", "public_lb", "runtime_minutes", "status", "interpretation",
]


@dataclass(frozen=True)
class Experiment:
    experiment_id: str
    purpose: str
    model: str = "catboost"
    feature_set: str = "base"
    baseline_id: str = ""
    changed_element: str = "첫 기준 모델"
    interpretation: str = ""
    model_profile: str = "baseline"


@dataclass(frozen=True)
class FitResult:
    valid_prediction: np.ndarray
    test_prediction: np.ndarray
    best_iteration: int
    feature_importance: dict[str, float]


EXPERIMENTS = {
    "E001": Experiment(
        "E001", "Kaggle 점수 트랙 기준선", interpretation="제공된 16개 피처를 사용하는 첫 CatBoost 기준선",
    ),
    "E002": Experiment(
        "E002", "통화 전 현실 트랙 기준선", feature_set="no_duration",
        baseline_id="E001", changed_element="duration 제외",
        interpretation="예측 시점에 알 수 없는 duration의 성능 기여를 분리",
    ),
    "E003": Experiment(
        "E003", "이전 연락 상태 피처 검증", feature_set="previous_contacted",
        baseline_id="E001", changed_element="previous_contacted 추가",
        interpretation="pdays=-1의 공식 의미를 명시적 상태 피처로 분리",
    ),
    "E004": Experiment(
        "E004", "LightGBM 범주형 처리 검증", model="lightgbm",
        baseline_id="E001-LIGHTGBM", changed_element="범주형 피처 명시",
        interpretation="같은 파라미터에서 범주형 열을 LightGBM 범주형 피처로 명시",
        model_profile="native_categorical",
    ),
    "E005": Experiment(
        "E005", "LightGBM 학습 상한 검증", model="lightgbm",
        baseline_id="E004", changed_element="n_estimators 1500 → 3000",
        interpretation="E004의 모든 fold가 학습 상한에 도달해 최대 트리 수만 확대",
        model_profile="native_cat_more_trees",
    ),
    "E006": Experiment(
        "E006", "LightGBM 리프 복잡도 검증", model="lightgbm",
        baseline_id="E005", changed_element="num_leaves 31 → 63",
        interpretation="E005를 유지하고 트리 하나가 표현할 수 있는 분할 수만 확대",
        model_profile="native_cat_more_trees_more_leaves",
    ),
    "E007": Experiment(
        "E007", "LightGBM 리프 최소 표본 검증", model="lightgbm",
        baseline_id="E006", changed_element="min_child_samples 20 → 50",
        interpretation="E006을 유지하고 작은 리프에 대한 일반화 제약만 강화",
        model_profile="native_cat_more_trees_more_leaves_larger_min_child",
    ),
    "E008": Experiment(
        "E008", "LightGBM 행 샘플링 검증", model="lightgbm",
        baseline_id="E006", changed_element="subsample_freq 0 → 1",
        interpretation="E006을 유지하고 기존 subsample=0.8 행 샘플링만 활성화",
        model_profile="native_cat_more_trees_more_leaves_active_bagging",
    ),
}


LIGHTGBM_PROFILES = {
    "baseline": {},
    "native_categorical": {"categorical_feature": True},
    "native_cat_more_trees": {"categorical_feature": True, "n_estimators": 3000},
    "native_cat_more_trees_more_leaves": {
        "categorical_feature": True, "n_estimators": 3000, "num_leaves": 63,
    },
    "native_cat_more_trees_more_leaves_larger_min_child": {
        "categorical_feature": True, "n_estimators": 3000, "num_leaves": 63,
        "min_child_samples": 50,
    },
    "native_cat_more_trees_more_leaves_active_bagging": {
        "categorical_feature": True, "n_estimators": 3000, "num_leaves": 63,
        "subsample_freq": 1,
    },
    "more_trees": {"n_estimators": 3000},
    "more_leaves": {"num_leaves": 63},
    "larger_min_child": {"min_child_samples": 50},
    "active_bagging": {"subsample_freq": 1},
}


def lightgbm_parameters(profile: str) -> tuple[dict[str, object], bool]:
    if profile not in LIGHTGBM_PROFILES:
        raise ValueError(f"Unknown LightGBM profile: {profile}")
    overrides = dict(LIGHTGBM_PROFILES[profile])
    use_categorical = bool(overrides.pop("categorical_feature", False))
    parameters: dict[str, object] = {
        "n_estimators": 1500,
        "learning_rate": 0.04,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 0.8,
        "subsample_freq": 0,
        "colsample_bytree": 0.8,
        "reg_lambda": 2.0,
    }
    parameters.update(overrides)
    return parameters, use_categorical


def apply_feature_set(frame: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    result = frame.drop(columns=[ID_COLUMN, TARGET], errors="ignore").copy()
    if feature_set == "no_duration":
        result = result.drop(columns=["duration"])
    elif feature_set == "previous_contacted":
        result["previous_contacted"] = (result["pdays"] != -1).astype("int8")
    elif feature_set != "base":
        raise ValueError(f"Unknown feature set: {feature_set}")
    return result


def make_folds(target: pd.Series, folds: int = FOLDS, seed: int = SEED) -> np.ndarray:
    assignments = np.full(len(target), -1, dtype=np.int8)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold, (_, valid_idx) in enumerate(splitter.split(np.zeros(len(target)), target)):
        assignments[valid_idx] = fold
    validate_fold_assignments(assignments, target, folds)
    return assignments


def validate_fold_assignments(assignments: np.ndarray, target: pd.Series, folds: int) -> None:
    if len(assignments) != len(target) or np.any(assignments < 0):
        raise ValueError("Every training row must belong to exactly one validation fold")
    if set(np.unique(assignments)) != set(range(folds)):
        raise ValueError("Fold identifiers are incomplete")
    overall = float(target.mean())
    for fold in range(folds):
        rate = float(target.iloc[np.flatnonzero(assignments == fold)].mean())
        if abs(rate - overall) > 0.002:
            raise ValueError(f"Fold {fold} target rate differs too much: {rate:.6f}")


def validate_predictions(values: np.ndarray, expected_length: int, name: str) -> None:
    if len(values) != expected_length:
        raise ValueError(f"{name} length mismatch: {len(values)} != {expected_length}")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    if np.any((values < 0) | (values > 1)):
        raise ValueError(f"{name} must contain probabilities in [0, 1]")


def fit_catboost(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    valid_x: pd.DataFrame,
    valid_y: pd.Series,
    test_x: pd.DataFrame,
    seed: int,
    experiment: Experiment,
) -> FitResult:
    from catboost import CatBoostClassifier

    categorical = list(train_x.select_dtypes(exclude=np.number).columns)
    model = CatBoostClassifier(
        iterations=260,
        depth=6,
        learning_rate=0.10,
        loss_function="Logloss",
        eval_metric="AUC",
        l2_leaf_reg=5.0,
        random_strength=0.5,
        bootstrap_type="Bernoulli",
        subsample=0.8,
        random_seed=seed,
        thread_count=-1,
        allow_writing_files=False,
        verbose=False,
    )
    model.fit(
        train_x, train_y, cat_features=categorical,
        eval_set=(valid_x, valid_y), early_stopping_rounds=40, use_best_model=True,
    )
    return FitResult(
        model.predict_proba(valid_x)[:, 1],
        model.predict_proba(test_x)[:, 1],
        int(model.get_best_iteration() + 1),
        dict(zip(train_x.columns, map(float, model.get_feature_importance()))),
    )


def ordinal_encode(
    train_x: pd.DataFrame, valid_x: pd.DataFrame, test_x: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[int]]:
    categorical = list(train_x.select_dtypes(exclude=np.number).columns)
    numeric = [column for column in train_x.columns if column not in categorical]
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    train_cat = encoder.fit_transform(train_x[categorical]) if categorical else np.empty((len(train_x), 0))
    valid_cat = encoder.transform(valid_x[categorical]) if categorical else np.empty((len(valid_x), 0))
    test_cat = encoder.transform(test_x[categorical]) if categorical else np.empty((len(test_x), 0))
    feature_names = numeric + categorical
    categorical_indices = list(range(len(numeric), len(feature_names)))
    return (
        np.column_stack([train_x[numeric].to_numpy(), train_cat]),
        np.column_stack([valid_x[numeric].to_numpy(), valid_cat]),
        np.column_stack([test_x[numeric].to_numpy(), test_cat]),
        feature_names,
        categorical_indices,
    )


def fit_lightgbm(train_x, train_y, valid_x, valid_y, test_x, seed, experiment):
    from lightgbm import LGBMClassifier, early_stopping, log_evaluation

    encoded_train, encoded_valid, encoded_test, feature_names, categorical_indices = ordinal_encode(
        train_x, valid_x, test_x,
    )
    parameters, use_categorical = lightgbm_parameters(experiment.model_profile)
    model = LGBMClassifier(
        **parameters, random_state=seed, n_jobs=-1, verbosity=-1,
    )
    fit_kwargs = {}
    if use_categorical:
        fit_kwargs["categorical_feature"] = categorical_indices
    model.fit(
        encoded_train, train_y, eval_set=[(encoded_valid, valid_y)], eval_metric="auc",
        callbacks=[early_stopping(100, verbose=False), log_evaluation(0)],
        **fit_kwargs,
    )
    return FitResult(
        model.predict_proba(encoded_valid)[:, 1],
        model.predict_proba(encoded_test)[:, 1],
        int(model.best_iteration_),
        dict(zip(feature_names, map(float, model.booster_.feature_importance(importance_type="gain")))),
    )


def fit_xgboost(train_x, train_y, valid_x, valid_y, test_x, seed, experiment):
    from xgboost import XGBClassifier

    encoded_train, encoded_valid, encoded_test, feature_names, _ = ordinal_encode(train_x, valid_x, test_x)
    model = XGBClassifier(
        n_estimators=900, learning_rate=0.05, max_depth=7, min_child_weight=5,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0,
        objective="binary:logistic", eval_metric="auc", tree_method="hist",
        random_state=seed, n_jobs=-1, early_stopping_rounds=80,
    )
    model.fit(encoded_train, train_y, eval_set=[(encoded_valid, valid_y)], verbose=False)
    return FitResult(
        model.predict_proba(encoded_valid)[:, 1],
        model.predict_proba(encoded_test)[:, 1],
        int(model.best_iteration + 1),
        dict(zip(feature_names, map(float, model.feature_importances_))),
    )


MODEL_FITTERS = {"catboost": fit_catboost, "lightgbm": fit_lightgbm, "xgboost": fit_xgboost}


def segment_auc(train: pd.DataFrame, target: pd.Series, predictions: np.ndarray) -> list[dict]:
    segments: list[tuple[str, pd.Series]] = [
        ("month", train["month"].astype(str)),
        ("poutcome", train["poutcome"].astype(str)),
        ("previous_contacted", (train["pdays"] != -1).map({True: "yes", False: "no"})),
        ("id_decile", pd.qcut(train[ID_COLUMN], 10, labels=False).astype(str)),
    ]
    rows = []
    for variable, groups in segments:
        for value in sorted(groups.unique()):
            mask = groups.eq(value).to_numpy()
            if mask.sum() < 100 or target[mask].nunique() < 2:
                continue
            rows.append({
                "segment": variable, "value": value, "rows": int(mask.sum()),
                "positive_rate": float(target[mask].mean()),
                "auc": float(roc_auc_score(target[mask], predictions[mask])),
            })
    return rows


def upsert_registry(row: dict[str, object]) -> None:
    existing = []
    if REGISTRY_PATH.exists():
        with REGISTRY_PATH.open(encoding="utf-8-sig", newline="") as handle:
            existing = list(csv.DictReader(handle))
    normalized = [{field: item.get(field, "") for field in REGISTRY_FIELDS} for item in existing]
    normalized = [item for item in normalized if item["experiment_id"] != row["experiment_id"]]
    normalized.append({field: row.get(field, "") for field in REGISTRY_FIELDS})
    with REGISTRY_PATH.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REGISTRY_FIELDS)
        writer.writeheader()
        writer.writerows(normalized)


def train_experiment(experiment: Experiment, folds: int = FOLDS, seed: int = SEED) -> dict:
    start = time.perf_counter()
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
    train_x = apply_feature_set(train, experiment.feature_set)
    test_x = apply_feature_set(test, experiment.feature_set)
    target = train[TARGET].astype(int)
    assignments = make_folds(target, folds=folds, seed=seed)
    oof = np.zeros(len(train), dtype=np.float64)
    test_prediction = np.zeros(len(test), dtype=np.float64)
    fold_scores, best_iterations, fold_importances = [], [], []
    fitter = MODEL_FITTERS[experiment.model]

    for fold in range(folds):
        valid_mask = assignments == fold
        train_idx, valid_idx = np.flatnonzero(~valid_mask), np.flatnonzero(valid_mask)
        fit_result = fitter(
            train_x.iloc[train_idx], target.iloc[train_idx],
            train_x.iloc[valid_idx], target.iloc[valid_idx], test_x, seed + fold, experiment,
        )
        valid_pred = fit_result.valid_prediction
        fold_test_pred = fit_result.test_prediction
        best_iteration = fit_result.best_iteration
        validate_predictions(valid_pred, len(valid_idx), f"fold {fold} prediction")
        oof[valid_idx] = valid_pred
        test_prediction += fold_test_pred / folds
        score = float(roc_auc_score(target.iloc[valid_idx], valid_pred))
        fold_scores.append(score)
        best_iterations.append(best_iteration)
        fold_importances.append(fit_result.feature_importance)
        print(f"fold={fold} auc={score:.6f} best_iteration={best_iteration}", flush=True)

    validate_predictions(oof, len(train), "OOF prediction")
    validate_predictions(test_prediction, len(test), "test prediction")
    cv_auc = float(roc_auc_score(target, oof))
    runtime_minutes = (time.perf_counter() - start) / 60
    output = OUTPUT_DIR / experiment.experiment_id
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "predictions.npz", oof=oof, test=test_prediction, folds=assignments)
    submission = sample.copy()
    if list(submission.columns) != [ID_COLUMN, TARGET] or len(submission) != len(test):
        raise ValueError("sample_submission.csv schema or row count is unexpected")
    if not submission[ID_COLUMN].equals(test[ID_COLUMN]):
        raise ValueError("Submission IDs do not match test IDs in order")
    submission[TARGET] = test_prediction
    submission.to_csv(output / "submission.csv", index=False)
    importance_frame = pd.DataFrame(fold_importances).fillna(0.0)
    feature_importance = [
        {
            "feature": feature,
            "mean_gain": float(importance_frame[feature].mean()),
            "fold_std": float(importance_frame[feature].std(ddof=0)),
        }
        for feature in importance_frame.mean().sort_values(ascending=False).index
    ]
    metrics = {
        **asdict(experiment), "seed": seed, "folds": folds, "rows": len(train),
        "feature_count": train_x.shape[1], "cv_auc": cv_auc,
        "fold_scores": fold_scores, "fold_std": float(np.std(fold_scores)),
        "best_iterations": best_iterations, "runtime_minutes": runtime_minutes,
        "feature_importance": feature_importance,
        "segments": segment_auc(train, target, oof),
    }
    (output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    upsert_registry({
        "experiment_id": experiment.experiment_id, "date": date.today().isoformat(),
        "purpose": experiment.purpose, "baseline_id": experiment.baseline_id,
        "changed_element": experiment.changed_element, "model": experiment.model,
        "features": f"{experiment.feature_set} ({train_x.shape[1]})",
        "validation": f"Stratified {folds}-Fold OOF", "seed": seed,
        "cv_auc": f"{cv_auc:.6f}", "fold_std": f"{np.std(fold_scores):.6f}",
        "fold_scores": "|".join(f"{score:.6f}" for score in fold_scores),
        "public_lb": "", "runtime_minutes": f"{runtime_minutes:.2f}",
        "status": "완료", "interpretation": experiment.interpretation,
    })
    print(json.dumps({key: metrics[key] for key in ["experiment_id", "cv_auc", "fold_std", "runtime_minutes"]}, indent=2))
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=sorted(EXPERIMENTS), required=True)
    parser.add_argument("--model", choices=sorted(MODEL_FITTERS), help="Override the configured model")
    parser.add_argument("--folds", type=int, default=FOLDS)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment = EXPERIMENTS[args.experiment]
    if args.model:
        values = asdict(experiment)
        values.update({
            "experiment_id": f"{experiment.experiment_id}-{args.model.upper()}",
            "baseline_id": experiment.experiment_id,
            "changed_element": f"model {experiment.model} → {args.model}",
            "model": args.model,
            "interpretation": f"같은 피처와 fold에서 {args.model} 모델만 비교",
        })
        experiment = Experiment(**values)
    train_experiment(experiment, folds=args.folds, seed=args.seed)


if __name__ == "__main__":
    main()
