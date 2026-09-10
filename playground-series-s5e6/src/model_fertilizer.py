"""Reproducible MAP@3 experiments for the fertilizer competition.

Raw CSV files are read-only. Derived predictions and submissions are written below
``outputs/`` while the compact experiment registry is tracked in ``benchmarks``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCAL_PACKAGES = ROOT / ".runtime_pkgs"
if LOCAL_PACKAGES.exists():
    sys.path.insert(0, str(LOCAL_PACKAGES))

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder


DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
REGISTRY_PATH = ROOT / "benchmarks" / "our_experiments.csv"
TARGET = "Fertilizer Name"
ID_COLUMN = "id"
SEED = 326
FOLDS = 5

BASE_FEATURES = [
    "Temparature", "Humidity", "Moisture", "Soil Type", "Crop Type",
    "Nitrogen", "Potassium", "Phosphorous",
]
CATEGORICAL_BASE = ["Soil Type", "Crop Type"]
PAIR_FEATURES = [
    ("Nitrogen", "Phosphorous"),
    ("Moisture", "Phosphorous"),
    ("Nitrogen", "Potassium"),
    ("Potassium", "Phosphorous"),
    ("Moisture", "Nitrogen"),
]
TE_GROUPS = [(column,) for column in BASE_FEATURES] + PAIR_FEATURES

REGISTRY_FIELDS = [
    "experiment_id", "date", "purpose", "baseline_id", "changed_element",
    "model", "features", "validation", "seed", "cv_map3", "fold_std",
    "log_loss", "fold_scores", "public_lb", "runtime_minutes", "status",
    "interpretation",
]


@dataclass(frozen=True)
class Experiment:
    experiment_id: str
    purpose: str
    model: str
    feature_set: str
    baseline_id: str = ""
    changed_element: str = "첫 기준선"
    interpretation: str = ""


EXPERIMENTS = {
    "E000": Experiment(
        "E000", "빈도 기준선", "frequency_prior", "base",
        interpretation="학습 fold의 클래스 빈도 상위 3개를 모든 행에 동일하게 예측",
    ),
    "E001": Experiment(
        "E001", "CatBoost 기준선", "catboost", "base", "E000",
        "학습 모델 도입", "범주형 2개를 직접 처리하는 첫 모델",
    ),
    "E002": Experiment(
        "E002", "LightGBM 비교", "lightgbm", "base", "E001",
        "모델만 변경", "같은 특성과 fold에서 모델 계열의 차이를 확인",
    ),
    "E003": Experiment(
        "E003", "모든 특성 범주형 처리", "catboost", "all_categorical", "E001",
        "특성 타입만 변경", "낮은 cardinality 정수값을 범주로 처리하는 가설",
    ),
    "E004": Experiment(
        "E004", "쌍 상호작용 검증", "catboost", "pair_interactions", "E001",
        "쌍 상호작용만 추가", "영양소와 수분 조합의 추가 정보를 검증",
    ),
    "E005": Experiment(
        "E005", "타깃 인코딩 검증", "catboost_te", "target_encoding", "E004",
        "fold-safe 타깃 인코딩만 추가", "학습 fold 통계로 조합별 클래스 경향을 표현",
    ),
    "E006": Experiment(
        "E006", "확률 평균 앙상블", "probability_blend", "completed_models",
        changed_element="완료 모델 OOF 확률 평균",
        interpretation="서로 다른 모델의 오류가 보완되는지 확인",
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, str]]:
    paths = {name: DATA_DIR / name for name in ["train.csv", "test.csv", "sample_submission.csv"]}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing competition files: {missing}")
    before = {name: sha256_file(path) for name, path in paths.items()}
    train = pd.read_csv(paths["train.csv"])
    test = pd.read_csv(paths["test.csv"])
    sample = pd.read_csv(paths["sample_submission.csv"])
    validate_schema(train, test, sample)
    return train, test, sample, before


def validate_schema(train: pd.DataFrame, test: pd.DataFrame, sample: pd.DataFrame) -> None:
    expected_train = [ID_COLUMN, *BASE_FEATURES, TARGET]
    expected_test = [ID_COLUMN, *BASE_FEATURES]
    if list(train.columns) != expected_train:
        raise ValueError(f"Unexpected train schema: {list(train.columns)}")
    if list(test.columns) != expected_test:
        raise ValueError(f"Unexpected test schema: {list(test.columns)}")
    if list(sample.columns) != [ID_COLUMN, TARGET]:
        raise ValueError(f"Unexpected submission schema: {list(sample.columns)}")
    if not train[ID_COLUMN].is_unique or not test[ID_COLUMN].is_unique:
        raise ValueError("IDs must be unique within each split")
    if len(sample) != len(test) or not sample[ID_COLUMN].equals(test[ID_COLUMN]):
        raise ValueError("Submission IDs must match test IDs in order")
    if train[TARGET].nunique() != 7:
        raise ValueError("Expected exactly seven fertilizer labels")


def make_folds(target: pd.Series, folds: int = FOLDS, seed: int = SEED) -> np.ndarray:
    if folds < 2:
        raise ValueError("At least two folds are required")
    assignments = np.full(len(target), -1, dtype=np.int8)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold, (_, valid_idx) in enumerate(splitter.split(np.zeros(len(target)), target)):
        assignments[valid_idx] = fold
    if np.any(assignments < 0) or set(np.unique(assignments)) != set(range(folds)):
        raise ValueError("Every row must belong to exactly one validation fold")
    for fold in range(folds):
        if target.iloc[np.flatnonzero(assignments == fold)].nunique() != target.nunique():
            raise ValueError(f"Fold {fold} does not contain every class")
    return assignments


def map_at_3(y_true: np.ndarray | pd.Series, probabilities: np.ndarray, classes: np.ndarray) -> float:
    truth = np.asarray(y_true, dtype=object)
    probabilities = np.asarray(probabilities, dtype=float)
    classes = np.asarray(classes, dtype=object)
    if probabilities.shape != (len(truth), len(classes)):
        raise ValueError("Probability matrix shape does not match labels/classes")
    top = classes[np.argsort(-probabilities, axis=1, kind="stable")[:, :3]]
    scores = np.zeros(len(truth), dtype=float)
    for rank in range(top.shape[1]):
        scores[(scores == 0) & (top[:, rank] == truth)] = 1.0 / (rank + 1)
    return float(scores.mean())


def top3_labels(probabilities: np.ndarray, classes: np.ndarray) -> list[str]:
    top = np.asarray(classes, dtype=object)[np.argsort(-probabilities, axis=1, kind="stable")[:, :3]]
    return [" ".join(map(str, row)) for row in top]


def validate_probabilities(values: np.ndarray, rows: int, classes: int, name: str) -> None:
    if values.shape != (rows, classes):
        raise ValueError(f"{name} shape mismatch: {values.shape}")
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
        raise ValueError(f"{name} contains invalid probabilities")
    if not np.allclose(values.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError(f"{name} rows must sum to one")


def validate_submission(submission: pd.DataFrame, test: pd.DataFrame, classes: np.ndarray) -> None:
    if len(submission) != len(test) or not submission[ID_COLUMN].equals(test[ID_COLUMN]):
        raise ValueError("Submission rows or ID order do not match test")
    allowed = set(map(str, classes))
    for value in submission[TARGET]:
        labels = str(value).split()
        if len(labels) != 3 or len(set(labels)) != 3 or not set(labels).issubset(allowed):
            raise ValueError(f"Invalid top-3 prediction: {value}")


def apply_feature_set(frame: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    result = frame[BASE_FEATURES].copy()
    if feature_set == "all_categorical":
        for column in BASE_FEATURES:
            result[column] = result[column].astype(str)
    elif feature_set in {"pair_interactions", "target_encoding"}:
        for left, right in PAIR_FEATURES:
            result[f"{left}__{right}"] = result[left].astype(str) + "|" + result[right].astype(str)
    elif feature_set != "base":
        raise ValueError(f"Unknown feature set: {feature_set}")
    return result


def fit_frequency_prior(train_y, valid_y, test_rows, classes):
    counts = train_y.value_counts().reindex(classes, fill_value=0).to_numpy(dtype=float)
    probabilities = counts / counts.sum()
    return (
        np.tile(probabilities, (len(valid_y), 1)),
        np.tile(probabilities, (test_rows, 1)),
        0,
    )


def fit_catboost(train_x, train_y, valid_x, valid_y, test_x, seed, classes):
    from catboost import CatBoostClassifier

    categorical = list(train_x.select_dtypes(exclude=np.number).columns)
    model = CatBoostClassifier(
        iterations=180,
        depth=6,
        learning_rate=0.10,
        loss_function="MultiClass",
        eval_metric="MultiClass",
        l2_leaf_reg=6.0,
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
        eval_set=(valid_x, valid_y), early_stopping_rounds=30, use_best_model=True,
    )
    model_classes = np.asarray(model.classes_, dtype=object)
    return (
        align_probabilities(model.predict_proba(valid_x), model_classes, classes),
        align_probabilities(model.predict_proba(test_x), model_classes, classes),
        int(model.get_best_iteration() + 1),
    )


def ordinal_encode(train_x, valid_x, test_x):
    categorical = list(train_x.select_dtypes(exclude=np.number).columns)
    numeric = [column for column in train_x.columns if column not in categorical]
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    train_cat = encoder.fit_transform(train_x[categorical]) if categorical else np.empty((len(train_x), 0))
    valid_cat = encoder.transform(valid_x[categorical]) if categorical else np.empty((len(valid_x), 0))
    test_cat = encoder.transform(test_x[categorical]) if categorical else np.empty((len(test_x), 0))
    return (
        np.column_stack([train_x[numeric].to_numpy(), train_cat]),
        np.column_stack([valid_x[numeric].to_numpy(), valid_cat]),
        np.column_stack([test_x[numeric].to_numpy(), test_cat]),
    )


def fit_lightgbm(train_x, train_y, valid_x, valid_y, test_x, seed, classes):
    from lightgbm import LGBMClassifier, early_stopping, log_evaluation

    encoded_train, encoded_valid, encoded_test = ordinal_encode(train_x, valid_x, test_x)
    model = LGBMClassifier(
        objective="multiclass", n_estimators=1000, learning_rate=0.05,
        num_leaves=63, max_depth=-1, subsample=0.8, colsample_bytree=0.8,
        reg_lambda=4.0, random_state=seed, n_jobs=-1, verbosity=-1,
    )
    model.fit(
        encoded_train, train_y, eval_set=[(encoded_valid, valid_y)],
        eval_metric="multi_logloss",
        callbacks=[early_stopping(80, verbose=False), log_evaluation(0)],
    )
    model_classes = np.asarray(model.classes_, dtype=object)
    return (
        align_probabilities(model.predict_proba(encoded_valid), model_classes, classes),
        align_probabilities(model.predict_proba(encoded_test), model_classes, classes),
        int(model.best_iteration_),
    )


def align_probabilities(values: np.ndarray, source_classes: np.ndarray, target_classes: np.ndarray) -> np.ndarray:
    aligned = np.zeros((len(values), len(target_classes)), dtype=float)
    lookup = {label: index for index, label in enumerate(source_classes)}
    for index, label in enumerate(target_classes):
        aligned[:, index] = values[:, lookup[label]]
    return aligned


def target_encode(
    train_x, train_y, other_frames, classes, smoothing=20.0,
    leave_one_out_first=False,
):
    priors = train_y.value_counts(normalize=True).reindex(classes, fill_value=0).to_numpy()
    encoded = [pd.DataFrame(index=frame.index) for frame in other_frames]
    training = train_x.copy()
    training["__target__"] = train_y.to_numpy()
    for columns in TE_GROUPS:
        name = "__".join(columns)
        grouped = training.groupby(list(columns) + ["__target__"], observed=True).size().unstack(fill_value=0)
        grouped = grouped.reindex(columns=classes, fill_value=0)
        totals = grouped.sum(axis=1).to_numpy()[:, None]
        rates = (grouped.to_numpy() + smoothing * priors[None, :]) / (totals + smoothing)
        rate_frame = pd.DataFrame(rates, index=grouped.index, columns=[f"te_{name}_{label}" for label in classes])
        for position, (result, frame) in enumerate(zip(encoded, other_frames)):
            if len(columns) == 1:
                keys = frame[columns[0]].to_numpy()
            else:
                keys = pd.MultiIndex.from_frame(frame[list(columns)])
            if position == 0 and leave_one_out_first:
                if len(frame) != len(train_x) or not frame.index.equals(train_x.index):
                    raise ValueError("The first target-encoding frame must be the training frame")
                raw = grouped.reindex(keys, fill_value=0).to_numpy(dtype=float).copy()
                target_values = train_y.to_numpy()
                for class_index, label in enumerate(classes):
                    raw[:, class_index] -= target_values == label
                totals_without_self = raw.sum(axis=1)
                values = (raw + smoothing * priors[None, :]) / (
                    totals_without_self[:, None] + smoothing
                )
                mapped = pd.DataFrame(
                    values, columns=[f"te_{name}_{label}" for label in classes], index=frame.index,
                )
            else:
                mapped = rate_frame.reindex(keys)
                mapped.index = frame.index
            mapped = mapped.fillna(dict(zip(mapped.columns, priors))).astype("float32")
            for column in mapped.columns:
                result[column] = mapped[column]
    return encoded


def fit_catboost_te(train_x, train_y, valid_x, valid_y, test_x, seed, classes):
    encoded_train, encoded_valid, encoded_test = target_encode(
        train_x[BASE_FEATURES], train_y,
        [train_x[BASE_FEATURES], valid_x[BASE_FEATURES], test_x[BASE_FEATURES]], classes,
        leave_one_out_first=True,
    )
    augmented_train = pd.concat([train_x.reset_index(drop=True), encoded_train.reset_index(drop=True)], axis=1)
    augmented_valid = pd.concat([valid_x.reset_index(drop=True), encoded_valid.reset_index(drop=True)], axis=1)
    augmented_test = pd.concat([test_x.reset_index(drop=True), encoded_test.reset_index(drop=True)], axis=1)
    return fit_catboost(
        augmented_train, train_y.reset_index(drop=True), augmented_valid,
        valid_y.reset_index(drop=True), augmented_test, seed, classes,
    )


FITTERS = {
    "catboost": fit_catboost,
    "lightgbm": fit_lightgbm,
    "catboost_te": fit_catboost_te,
}


def per_class_diagnostics(target, probabilities, classes):
    top = classes[np.argsort(-probabilities, axis=1, kind="stable")[:, :3]]
    rows = []
    target_array = np.asarray(target, dtype=object)
    for label in classes:
        mask = target_array == label
        label_top = top[mask]
        rows.append({
            "label": str(label), "rows": int(mask.sum()),
            "top1_accuracy": float((label_top[:, 0] == label).mean()) if len(label_top) else 0.0,
            "top3_recall": float((label_top == label).any(axis=1).mean()) if len(label_top) else 0.0,
        })
    return rows


def refresh_saved_diagnostics(experiment_id: str) -> dict:
    train, _, _, _ = load_data()
    output = OUTPUT_DIR / experiment_id
    metrics_path = output / "metrics.json"
    predictions_path = output / "predictions.npz"
    if not metrics_path.exists() or not predictions_path.exists():
        raise FileNotFoundError(f"Saved result is incomplete: {output}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    saved = np.load(predictions_path, allow_pickle=True)
    classes = np.asarray(saved["classes"], dtype=object)
    if len(saved["oof"]) != len(train):
        raise ValueError("Saved OOF rows do not match the full training data")
    metrics["per_class"] = per_class_diagnostics(train[TARGET], saved["oof"], classes)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def upsert_registry(row: dict[str, object]) -> None:
    existing = []
    if REGISTRY_PATH.exists():
        with REGISTRY_PATH.open(encoding="utf-8-sig", newline="") as handle:
            existing = list(csv.DictReader(handle))
    normalized = [{field: item.get(field, "") for field in REGISTRY_FIELDS} for item in existing]
    normalized = [item for item in normalized if item["experiment_id"] != row["experiment_id"]]
    normalized.append({field: row.get(field, "") for field in REGISTRY_FIELDS})
    normalized.sort(key=lambda item: item["experiment_id"])
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REGISTRY_PATH.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REGISTRY_FIELDS)
        writer.writeheader()
        writer.writerows(normalized)


def save_result(experiment, train, test, sample, classes, folds, oof, test_prob, metrics, hashes):
    output = OUTPUT_DIR / experiment.experiment_id
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "predictions.npz", oof=oof.astype("float32"),
        test=test_prob.astype("float32"), folds=folds, classes=classes,
    )
    submission = sample.copy()
    submission[TARGET] = top3_labels(test_prob, classes)
    validate_submission(submission, test, classes)
    submission.to_csv(output / "submission.csv", index=False)
    metrics["source_sha256"] = hashes
    (output / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8",
    )


def train_experiment(experiment: Experiment, folds_count=FOLDS, seed=SEED, smoke=False):
    start = time.perf_counter()
    train, test, sample, hashes = load_data()
    if smoke:
        per_class = min(5000, int(train[TARGET].value_counts().min()))
        train = train.groupby(TARGET, group_keys=False).sample(per_class, random_state=seed).sort_index()
        test = test.head(10000).copy()
        sample = sample.head(len(test)).copy()
        folds_count = min(folds_count, 2)
    classes = np.asarray(sorted(train[TARGET].unique()), dtype=object)
    assignments = make_folds(train[TARGET], folds_count, seed)
    train_x = apply_feature_set(train, experiment.feature_set)
    test_x = apply_feature_set(test, experiment.feature_set)
    oof = np.zeros((len(train), len(classes)), dtype=float)
    test_probability = np.zeros((len(test), len(classes)), dtype=float)
    fold_scores, best_iterations = [], []

    for fold in range(folds_count):
        valid_mask = assignments == fold
        train_idx, valid_idx = np.flatnonzero(~valid_mask), np.flatnonzero(valid_mask)
        train_y = train[TARGET].iloc[train_idx]
        valid_y = train[TARGET].iloc[valid_idx]
        if experiment.model == "frequency_prior":
            valid_probability, fold_test, best_iteration = fit_frequency_prior(
                train_y, valid_y, len(test), classes,
            )
        else:
            valid_probability, fold_test, best_iteration = FITTERS[experiment.model](
                train_x.iloc[train_idx], train_y, train_x.iloc[valid_idx], valid_y,
                test_x, seed + fold, classes,
            )
        validate_probabilities(valid_probability, len(valid_idx), len(classes), f"fold {fold}")
        validate_probabilities(fold_test, len(test), len(classes), f"fold {fold} test")
        oof[valid_idx] = valid_probability
        test_probability += fold_test / folds_count
        score = map_at_3(valid_y, valid_probability, classes)
        fold_scores.append(score)
        best_iterations.append(best_iteration)
        print(f"fold={fold} map3={score:.6f} best_iteration={best_iteration}", flush=True)

    validate_probabilities(oof, len(train), len(classes), "OOF")
    validate_probabilities(test_probability, len(test), len(classes), "test")
    cv_map3 = map_at_3(train[TARGET], oof, classes)
    cv_logloss = float(log_loss(train[TARGET], oof, labels=classes))
    runtime_minutes = (time.perf_counter() - start) / 60
    metrics = {
        **asdict(experiment), "seed": seed, "folds": folds_count,
        "rows": len(train), "test_rows": len(test), "feature_count": train_x.shape[1],
        "classes": list(map(str, classes)), "cv_map3": cv_map3,
        "fold_scores": fold_scores, "fold_std": float(np.std(fold_scores)),
        "log_loss": cv_logloss, "best_iterations": best_iterations,
        "runtime_minutes": runtime_minutes, "smoke": smoke,
        "per_class": per_class_diagnostics(train[TARGET], oof, classes),
    }
    save_result(
        experiment, train, test, sample, classes, assignments, oof,
        test_probability, metrics, hashes,
    )
    upsert_registry({
        "experiment_id": experiment.experiment_id, "date": date.today().isoformat(),
        "purpose": experiment.purpose, "baseline_id": experiment.baseline_id,
        "changed_element": experiment.changed_element, "model": experiment.model,
        "features": f"{experiment.feature_set} ({train_x.shape[1]})",
        "validation": f"Stratified {folds_count}-Fold OOF" + (" smoke" if smoke else ""),
        "seed": seed, "cv_map3": f"{cv_map3:.6f}",
        "fold_std": f"{np.std(fold_scores):.6f}", "log_loss": f"{cv_logloss:.6f}",
        "fold_scores": "|".join(f"{score:.6f}" for score in fold_scores),
        "runtime_minutes": f"{runtime_minutes:.2f}",
        "status": "스모크 완료" if smoke else "완료",
        "interpretation": experiment.interpretation,
    })
    after = {name: sha256_file(DATA_DIR / name) for name in hashes}
    if hashes != after:
        raise RuntimeError("A raw data file changed during the experiment")
    print(json.dumps({
        "experiment_id": experiment.experiment_id,
        "cv_map3": cv_map3, "fold_std": float(np.std(fold_scores)),
        "log_loss": cv_logloss, "runtime_minutes": runtime_minutes,
    }, ensure_ascii=False, indent=2))
    return metrics


def blend_experiment(source_ids: list[str], seed=SEED):
    if len(source_ids) < 2:
        raise ValueError("E006 requires at least two source experiments")
    train, test, sample, hashes = load_data()
    loaded = []
    for source_id in source_ids:
        path = OUTPUT_DIR / source_id / "predictions.npz"
        if not path.exists():
            raise FileNotFoundError(f"Missing predictions for {source_id}: {path}")
        loaded.append(np.load(path, allow_pickle=True))
    reference_classes = loaded[0]["classes"]
    reference_folds = loaded[0]["folds"]
    if any(not np.array_equal(item["classes"], reference_classes) for item in loaded[1:]):
        raise ValueError("Blend source class orders differ")
    if any(not np.array_equal(item["folds"], reference_folds) for item in loaded[1:]):
        raise ValueError("Blend source fold assignments differ")
    oof = np.mean([item["oof"] for item in loaded], axis=0)
    test_probability = np.mean([item["test"] for item in loaded], axis=0)
    classes = np.asarray(reference_classes, dtype=object)
    validate_probabilities(oof, len(train), len(classes), "blend OOF")
    validate_probabilities(test_probability, len(test), len(classes), "blend test")
    score = map_at_3(train[TARGET], oof, classes)
    loss = float(log_loss(train[TARGET], oof, labels=classes))
    experiment = EXPERIMENTS["E006"]
    metrics = {
        **asdict(experiment), "seed": seed, "folds": int(reference_folds.max() + 1),
        "rows": len(train), "test_rows": len(test), "classes": list(map(str, classes)),
        "source_experiments": source_ids, "cv_map3": score, "log_loss": loss,
        "fold_std": float(np.std([
            map_at_3(train.loc[reference_folds == fold, TARGET], oof[reference_folds == fold], classes)
            for fold in np.unique(reference_folds)
        ])), "runtime_minutes": 0.0, "smoke": False,
    }
    save_result(
        experiment, train, test, sample, classes, reference_folds, oof,
        test_probability, metrics, hashes,
    )
    upsert_registry({
        "experiment_id": "E006", "date": date.today().isoformat(),
        "purpose": experiment.purpose, "changed_element": "+".join(source_ids),
        "model": experiment.model, "features": experiment.feature_set,
        "validation": "OOF probability blend", "seed": seed,
        "cv_map3": f"{score:.6f}", "fold_std": f"{metrics['fold_std']:.6f}",
        "log_loss": f"{loss:.6f}", "status": "완료",
        "interpretation": experiment.interpretation,
    })
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=sorted(EXPERIMENTS), required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--folds", type=int, default=FOLDS)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--refresh-saved", action="store_true")
    parser.add_argument("--blend", nargs="+", default=["E001", "E003"])
    return parser.parse_args()


def main():
    args = parse_args()
    if args.refresh_saved:
        result = refresh_saved_diagnostics(args.experiment)
        print(json.dumps({"experiment_id": args.experiment, "refreshed": True}, ensure_ascii=False))
    elif args.experiment == "E006":
        result = blend_experiment(args.blend, seed=args.seed)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        train_experiment(
            EXPERIMENTS[args.experiment], folds_count=args.folds,
            seed=args.seed, smoke=args.smoke,
        )


if __name__ == "__main__":
    main()
