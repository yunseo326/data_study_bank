"""Compare completed OOF experiments and write compact report diagnostics."""

from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LOCAL_PACKAGES = ROOT / "bank" / ".runtime_pkgs"
if LOCAL_PACKAGES.exists():
    sys.path.insert(0, str(LOCAL_PACKAGES))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


BANK_DIR = ROOT / "bank"
DATA_PATH = BANK_DIR / "data" / "train.csv"
REGISTRY_PATH = BANK_DIR / "benchmarks" / "our_experiments.csv"
OUTPUT_DIR = BANK_DIR / "outputs"
RESULT_PATH = BANK_DIR / "logs" / "result" / "model_analysis.json"


def load_completed() -> pd.DataFrame:
    registry = pd.read_csv(REGISTRY_PATH)
    registry["cv_auc_num"] = pd.to_numeric(registry["cv_auc"], errors="coerce")
    registry = registry.dropna(subset=["cv_auc_num"]).copy()
    registry = registry[
        registry["experiment_id"].map(lambda value: (OUTPUT_DIR / value / "predictions.npz").exists())
    ]
    if registry.empty:
        raise ValueError("No completed experiments with OOF predictions were found")
    return registry


def core_experiment_ids(registry: pd.DataFrame, best_id: str) -> list[str]:
    preferred = ["E001", "E001-LIGHTGBM", "E001-XGBOOST", best_id]
    available = set(registry["experiment_id"])
    return list(dict.fromkeys(value for value in preferred if value in available))


def main() -> None:
    registry = load_completed()
    best_row = registry.loc[registry["cv_auc_num"].idxmax()]
    best_id = str(best_row["experiment_id"])
    ids = core_experiment_ids(registry, best_id)
    target = pd.read_csv(DATA_PATH, usecols=["y"])["y"].to_numpy()
    predictions = {
        experiment_id: np.load(OUTPUT_DIR / experiment_id / "predictions.npz")["oof"]
        for experiment_id in ids
    }

    correlation = pd.DataFrame(predictions).corr()
    correlation_rows = [
        {"left": left, "right": right, "correlation": float(correlation.loc[left, right])}
        for left, right in combinations(ids, 2)
    ]
    blend_rows = []
    for left, right in combinations(ids, 2):
        blend_score = float(roc_auc_score(target, 0.5 * predictions[left] + 0.5 * predictions[right]))
        blend_rows.append({"left": left, "right": right, "weight": "50:50", "oof_auc": blend_score})
    blend_rows.sort(key=lambda row: row["oof_auc"], reverse=True)

    metrics = json.loads((OUTPUT_DIR / best_id / "metrics.json").read_text(encoding="utf-8"))
    importance = metrics.get("feature_importance", [])
    total_gain = sum(float(row["mean_gain"]) for row in importance)
    importance_rows = [
        {**row, "gain_share": float(row["mean_gain"]) / total_gain if total_gain else 0.0}
        for row in importance
    ]

    segments = pd.DataFrame(metrics["segments"])
    stable_segments = segments[segments["rows"] >= 1000].sort_values("auc")
    weakest_segments = stable_segments.head(8).to_dict("records")

    experiment_rows = []
    scores = dict(zip(registry["experiment_id"], registry["cv_auc_num"]))
    for _, row in registry.iterrows():
        baseline = row.get("baseline_id")
        baseline_score = scores.get(baseline)
        experiment_rows.append({
            "experiment_id": row["experiment_id"],
            "model": row["model"],
            "cv_auc": float(row["cv_auc_num"]),
            "baseline_id": "" if pd.isna(baseline) else baseline,
            "delta": None if baseline_score is None else float(row["cv_auc_num"] - baseline_score),
            "runtime_minutes": None if pd.isna(row["runtime_minutes"]) else float(row["runtime_minutes"]),
        })

    result = {
        "best_experiment": best_id,
        "best_oof_auc": float(best_row["cv_auc_num"]),
        "target_gap": float(0.970 - best_row["cv_auc_num"]),
        "experiments": experiment_rows,
        "prediction_correlations": correlation_rows,
        "fixed_half_blends": blend_rows,
        "feature_importance": importance_rows,
        "weakest_segments": weakest_segments,
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "best_experiment": result["best_experiment"],
        "best_oof_auc": result["best_oof_auc"],
        "best_fixed_blend": blend_rows[0] if blend_rows else None,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
