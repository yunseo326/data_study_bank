from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
from threadpoolctl import threadpool_limits

from src.artifacts import write_oof_predictions
from src.config import ExperimentConfig
from src.data import (
    infer_id_columns,
    infer_target,
    infer_task_type,
    read_csv,
    sample_training_data,
)
from src.features import infer_feature_schema
from src.train import build_training_pipeline
from src.validation import cross_validate


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in ("numpy", "pandas", "scikit-learn", "threadpoolctl", "pyarrow"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _write_run_manifest(
    path: Path,
    experiment_id: str,
    config_path: str,
    artifact_paths: list[Path],
    experiment_dir: Path,
) -> None:
    artifacts = {
        artifact.relative_to(experiment_dir).as_posix(): {
            "sha256": _sha256(artifact),
            "size_bytes": artifact.stat().st_size,
        }
        for artifact in artifact_paths
    }
    _write_json(
        path,
        {
            "artifact_layout_version": 2,
            "experiment_id": experiment_id,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "entrypoint": "python -m src.run_experiment",
            "arguments": {
                "config": str(config_path),
                "experiment_id": experiment_id,
            },
            "environment": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "packages": _package_versions(),
            },
            "verification": {
                "algorithm": "sha256",
                "artifacts": artifacts,
            },
        },
    )


def _experiment_markdown(
    experiment_id: str,
    target: str,
    task_type: str,
    metrics: dict[str, float],
    elapsed: float,
    sample_rows: int,
) -> str:
    metric_lines = "\n".join(f"- {name}: `{value:.6f}`" for name, value in metrics.items())
    return f"""# {experiment_id}

## Parent

- None (baseline)

## Source Idea

- Configuration-driven fast baseline for validating the project template.

## Hypothesis

- A bounded sample and fixed cross-validation can produce a reproducible baseline within the runtime budget.

## Change

- Target: `{target}`
- Task: `{task_type}`
- Training rows: `{sample_rows}`

## Result

{metric_lines}
- Runtime: `{elapsed:.3f}s`

## Reviewer

- Pending. Apply `reviewer_rule.md` before assigning the final verdict.

## Verdict

- INCONCLUSIVE

## Conclusion

- This run verifies pipeline execution. It does not establish the best domain model.
"""


def _append_registry(
    registry_path: Path,
    experiment_id: str,
    metrics: dict[str, float],
    target: str,
    experiment_path: Path,
) -> None:
    fields = [
        "experiment_id",
        "parent_id",
        "created_at",
        "status",
        "decision",
        "hypothesis",
        "model",
        "validation_version",
        "primary_metric",
        "primary_score",
        "delta_vs_parent",
        "data_version",
        "git_commit",
        "experiment_path",
    ]
    existing_ids: set[str] = set()
    if registry_path.exists():
        with registry_path.open(encoding="utf-8", newline="") as file:
            existing_ids = {row["experiment_id"] for row in csv.DictReader(file)}
    if experiment_id in existing_ids:
        raise FileExistsError(f"Experiment ID already exists in registry: {experiment_id}")
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not registry_path.exists()
    primary_metric = "map_at_3" if "map_at_3" in metrics else "rmse"
    with registry_path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "experiment_id": experiment_id,
                "parent_id": "",
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "status": "COMPLETED",
                "decision": "INCONCLUSIVE",
                "hypothesis": "Fast reproducible baseline within runtime budget",
                "model": "hist_gradient_boosting",
                "validation_version": "v1_fixed_cv",
                "primary_metric": primary_metric,
                "primary_score": metrics[primary_metric],
                "delta_vs_parent": "",
                "data_version": Path(target).name if os.path.exists(target) else "local_csv",
                "git_commit": "",
                "experiment_path": experiment_path.as_posix(),
            }
        )


def run_experiment(config_path: str, experiment_id: str) -> dict[str, Any]:
    started = time.perf_counter()
    config = ExperimentConfig.from_json(config_path)
    config.validate()
    experiment_dir = Path("experiments") / experiment_id
    if experiment_dir.exists():
        raise FileExistsError(f"Experiment directory already exists: {experiment_dir}")
    results_dir = experiment_dir / "results"
    reproducibility_dir = experiment_dir / "reproducibility"

    train = read_csv(config.data.train_path)
    test = read_csv(config.data.test_path) if config.data.test_path else None
    target = infer_target(train, test, config.data.target)
    task_type = infer_task_type(train[target])
    id_columns = infer_id_columns(train, config.data.id_columns)
    sampled = sample_training_data(
        train,
        target=target,
        max_rows=config.data.max_train_rows,
        task_type=task_type,
        random_state=config.validation.random_state,
    )
    schema = infer_feature_schema(sampled, target, id_columns)
    features = sampled[schema.features]
    labels = sampled[target]
    pipeline = build_training_pipeline(
        schema=schema,
        task_type=task_type,
        model_name=config.model.name,
        model_params=config.model.params,
        random_state=config.validation.random_state,
    )
    deadline = started + config.runtime.time_budget_seconds

    with threadpool_limits(limits=config.runtime.n_threads):
        validation = cross_validate(
            pipeline=pipeline,
            features=features,
            target=labels,
            task_type=task_type,
            n_splits=config.validation.n_splits,
            shuffle=config.validation.shuffle,
            random_state=config.validation.random_state,
            deadline=deadline,
        )

    elapsed = time.perf_counter() - started
    if elapsed > config.runtime.time_budget_seconds:
        raise TimeoutError(
            f"Experiment exceeded time budget: {elapsed:.3f}s > "
            f"{config.runtime.time_budget_seconds:.3f}s"
        )

    results_dir.mkdir(parents=True)
    reproducibility_dir.mkdir()
    config_snapshot_path = reproducibility_dir / "config.json"
    _write_json(config_snapshot_path, config.to_dict())
    metrics_payload = {
        "overall": validation.overall_metrics,
        "folds": validation.fold_metrics,
        "runtime_seconds": elapsed,
        "training_rows": len(sampled),
        "task_type": task_type,
        "target": target,
        "feature_schema": {
            "numeric": schema.numeric,
            "categorical": schema.categorical,
            "dropped": schema.dropped,
        },
    }
    result_paths = [results_dir / "metrics.json", results_dir / "experiment.md"]
    _write_json(result_paths[0], metrics_payload)
    result_paths[1].write_text(
        _experiment_markdown(
            experiment_id,
            target,
            task_type,
            validation.overall_metrics,
            elapsed,
            len(sampled),
        ),
        encoding="utf-8",
    )
    if task_type == "classification":
        oof = pd.DataFrame(validation.predictions, columns=[f"proba_{c}" for c in validation.classes])
        oof.insert(0, "target", labels.to_numpy())
    else:
        oof = pd.DataFrame({"target": labels.to_numpy(), "prediction": validation.predictions})
    oof_path = results_dir / "oof_predictions.parquet"
    write_oof_predictions(oof, oof_path)
    result_paths.append(oof_path)
    _write_run_manifest(
        reproducibility_dir / "run_manifest.json",
        experiment_id,
        config_path,
        [config_snapshot_path, *result_paths],
        experiment_dir,
    )
    _append_registry(
        Path("experiments/registry.csv"),
        experiment_id,
        validation.overall_metrics,
        config.data.train_path,
        experiment_dir,
    )
    return metrics_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a formal baseline experiment")
    parser.add_argument("--config", default="configs/baseline.json")
    parser.add_argument("--experiment-id", required=True)
    args = parser.parse_args()
    result = run_experiment(args.config, args.experiment_id)
    print(json.dumps({"status": "ok", **result["overall"], "runtime_seconds": result["runtime_seconds"]}))


if __name__ == "__main__":
    main()
