from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import ExperimentConfig
from src.data import infer_id_columns, infer_target, infer_task_type, read_csv
from src.EDA.visuals import render_tabular_visuals


def _json_value(value: Any) -> Any:
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if pd.isna(value):
        return None
    return value


def _sample(frame: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    if len(frame) <= max_rows:
        return frame
    return frame.sample(n=max_rows, random_state=seed)


def build_eda_summary(config: ExperimentConfig) -> dict[str, Any]:
    started = time.perf_counter()
    train = read_csv(config.data.train_path)
    test = read_csv(config.data.test_path) if config.data.test_path else None
    target = infer_target(train, test, config.data.target)
    task_type = infer_task_type(train[target])
    ids = infer_id_columns(train, config.data.id_columns)
    sample = _sample(train, config.data.eda_max_rows, config.validation.random_state)
    feature_sample = sample.drop(columns=[target])

    columns = {}
    for column in train.columns:
        series = sample[column]
        entry: dict[str, Any] = {
            "dtype": str(train[column].dtype),
            "missing_rate": float(series.isna().mean()),
            "unique_in_sample": int(series.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(series):
            description = series.describe(percentiles=[0.25, 0.5, 0.75])
            entry["summary"] = {
                key: _json_value(value) for key, value in description.items()
            }
        else:
            entry["top_values"] = {
                str(key): int(value)
                for key, value in series.value_counts(dropna=False).head(10).items()
            }
        columns[column] = entry

    warnings = []
    constant_columns = [
        column for column in feature_sample.columns if feature_sample[column].nunique(dropna=False) <= 1
    ]
    if constant_columns:
        warnings.append(f"Constant columns: {constant_columns}")
    if ids:
        warnings.append(f"ID-like columns excluded from modeling: {ids}")
    if sample.duplicated().any():
        warnings.append(f"Duplicate rows in sample: {int(sample.duplicated().sum())}")
    if test is not None:
        for column in feature_sample.select_dtypes(exclude="number").columns:
            if column in test.columns:
                unseen = set(test[column].dropna().unique()) - set(train[column].dropna().unique())
                if unseen:
                    warnings.append(f"Unseen test categories in {column}: {sorted(map(str, unseen))[:10]}")

    target_counts = sample[target].value_counts()
    target_distribution = (
        {str(key): int(value) for key, value in target_counts.head(50).items()}
        if task_type == "classification"
        else {}
    )
    target_summary = None
    if task_type == "regression":
        target_summary = {
            key: _json_value(value)
            for key, value in sample[target].describe(percentiles=[0.25, 0.5, 0.75]).items()
        }

    return {
        "train_shape": list(train.shape),
        "test_shape": list(test.shape) if test is not None else None,
        "eda_sample_rows": len(sample),
        "target": target,
        "task_type": task_type,
        "id_columns": ids,
        "target_distribution": target_distribution,
        "target_distribution_truncated": task_type == "classification" and len(target_counts) > 50,
        "target_summary": target_summary,
        "columns": columns,
        "warnings": warnings,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# EDA Summary",
        "",
        f"- Train shape: `{summary['train_shape']}`",
        f"- Test shape: `{summary['test_shape']}`",
        f"- EDA sample rows: `{summary['eda_sample_rows']}`",
        f"- Target: `{summary['target']}`",
        f"- Task type: `{summary['task_type']}`",
        f"- ID columns excluded: `{summary['id_columns']}`",
        f"- Runtime: `{summary['elapsed_seconds']:.3f}s`",
        "",
    ]
    if summary["task_type"] == "classification":
        lines.extend(["## Target Distribution", "", "| Value | Count |", "|---|---:|"])
        lines.extend(f"| {key} | {value} |" for key, value in summary["target_distribution"].items())
        if summary.get("target_distribution_truncated"):
            lines.extend(["", "Only the 50 most frequent classes are shown."])
    else:
        value = summary["target_summary"]
        lines.extend([
            "## Target Summary",
            "",
            "| Count | Mean | Std | Min | Median | Max |",
            "|---:|---:|---:|---:|---:|---:|",
            f"| {value.get('count', 0):.0f} | {value.get('mean', 0):.4g} | {value.get('std', 0):.4g} | {value.get('min', 0):.4g} | {value.get('50%', 0):.4g} | {value.get('max', 0):.4g} |",
        ])
    lines.extend(["", "## Data Quality Warnings", ""])
    if summary["warnings"]:
        lines.extend(f"- {warning}" for warning in summary["warnings"])
    else:
        lines.append("- No warning detected by the fast checks.")
    lines.extend(["", "## Columns", "", "| Column | Type | Missing | Unique (sample) |", "|---|---|---:|---:|"])
    for column, details in summary["columns"].items():
        lines.append(
            f"| {column} | {details['dtype']} | {details['missing_rate']:.3%} | "
            f"{details['unique_in_sample']} |"
        )
    lines.append("")
    numeric = [(name, detail) for name, detail in summary["columns"].items() if "summary" in detail]
    if numeric:
        lines.extend(["## Numeric Descriptive Statistics", "", "| Column | Count | Mean | Std | Min | Median | Max |", "|---|---:|---:|---:|---:|---:|---:|"])
        for name, detail in numeric:
            value = detail["summary"]
            lines.append(f"| {name} | {value.get('count', 0):.0f} | {value.get('mean', 0):.4g} | {value.get('std', 0):.4g} | {value.get('min', 0):.4g} | {value.get('50%', 0):.4g} | {value.get('max', 0):.4g} |")
        lines.append("")
    if summary.get("visual_artifacts"):
        lines.extend(["## Visual EDA", ""])
        for artifact in summary["visual_artifacts"]:
            lines.append(f"![{artifact}](assets/{artifact})")
        lines.append("")
    return "\n".join(lines)


def run_eda(
    config_path: str,
    output_dir: str = "reports",
    render_visuals: bool = False,
) -> dict[str, Any]:
    run_started = time.perf_counter()
    config = ExperimentConfig.from_json(config_path)
    config.validate()
    summary = build_eda_summary(config)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    if render_visuals:
        frame = _sample(read_csv(config.data.train_path), config.data.eda_max_rows, config.validation.random_state)
        summary["visual_artifacts"] = render_tabular_visuals(
            frame,
            summary["target"],
            summary["id_columns"],
            destination / "assets",
            summary["task_type"],
        )
    summary["elapsed_seconds"] = round(time.perf_counter() - run_started, 3)
    (destination / "eda_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (destination / "EDA.md").write_text(render_markdown(summary), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run bounded EDA from an experiment config")
    parser.add_argument("--config", default="configs/baseline.json")
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--render-visuals", action="store_true")
    args = parser.parse_args()
    summary = run_eda(args.config, args.output_dir, args.render_visuals)
    print(json.dumps({"status": "ok", "elapsed_seconds": summary["elapsed_seconds"]}))


if __name__ == "__main__":
    main()
