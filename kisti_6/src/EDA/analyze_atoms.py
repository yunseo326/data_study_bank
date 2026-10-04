from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import ExperimentConfig
from src.data import infer_id_columns, infer_target, read_csv


MISSING = "<MISSING>"


def _json_value(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if pd.isna(value):
        return None
    return value


def _sample(frame: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    if len(frame) <= max_rows:
        return frame.copy()
    return frame.sample(n=max_rows, random_state=seed)


def _distribution(series: pd.Series) -> dict[str, float]:
    values = series.astype("string").fillna(MISSING)
    counts = values.value_counts(normalize=True, dropna=False)
    return {str(key): float(value) for key, value in counts.items()}


def _total_variation(left: pd.Series, right: pd.Series) -> float:
    left_dist = _distribution(left)
    right_dist = _distribution(right)
    keys = set(left_dist) | set(right_dist)
    return 0.5 * sum(abs(left_dist.get(key, 0.0) - right_dist.get(key, 0.0)) for key in keys)


def _cramers_v(feature: pd.Series, target: pd.Series) -> float | None:
    table = pd.crosstab(feature.astype("string").fillna(MISSING), target, dropna=False)
    if table.empty or min(table.shape) <= 1:
        return None
    observed = table.to_numpy(dtype=float)
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / observed.sum()
    chi_squared = np.divide(
        (observed - expected) ** 2,
        expected,
        out=np.zeros_like(observed),
        where=expected > 0,
    ).sum()
    denominator = observed.sum() * min(table.shape[0] - 1, table.shape[1] - 1)
    return float(math.sqrt(chi_squared / denominator)) if denominator else None


def _eta_squared(feature: pd.Series, target: pd.Series) -> float | None:
    valid = feature.notna() & target.notna()
    values = feature[valid].astype(float)
    labels = target[valid]
    if values.empty:
        return None
    total = float(((values - values.mean()) ** 2).sum())
    if total == 0:
        return None
    between = 0.0
    for _, group in values.groupby(labels):
        between += len(group) * float((group.mean() - values.mean()) ** 2)
    return between / total


def _target_purity(feature: pd.Series, target: pd.Series) -> float | None:
    valid = target.notna()
    if not valid.any():
        return None
    table = pd.crosstab(
        feature[valid].astype("string").fillna(MISSING),
        target[valid],
        dropna=False,
    )
    return float(table.max(axis=1).sum() / table.to_numpy().sum()) if not table.empty else None


def _numeric_psi(train: pd.Series, test: pd.Series, bins: int = 10) -> float | None:
    train_values = train.dropna().astype(float)
    test_values = test.dropna().astype(float)
    if train_values.empty or test_values.empty:
        return None
    edges = np.unique(train_values.quantile(np.linspace(0, 1, bins + 1)).to_numpy())
    if len(edges) < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    train_bins = pd.cut(train_values, bins=edges, include_lowest=True).value_counts(sort=False)
    test_bins = pd.cut(test_values, bins=edges, include_lowest=True).value_counts(sort=False)
    epsilon = 1e-6
    train_rate = np.clip(train_bins.to_numpy(dtype=float) / len(train_values), epsilon, None)
    test_rate = np.clip(test_bins.to_numpy(dtype=float) / len(test_values), epsilon, None)
    return float(np.sum((test_rate - train_rate) * np.log(test_rate / train_rate)))


def _numeric_summary(series: pd.Series) -> dict[str, Any]:
    description = series.describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99])
    return {str(key): _json_value(value) for key, value in description.items()}


def _class_numeric_summary(feature: pd.Series, target: pd.Series) -> list[dict[str, Any]]:
    frame = pd.DataFrame({"feature": feature, "target": target}).dropna(subset=["target"])
    grouped = frame.groupby("target", dropna=False)["feature"].agg(["count", "mean", "median", "std"])
    return [
        {
            "target": str(label),
            **{column: _json_value(value) for column, value in row.items()},
        }
        for label, row in grouped.iterrows()
    ]


def _class_categorical_summary(
    feature: pd.Series, target: pd.Series, max_categories: int = 20
) -> list[dict[str, Any]]:
    values = feature.astype("string").fillna(MISSING)
    top = values.value_counts(dropna=False).head(max_categories).index
    frame = pd.DataFrame({"feature": values, "target": target})
    rows = []
    for category in top:
        subset = frame.loc[frame["feature"] == category, "target"].dropna()
        counts = subset.value_counts()
        rows.append(
            {
                "category": str(category),
                "count": int(len(subset)),
                "dominant_target": str(counts.index[0]) if not counts.empty else None,
                "dominant_target_rate": float(counts.iloc[0] / len(subset)) if len(subset) else None,
            }
        )
    return rows


def analyze_atom(
    atom: str,
    train: pd.DataFrame,
    test: pd.DataFrame | None,
    target: str,
) -> dict[str, Any]:
    series = train[atom]
    result: dict[str, Any] = {
        "atom": atom,
        "dtype": str(series.dtype),
        "kind": "numeric" if pd.api.types.is_numeric_dtype(series) else "categorical",
        "train_missing_rate": float(series.isna().mean()),
        "train_unique": int(series.nunique(dropna=True)),
        "train_unique_rate": float(series.nunique(dropna=True) / max(len(series), 1)),
        "target_purity": _target_purity(series, train[target]),
        "warnings": [],
    }
    test_series = test[atom] if test is not None and atom in test.columns else None

    if result["kind"] == "numeric":
        result["train_summary"] = _numeric_summary(series)
        result["target_association_eta_squared"] = _eta_squared(series, train[target])
        result["target_by_class"] = _class_numeric_summary(series, train[target])
        if test_series is not None:
            result["test_missing_rate"] = float(test_series.isna().mean())
            result["test_summary"] = _numeric_summary(test_series)
            result["train_test_psi"] = _numeric_psi(series, test_series)
    else:
        train_values = series.astype("string").fillna(MISSING)
        result["top_train_values"] = {
            str(key): int(value) for key, value in train_values.value_counts(dropna=False).head(20).items()
        }
        result["target_association_cramers_v"] = _cramers_v(series, train[target])
        result["target_by_category"] = _class_categorical_summary(series, train[target])
        if test_series is not None:
            test_values = test_series.astype("string").fillna(MISSING)
            unseen = sorted(set(test_values.unique()) - set(train_values.unique()))
            result["test_missing_rate"] = float(test_series.isna().mean())
            result["top_test_values"] = {
                str(key): int(value) for key, value in test_values.value_counts(dropna=False).head(20).items()
            }
            result["unseen_test_categories"] = [str(value) for value in unseen[:50]]
            result["train_test_total_variation"] = _total_variation(series, test_series)

    if result["train_missing_rate"] > 0:
        result["warnings"].append("Missing values exist in train.")
    if result["train_unique_rate"] >= 0.95:
        result["warnings"].append("Near-unique values; inspect whether this atom is an identifier.")
    if result.get("unseen_test_categories"):
        result["warnings"].append("Test contains categories absent from the train sample.")
    association = result.get("target_association_eta_squared", result.get("target_association_cramers_v"))
    if association is not None and association >= 0.8:
        result["warnings"].append("Very strong target association; inspect for possible leakage.")
    return result


def build_atom_summary(
    config: ExperimentConfig,
    atoms: list[str] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    train_full = read_csv(config.data.train_path)
    test_full = read_csv(config.data.test_path) if config.data.test_path else None
    target = infer_target(train_full, test_full, config.data.target)
    ids = infer_id_columns(train_full, config.data.id_columns)
    available = [column for column in train_full.columns if column not in {target, *ids}]
    selected = atoms or available
    invalid = [atom for atom in selected if atom not in available]
    if invalid:
        raise ValueError(f"Unknown or excluded atoms: {invalid}. Available atoms: {available}")

    train = _sample(train_full[[*selected, target]], config.data.eda_max_rows, config.validation.random_state)
    test = None
    if test_full is not None:
        test = _sample(
            test_full[selected],
            config.data.eda_max_rows,
            config.validation.random_state,
        )
    analyses = {atom: analyze_atom(atom, train, test, target) for atom in selected}
    return {
        "scope": "sampled exploratory analysis; not a formal experiment",
        "target": target,
        "excluded_id_columns": ids,
        "available_atoms": available,
        "analyzed_atoms": selected,
        "train_rows_used": len(train),
        "test_rows_used": len(test) if test is not None else None,
        "sample_seed": config.validation.random_state,
        "atoms": analyses,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Atom EDA",
        "",
        f"- Scope: {summary['scope']}",
        f"- Target: `{summary['target']}`",
        f"- Analyzed atoms: `{summary['analyzed_atoms']}`",
        f"- Train/test rows used: `{summary['train_rows_used']}` / `{summary['test_rows_used']}`",
        f"- Sample seed: `{summary['sample_seed']}`",
        f"- Runtime: `{summary['elapsed_seconds']:.3f}s`",
        "",
        "## Overview",
        "",
        "| Atom | Kind | Missing | Unique | Target association | Train/test drift | Warnings |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for atom, result in summary["atoms"].items():
        association = result.get("target_association_eta_squared", result.get("target_association_cramers_v"))
        drift = result.get("train_test_psi", result.get("train_test_total_variation"))
        association_text = "-" if association is None else f"{association:.4f}"
        drift_text = "-" if drift is None else f"{drift:.4f}"
        warning_text = "<br>".join(result["warnings"]) if result["warnings"] else "None"
        lines.append(
            f"| {atom} | {result['kind']} | {result['train_missing_rate']:.3%} | "
            f"{result['train_unique']} | {association_text} | {drift_text} | {warning_text} |"
        )

    for atom, result in summary["atoms"].items():
        lines.extend(["", f"## {atom}", ""])
        if result["kind"] == "numeric":
            lines.extend([
                "Target association is eta-squared. Train/test drift is PSI over train-derived quantile bins.",
                "",
                "| Target | Count | Mean | Median | Std |",
                "|---|---:|---:|---:|---:|",
            ])
            for row in result["target_by_class"]:
                lines.append(
                    f"| {row['target']} | {row['count']} | {row['mean']:.3f} | "
                    f"{row['median']:.3f} | {row['std']:.3f} |"
                )
        else:
            lines.extend([
                "Target association is Cramer's V. Train/test drift is total variation distance.",
                "",
                "| Category | Count | Dominant target | Dominant rate |",
                "|---|---:|---|---:|",
            ])
            for row in result["target_by_category"]:
                rate = row["dominant_target_rate"]
                rate_text = "-" if rate is None else f"{rate:.3%}"
                lines.append(
                    f"| {row['category']} | {row['count']} | {row['dominant_target']} | {rate_text} |"
                )
            unseen = result.get("unseen_test_categories", [])
            lines.extend(["", f"Unseen test categories: `{unseen}`"])
    lines.append("")
    return "\n".join(lines)


def run_atom_eda(
    config_path: str,
    output_dir: str,
    atoms: list[str] | None = None,
) -> dict[str, Any]:
    config = ExperimentConfig.from_json(config_path)
    config.validate()
    summary = build_atom_summary(config, atoms)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "atom_eda_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (destination / "ATOM_EDA.md").write_text(render_markdown(summary), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run bounded, per-feature atom EDA")
    parser.add_argument("--config", default="configs/baseline.json")
    parser.add_argument("--output-dir", default="reports/atom_eda/all")
    parser.add_argument(
        "--atoms",
        nargs="+",
        help="Feature columns to analyze. Omit this option to analyze every non-ID feature.",
    )
    args = parser.parse_args()
    summary = run_atom_eda(args.config, args.output_dir, args.atoms)
    print(
        json.dumps(
            {
                "status": "ok",
                "analyzed_atoms": summary["analyzed_atoms"],
                "output_dir": args.output_dir,
                "elapsed_seconds": summary["elapsed_seconds"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
