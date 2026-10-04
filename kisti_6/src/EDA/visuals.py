from __future__ import annotations

from html import escape
from pathlib import Path

import numpy as np
import pandas as pd


_COLORS = ["#2563eb", "#0f766e", "#7c3aed", "#c2410c", "#be123c", "#4d7c0f"]


def _write_svg(path: Path, width: int, height: int, body: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;fill:#172033}'
        '.title{font-size:18px;font-weight:700}.label{font-size:12px}.note{font-size:11px;fill:#64748b}</style>',
        *body,
        "</svg>",
    ]
    path.write_text("\n".join(svg), encoding="utf-8")


def _numeric_columns(frame: pd.DataFrame, target: str, id_columns: list[str], limit: int) -> list[str]:
    excluded = {target, *id_columns}
    return [
        column
        for column in frame.select_dtypes(include="number").columns
        if column not in excluded and frame[column].nunique(dropna=True) > 1
    ][:limit]


def _render_target(frame: pd.DataFrame, target: str, task_type: str, path: Path) -> None:
    series = frame[target].dropna()
    body = [f'<text x="32" y="34" class="title">Target distribution: {escape(str(target))}</text>']
    if task_type == "classification":
        counts = series.astype(str).value_counts().head(30)
        maximum = max(int(counts.max()), 1) if len(counts) else 1
        row_height = 24
        canvas_height = max(140, 80 + row_height * len(counts))
        for index, (label, count) in enumerate(counts.items()):
            y = 62 + index * row_height
            bar_width = 560 * int(count) / maximum
            body.extend(
                [
                    f'<text x="32" y="{y + 13}" class="label">{escape(str(label)[:32])}</text>',
                    f'<rect x="210" y="{y}" width="{bar_width:.1f}" height="16" rx="2" fill="#2563eb"/>',
                    f'<text x="{218 + bar_width:.1f}" y="{y + 13}" class="label">{int(count)}</text>',
                ]
            )
        if series.nunique() > len(counts):
            body.append('<text x="32" y="54" class="note">Top 30 classes shown.</text>')
        _write_svg(path, 860, canvas_height, body)
        return

    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    counts, edges = np.histogram(values, bins="auto" if len(values) >= 2 else 1)
    counts = counts[:40]
    edges = edges[: len(counts) + 1]
    maximum = max(int(counts.max()), 1) if len(counts) else 1
    plot_x, plot_y, plot_width, plot_height = 56, 62, 740, 260
    bar_width = plot_width / max(len(counts), 1)
    for index, count in enumerate(counts):
        height = plot_height * int(count) / maximum
        body.append(
            f'<rect x="{plot_x + index * bar_width:.1f}" y="{plot_y + plot_height - height:.1f}" '
            f'width="{max(bar_width - 1, 1):.1f}" height="{height:.1f}" fill="#2563eb"/>'
        )
    if len(edges):
        body.extend(
            [
                f'<text x="{plot_x}" y="348" class="label">{edges[0]:.4g}</text>',
                f'<text x="{plot_x + plot_width}" y="348" text-anchor="end" class="label">{edges[-1]:.4g}</text>',
            ]
        )
    _write_svg(path, 860, 370, body)


def _render_histograms(frame: pd.DataFrame, columns: list[str], path: Path) -> None:
    panel_width, panel_height = 360, 210
    grid_columns = 2
    rows = (len(columns) + grid_columns - 1) // grid_columns
    body = ['<text x="28" y="32" class="title">Numeric feature histograms</text>']
    for index, column in enumerate(columns):
        x0 = 28 + (index % grid_columns) * panel_width
        y0 = 54 + (index // grid_columns) * panel_height
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if not len(values):
            continue
        low, high = np.quantile(values, [0.01, 0.99]) if len(values) >= 100 else (values.min(), values.max())
        clipped = values[(values >= low) & (values <= high)]
        counts, edges = np.histogram(clipped, bins=min(24, max(5, int(np.sqrt(len(clipped))))))
        maximum = max(int(counts.max()), 1)
        plot_x, plot_y, plot_width, plot_height = x0, y0 + 28, 310, 125
        width = plot_width / max(len(counts), 1)
        body.append(f'<text x="{x0}" y="{y0 + 16}" class="label">{escape(str(column))}</text>')
        for bin_index, count in enumerate(counts):
            height = plot_height * int(count) / maximum
            body.append(
                f'<rect x="{plot_x + bin_index * width:.1f}" y="{plot_y + plot_height - height:.1f}" '
                f'width="{max(width - 1, 1):.1f}" height="{height:.1f}" fill="{_COLORS[index % len(_COLORS)]}"/>'
            )
        body.extend(
            [
                f'<text x="{plot_x}" y="{plot_y + plot_height + 18}" class="note">{edges[0]:.3g}</text>',
                f'<text x="{plot_x + plot_width}" y="{plot_y + plot_height + 18}" text-anchor="end" class="note">{edges[-1]:.3g}</text>',
            ]
        )
    body.append(
        f'<text x="28" y="{68 + rows * panel_height}" class="note">For n &gt;= 100, axes show the 1st-99th percentile range; descriptive statistics retain full values.</text>'
    )
    _write_svg(path, 760, 90 + rows * panel_height, body)


def _correlation_color(value: float) -> str:
    if not np.isfinite(value):
        return "#e2e8f0"
    strength = abs(float(value))
    base = (37, 99, 235) if value >= 0 else (220, 38, 38)
    return "#%02x%02x%02x" % tuple(round(255 - (255 - channel) * strength) for channel in base)


def _render_correlation(frame: pd.DataFrame, columns: list[str], path: Path) -> None:
    correlation = frame[columns].corr(method="pearson", min_periods=3)
    cell = max(24, min(46, 640 // max(len(columns), 1)))
    margin = 180
    width = margin + cell * len(columns) + 32
    body = [
        '<text x="24" y="30" class="title">Pearson correlation heatmap</text>',
        '<text x="24" y="49" class="note">Linear pairwise association; blank cells have insufficient observations.</text>',
    ]
    top = 70
    for row_index, row_name in enumerate(columns):
        y = top + row_index * cell
        body.append(f'<text x="{margin - 8}" y="{y + cell * 0.68:.1f}" text-anchor="end" class="label">{escape(str(row_name)[:22])}</text>')
        for column_index, column_name in enumerate(columns):
            value = correlation.loc[row_name, column_name]
            x = margin + column_index * cell
            label = "" if not np.isfinite(value) else f"{value:.2f}"
            body.extend(
                [
                    f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{_correlation_color(value)}" stroke="#ffffff"/>',
                    f'<text x="{x + cell / 2:.1f}" y="{y + cell * 0.64:.1f}" text-anchor="middle" class="note">{label}</text>',
                ]
            )
    for index, column in enumerate(columns):
        x = margin + index * cell + cell / 2
        body.append(f'<text x="{x:.1f}" y="{top - 6}" transform="rotate(-45 {x:.1f} {top - 6})" class="label">{escape(str(column)[:22])}</text>')
    _write_svg(path, width, top + cell * len(columns) + 24, body)


def render_tabular_visuals(
    frame: pd.DataFrame,
    target: str,
    id_columns: list[str],
    output_dir: str | Path,
    task_type: str,
    max_numeric_features: int = 20,
) -> list[str]:
    """Render bounded, dependency-free SVGs for common tabular EDA."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    artifacts: list[str] = []

    target_path = destination / "target_distribution.svg"
    _render_target(frame, target, task_type, target_path)
    artifacts.append(target_path.name)

    numeric = _numeric_columns(frame, target, id_columns, max_numeric_features)
    if numeric:
        histogram_path = destination / "numeric_histograms.svg"
        _render_histograms(frame, numeric, histogram_path)
        artifacts.append(histogram_path.name)
    if len(numeric) >= 2:
        correlation_path = destination / "correlation_heatmap.svg"
        _render_correlation(frame, numeric, correlation_path)
        artifacts.append(correlation_path.name)
    return artifacts
