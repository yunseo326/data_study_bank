from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_oof_predictions(frame: pd.DataFrame, path: str | Path) -> Path:
    """Write OOF predictions in the canonical compressed Parquet format."""
    destination = Path(path)
    if destination.suffix.lower() != ".parquet":
        raise ValueError("OOF artifact path must use the .parquet extension")
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(
        destination,
        index=False,
        engine="pyarrow",
        compression="zstd",
    )
    return destination

