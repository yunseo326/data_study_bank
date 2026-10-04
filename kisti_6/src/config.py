from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DataConfig:
    train_path: str
    test_path: str | None = None
    target: str | None = None
    id_columns: list[str] = field(default_factory=list)
    max_train_rows: int = 60_000
    eda_max_rows: int = 100_000


@dataclass
class ValidationConfig:
    n_splits: int = 3
    shuffle: bool = True
    random_state: int = 42


@dataclass
class ModelConfig:
    name: str = "hist_gradient_boosting"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class RuntimeConfig:
    time_budget_seconds: float = 55.0
    n_threads: int = 4


@dataclass
class ExperimentConfig:
    data: DataConfig
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    @classmethod
    def from_json(cls, path: str | Path) -> "ExperimentConfig":
        with Path(path).open(encoding="utf-8") as file:
            raw = json.load(file)
        return cls(
            data=DataConfig(**raw["data"]),
            validation=ValidationConfig(**raw.get("validation", {})),
            model=ModelConfig(**raw.get("model", {})),
            runtime=RuntimeConfig(**raw.get("runtime", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        if self.validation.n_splits < 2:
            raise ValueError("validation.n_splits must be at least 2")
        if self.data.max_train_rows < self.validation.n_splits * 10:
            raise ValueError("max_train_rows is too small for stable cross-validation")
        if self.runtime.time_budget_seconds <= 0:
            raise ValueError("time_budget_seconds must be positive")
