from __future__ import annotations

import argparse
import json
import time

from src.EDA.run_eda import run_eda
from src.run_experiment import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EDA and a baseline experiment")
    parser.add_argument("--config", default="configs/baseline.json")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--skip-eda", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    if not args.skip_eda:
        run_eda(args.config)
    result = run_experiment(args.config, args.experiment_id)
    print(
        json.dumps(
            {
                "status": "ok",
                "experiment_id": args.experiment_id,
                "total_runtime_seconds": time.perf_counter() - started,
                "metrics": result["overall"],
            }
        )
    )


if __name__ == "__main__":
    main()

