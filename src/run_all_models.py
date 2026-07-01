"""Run all model evaluations with a shared run_id.

This script executes baselines, advanced models, and standalone T-BPR in a
single session so outputs are easy to align and audit.
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timezone

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.run_baselines import main as run_baselines_main
from src.run_repeat_advanced import main as run_advanced_main
from src.run_tbpr import main as run_tbpr_main

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "data"))
BASELINES_RESULTS = os.path.join(DATA_DIR, "baselines_results.csv")
ADV_RESULTS = os.path.join(DATA_DIR, "repeat_advanced_results.csv")
TBPR_RESULTS = os.path.join(DATA_DIR, "tbpr_results.csv")
ALL_RESULTS = os.path.join(DATA_DIR, "all_models_results.csv")


def main() -> None:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    t0 = time.perf_counter()
    print(f"Run completo con run_id={run_id}")

    run_baselines_main(run_id=run_id)
    run_advanced_main(run_id=run_id)
    run_tbpr_main(run_id=run_id)

    baselines = pd.read_csv(BASELINES_RESULTS)
    advanced = pd.read_csv(ADV_RESULTS)
    tbpr = pd.read_csv(TBPR_RESULTS)

    all_results = pd.concat([baselines, advanced, tbpr], ignore_index=True)
    all_results.to_csv(ALL_RESULTS, index=False)

    print(f"\nResultados consolidados en {ALL_RESULTS}")
    print(f"Tiempo total: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
