"""Entrena y evalua Temporal BPR (T-BPR) en el subset H1.

Usa el mismo protocolo temporal leave-one-out de src/evaluation.py para que
la comparacion con baselines sea directa.
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timezone

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation import evaluate_recommender, temporal_loo_split
from src.models import TemporalBPRRecommender
from src.tbpr_config import load_tbpr_config

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "data"))
SUBSET_PATH = os.path.join(DATA_DIR, "lastfm_100_users_h1_fixed.parquet")
RESULTS_CSV = os.path.join(DATA_DIR, "tbpr_results.csv")
HITS_CSV = os.path.join(DATA_DIR, "tbpr_hits.csv")
TBPR_BEST_CFG = os.path.join(DATA_DIR, "tbpr_best_config.json")

K = 10


def main(run_id: str | None = None) -> None:
    print(f"Cargando {SUBSET_PATH}")
    t0 = time.perf_counter()
    run_id = run_id or os.environ.get("MODEL_RUN_ID")
    if not run_id:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    df = pd.read_parquet(SUBSET_PATH)
    print(
        f"  {len(df):,} filas | {df['user_id'].nunique()} usuarios "
        f"| {df['item_id'].nunique():,} items"
    )

    train_df, test_df = temporal_loo_split(df)
    print(f"Split temporal LOO -> train: {len(train_df):,} | test: {len(test_df):,}")

    tbpr_cfg = load_tbpr_config(TBPR_BEST_CFG)
    print(f"Config TemporalBPR: {tbpr_cfg}")
    print(f"run_id: {run_id}")
    rec = TemporalBPRRecommender(**tbpr_cfg)

    print("\nEntrenando + evaluando T-BPR ...", flush=True)
    t_eval = time.perf_counter()
    m, per_user = evaluate_recommender(
        rec, train_df, test_df, k=K, return_per_user=True
    )
    dt = time.perf_counter() - t_eval

    out = pd.DataFrame(
        [
            {
                "run_id": run_id,
                "model": "TemporalBPR",
                "recall@10": m["recall@10"],
                "ndcg@10": m["ndcg@10"],
                "mrr": m["mrr"],
                "repeat_ratio": m["repeat_ratio"],
                "hits": int(m["hits"]),
                "n_users": int(m["n_users"]),
                "eval_seconds": round(dt, 2),
            }
        ]
    )
    out.to_csv(RESULTS_CSV, index=False)

    per_user.insert(0, "model", "TemporalBPR")
    per_user.insert(0, "run_id", run_id)
    hits_df = per_user[per_user["hit"] == 1].copy()
    expected_hits = int(m["hits"])
    if len(hits_df) != expected_hits:
        raise RuntimeError(
            f"Inconsistencia entre resultados e hits: expected={expected_hits}, got={len(hits_df)}"
        )
    hits_df.to_csv(HITS_CSV, index=False)

    print(f"\nResultados guardados en {RESULTS_CSV}")
    print(f"Detalle de hits guardado en {HITS_CSV} ({len(hits_df)} filas)")
    print(f"Tiempo total: {time.perf_counter() - t0:.1f}s\n")

    for kpi in ("recall@10", "ndcg@10", "mrr", "repeat_ratio"):
        print(f"  {kpi:<14}{m[kpi]:.4f}")
    print(f"  hits          {int(m['hits'])}/{int(m['n_users'])}  ({dt:.1f}s)")


if __name__ == "__main__":
    main()
