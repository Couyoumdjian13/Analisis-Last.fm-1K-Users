"""Evaluacion escalada sobre los 992 usuarios del dataset completo.

En lugar de evaluar sobre los 992 usuarios simultaneamente (lo que puede ser
costoso para modelos como T-BPR), se usa una metodologia de submuestreo:

    - Se realiza el split temporal LOO sobre todos los 992 usuarios.
    - En cada ronda se samplea una muestra aleatoria de N_SAMPLE usuarios
      (default: 100) usando una semilla determinista.
    - Se entrena cada modelo en el subconjunto de entrenamiento de esa muestra
      y se evalua en el subconjunto de test correspondiente.
    - Se repite N_ROUNDS veces (semillas base_seed, base_seed+1, ...).
    - El resultado final es el promedio de metricas sobre todas las rondas.

Esta metodologia permite estimar el desempeno en el dataset completo con
intervalos de incertidumbre, a un costo computacional controlado.

Salidas:
    data/scaled_eval_per_round.csv   — metricas por ronda y modelo
    data/scaled_eval_summary.csv     — promedios finales con std
"""

from __future__ import annotations

import os
import sys
import time
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation import evaluate_recommender, temporal_loo_split
from src.models import (
    MostPopularRecommender,
    PISARecommender,
    RepeatNetRecommender,
    SimpleRepeatRecencyRecommender,
    TemporalBPRRecommender,
)
from src.tbpr_config import load_tbpr_config

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "data"))
FULL_DATA_PATH = os.path.join(DATA_DIR, "lastfm_1k_complete_fixed.parquet")
TBPR_BEST_CFG = os.path.join(DATA_DIR, "tbpr_best_config.json")
PER_ROUND_CSV = os.path.join(DATA_DIR, "scaled_eval_per_round.csv")
SUMMARY_CSV = os.path.join(DATA_DIR, "scaled_eval_summary.csv")

K = 10
N_SAMPLE = 100   # usuarios por ronda
N_ROUNDS = 5     # numero de rondas de submuestreo
BASE_SEED = 42   # semilla base; ronda k usa BASE_SEED + k


def _build_recommenders() -> Dict[str, object]:
    tbpr_cfg = load_tbpr_config(TBPR_BEST_CFG)
    return {
        "MostPopular": MostPopularRecommender(),
        "SimpleRepeat-Recency": SimpleRepeatRecencyRecommender(),
        "TemporalBPR": TemporalBPRRecommender(**tbpr_cfg),
        "PISA": PISARecommender(
            decay=0.6,
            w_activation=0.65,
            w_context=0.25,
            w_popularity=0.10,
            context_window=5,
            top_pop_fallback=500,
        ),
        "RepeatNet": RepeatNetRecommender(
            w_repeat_recency=0.55,
            w_repeat_freq=0.45,
            w_explore_transition=0.70,
            w_explore_pop=0.30,
            top_pop_fallback=500,
        ),
    }


def main() -> None:
    t0 = time.perf_counter()

    print(f"Cargando dataset completo desde {FULL_DATA_PATH}")
    df = pd.read_parquet(FULL_DATA_PATH)
    all_users = df["user_id"].unique()
    print(
        f"  {len(df):,} filas | {len(all_users)} usuarios | "
        f"{df['item_id'].nunique():,} items unicos"
    )

    print("Generando split temporal LOO sobre todos los usuarios ...")
    train_full, test_full = temporal_loo_split(df)
    print(f"  train: {len(train_full):,} | test: {len(test_full):,}")

    print(
        f"\nIniciando evaluacion escalada: {N_ROUNDS} rondas x {N_SAMPLE} usuarios "
        f"(seed base={BASE_SEED})\n"
    )

    per_round_rows: List[Dict] = []
    model_names = list(_build_recommenders().keys())

    for round_idx in range(N_ROUNDS):
        seed = BASE_SEED + round_idx
        rng = np.random.default_rng(seed)
        sampled_users = rng.choice(all_users, size=min(N_SAMPLE, len(all_users)), replace=False)
        sampled_set = set(sampled_users)

        train_r = train_full[train_full["user_id"].isin(sampled_set)].copy()
        test_r = test_full[test_full["user_id"].isin(sampled_set)].copy()

        print(
            f"  Ronda {round_idx + 1}/{N_ROUNDS}  seed={seed}  "
            f"train={len(train_r):,}  test={len(test_r):,}"
        )

        recs = _build_recommenders()
        for name, rec in recs.items():
            t_m = time.perf_counter()
            m = evaluate_recommender(rec, train_r, test_r, k=K, return_per_user=False)
            dt = time.perf_counter() - t_m
            per_round_rows.append(
                {
                    "round": round_idx + 1,
                    "seed": seed,
                    "model": name,
                    "recall@10": m["recall@10"],
                    "ndcg@10": m["ndcg@10"],
                    "mrr": m["mrr"],
                    "repeat_ratio": m["repeat_ratio"],
                    "hits": int(m["hits"]),
                    "n_users": int(m["n_users"]),
                    "eval_seconds": round(dt, 1),
                }
            )
            print(
                f"    {name:<22} recall={m['recall@10']:.4f}  "
                f"ndcg={m['ndcg@10']:.4f}  mrr={m['mrr']:.4f}  ({dt:.1f}s)"
            )

    per_round_df = pd.DataFrame(per_round_rows)
    per_round_df.to_csv(PER_ROUND_CSV, index=False)
    print(f"\nResultados por ronda guardados en {PER_ROUND_CSV}")

    # Calcula promedio y desviacion estandar sobre rondas.
    metrics = ["recall@10", "ndcg@10", "mrr", "repeat_ratio"]
    summary_rows = []
    for name in model_names:
        sub = per_round_df[per_round_df["model"] == name]
        row: Dict = {"model": name}
        for metric in metrics:
            row[f"{metric}_mean"] = round(sub[metric].mean(), 4)
            row[f"{metric}_std"] = round(sub[metric].std(), 4)
        row["total_hits"] = int(sub["hits"].sum())
        row["total_users"] = int(sub["n_users"].sum())
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SUMMARY_CSV, index=False)
    print(f"Resumen consolidado guardado en {SUMMARY_CSV}")

    print("\n=== Resumen (promedio ± std sobre rondas) ===")
    print(
        summary_df[
            ["model", "recall@10_mean", "recall@10_std", "ndcg@10_mean", "ndcg@10_std",
             "mrr_mean", "mrr_std", "repeat_ratio_mean"]
        ].to_string(index=False)
    )
    print(f"\nTiempo total: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
