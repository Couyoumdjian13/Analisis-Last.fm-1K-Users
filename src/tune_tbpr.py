"""Busqueda de hiperparametros para Temporal BPR (T-BPR).

Ejecuta varias configuraciones sobre el subset H1 bajo el mismo protocolo
temporal LOO y guarda:
    - data/tbpr_tuning_results.csv
    - data/tbpr_best_config.json

Ademas reporta comparacion contra el baseline SimpleRepeat-Recency.

La seleccion de mejor configuracion se hace sobre validacion temporal por
usuario (penultima interaccion) y luego se reporta desempeno final en test
(ultima interaccion) reentrenando con train+valid.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Dict, List

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation import evaluate_recommender
from src.models import SimpleRepeatRecencyRecommender, TemporalBPRRecommender


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "data"))
SUBSET_PATH = os.path.join(DATA_DIR, "lastfm_100_users_h1_fixed.parquet")
OUT_RESULTS = os.path.join(DATA_DIR, "tbpr_tuning_results.csv")
OUT_BEST_CFG = os.path.join(DATA_DIR, "tbpr_best_config.json")

K = 10


def _candidate_configs() -> List[Dict[str, float]]:
    """Grid acotado con parametros nuevos de T-BPR."""
    return [
        {"factors": 32, "learning_rate": 0.010, "reg": 1e-4, "epochs": 10, "alpha": 0.6, "beta": 1.2, "recency_boost": 0.7, "max_recent_boost": 80, "temporal_activation_weight": 0.2, "temporal_activation_decay": 0.6, "hard_negative_ratio": 0.2, "hard_negative_pool": 18},
        {"factors": 32, "learning_rate": 0.015, "reg": 1e-4, "epochs": 12, "alpha": 0.7, "beta": 1.2, "recency_boost": 1.0, "max_recent_boost": 100, "temporal_activation_weight": 0.3, "temporal_activation_decay": 0.6, "hard_negative_ratio": 0.3, "hard_negative_pool": 24},
        {"factors": 48, "learning_rate": 0.010, "reg": 5e-5, "epochs": 12, "alpha": 0.7, "beta": 1.1, "recency_boost": 1.0, "max_recent_boost": 120, "temporal_activation_weight": 0.4, "temporal_activation_decay": 0.6, "hard_negative_ratio": 0.3, "hard_negative_pool": 24},
        {"factors": 48, "learning_rate": 0.015, "reg": 1e-4, "epochs": 14, "alpha": 0.8, "beta": 1.1, "recency_boost": 1.2, "max_recent_boost": 150, "temporal_activation_weight": 0.4, "temporal_activation_decay": 0.5, "hard_negative_ratio": 0.4, "hard_negative_pool": 30},
        {"factors": 64, "learning_rate": 0.008, "reg": 5e-5, "epochs": 14, "alpha": 0.8, "beta": 1.1, "recency_boost": 1.2, "max_recent_boost": 150, "temporal_activation_weight": 0.5, "temporal_activation_decay": 0.6, "hard_negative_ratio": 0.4, "hard_negative_pool": 30},
        {"factors": 64, "learning_rate": 0.012, "reg": 1e-4, "epochs": 16, "alpha": 0.9, "beta": 1.1, "recency_boost": 1.5, "max_recent_boost": 180, "temporal_activation_weight": 0.6, "temporal_activation_decay": 0.5, "hard_negative_ratio": 0.5, "hard_negative_pool": 36},
        {"factors": 64, "learning_rate": 0.020, "reg": 1e-4, "epochs": 12, "alpha": 0.8, "beta": 1.2, "recency_boost": 1.3, "max_recent_boost": 120, "temporal_activation_weight": 0.5, "temporal_activation_decay": 0.7, "hard_negative_ratio": 0.3, "hard_negative_pool": 24},
        {"factors": 96, "learning_rate": 0.010, "reg": 1e-4, "epochs": 16, "alpha": 0.9, "beta": 1.1, "recency_boost": 1.5, "max_recent_boost": 200, "temporal_activation_weight": 0.7, "temporal_activation_decay": 0.5, "hard_negative_ratio": 0.5, "hard_negative_pool": 36},
        {"factors": 96, "learning_rate": 0.006, "reg": 5e-4, "epochs": 18, "alpha": 0.85, "beta": 1.15, "recency_boost": 1.0, "max_recent_boost": 180, "temporal_activation_weight": 0.6, "temporal_activation_decay": 0.6, "hard_negative_ratio": 0.4, "hard_negative_pool": 30},
        {"factors": 48, "learning_rate": 0.020, "reg": 1e-3, "epochs": 10, "alpha": 0.7, "beta": 1.2, "recency_boost": 0.8, "max_recent_boost": 100, "temporal_activation_weight": 0.3, "temporal_activation_decay": 0.7, "hard_negative_ratio": 0.2, "hard_negative_pool": 18},
    ]


def _evaluate_baseline(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, float]:
    baseline = SimpleRepeatRecencyRecommender()
    return evaluate_recommender(baseline, train_df, test_df, k=K, return_per_user=False)


def temporal_train_valid_test_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split temporal por usuario: train (todo-2), valid (penultima), test (ultima)."""
    df_sorted = df.sort_values(["user_id", "timestamp"], kind="mergesort")

    test_df = df_sorted.groupby("user_id", observed=True).tail(1)
    rem_df = df_sorted.drop(test_df.index)
    valid_df = rem_df.groupby("user_id", observed=True).tail(1)
    train_df = rem_df.drop(valid_df.index)

    return train_df, valid_df, test_df


def main() -> None:
    t0 = time.perf_counter()
    print(f"Cargando {SUBSET_PATH}")
    df = pd.read_parquet(SUBSET_PATH)
    train_df, valid_df, test_df = temporal_train_valid_test_split(df)
    train_valid_df = pd.concat([train_df, valid_df], ignore_index=True)
    print(
        f"  train: {len(train_df):,} | valid: {len(valid_df):,} "
        f"| test: {len(test_df):,} | users: {test_df['user_id'].nunique()}"
    )

    baseline_m = _evaluate_baseline(train_df, valid_df)
    print(
        "Baseline SimpleRepeat-Recency (valid) -> "
        f"recall@10={baseline_m['recall@10']:.4f} "
        f"ndcg@10={baseline_m['ndcg@10']:.4f} "
        f"mrr={baseline_m['mrr']:.4f} "
        f"repeat_ratio={baseline_m['repeat_ratio']:.4f}"
    )

    rows = []
    cfgs = _candidate_configs()
    for idx, cfg in enumerate(cfgs, start=1):
        print(f"\n[{idx}/{len(cfgs)}] Evaluando config: {cfg}", flush=True)
        rec = TemporalBPRRecommender(
            factors=int(cfg["factors"]),
            learning_rate=float(cfg["learning_rate"]),
            reg=float(cfg["reg"]),
            epochs=int(cfg["epochs"]),
            alpha=float(cfg["alpha"]),
            beta=float(cfg["beta"]),
            recency_boost=float(cfg.get("recency_boost", 0.0)),
            max_recent_boost=int(cfg.get("max_recent_boost", 100)),
            temporal_activation_weight=float(cfg.get("temporal_activation_weight", 0.0)),
            temporal_activation_decay=float(cfg.get("temporal_activation_decay", 0.6)),
            hard_negative_ratio=float(cfg.get("hard_negative_ratio", 0.0)),
            hard_negative_pool=int(cfg.get("hard_negative_pool", 24)),
            random_state=42,
            verbose=False,
        )

        t_cfg = time.perf_counter()
        m = evaluate_recommender(rec, train_df, valid_df, k=K, return_per_user=False)
        dt = time.perf_counter() - t_cfg

        row = {
            **cfg,
            "valid_recall@10": m["recall@10"],
            "valid_ndcg@10": m["ndcg@10"],
            "valid_mrr": m["mrr"],
            "valid_repeat_ratio": m["repeat_ratio"],
            "valid_hits": int(m["hits"]),
            "valid_n_users": int(m["n_users"]),
            "eval_seconds": round(dt, 2),
            "delta_valid_recall_vs_recency": m["recall@10"] - baseline_m["recall@10"],
            "delta_valid_ndcg_vs_recency": m["ndcg@10"] - baseline_m["ndcg@10"],
        }
        rows.append(row)
        print(
            f"  valid_recall@10={row['valid_recall@10']:.4f} "
            f"valid_ndcg@10={row['valid_ndcg@10']:.4f} "
            f"valid_mrr={row['valid_mrr']:.4f} valid_repeat_ratio={row['valid_repeat_ratio']:.4f} "
            f"({dt:.1f}s)"
        )

    out = pd.DataFrame(rows)
    out = out.sort_values(["valid_ndcg@10", "valid_recall@10", "valid_mrr"], ascending=False).reset_index(drop=True)
    out.to_csv(OUT_RESULTS, index=False)

    best_cfg = out.iloc[0].to_dict()

    # Reentrena mejor configuracion sobre train+valid y evalua en test final.
    best_rec = TemporalBPRRecommender(
        factors=int(best_cfg["factors"]),
        learning_rate=float(best_cfg["learning_rate"]),
        reg=float(best_cfg["reg"]),
        epochs=int(best_cfg["epochs"]),
        alpha=float(best_cfg["alpha"]),
        beta=float(best_cfg["beta"]),
        recency_boost=float(best_cfg.get("recency_boost", 0.0)),
        max_recent_boost=int(best_cfg.get("max_recent_boost", 100)),
        temporal_activation_weight=float(best_cfg.get("temporal_activation_weight", 0.0)),
        temporal_activation_decay=float(best_cfg.get("temporal_activation_decay", 0.6)),
        hard_negative_ratio=float(best_cfg.get("hard_negative_ratio", 0.0)),
        hard_negative_pool=int(best_cfg.get("hard_negative_pool", 24)),
        random_state=42,
        verbose=False,
    )
    test_m = evaluate_recommender(best_rec, train_valid_df, test_df, k=K, return_per_user=False)

    best = {
        "factors": int(best_cfg["factors"]),
        "learning_rate": float(best_cfg["learning_rate"]),
        "reg": float(best_cfg["reg"]),
        "epochs": int(best_cfg["epochs"]),
        "alpha": float(best_cfg["alpha"]),
        "beta": float(best_cfg["beta"]),
        "recency_boost": float(best_cfg.get("recency_boost", 0.0)),
        "max_recent_boost": int(best_cfg.get("max_recent_boost", 100)),
        "temporal_activation_weight": float(best_cfg.get("temporal_activation_weight", 0.0)),
        "temporal_activation_decay": float(best_cfg.get("temporal_activation_decay", 0.6)),
        "hard_negative_ratio": float(best_cfg.get("hard_negative_ratio", 0.0)),
        "hard_negative_pool": int(best_cfg.get("hard_negative_pool", 24)),
        "valid_recall@10": float(best_cfg["valid_recall@10"]),
        "valid_ndcg@10": float(best_cfg["valid_ndcg@10"]),
        "valid_mrr": float(best_cfg["valid_mrr"]),
        "test_recall@10": float(test_m["recall@10"]),
        "test_ndcg@10": float(test_m["ndcg@10"]),
        "test_mrr": float(test_m["mrr"]),
        "test_repeat_ratio": float(test_m["repeat_ratio"]),
        "test_hits": int(test_m["hits"]),
        "test_n_users": int(test_m["n_users"]),
    }

    with open(OUT_BEST_CFG, "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)

    print("\n=== Top-5 configs T-BPR ===")
    print(
        out[
            [
                "factors",
                "learning_rate",
                "reg",
                "epochs",
                "alpha",
                "beta",
                "recency_boost",
                "max_recent_boost",
                "temporal_activation_weight",
                "temporal_activation_decay",
                "hard_negative_ratio",
                "hard_negative_pool",
                "valid_recall@10",
                "valid_ndcg@10",
                "valid_mrr",
                "valid_repeat_ratio",
                "eval_seconds",
            ]
        ]
        .head(5)
        .to_string(index=False)
    )

    print("\n=== Mejor config evaluada en TEST final ===")
    print(
        f"  test_recall@10={best['test_recall@10']:.4f} "
        f"test_ndcg@10={best['test_ndcg@10']:.4f} "
        f"test_mrr={best['test_mrr']:.4f} "
        f"test_repeat_ratio={best['test_repeat_ratio']:.4f} "
        f"hits={best['test_hits']}/{best['test_n_users']}"
    )

    print(f"\nResultados completos: {OUT_RESULTS}")
    print(f"Mejor config JSON: {OUT_BEST_CFG}")
    print(f"Tiempo total: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
