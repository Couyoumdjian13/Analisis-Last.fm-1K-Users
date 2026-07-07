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

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Dict, List, Sequence

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation import evaluate_recommender
from src.models import SimpleRepeatRecencyRecommender, TemporalBPRRecommender
from src.tbpr_config import DEFAULT_TBPR_CONFIG, normalize_tbpr_config

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "data"))
SUBSET_PATH = os.path.join(DATA_DIR, "lastfm_100_users_h1_fixed.parquet")
OUT_RESULTS = os.path.join(DATA_DIR, "tbpr_tuning_results.csv")
OUT_BEST_CFG = os.path.join(DATA_DIR, "tbpr_best_config.json")

K = 10


def _candidate_configs() -> List[Dict[str, float]]:
    """Grid acotado con parametros nuevos de T-BPR."""
    return [
        {
            "factors": 32,
            "learning_rate": 0.010,
            "reg": 1e-4,
            "epochs": 10,
            "alpha": 0.7,
            "beta": 1.2,
            "recency_boost": 0.9,
            "max_recent_boost": 100,
            "temporal_activation_weight": 0.2,
            "temporal_activation_decay": 0.6,
            "hard_negative_ratio": 0.2,
            "hard_negative_pool": 18,
            "target_repeat_ratio": 0.35,
            "calibration_strength": 1.0,
            "calibration_iterations": 16,
        },
        {
            "factors": 48,
            "learning_rate": 0.012,
            "reg": 1e-4,
            "epochs": 12,
            "alpha": 0.8,
            "beta": 1.15,
            "recency_boost": 1.1,
            "max_recent_boost": 140,
            "temporal_activation_weight": 0.3,
            "temporal_activation_decay": 0.6,
            "hard_negative_ratio": 0.25,
            "hard_negative_pool": 24,
            "target_repeat_ratio": 0.35,
            "calibration_strength": 1.0,
            "calibration_iterations": 16,
        },
        {
            "factors": 64,
            "learning_rate": 0.012,
            "reg": 1e-4,
            "epochs": 16,
            "alpha": 0.9,
            "beta": 1.1,
            "recency_boost": 1.5,
            "max_recent_boost": 180,
            "temporal_activation_weight": 0.6,
            "temporal_activation_decay": 0.5,
            "hard_negative_ratio": 0.3,
            "hard_negative_pool": 30,
            "target_repeat_ratio": 0.35,
            "calibration_strength": 1.0,
            "calibration_iterations": 16,
        },
        {
            "factors": 64,
            "learning_rate": 0.008,
            "reg": 5e-5,
            "epochs": 14,
            "alpha": 0.85,
            "beta": 1.1,
            "recency_boost": 1.2,
            "max_recent_boost": 160,
            "temporal_activation_weight": 0.5,
            "temporal_activation_decay": 0.6,
            "hard_negative_ratio": 0.2,
            "hard_negative_pool": 24,
            "target_repeat_ratio": 0.33,
            "calibration_strength": 1.0,
            "calibration_iterations": 16,
        },
        {
            "factors": 96,
            "learning_rate": 0.010,
            "reg": 1e-4,
            "epochs": 14,
            "alpha": 0.9,
            "beta": 1.1,
            "recency_boost": 1.4,
            "max_recent_boost": 200,
            "temporal_activation_weight": 0.6,
            "temporal_activation_decay": 0.5,
            "hard_negative_ratio": 0.25,
            "hard_negative_pool": 30,
            "target_repeat_ratio": 0.35,
            "calibration_strength": 1.0,
            "calibration_iterations": 16,
        },
        {
            "factors": 48,
            "learning_rate": 0.015,
            "reg": 1e-4,
            "epochs": 12,
            "alpha": 0.8,
            "beta": 1.2,
            "recency_boost": 1.0,
            "max_recent_boost": 120,
            "temporal_activation_weight": 0.4,
            "temporal_activation_decay": 0.7,
            "hard_negative_ratio": 0.3,
            "hard_negative_pool": 24,
            "target_repeat_ratio": 0.30,
            "calibration_strength": 1.0,
            "calibration_iterations": 16,
        },
    ]


def _evaluate_baseline(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> Dict[str, float]:
    baseline = SimpleRepeatRecencyRecommender()
    return evaluate_recommender(baseline, train_df, test_df, k=K, return_per_user=False)


def temporal_train_valid_test_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split temporal por usuario: train (todo-2), valid (penultima), test (ultima)."""
    df_sorted = df.sort_values(["user_id", "timestamp"], kind="mergesort")

    test_df = df_sorted.groupby("user_id", observed=True).tail(1)
    rem_df = df_sorted.drop(test_df.index)
    valid_df = rem_df.groupby("user_id", observed=True).tail(1)
    train_df = rem_df.drop(valid_df.index)

    return train_df, valid_df, test_df


def temporal_rolling_valid_splits(
    df: pd.DataFrame, holdout_offsets: Sequence[int]
) -> List[tuple[pd.DataFrame, pd.DataFrame]]:
    """Genera folds temporales: valid es la interaccion -offset por usuario."""
    df_sorted = df.sort_values(["user_id", "timestamp"], kind="mergesort")
    folds: List[tuple[pd.DataFrame, pd.DataFrame]] = []

    for offset in holdout_offsets:
        valid_parts = []
        train_parts = []
        for _, group in df_sorted.groupby("user_id", observed=True, sort=False):
            n = len(group)
            if n <= offset:
                continue
            valid_pos = n - offset
            valid_parts.append(group.iloc[[valid_pos]])
            train_parts.append(group.iloc[:valid_pos])

        if not valid_parts:
            continue

        valid_df = pd.concat(valid_parts, ignore_index=True)
        train_df = pd.concat(train_parts, ignore_index=True)
        folds.append((train_df, valid_df))

    return folds


def _build_tbpr(cfg: Dict[str, float]) -> TemporalBPRRecommender:
    full_cfg = dict(DEFAULT_TBPR_CONFIG)
    # Solo tomamos claves de configuracion validas: la fila ganadora del grid
    # arrastra columnas de metricas (rolling_*, eval_seconds, delta_*) que no
    # son argumentos del constructor de TemporalBPRRecommender.
    full_cfg.update({k: v for k, v in cfg.items() if k in DEFAULT_TBPR_CONFIG})
    full_cfg = normalize_tbpr_config(full_cfg)
    return TemporalBPRRecommender(**full_cfg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tuning TemporalBPR con validacion rolling"
    )
    parser.add_argument(
        "--max-configs",
        type=int,
        default=0,
        help="Limita numero de configuraciones evaluadas (0 = todas)",
    )
    parser.add_argument(
        "--fold-offsets",
        type=str,
        default="2,3,4",
        help="Offsets de validacion temporal separados por coma, p.ej. 2,3,4",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.perf_counter()
    print(f"Cargando {SUBSET_PATH}")
    df = pd.read_parquet(SUBSET_PATH)
    train_df, valid_df, test_df = temporal_train_valid_test_split(df)
    train_valid_df = pd.concat([train_df, valid_df], ignore_index=True)
    offsets = [int(x.strip()) for x in args.fold_offsets.split(",") if x.strip()]
    rolling_folds = temporal_rolling_valid_splits(df, offsets)
    if not rolling_folds:
        raise RuntimeError(
            "No se pudieron generar folds rolling; revisa --fold-offsets"
        )
    print(
        f"  train: {len(train_df):,} | valid: {len(valid_df):,} "
        f"| test: {len(test_df):,} | users: {test_df['user_id'].nunique()}"
    )
    print(f"  rolling folds: {len(rolling_folds)} | offsets={offsets}")

    baseline_fold_rows = []
    for fold_idx, (fold_train, fold_valid) in enumerate(rolling_folds, start=1):
        bm = _evaluate_baseline(fold_train, fold_valid)
        baseline_fold_rows.append(bm)
        print(
            f"  Baseline fold {fold_idx}: recall@10={bm['recall@10']:.4f} "
            f"ndcg@10={bm['ndcg@10']:.4f} mrr={bm['mrr']:.4f} repeat_ratio={bm['repeat_ratio']:.4f}"
        )

    baseline_m = {
        "recall@10": sum(b["recall@10"] for b in baseline_fold_rows)
        / len(baseline_fold_rows),
        "ndcg@10": sum(b["ndcg@10"] for b in baseline_fold_rows)
        / len(baseline_fold_rows),
        "mrr": sum(b["mrr"] for b in baseline_fold_rows) / len(baseline_fold_rows),
        "repeat_ratio": sum(b["repeat_ratio"] for b in baseline_fold_rows)
        / len(baseline_fold_rows),
    }
    print(
        "Baseline SimpleRepeat-Recency (rolling avg) -> "
        f"recall@10={baseline_m['recall@10']:.4f} "
        f"ndcg@10={baseline_m['ndcg@10']:.4f} "
        f"mrr={baseline_m['mrr']:.4f} "
        f"repeat_ratio={baseline_m['repeat_ratio']:.4f}"
    )

    rows = []
    cfgs = _candidate_configs()
    if args.max_configs > 0:
        cfgs = cfgs[: args.max_configs]

    for idx, cfg in enumerate(cfgs, start=1):
        print(f"\n[{idx}/{len(cfgs)}] Evaluando config: {cfg}", flush=True)
        t_cfg = time.perf_counter()
        fold_metrics = []
        for fold_idx, (fold_train, fold_valid) in enumerate(rolling_folds, start=1):
            rec = _build_tbpr(cfg)
            m = evaluate_recommender(
                rec, fold_train, fold_valid, k=K, return_per_user=False
            )
            fold_metrics.append(m)
            print(
                f"    fold {fold_idx}: recall@10={m['recall@10']:.4f} "
                f"ndcg@10={m['ndcg@10']:.4f} mrr={m['mrr']:.4f} repeat_ratio={m['repeat_ratio']:.4f}"
            )
        dt = time.perf_counter() - t_cfg

        m = {
            "recall@10": sum(x["recall@10"] for x in fold_metrics) / len(fold_metrics),
            "ndcg@10": sum(x["ndcg@10"] for x in fold_metrics) / len(fold_metrics),
            "mrr": sum(x["mrr"] for x in fold_metrics) / len(fold_metrics),
            "repeat_ratio": sum(x["repeat_ratio"] for x in fold_metrics)
            / len(fold_metrics),
            "hits": sum(int(x["hits"]) for x in fold_metrics),
            "n_users": sum(int(x["n_users"]) for x in fold_metrics),
        }

        row = {
            **cfg,
            "rolling_recall@10": m["recall@10"],
            "rolling_ndcg@10": m["ndcg@10"],
            "rolling_mrr": m["mrr"],
            "rolling_repeat_ratio": m["repeat_ratio"],
            "rolling_hits": int(m["hits"]),
            "rolling_n_users": int(m["n_users"]),
            "eval_seconds": round(dt, 2),
            "delta_rolling_recall_vs_recency": m["recall@10"] - baseline_m["recall@10"],
            "delta_rolling_ndcg_vs_recency": m["ndcg@10"] - baseline_m["ndcg@10"],
        }
        rows.append(row)
        print(
            f"  rolling_recall@10={row['rolling_recall@10']:.4f} "
            f"rolling_ndcg@10={row['rolling_ndcg@10']:.4f} "
            f"rolling_mrr={row['rolling_mrr']:.4f} rolling_repeat_ratio={row['rolling_repeat_ratio']:.4f} "
            f"({dt:.1f}s)"
        )

    out = pd.DataFrame(rows)
    out = out.sort_values(
        ["rolling_ndcg@10", "rolling_recall@10", "rolling_mrr"], ascending=False
    ).reset_index(drop=True)
    out.to_csv(OUT_RESULTS, index=False)

    best_cfg = out.iloc[0].to_dict()

    # Reentrena mejor configuracion sobre train+valid y evalua en test final.
    best_rec = _build_tbpr(best_cfg)
    test_m = evaluate_recommender(
        best_rec, train_valid_df, test_df, k=K, return_per_user=False
    )

    best = {
        "factors": int(best_cfg["factors"]),
        "learning_rate": float(best_cfg["learning_rate"]),
        "reg": float(best_cfg["reg"]),
        "epochs": int(best_cfg["epochs"]),
        "alpha": float(best_cfg["alpha"]),
        "beta": float(best_cfg["beta"]),
        "recency_boost": float(best_cfg.get("recency_boost", 0.0)),
        "max_recent_boost": int(best_cfg.get("max_recent_boost", 100)),
        "temporal_activation_weight": float(
            best_cfg.get("temporal_activation_weight", 0.0)
        ),
        "temporal_activation_decay": float(
            best_cfg.get("temporal_activation_decay", 0.6)
        ),
        "hard_negative_ratio": float(best_cfg.get("hard_negative_ratio", 0.0)),
        "hard_negative_pool": int(best_cfg.get("hard_negative_pool", 24)),
        "target_repeat_ratio": float(best_cfg.get("target_repeat_ratio", 0.35)),
        "calibration_strength": float(best_cfg.get("calibration_strength", 1.0)),
        "calibration_iterations": int(best_cfg.get("calibration_iterations", 16)),
        "rolling_recall@10": float(best_cfg["rolling_recall@10"]),
        "rolling_ndcg@10": float(best_cfg["rolling_ndcg@10"]),
        "rolling_mrr": float(best_cfg["rolling_mrr"]),
        "test_recall@10": float(test_m["recall@10"]),
        "test_ndcg@10": float(test_m["ndcg@10"]),
        "test_mrr": float(test_m["mrr"]),
        "test_repeat_ratio": float(test_m["repeat_ratio"]),
        "test_hits": int(test_m["hits"]),
        "test_n_users": int(test_m["n_users"]),
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "selection_protocol": "rolling_offsets",
        "selection_offsets": offsets,
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
                "target_repeat_ratio",
                "calibration_strength",
                "calibration_iterations",
                "rolling_recall@10",
                "rolling_ndcg@10",
                "rolling_mrr",
                "rolling_repeat_ratio",
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
