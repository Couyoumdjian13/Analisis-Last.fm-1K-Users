"""Analisis de sensibilidad sobre el intervalo de repeticion W_u = (p_low, p_high).

El objetivo es cuantificar como cambia el desempeno de T-BPR al variar los
percentiles que definen la ventana de repeticion optima por usuario.

Por defecto se prueban todas las combinaciones de:
    lower_pct in [10, 15, 20, 25, 30, 35, 40]
    upper_pct in [60, 65, 70, 75, 80, 85, 90]
    con la restriccion lower_pct < upper_pct

Se usa el subset H1 (100 usuarios) con el protocolo temporal LOO y se reporta
Recall@10, nDCG@10 y MRR para cada configuracion.

Salida:
    data/sensitivity_interval_results.csv
    docs/figures/fig_sensitivity_interval.png  (opcional, requiere matplotlib)
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timezone
from typing import List, Tuple

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation import evaluate_recommender, temporal_loo_split
from src.models import TemporalBPRRecommender
from src.tbpr_config import DEFAULT_TBPR_CONFIG, normalize_tbpr_config

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "data"))
SUBSET_PATH = os.path.join(DATA_DIR, "lastfm_100_users_h1_fixed.parquet")
RESULTS_CSV = os.path.join(DATA_DIR, "sensitivity_interval_results.csv")
FIGURES_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "docs", "figures"))

K = 10

# Percentile combinations to evaluate.
LOWER_PCTS = [10, 15, 20, 25, 30, 35, 40]
UPPER_PCTS = [60, 65, 70, 75, 80, 85, 90]


def _build_tbpr(low_pct: int, high_pct: int) -> TemporalBPRRecommender:
    cfg = dict(DEFAULT_TBPR_CONFIG)
    cfg["window_low_pct"] = low_pct
    cfg["window_high_pct"] = high_pct
    cfg["verbose"] = False
    return TemporalBPRRecommender(**normalize_tbpr_config(cfg))


def _interval_pairs() -> List[Tuple[int, int]]:
    pairs = []
    for lo in LOWER_PCTS:
        for hi in UPPER_PCTS:
            if lo < hi:
                pairs.append((lo, hi))
    return pairs


def main() -> None:
    t0 = time.perf_counter()
    print(f"Cargando {SUBSET_PATH}")
    df = pd.read_parquet(SUBSET_PATH)
    train_df, test_df = temporal_loo_split(df)
    print(
        f"Split LOO: train={len(train_df):,} | test={len(test_df):,} | "
        f"usuarios={test_df['user_id'].nunique()}"
    )

    pairs = _interval_pairs()
    print(f"Evaluando {len(pairs)} combinaciones (low_pct, high_pct) ...\n")

    rows = []
    for idx, (lo, hi) in enumerate(pairs, start=1):
        t_cfg = time.perf_counter()
        rec = _build_tbpr(lo, hi)
        m = evaluate_recommender(rec, train_df, test_df, k=K, return_per_user=False)
        dt = time.perf_counter() - t_cfg

        row = {
            "window_low_pct": lo,
            "window_high_pct": hi,
            "window_label": f"p{lo}-p{hi}",
            "recall@10": round(m["recall@10"], 4),
            "ndcg@10": round(m["ndcg@10"], 4),
            "mrr": round(m["mrr"], 4),
            "repeat_ratio": round(m["repeat_ratio"], 4),
            "hits": int(m["hits"]),
            "n_users": int(m["n_users"]),
            "eval_seconds": round(dt, 1),
        }
        rows.append(row)
        print(
            f"  [{idx:>2}/{len(pairs)}] p{lo}-p{hi}: "
            f"recall={row['recall@10']:.4f}  ndcg={row['ndcg@10']:.4f}  "
            f"mrr={row['mrr']:.4f}  repeat_ratio={row['repeat_ratio']:.4f}  "
            f"({dt:.1f}s)"
        )

    out = pd.DataFrame(rows).sort_values(
        ["ndcg@10", "recall@10", "mrr"], ascending=False
    ).reset_index(drop=True)
    out.to_csv(RESULTS_CSV, index=False)
    print(f"\nResultados guardados en {RESULTS_CSV}")
    print("\n=== Top-10 configuraciones por nDCG@10 ===")
    print(
        out[["window_label", "recall@10", "ndcg@10", "mrr", "repeat_ratio"]]
        .head(10)
        .to_string(index=False)
    )

    _try_plot(out)
    print(f"\nTiempo total: {time.perf_counter() - t0:.1f}s")


def _try_plot(results: pd.DataFrame) -> None:
    """Genera heatmap de nDCG@10 por (low_pct, high_pct) si matplotlib esta disponible."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        pivot = results.pivot(
            index="window_low_pct", columns="window_high_pct", values="ndcg@10"
        )

        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_yticks(range(len(pivot.index)))
        ax.set_xticklabels([f"p{c}" for c in pivot.columns])
        ax.set_yticklabels([f"p{r}" for r in pivot.index])
        ax.set_xlabel("Percentil superior de la ventana W_u")
        ax.set_ylabel("Percentil inferior de la ventana W_u")
        ax.set_title("Sensibilidad de T-BPR al intervalo W_u: nDCG@10")

        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    ax.text(
                        j, i, f"{val:.3f}", ha="center", va="center", fontsize=7,
                        color="black"
                    )

        plt.colorbar(im, ax=ax, label="nDCG@10")
        plt.tight_layout()

        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig_path = os.path.join(FIGURES_DIR, "fig_sensitivity_interval.png")
        plt.savefig(fig_path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"Figura guardada en {fig_path}")
    except ImportError:
        print("matplotlib no disponible; omitiendo figura de sensibilidad.")


if __name__ == "__main__":
    main()
