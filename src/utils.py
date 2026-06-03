"""Metricas de evaluacion para Repeat-Aware Recommendation bajo LOO temporal.

En el protocolo temporal leave-one-out hay exactamente un item relevante
por usuario, por lo que las definiciones se simplifican:
    Recall@K = HitRate@K     (binario)
    nDCG@K   = 1/log2(rank+1) si hit, 0 si no
    MRR      = 1/rank         si hit, 0 si no
RepeatRatio mide la fraccion del top-K que pertenece al historial del
usuario, sirviendo como diagnostico del balance repeat/explore.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence, Set


def recall_at_k(ranked: Sequence[str], ground_truth: str, k: int = 10) -> float:
    return 1.0 if ground_truth in ranked[:k] else 0.0


def ndcg_at_k(ranked: Sequence[str], ground_truth: str, k: int = 10) -> float:
    for i, item in enumerate(ranked[:k]):
        if item == ground_truth:
            return 1.0 / math.log2(i + 2)
    return 0.0


def reciprocal_rank(ranked: Sequence[str], ground_truth: str) -> float:
    for i, item in enumerate(ranked):
        if item == ground_truth:
            return 1.0 / (i + 1)
    return 0.0


def repeat_ratio(ranked: Sequence[str], user_history: Set[str], k: int = 10) -> float:
    top = ranked[:k]
    if not top:
        return 0.0
    return sum(1 for it in top if it in user_history) / len(top)
