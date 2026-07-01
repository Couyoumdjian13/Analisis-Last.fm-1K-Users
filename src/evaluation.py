"""Pipeline de evaluacion bajo protocolo temporal leave-one-out (LOO).

Para cada usuario se reserva su ultima interaccion cronologica como item
relevante (ground truth) y el resto del historial se usa como entrenamiento.
El recomendador propone una lista top-K sobre el catalogo COMPLETO (sin
excluir el historial), permitiendo que los modelos repeat-aware sugieran
items ya consumidos.
"""

from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd

from .utils import ndcg_at_k, recall_at_k, reciprocal_rank, repeat_ratio


def temporal_loo_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Devuelve (train, test): test = ultima interaccion por usuario."""
    df_sorted = df.sort_values(["user_id", "timestamp"], kind="mergesort")
    last_idx = df_sorted.groupby("user_id", observed=True).tail(1).index
    test_df = df_sorted.loc[last_idx]
    train_df = df_sorted.drop(last_idx)
    return train_df, test_df


def evaluate_recommender(
    rec,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    k: int = 10,
    return_per_user: bool = False,
):
    """Ajusta el recomendador en train y reporta metricas promedio en test.

    Si return_per_user=True, devuelve una tupla (metricas_agregadas, df_por_usuario)
    donde df_por_usuario contiene (user_id, ground_truth, rank, hit) — util para
    auditar en que usuarios el modelo acierta y en que posicion.
    """
    rec.fit(train_df)

    user_histories = (
        train_df.groupby("user_id", observed=True)["item_id"].apply(set).to_dict()
    )

    n = len(test_df)
    sums = {"recall@10": 0.0, "ndcg@10": 0.0, "mrr": 0.0, "repeat_ratio": 0.0}
    hits = 0
    per_user_rows = []

    for row in test_df.itertuples(index=False):
        u = row.user_id
        gt = row.item_id
        hist = user_histories.get(u, set())
        ranked = rec.recommend(u, hist, k=k)

        r = recall_at_k(ranked, gt, k)
        sums["recall@10"] += r
        sums["ndcg@10"] += ndcg_at_k(ranked, gt, k)
        sums["mrr"] += reciprocal_rank(ranked, gt)
        sums["repeat_ratio"] += repeat_ratio(ranked, hist, k)
        hits += int(r > 0)

        if return_per_user:
            rank = next((i + 1 for i, it in enumerate(ranked[:k]) if it == gt), None)
            per_user_rows.append(
                {
                    "user_id": u,
                    "ground_truth": gt,
                    "rank": rank if rank is not None else 0,
                    "hit": int(r > 0),
                }
            )

    out = {m: sums[m] / n for m in sums}
    out["hits"] = hits
    out["n_users"] = n

    if return_per_user:
        return out, pd.DataFrame(per_user_rows)
    return out
