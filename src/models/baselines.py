"""Baselines repeat-aware con interfaz comun fit / recommend.

Diseño explicito: bajo el protocolo de evaluacion temporal leave-one-out,
NO se excluye el historial del usuario del espacio de candidatos. Excluirlo
sesga el experimento contra los modelos repeat-aware (un recomendador que
solo propone items del historial obtiene 0 si se filtran los items vistos).
"""

from __future__ import annotations

import random as _random
from collections import Counter
from typing import Dict, List, Sequence, Set

import pandas as pd


class RandomRecommender:
    """Muestreo uniforme sobre el catalogo completo."""

    def __init__(self, seed: int = 42) -> None:
        self._rng = _random.Random(seed)
        self._items: List[str] = []

    def fit(self, train_df: pd.DataFrame) -> None:
        self._items = train_df["item_id"].unique().tolist()

    def recommend(self, user_id: str, user_history: Set[str], k: int = 10) -> List[str]:
        if not self._items:
            return []
        k = min(k, len(self._items))
        return self._rng.sample(self._items, k=k)


class MostPopularRecommender:
    """Ranking global por numero de reproducciones."""

    def __init__(self) -> None:
        self._top_items: List[str] = []

    def fit(self, train_df: pd.DataFrame) -> None:
        self._top_items = train_df["item_id"].value_counts().index.tolist()

    def recommend(self, user_id: str, user_history: Set[str], k: int = 10) -> List[str]:
        return self._top_items[:k]


class SimpleRepeatRecommender:
    """Recomienda los items mas frecuentes del historial del usuario.

    Si el usuario tiene menos de K items unicos en su historial, se completa
    con los items globalmente mas populares (que no esten ya en la lista).
    """

    def __init__(self) -> None:
        self._user_counts: Dict[str, Counter] = {}
        self._top_items: List[str] = []

    def fit(self, train_df: pd.DataFrame) -> None:
        self._user_counts = (
            train_df.groupby("user_id", observed=True)["item_id"]
            .apply(lambda s: Counter(s))
            .to_dict()
        )
        self._top_items = train_df["item_id"].value_counts().index.tolist()

    def recommend(self, user_id: str, user_history: Set[str], k: int = 10) -> List[str]:
        counts = self._user_counts.get(user_id, Counter())
        ranked = [item for item, _ in counts.most_common(k)]
        if len(ranked) < k:
            seen = set(ranked)
            for it in self._top_items:
                if it not in seen:
                    ranked.append(it)
                    seen.add(it)
                    if len(ranked) >= k:
                        break
        return ranked


class SimpleRepeatRecencyRecommender:
    """Variante recency-aware del baseline SimpleRepeat.

    En lugar de rankear el historial por frecuencia, lo hace por recencia:
    devuelve los K ultimos items unicos consumidos por el usuario (de mas
    reciente a menos reciente). Es la operacionalizacion mas simple posible
    del fenomeno de localidad temporal observado en el EDA.
    """

    def __init__(self) -> None:
        self._user_recent: Dict[str, List[str]] = {}
        self._top_items: List[str] = []

    def fit(self, train_df: pd.DataFrame) -> None:
        df_sorted = train_df.sort_values(
            ["user_id", "timestamp"], ascending=[True, False], kind="mergesort"
        )
        self._user_recent = {}
        for u, group in df_sorted.groupby("user_id", observed=True, sort=False):
            seen: Set[str] = set()
            ordered: List[str] = []
            for it in group["item_id"]:
                if it not in seen:
                    seen.add(it)
                    ordered.append(it)
            self._user_recent[u] = ordered
        self._top_items = train_df["item_id"].value_counts().index.tolist()

    def recommend(self, user_id: str, user_history: Set[str], k: int = 10) -> List[str]:
        recent = self._user_recent.get(user_id, [])
        ranked = recent[:k]
        if len(ranked) < k:
            seen = set(ranked)
            for it in self._top_items:
                if it not in seen:
                    ranked.append(it)
                    seen.add(it)
                    if len(ranked) >= k:
                        break
        return ranked
