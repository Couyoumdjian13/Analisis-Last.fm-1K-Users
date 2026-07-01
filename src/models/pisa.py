"""PISA-inspired recommender for repeat-aware recommendation.

This is a lightweight implementation inspired by the RecSys'24 PISA idea:
- Temporal memory activation (ACT-R style decay over past occurrences).
- Short-term session context via recent items.
- Popularity prior as exploration signal.

The class follows the project interface: fit(train_df) / recommend(...).
"""

from __future__ import annotations

from collections import defaultdict
from typing import DefaultDict, Dict, List, Set, Tuple

import numpy as np
import pandas as pd


class PISARecommender:
    """Temporal-memory recommender inspired by PISA."""

    def __init__(
        self,
        decay: float = 0.6,
        w_activation: float = 0.65,
        w_context: float = 0.25,
        w_popularity: float = 0.10,
        context_window: int = 5,
        top_pop_fallback: int = 400,
    ) -> None:
        if not (0.1 <= decay <= 1.5):
            raise ValueError("decay debe estar entre 0.1 y 1.5")
        self.decay = decay
        self.w_activation = w_activation
        self.w_context = w_context
        self.w_popularity = w_popularity
        self.context_window = context_window
        self.top_pop_fallback = top_pop_fallback

        self._all_items: List[str] = []
        self._pop_scores: Dict[str, float] = {}
        self._top_items: List[str] = []

        self._user_item_times_h: Dict[str, Dict[str, List[float]]] = {}
        self._user_recent_unique: Dict[str, List[str]] = {}
        self._user_last_t_h: Dict[str, float] = {}

        self._item_context_strength: Dict[str, Dict[str, float]] = {}

    def fit(self, train_df: pd.DataFrame) -> None:
        needed = {"user_id", "item_id", "timestamp"}
        if not needed.issubset(train_df.columns):
            missing = sorted(needed - set(train_df.columns))
            raise ValueError(f"Faltan columnas requeridas: {missing}")

        df = train_df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=False)
        df = df.sort_values(["user_id", "timestamp"], kind="mergesort").reset_index(
            drop=True
        )

        self._all_items = df["item_id"].drop_duplicates().tolist()

        pop = df["item_id"].value_counts()
        if len(pop) > 0:
            max_pop = float(pop.iloc[0])
            self._pop_scores = {it: float(cnt) / max_pop for it, cnt in pop.items()}
            self._top_items = pop.index.tolist()
        else:
            self._pop_scores = {}
            self._top_items = []

        self._build_user_memories(df)
        self._build_context_strength(df)

    def recommend(self, user_id: str, user_history: Set[str], k: int = 10) -> List[str]:
        if k <= 0:
            return []
        if not self._all_items:
            return []

        if user_id not in self._user_item_times_h:
            return self._top_items[:k]

        repeat_candidates = list(self._user_item_times_h[user_id].keys())
        explore_candidates = self._top_items[: self.top_pop_fallback]

        candidates: List[str] = []
        seen: Set[str] = set()
        for it in repeat_candidates + explore_candidates:
            if it not in seen:
                candidates.append(it)
                seen.add(it)

        if not candidates:
            return self._top_items[:k]

        cutoff_h = self._user_last_t_h[user_id]
        recent_ctx = self._user_recent_unique.get(user_id, [])[: self.context_window]

        scored: List[Tuple[str, float]] = []
        for item in candidates:
            act = self._activation_score(user_id, item, cutoff_h)
            ctx = self._context_score(item, recent_ctx)
            pop = self._pop_scores.get(item, 0.0)
            score = (
                self.w_activation * act + self.w_context * ctx + self.w_popularity * pop
            )
            scored.append((item, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [it for it, _ in scored[:k]]

    def _build_user_memories(self, df: pd.DataFrame) -> None:
        self._user_item_times_h = {}
        self._user_recent_unique = {}
        self._user_last_t_h = {}

        epoch = pd.Timestamp("1970-01-01")

        for user_id, group in df.groupby("user_id", observed=True, sort=False):
            g = group.sort_values("timestamp", kind="mergesort")
            item_times: DefaultDict[str, List[float]] = defaultdict(list)
            for row in g.itertuples(index=False):
                t_hours = (row.timestamp - epoch).total_seconds() / 3600.0
                item_times[row.item_id].append(float(t_hours))

            self._user_item_times_h[user_id] = dict(item_times)
            if len(g) > 0:
                last_t = (g.iloc[-1]["timestamp"] - epoch).total_seconds() / 3600.0
                self._user_last_t_h[user_id] = float(last_t)
            else:
                self._user_last_t_h[user_id] = 0.0

            recent_unique: List[str] = []
            seen: Set[str] = set()
            for item in reversed(g["item_id"].tolist()):
                if item not in seen:
                    seen.add(item)
                    recent_unique.append(item)
            self._user_recent_unique[user_id] = recent_unique

    def _build_context_strength(self, df: pd.DataFrame) -> None:
        """Build item-item directional strengths from adjacent interactions."""
        pair_counts: DefaultDict[str, DefaultDict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )

        for _, group in df.groupby("user_id", observed=True, sort=False):
            items = group.sort_values("timestamp", kind="mergesort")["item_id"].tolist()
            for prev_item, next_item in zip(items[:-1], items[1:]):
                if prev_item != next_item:
                    pair_counts[prev_item][next_item] += 1

        self._item_context_strength = {}
        for src_item, targets in pair_counts.items():
            if not targets:
                continue
            total = float(sum(targets.values()))
            self._item_context_strength[src_item] = {
                tgt_item: cnt / total for tgt_item, cnt in targets.items()
            }

    def _activation_score(self, user_id: str, item: str, cutoff_h: float) -> float:
        times = self._user_item_times_h.get(user_id, {}).get(item, [])
        if not times:
            return 0.0

        # ACT-R style base-level activation approximation.
        score = 0.0
        eps = 1.0 / 3600.0
        for t in times:
            delta = max(cutoff_h - t, eps)
            score += delta ** (-self.decay)
        return float(score)

    def _context_score(self, item: str, recent_ctx: List[str]) -> float:
        if not recent_ctx:
            return 0.0

        score = 0.0
        for pos, ctx_item in enumerate(recent_ctx):
            weight = 1.0 / (1.0 + pos)
            score += weight * self._item_context_strength.get(ctx_item, {}).get(
                item, 0.0
            )
        return float(score)
