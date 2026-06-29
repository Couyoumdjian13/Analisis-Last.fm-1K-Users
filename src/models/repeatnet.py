"""RepeatNet-inspired recommender.

This is a lightweight approximation of RepeatNet ideas:
- Explicit repeat vs explore branches.
- User-dependent gate controlling branch mixture.
- Next-item transition signal for exploration.

The class follows the project interface: fit(train_df) / recommend(...).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import DefaultDict, Dict, List, Set, Tuple

import pandas as pd


class RepeatNetRecommender:
    """Gated repeat/explore recommender inspired by RepeatNet."""

    def __init__(
        self,
        w_repeat_recency: float = 0.55,
        w_repeat_freq: float = 0.45,
        w_explore_transition: float = 0.70,
        w_explore_pop: float = 0.30,
        top_pop_fallback: int = 400,
    ) -> None:
        self.w_repeat_recency = w_repeat_recency
        self.w_repeat_freq = w_repeat_freq
        self.w_explore_transition = w_explore_transition
        self.w_explore_pop = w_explore_pop
        self.top_pop_fallback = top_pop_fallback

        self._user_counts: Dict[str, Counter] = {}
        self._user_recent_unique: Dict[str, List[str]] = {}
        self._user_repeat_prob: Dict[str, float] = {}
        self._user_last_item: Dict[str, str] = {}

        self._global_pop: Dict[str, float] = {}
        self._top_items: List[str] = []
        self._transitions: Dict[str, Dict[str, float]] = {}

    def fit(self, train_df: pd.DataFrame) -> None:
        needed = {"user_id", "item_id", "timestamp"}
        if not needed.issubset(train_df.columns):
            missing = sorted(needed - set(train_df.columns))
            raise ValueError(f"Faltan columnas requeridas: {missing}")

        df = train_df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=False)
        df = df.sort_values(["user_id", "timestamp"], kind="mergesort").reset_index(drop=True)

        pop = df["item_id"].value_counts()
        self._top_items = pop.index.tolist()
        if len(pop) > 0:
            max_pop = float(pop.iloc[0])
            self._global_pop = {it: float(cnt) / max_pop for it, cnt in pop.items()}
        else:
            self._global_pop = {}

        self._build_user_stats(df)
        self._build_transitions(df)

    def recommend(self, user_id: str, user_history: Set[str], k: int = 10) -> List[str]:
        if k <= 0:
            return []
        if not self._top_items:
            return []

        if user_id not in self._user_counts:
            return self._top_items[:k]

        repeat_scores = self._repeat_branch_scores(user_id)
        explore_scores = self._explore_branch_scores(user_id)

        gate = self._user_repeat_prob.get(user_id, 0.5)
        candidates: List[str] = []
        seen: Set[str] = set()

        for it in repeat_scores.keys():
            if it not in seen:
                candidates.append(it)
                seen.add(it)
        for it in explore_scores.keys():
            if it not in seen:
                candidates.append(it)
                seen.add(it)
        for it in self._top_items[: self.top_pop_fallback]:
            if it not in seen:
                candidates.append(it)
                seen.add(it)

        scored: List[Tuple[str, float]] = []
        for item in candidates:
            r = repeat_scores.get(item, 0.0)
            e = explore_scores.get(item, self._global_pop.get(item, 0.0))
            final_score = gate * r + (1.0 - gate) * e
            scored.append((item, final_score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [it for it, _ in scored[:k]]

    def _build_user_stats(self, df: pd.DataFrame) -> None:
        self._user_counts = {}
        self._user_recent_unique = {}
        self._user_repeat_prob = {}
        self._user_last_item = {}

        for user_id, group in df.groupby("user_id", observed=True, sort=False):
            g = group.sort_values("timestamp", kind="mergesort")
            items = g["item_id"].tolist()
            if not items:
                continue

            counts = Counter(items)
            self._user_counts[user_id] = counts
            self._user_last_item[user_id] = items[-1]

            seen: Set[str] = set()
            recent_unique: List[str] = []
            for it in reversed(items):
                if it not in seen:
                    seen.add(it)
                    recent_unique.append(it)
            self._user_recent_unique[user_id] = recent_unique

            repeats = sum(max(c - 1, 0) for c in counts.values())
            denom = max(len(items) - 1, 1)
            repeat_prob = repeats / denom
            self._user_repeat_prob[user_id] = float(min(max(repeat_prob, 0.05), 0.95))

    def _build_transitions(self, df: pd.DataFrame) -> None:
        trans_counts: DefaultDict[str, DefaultDict[str, int]] = defaultdict(lambda: defaultdict(int))

        for _, group in df.groupby("user_id", observed=True, sort=False):
            items = group.sort_values("timestamp", kind="mergesort")["item_id"].tolist()
            for prev_item, next_item in zip(items[:-1], items[1:]):
                if prev_item != next_item:
                    trans_counts[prev_item][next_item] += 1

        self._transitions = {}
        for src_item, targets in trans_counts.items():
            total = float(sum(targets.values()))
            if total <= 0:
                continue
            self._transitions[src_item] = {tgt: cnt / total for tgt, cnt in targets.items()}

    def _repeat_branch_scores(self, user_id: str) -> Dict[str, float]:
        counts = self._user_counts.get(user_id, Counter())
        recent_unique = self._user_recent_unique.get(user_id, [])
        if not counts:
            return {}

        max_count = float(max(counts.values())) if counts else 1.0
        recency_rank = {it: pos for pos, it in enumerate(recent_unique)}

        out: Dict[str, float] = {}
        for item, cnt in counts.items():
            freq = float(cnt) / max_count
            pos = recency_rank.get(item, len(recent_unique))
            rec = 1.0 / (1.0 + pos)
            out[item] = self.w_repeat_recency * rec + self.w_repeat_freq * freq
        return out

    def _explore_branch_scores(self, user_id: str) -> Dict[str, float]:
        out: Dict[str, float] = {}
        last_item = self._user_last_item.get(user_id)

        if last_item is not None:
            for item, p in self._transitions.get(last_item, {}).items():
                out[item] = self.w_explore_transition * p + self.w_explore_pop * self._global_pop.get(item, 0.0)

        if not out:
            for item in self._top_items[: self.top_pop_fallback]:
                out[item] = self._global_pop.get(item, 0.0)

        return out
