"""Temporal BPR (T-BPR) con muestreo negativo dependiente del tiempo.

Implementa una variante de BPR-MF para escenarios repeat-aware.
Para cada interaccion positiva (u, i, t), el item negativo j se muestrea con
pesos distintos segun la ventana temporal de repeticion W_u = (p25, p75):

    - j repetido y dentro de W_u  -> peso alpha  (penalizacion suave)
    - j no visto previamente       -> peso 1.0
    - j repetido fuera de W_u      -> peso beta   (penalizacion fuerte)

La clase cumple la interfaz del proyecto: fit(train_df) / recommend(...).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd


@dataclass
class _Event:
    item_idx: int
    timestamp_ns: np.int64


class TemporalBPRRecommender:
    """Matrix Factorization con perdida BPR y muestreo negativo temporal."""

    def __init__(
        self,
        factors: int = 64,
        learning_rate: float = 0.05,
        reg: float = 1e-4,
        epochs: int = 8,
        alpha: float = 0.2,
        beta: float = 2.0,
        random_state: int = 42,
        top_pop_fallback: int = 200,
        recency_boost: float = 0.0,
        max_recent_boost: int = 100,
        temporal_activation_weight: float = 0.0,
        temporal_activation_decay: float = 0.6,
        hard_negative_ratio: float = 0.0,
        hard_negative_pool: int = 24,
        target_repeat_ratio: Optional[float] = 0.33,
        calibration_strength: float = 1.0,
        calibration_iterations: int = 16,
        verbose: bool = True,
    ) -> None:
        if not (0 < alpha < 1):
            raise ValueError("alpha debe cumplir 0 < alpha < 1")
        if not (beta > 1):
            raise ValueError("beta debe cumplir beta > 1")
        self.factors = factors
        self.learning_rate = learning_rate
        self.reg = reg
        self.epochs = epochs
        self.alpha = alpha
        self.beta = beta
        self.random_state = random_state
        self.top_pop_fallback = top_pop_fallback
        self.recency_boost = recency_boost
        self.max_recent_boost = max_recent_boost
        self.temporal_activation_weight = temporal_activation_weight
        self.temporal_activation_decay = temporal_activation_decay
        self.hard_negative_ratio = hard_negative_ratio
        self.hard_negative_pool = hard_negative_pool
        self.target_repeat_ratio = target_repeat_ratio
        self.calibration_strength = calibration_strength
        self.calibration_iterations = calibration_iterations
        self.verbose = verbose

        if not (0.0 <= self.temporal_activation_weight <= 5.0):
            raise ValueError("temporal_activation_weight debe estar entre 0 y 5")
        if not (0.1 <= self.temporal_activation_decay <= 1.5):
            raise ValueError("temporal_activation_decay debe estar entre 0.1 y 1.5")
        if not (0.0 <= self.hard_negative_ratio <= 1.0):
            raise ValueError("hard_negative_ratio debe estar entre 0 y 1")
        if self.hard_negative_pool < 4:
            raise ValueError("hard_negative_pool debe ser al menos 4")
        if self.target_repeat_ratio is not None and not (
            0.0 <= self.target_repeat_ratio <= 1.0
        ):
            raise ValueError("target_repeat_ratio debe estar entre 0 y 1 o ser None")
        if not (0.0 <= self.calibration_strength <= 3.0):
            raise ValueError("calibration_strength debe estar entre 0 y 3")
        if self.calibration_iterations < 1:
            raise ValueError("calibration_iterations debe ser al menos 1")

        self._rng = np.random.default_rng(random_state)

        self._user_to_idx: Dict[str, int] = {}
        self._idx_to_user: List[str] = []
        self._item_to_idx: Dict[str, int] = {}
        self._idx_to_item: List[str] = []

        self._P: Optional[np.ndarray] = None
        self._Q: Optional[np.ndarray] = None
        self._item_bias: Optional[np.ndarray] = None

        self._user_items_all: Dict[int, Set[int]] = {}
        self._user_windows_ns: Dict[int, Tuple[np.int64, np.int64]] = {}
        self._pop_items_idx: List[int] = []
        self._user_events: Dict[int, List[_Event]] = {}
        self._user_recent_unique: Dict[int, List[int]] = {}
        self._user_item_times_ns: Dict[int, Dict[int, List[np.int64]]] = {}
        self._user_last_ts_ns: Dict[int, np.int64] = {}

    def fit(self, train_df: pd.DataFrame) -> None:
        """Entrena T-BPR sobre `train_df`.

        train_df debe contener columnas: user_id, item_id, timestamp.
        """
        needed = {"user_id", "item_id", "timestamp"}
        if not needed.issubset(train_df.columns):
            missing = sorted(needed - set(train_df.columns))
            raise ValueError(f"Faltan columnas requeridas: {missing}")

        df = train_df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values(["user_id", "timestamp"], kind="mergesort").reset_index(
            drop=True
        )

        self._build_indexers(df)
        self._build_user_windows(df)
        self._build_popularity(df)

        self._build_events(df)
        self._init_params(
            n_users=len(self._idx_to_user), n_items=len(self._idx_to_item)
        )

        for epoch in range(1, self.epochs + 1):
            avg_loss = self._train_one_epoch()
            if self.verbose:
                print(
                    f"  [T-BPR] epoch {epoch:>2}/{self.epochs}  avg_bpr_loss={avg_loss:.5f}",
                    flush=True,
                )

    def recommend(self, user_id: str, user_history: Set[str], k: int = 10) -> List[str]:
        """Devuelve top-k items sin excluir historial (protocolo repeat-aware)."""
        if k <= 0:
            return []

        if self._Q is None or self._P is None or self._item_bias is None:
            return []

        u_idx = self._user_to_idx.get(user_id)
        if u_idx is None:
            return [self._idx_to_item[i] for i in self._pop_items_idx[:k]]

        scores = self._P[u_idx] @ self._Q.T + self._item_bias

        # Mezcla ligera con recencia para capturar patrones de corto plazo.
        if self.recency_boost > 0:
            recent = self._user_recent_unique.get(u_idx, [])
            for pos, item_idx in enumerate(recent[: self.max_recent_boost]):
                scores[item_idx] += self.recency_boost / (1.0 + pos)

        # Activacion temporal tipo memoria para items ya observados por el usuario.
        if self.temporal_activation_weight > 0:
            user_times = self._user_item_times_ns.get(u_idx, {})
            cutoff_ns = self._user_last_ts_ns.get(u_idx)
            if cutoff_ns is not None:
                for item_idx, times in user_times.items():
                    scores[
                        item_idx
                    ] += self.temporal_activation_weight * self._activation_score(
                        cutoff_ns, times
                    )

        if self.target_repeat_ratio is not None and self.calibration_strength > 0:
            seen_items = self._user_items_all.get(u_idx, set())
            if seen_items:
                seen_mask = np.zeros(len(scores), dtype=bool)
                seen_idx = np.fromiter(seen_items, dtype=np.int64)
                seen_mask[seen_idx] = True
                scores = self._calibrate_scores_for_target_repeat_ratio(
                    scores=scores,
                    seen_mask=seen_mask,
                    k=k,
                    target=self.target_repeat_ratio,
                )

        if k >= len(scores):
            top_idx = np.argsort(-scores)
        else:
            cand = np.argpartition(-scores, kth=k - 1)[:k]
            top_idx = cand[np.argsort(-scores[cand])]
        return [self._idx_to_item[i] for i in top_idx.tolist()]

    def _calibrate_scores_for_target_repeat_ratio(
        self,
        scores: np.ndarray,
        seen_mask: np.ndarray,
        k: int,
        target: float,
    ) -> np.ndarray:
        """Ajusta scores con un offset para aproximar repeat_ratio objetivo en top-k."""
        if k <= 0:
            return scores

        n_items = len(scores)
        if n_items == 0:
            return scores

        k_eff = min(k, n_items)
        if seen_mask.sum() == 0 or seen_mask.sum() == n_items:
            return scores

        centered_target = (
            target * self.calibration_strength + (1.0 - self.calibration_strength) * 0.5
        )
        centered_target = float(np.clip(centered_target, 0.0, 1.0))

        scores_adj = scores.copy()
        score_scale = float(np.std(scores_adj))
        base = max(1.0, score_scale * 8.0)
        lo, hi = -base, base

        for _ in range(self.calibration_iterations):
            mid = 0.5 * (lo + hi)
            tmp_scores = scores_adj.copy()
            tmp_scores[~seen_mask] += mid

            if k_eff >= len(tmp_scores):
                top_idx = np.argsort(-tmp_scores)
            else:
                cand = np.argpartition(-tmp_scores, kth=k_eff - 1)[:k_eff]
                top_idx = cand[np.argsort(-tmp_scores[cand])]

            observed_ratio = float(seen_mask[top_idx].mean())
            if observed_ratio > centered_target:
                lo = mid
            else:
                hi = mid

        shift = 0.5 * (lo + hi)
        scores_adj[~seen_mask] += shift
        return scores_adj

    def _build_indexers(self, df: pd.DataFrame) -> None:
        users = df["user_id"].drop_duplicates().tolist()
        items = df["item_id"].drop_duplicates().tolist()

        self._idx_to_user = users
        self._idx_to_item = items
        self._user_to_idx = {u: i for i, u in enumerate(users)}
        self._item_to_idx = {it: i for i, it in enumerate(items)}

    def _build_events(self, df: pd.DataFrame) -> None:
        self._user_items_all = {}
        self._user_events = {}
        self._user_recent_unique = {}
        self._user_item_times_ns = {}
        self._user_last_ts_ns = {}
        for row in df.itertuples(index=False):
            u_idx = self._user_to_idx[row.user_id]
            i_idx = self._item_to_idx[row.item_id]
            ts_ns = np.int64(row.timestamp.value)
            self._user_events.setdefault(u_idx, []).append(
                _Event(item_idx=i_idx, timestamp_ns=ts_ns)
            )
            self._user_items_all.setdefault(u_idx, set()).add(i_idx)
            self._user_item_times_ns.setdefault(u_idx, {}).setdefault(i_idx, []).append(
                ts_ns
            )
            self._user_last_ts_ns[u_idx] = ts_ns

        for u_idx, events in self._user_events.items():
            seen: Set[int] = set()
            recent_unique: List[int] = []
            for ev in reversed(events):
                if ev.item_idx not in seen:
                    seen.add(ev.item_idx)
                    recent_unique.append(ev.item_idx)
            self._user_recent_unique[u_idx] = recent_unique

    def _build_popularity(self, df: pd.DataFrame) -> None:
        vc = df["item_id"].value_counts()
        self._pop_items_idx = [self._item_to_idx[it] for it in vc.index.tolist()]

    def _build_user_windows(self, df: pd.DataFrame) -> None:
        """Construye W_u = (p25, p75) en nanosegundos para cada usuario."""
        self._user_windows_ns = {}
        for user_id, group in df.groupby("user_id", observed=True, sort=False):
            g = group.sort_values("timestamp", kind="mergesort")
            g = g[["item_id", "timestamp"]].copy()
            g["prev_ts"] = g.groupby("item_id", observed=True)["timestamp"].shift(1)
            delta_h = (
                (g["timestamp"] - g["prev_ts"]).dt.total_seconds().div(3600.0).dropna()
            )

            if len(delta_h) == 0:
                a_h, b_h = 24.0, 24.0 * 7.0
            else:
                a_h = float(np.percentile(delta_h, 25))
                b_h = float(np.percentile(delta_h, 75))
                if b_h < a_h:
                    a_h, b_h = b_h, a_h

            u_idx = self._user_to_idx[user_id]
            self._user_windows_ns[u_idx] = (
                np.int64(a_h * 3600 * 1e9),
                np.int64(b_h * 3600 * 1e9),
            )

    def _init_params(self, n_users: int, n_items: int) -> None:
        scale = 0.01
        self._P = self._rng.normal(0.0, scale, size=(n_users, self.factors)).astype(
            np.float64
        )
        self._Q = self._rng.normal(0.0, scale, size=(n_items, self.factors)).astype(
            np.float64
        )
        self._item_bias = np.zeros(n_items, dtype=np.float64)

    def _train_one_epoch(self) -> float:
        assert (
            self._P is not None and self._Q is not None and self._item_bias is not None
        )

        n_items = self._Q.shape[0]
        total_loss = 0.0
        n_updates = 0

        users = np.array(list(self._user_events.keys()), dtype=np.int64)
        self._rng.shuffle(users)

        for u in users.tolist():
            events = self._user_events.get(u, [])
            past_seen: Set[int] = set()
            last_seen_ts_ns: Dict[int, np.int64] = {}

            for ev in events:
                i = ev.item_idx
                t_ns = ev.timestamp_ns

                j = self._sample_negative(
                    user_idx=u,
                    pos_item_idx=i,
                    timestamp_ns=t_ns,
                    user_past_seen=past_seen,
                    user_last_seen_ts_ns=last_seen_ts_ns,
                    n_items=n_items,
                )
                if j is not None:
                    self._sgd_step(u, i, j)
                    total_loss += self._bpr_loss(u, i, j)
                    n_updates += 1

                past_seen.add(i)
                last_seen_ts_ns[i] = t_ns

        if n_updates == 0:
            return 0.0
        return total_loss / n_updates

    def _sample_negative(
        self,
        user_idx: int,
        pos_item_idx: int,
        timestamp_ns: np.int64,
        user_past_seen: Set[int],
        user_last_seen_ts_ns: Dict[int, np.int64],
        n_items: int,
    ) -> Optional[int]:
        """Muestrea j via rechazo ponderado por clase temporal.

        Proposal: uniforme sobre items != i.
        Aceptacion: w_j / beta, con w_j en {alpha, 1, beta}.
        """
        if n_items <= 1:
            return None

        window = self._user_windows_ns.get(
            user_idx,
            (np.int64(24 * 3600 * 1e9), np.int64(7 * 24 * 3600 * 1e9)),
        )

        if (
            self.hard_negative_ratio > 0
            and self._rng.random() < self.hard_negative_ratio
        ):
            hard = self._sample_hard_negative(
                user_idx=user_idx,
                pos_item_idx=pos_item_idx,
                timestamp_ns=timestamp_ns,
                user_past_seen=user_past_seen,
                user_last_seen_ts_ns=user_last_seen_ts_ns,
                n_items=n_items,
                window=window,
            )
            if hard is not None:
                return hard

        for _ in range(64):
            j = int(self._rng.integers(0, n_items))
            if j == pos_item_idx:
                continue
            weight = self._negative_weight(
                j, timestamp_ns, user_past_seen, user_last_seen_ts_ns, window
            )
            if self._rng.random() < (weight / self.beta):
                return j

        # Fallback determinista para evitar loops patologicos.
        return self._random_item_excluding({pos_item_idx}, n_items)

    def _sample_hard_negative(
        self,
        user_idx: int,
        pos_item_idx: int,
        timestamp_ns: np.int64,
        user_past_seen: Set[int],
        user_last_seen_ts_ns: Dict[int, np.int64],
        n_items: int,
        window: Tuple[np.int64, np.int64],
    ) -> Optional[int]:
        if n_items <= 1:
            return None

        a_ns, b_ns = window
        in_window: List[int] = []
        out_window: List[int] = []
        for item_idx in user_past_seen:
            if item_idx == pos_item_idx:
                continue
            last_ts = user_last_seen_ts_ns.get(item_idx)
            if last_ts is None:
                continue
            delta = timestamp_ns - last_ts
            if a_ns <= delta <= b_ns:
                in_window.append(item_idx)
            else:
                out_window.append(item_idx)

        per_group = max(2, self.hard_negative_pool // 3)
        candidates: List[int] = []
        candidates.extend(self._sample_list(in_window, per_group))
        candidates.extend(self._sample_list(out_window, per_group))

        unseen_needed = self.hard_negative_pool - len(candidates)
        if unseen_needed > 0:
            forbidden = set(user_past_seen)
            forbidden.add(pos_item_idx)
            candidates.extend(
                self._sample_unseen(
                    forbidden=forbidden, n_items=n_items, n_samples=unseen_needed
                )
            )

        if not candidates:
            return None

        uniq: List[int] = []
        seen: Set[int] = set()
        for c in candidates:
            if c != pos_item_idx and c not in seen:
                uniq.append(c)
                seen.add(c)

        if not uniq:
            return None

        # Hard negative: item no observado con mayor score actual del modelo.
        best_j = max(uniq, key=lambda j: self._score(user_idx, j))
        return int(best_j)

    def _negative_weight(
        self,
        item_idx: int,
        timestamp_ns: np.int64,
        user_past_seen: Set[int],
        user_last_seen_ts_ns: Dict[int, np.int64],
        window: Tuple[np.int64, np.int64],
    ) -> float:
        if item_idx not in user_past_seen:
            return 1.0
        last_ts = user_last_seen_ts_ns.get(item_idx)
        if last_ts is None:
            return 1.0
        a_ns, b_ns = window
        delta_ns = np.int64(timestamp_ns - last_ts)
        if a_ns <= delta_ns <= b_ns:
            return self.alpha
        return self.beta

    def _sample_list(self, items: List[int], n_samples: int) -> List[int]:
        if not items or n_samples <= 0:
            return []
        if len(items) <= n_samples:
            return items.copy()
        idx = self._rng.choice(len(items), size=n_samples, replace=False)
        return [items[int(i)] for i in idx.tolist()]

    def _sample_unseen(
        self, forbidden: Set[int], n_items: int, n_samples: int
    ) -> List[int]:
        out: List[int] = []
        used: Set[int] = set()
        max_tries = max(20, n_samples * 15)
        for _ in range(max_tries):
            if len(out) >= n_samples:
                break
            j = int(self._rng.integers(0, n_items))
            if j in forbidden or j in used:
                continue
            out.append(j)
            used.add(j)
        return out

    def _activation_score(self, cutoff_ns: np.int64, times_ns: List[np.int64]) -> float:
        total = 0.0
        eps_h = 1.0 / 3600.0
        for t_ns in times_ns:
            delta_h = max(float(cutoff_ns - t_ns) / 3_600_000_000_000.0, eps_h)
            total += delta_h ** (-self.temporal_activation_decay)
        # Log para estabilizar magnitud y evitar dominar el score MF.
        return math.log1p(total)

    def _random_item_excluding(
        self, forbidden: Set[int], n_items: int
    ) -> Optional[int]:
        if len(forbidden) >= n_items:
            return None
        for _ in range(50):
            j = int(self._rng.integers(0, n_items))
            if j not in forbidden:
                return j
        allowed = np.setdiff1d(
            np.arange(n_items, dtype=np.int64),
            np.array(list(forbidden), dtype=np.int64),
        )
        if len(allowed) == 0:
            return None
        return int(allowed[self._rng.integers(0, len(allowed))])

    def _score(self, u: int, i: int) -> float:
        assert (
            self._P is not None and self._Q is not None and self._item_bias is not None
        )
        return float(self._P[u] @ self._Q[i] + self._item_bias[i])

    def _bpr_loss(self, u: int, i: int, j: int) -> float:
        x_uij = self._score(u, i) - self._score(u, j)
        # softplus(-x) = log(1 + exp(-x))
        if x_uij > 0:
            return math.log1p(math.exp(-x_uij))
        return -x_uij + math.log1p(math.exp(x_uij))

    def _sgd_step(self, u: int, i: int, j: int) -> None:
        assert (
            self._P is not None and self._Q is not None and self._item_bias is not None
        )

        pu = self._P[u]
        qi = self._Q[i]
        qj = self._Q[j]

        x_uij = float(pu @ (qi - qj) + self._item_bias[i] - self._item_bias[j])
        grad = 1.0 / (1.0 + math.exp(x_uij))

        pu_old = pu.copy()
        qi_old = qi.copy()
        qj_old = qj.copy()

        self._P[u] += self.learning_rate * (
            grad * (qi_old - qj_old) - self.reg * pu_old
        )
        self._Q[i] += self.learning_rate * (grad * pu_old - self.reg * qi_old)
        self._Q[j] += self.learning_rate * (-grad * pu_old - self.reg * qj_old)

        self._item_bias[i] += self.learning_rate * (
            grad - self.reg * self._item_bias[i]
        )
        self._item_bias[j] += self.learning_rate * (
            -grad - self.reg * self._item_bias[j]
        )
