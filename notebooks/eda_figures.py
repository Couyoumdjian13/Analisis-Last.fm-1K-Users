"""Regenera las figuras del EDA referenciadas en docs/H2_Midterm.md.

Trabaja sobre el subset H1 (lastfm_100_users_h1_fixed.parquet) y produce
tres PNG en docs/figures/:
    fig_repeat_ratio.png       - distribucion de repeat ratio por usuario
    fig_repeat_intervals.png   - log(1+horas) entre repeticiones
    fig_plays_per_user.png     - total de plays por usuario

Tambien imprime el coeficiente de correlacion de Spearman entre la
frecuencia historica del item y su probabilidad empirica de ser repetido
(parte del OE1 del H1).
"""

from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(REPO_ROOT, "data")
FIG_DIR = os.path.join(REPO_ROOT, "docs", "figures")
SUBSET_PATH = os.path.join(DATA_DIR, "lastfm_100_users_h1_fixed.parquet")

os.makedirs(FIG_DIR, exist_ok=True)


def load_subset() -> pd.DataFrame:
    if not os.path.exists(SUBSET_PATH):
        print(
            f"ERROR: no encontre {SUBSET_PATH}. Corre antes notebooks/preprocessing.py"
        )
        sys.exit(1)
    df = pd.read_parquet(SUBSET_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["user_id", "timestamp"], kind="mergesort").reset_index(
        drop=True
    )
    return df


def per_user_stats(df: pd.DataFrame) -> pd.DataFrame:
    df["is_repeat"] = df.groupby("user_id", observed=True)["item_id"].transform(
        lambda x: x.duplicated(keep=False)
    )
    u = df.groupby("user_id", observed=True).agg(
        plays=("item_id", "count"),
        unique_items=("item_id", "nunique"),
        repeats=("is_repeat", "sum"),
    )
    u["repeat_ratio"] = u["repeats"] / u["plays"]
    return u


def interval_hours(df: pd.DataFrame) -> pd.Series:
    rep = df[df["is_repeat"]].copy()
    rep["prev_ts"] = rep.groupby(["user_id", "item_id"], observed=True)[
        "timestamp"
    ].shift(1)
    rep["interval_h"] = (rep["timestamp"] - rep["prev_ts"]).dt.total_seconds() / 3600.0
    return rep["interval_h"].dropna()


def figure_repeat_ratio(per_user: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.hist(per_user["repeat_ratio"], bins=25, color="#3498db", edgecolor="white")
    ax.axvline(
        per_user["repeat_ratio"].mean(),
        color="#e74c3c",
        linestyle="--",
        linewidth=1.2,
        label=f"media = {per_user['repeat_ratio'].mean():.3f}",
    )
    ax.set_xlabel("Repeat ratio del usuario")
    ax.set_ylabel("# de usuarios")
    ax.set_title("Distribucion de repeat ratio por usuario (n=100)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_repeat_ratio.png"), dpi=160)
    plt.close(fig)


def figure_intervals(interval_h: pd.Series) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    log_h = np.log1p(interval_h)
    ax.hist(log_h, bins=50, color="#2ecc71", edgecolor="white")
    p25 = np.log1p(1.74)
    p50 = np.log1p(27.04)
    ax.axvline(
        p25, color="#e74c3c", linestyle="--", linewidth=1.0, label="p25 ~ 1.74 h"
    )
    ax.axvline(
        p50, color="#8e44ad", linestyle="--", linewidth=1.0, label="mediana ~ 27 h"
    )
    ax.set_xlabel("log(1 + intervalo entre repeticiones [h])")
    ax.set_ylabel("# de repeticiones")
    ax.set_title(f"Intervalos entre repeticiones (n={len(interval_h):,})")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_repeat_intervals.png"), dpi=160)
    plt.close(fig)


def figure_plays(per_user: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.hist(per_user["unique_items"], bins=30, color="#f39c12", edgecolor="white")
    ax.axvline(
        per_user["unique_items"].median(),
        color="#c0392b",
        linestyle="--",
        linewidth=1.2,
        label=f"mediana = {int(per_user['unique_items'].median())}",
    )
    ax.set_xlabel("# de items unicos por usuario")
    ax.set_ylabel("# de usuarios")
    ax.set_title("Items unicos por usuario en el subset H1 (top 2000 plays / user)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_plays_per_user.png"), dpi=160)
    plt.close(fig)


def spearman_freq_vs_repeat(df: pd.DataFrame) -> tuple[float, float, int]:
    """Correlacion de Spearman entre frecuencia historica del item en
    la primera mitad cronologica del historial de cada usuario y la
    probabilidad de que ese item reaparezca en la segunda mitad
    (operacionalizacion del OE1 del H1).

    Setup:
        - Split temporal 50/50 por usuario (primera mitad cronologica = "pasado",
          segunda mitad = "futuro").
        - Para cada (user, item) con freq_past >= 1 calculamos:
              freq_past   = numero de plays en la primera mitad
              reappears   = 1 si el item aparece en la segunda mitad, 0 si no
        - Spearman pooled across users sobre todos los pares (user, item).
    """
    df = df.sort_values(["user_id", "timestamp"], kind="mergesort").reset_index(
        drop=True
    )
    df["rank"] = df.groupby("user_id", observed=True).cumcount()
    df["n"] = df.groupby("user_id", observed=True)["rank"].transform("count")
    df["half"] = np.where(df["rank"] < df["n"] / 2, "past", "future")

    past = (
        df[df["half"] == "past"]
        .groupby(["user_id", "item_id"], observed=True)
        .size()
        .reset_index(name="freq_past")
    )
    future_items = (
        df[df["half"] == "future"]
        .groupby(["user_id", "item_id"], observed=True)
        .size()
        .reset_index(name="freq_future")
    )
    merged = past.merge(future_items, on=["user_id", "item_id"], how="left")
    merged["reappears"] = (merged["freq_future"].fillna(0) > 0).astype(int)

    rho, p = spearmanr(merged["freq_past"], merged["reappears"])
    return float(rho), float(p), int(len(merged))


def main() -> None:
    df = load_subset()
    print(f"Cargado subset H1: {len(df):,} filas, {df['user_id'].nunique()} usuarios")

    per_user = per_user_stats(df)
    interval_h = interval_hours(df)

    figure_repeat_ratio(per_user)
    figure_intervals(interval_h)
    figure_plays(per_user)
    print(f"3 figuras escritas en {FIG_DIR}")

    rho, p, n = spearman_freq_vs_repeat(df)
    print(
        f"\nOE1 - correlacion Spearman(frecuencia historica, item_es_repetido):"
        f"\n  rho = {rho:.4f}  p-value = {p:.2e}  n = {n:,} pares (usuario, item)"
    )


if __name__ == "__main__":
    main()
