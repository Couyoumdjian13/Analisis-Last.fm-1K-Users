#!/usr/bin/env python
# coding: utf-8
"""
Preprocesamiento del dataset Last.fm 1K Users.

Pipeline en dos pasadas, disenado para correr en maquinas con poca RAM
(< 8 GB) sobre un TSV de ~2.5 GB / 19M filas:

Pasada 1 (TSV -> parquet completo)
    - Lectura chunked con pandas; descarta artist_id/track_id (no se usan).
    - Construye item_id = "artist_name - track_name" para evitar los NaNs
      de MusicBrainz declarados en la hipotesis H1.
    - Escribe lastfm_1k_complete_fixed.parquet de forma incremental.
    - Acumula conteos por user_id para identificar el top-N.

Pasada 2 (parquet -> subset H1)
    - Relee el parquet con pushdown de filtro por top-N users.
    - Ordena cronologicamente y recorta a MAX_PLAYS_PER_USER por user.
    - Escribe lastfm_100_users_h1_fixed.parquet.
"""

import os
import time
from collections import Counter

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "data"))
DIRNAME = os.environ.get("LASTFM_1K_DIRNAME", DEFAULT_DATA_DIR)

INPUT_FILENAME = "userid-timestamp-artid-artname-traid-traname.tsv"
FULL_PARQUET_NAME = "lastfm_1k_complete_fixed.parquet"
SUBSET_PARQUET_NAME = "lastfm_100_users_h1_fixed.parquet"

N_USERS = 100
MAX_PLAYS_PER_USER = 2000
CHUNK_SIZE = 1_000_000

# Filas del TSV con campos corruptos (offset 1-indexed del archivo original).
CORRUPTED_ROWS = [
    2120260 - 1,
    2446318 - 1,
    11141081 - 1,
    11152099 - 1,
    11152402 - 1,
    11882087 - 1,
    12902539 - 1,
    12935044 - 1,
    17589539 - 1,
]

# Schema fijo del parquet de salida (necesario para escritura incremental).
PARQUET_SCHEMA = pa.schema(
    [
        pa.field("user_id", pa.string()),
        pa.field("timestamp", pa.timestamp("ns")),
        pa.field("item_id", pa.string()),
    ]
)


def iter_chunks(filepath: str):
    """Itera el TSV en chunks descartando columnas innecesarias."""
    return pd.read_csv(
        filepath,
        sep="\t",
        header=None,
        names=[
            "user_id",
            "timestamp",
            "artist_id",
            "artist_name",
            "track_id",
            "track_name",
        ],
        usecols=["user_id", "timestamp", "artist_name", "track_name"],
        dtype={"user_id": "string", "artist_name": "string", "track_name": "string"},
        skiprows=CORRUPTED_ROWS,
        chunksize=CHUNK_SIZE,
    )


def write_full_parquet(filepath: str, full_parquet_path: str) -> Counter:
    """Pasada 1: escribe el parquet completo y devuelve el Counter de user_id."""
    user_counts: Counter = Counter()
    writer = pq.ParquetWriter(full_parquet_path, PARQUET_SCHEMA, compression="snappy")
    total_rows = 0
    try:
        for i, chunk in enumerate(iter_chunks(filepath), start=1):
            chunk["timestamp"] = pd.to_datetime(chunk["timestamp"])
            chunk["item_id"] = (
                chunk["artist_name"].fillna("UNKNOWN").astype(str)
                + " - "
                + chunk["track_name"].fillna("UNKNOWN").astype(str)
            )
            user_counts.update(chunk["user_id"].value_counts().to_dict())

            out = chunk[["user_id", "timestamp", "item_id"]]
            table = pa.Table.from_pandas(
                out, schema=PARQUET_SCHEMA, preserve_index=False
            )
            writer.write_table(table)

            total_rows += len(chunk)
            print(
                f"  chunk {i:>3}: +{len(chunk):>9,} filas (acumulado: {total_rows:,})",
                flush=True,
            )
    finally:
        writer.close()

    print(f"Pasada 1 lista. Total filas escritas: {total_rows:,}")
    print(f"Usuarios unicos detectados: {len(user_counts):,}")
    return user_counts


def write_h1_subset(
    full_parquet_path: str, subset_parquet_path: str, top_users: list[str]
) -> int:
    """Pasada 2: lee el parquet con pushdown, recorta a tail-N por user."""
    table = pq.read_table(
        full_parquet_path, filters=[("user_id", "in", set(top_users))]
    )
    subset_df = table.to_pandas()
    subset_df.sort_values(["user_id", "timestamp"], inplace=True)
    subset_df = subset_df.groupby("user_id", group_keys=False).tail(MAX_PLAYS_PER_USER)

    subset_df[["user_id", "timestamp", "item_id"]].to_parquet(
        subset_parquet_path, compression="snappy", index=False
    )
    return len(subset_df)


def main() -> None:
    filepath = os.path.join(DIRNAME, INPUT_FILENAME)
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"No se encontro el TSV en {filepath}. "
            "Defini LASTFM_1K_DIRNAME apuntando al directorio que lo contiene."
        )

    full_parquet_path = os.path.join(DIRNAME, FULL_PARQUET_NAME)
    subset_parquet_path = os.path.join(DIRNAME, SUBSET_PARQUET_NAME)

    print(f"Leyendo {filepath}")
    print(f"  chunksize={CHUNK_SIZE:,} filas | dir salida={DIRNAME}", flush=True)

    t0 = time.perf_counter()
    print("\n=== Pasada 1: TSV -> parquet completo ===", flush=True)
    user_counts = write_full_parquet(filepath, full_parquet_path)
    print(f"Parquet completo: {full_parquet_path}")
    print(f"Tiempo pasada 1: {time.perf_counter() - t0:.1f}s\n")

    top_users = [uid for uid, _ in user_counts.most_common(N_USERS)]
    print(
        f"=== Pasada 2: subset H1 (top {N_USERS} usuarios, tail {MAX_PLAYS_PER_USER}) ===",
        flush=True,
    )
    t1 = time.perf_counter()
    n_subset = write_h1_subset(full_parquet_path, subset_parquet_path, top_users)
    print(f"Subset H1: {subset_parquet_path}")
    print(f"  Registros: {n_subset:,} (esperado ~{N_USERS * MAX_PLAYS_PER_USER:,})")
    print(f"Tiempo pasada 2: {time.perf_counter() - t1:.1f}s")
    print(f"\nTiempo total: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
