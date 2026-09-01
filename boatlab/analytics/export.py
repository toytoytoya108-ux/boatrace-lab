"""モデリング用 parquet 書き出し（DuckDB 経由で高速に）。"""
from __future__ import annotations

from pathlib import Path

import duckdb

from boatlab.config import DATABASE_URL

TABLES = ["races", "entries", "preview_snapshots", "race_conditions", "results", "result_entries",
          "odds_snapshots", "racers"]


def export_all(out_dir: str) -> dict[str, int]:
    assert DATABASE_URL.startswith("sqlite:///"), "export_parquet は SQLite 前提（Postgres は pg_dump を使う）"
    db_path = DATABASE_URL.replace("sqlite:///", "")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("INSTALL sqlite; LOAD sqlite;")
    con.execute(f"ATTACH '{db_path}' AS lab (TYPE SQLITE, READ_ONLY);")
    counts = {}
    for t in TABLES:
        target = out / f"{t}.parquet"
        con.execute(f"COPY (SELECT * FROM lab.{t}) TO '{target}' (FORMAT PARQUET, COMPRESSION ZSTD);")
        counts[t] = con.execute(f"SELECT COUNT(*) FROM lab.{t}").fetchone()[0]
    con.close()
    return counts
