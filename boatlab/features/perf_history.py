"""実績ログ用の軽量読込（perf_log に必要な列だけ）。メモリ対策。"""
from __future__ import annotations

from datetime import date

import pandas as pd
from sqlalchemy import text

from boatlab.store.db import get_engine


def load_perf_frames(d0: date, d1: date):
    eng = get_engine()
    races = pd.read_sql_query(text(
        "SELECT id, race_date, stadium_code FROM races WHERE race_date BETWEEN :d0 AND :d1"),
        eng, params={"d0": str(d0), "d1": str(d1)})
    races["race_date"] = pd.to_datetime(races["race_date"])
    result_entries = pd.read_sql_query(text("""
        SELECT x.race_id, x.lane, x.regno, x.finish_pos, x.course, x.st, x.abnormal
        FROM result_entries x JOIN races r ON r.id = x.race_id
        WHERE r.race_date BETWEEN :d0 AND :d1"""), eng, params={"d0": str(d0), "d1": str(d1)})
    entries = pd.read_sql_query(text("""
        SELECT e.race_id, e.lane, e.motor_no, e.boat_no
        FROM entries e JOIN races r ON r.id = e.race_id
        WHERE r.race_date BETWEEN :d0 AND :d1"""), eng, params={"d0": str(d0), "d1": str(d1)})
    return races, entries, result_entries
