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


def load_ext_frames(d0: date, d1: date):
    """fs2 拡張用の追加フレーム（決まり手・展示タイム・風）。軽量列のみ。"""
    eng = get_engine()
    par = {"d0": str(d0), "d1": str(d1)}
    results = pd.read_sql_query(text("""
        SELECT res.race_id, res.kimarite FROM results res JOIN races r ON r.id = res.race_id
        WHERE r.race_date BETWEEN :d0 AND :d1"""), eng, params=par)
    previews = pd.read_sql_query(text("""
        SELECT p.race_id, p.lane, p.exhibition_time
        FROM preview_snapshots p JOIN races r ON r.id = p.race_id
        WHERE r.race_date BETWEEN :d0 AND :d1 AND p.exhibition_time IS NOT NULL
          AND (r.closed_at IS NULL OR p.fetched_at <= r.closed_at)"""), eng, params=par)
    conditions = pd.read_sql_query(text("""
        SELECT c.race_id, c.wind_dir, c.wind_speed_m
        FROM race_conditions c JOIN races r ON r.id = c.race_id
        WHERE r.race_date BETWEEN :d0 AND :d1 AND c.phase='preview'"""), eng, params=par)
    conditions = conditions.drop_duplicates("race_id")
    return results, previews, conditions
