"""履歴フレームの読み込み。

当日予想もバックテストも、ここで作る同じ形の DataFrame から特徴量を作る。
previews は「締切時刻以前に取得した最新スナップショット」だけを採用する（as-of 原則）。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd
from sqlalchemy import text

from boatlab.store.db import get_engine


@dataclass
class HistoryFrames:
    races: pd.DataFrame          # id, race_date, stadium_code, race_no, closed_at, grade, race_type, distance_m, day_no, status
    entries: pd.DataFrame        # race_id, lane, regno, ... (番組表)
    previews: pd.DataFrame       # race_id, lane, course, st_exh, exhibition_time, tilt, weight_adj, fetched_at
    conditions: pd.DataFrame     # race_id, weather, temp_c, water_temp_c, wind_dir, wind_speed_m, wave_cm
    results: pd.DataFrame        # race_id, trifecta, trifecta_payout, kimarite, is_irregular, refunds
    result_entries: pd.DataFrame # race_id, lane, regno, finish_pos, course, st, abnormal


def _read(sql: str, params: dict | None = None) -> pd.DataFrame:
    return pd.read_sql_query(text(sql), get_engine(), params=params or {})


def load_history(d0: date | None = None, d1: date | None = None,
                 preview_sources: tuple[str, ...] = ("openapi_v3_hist", "turnmark_hist", "openapi_api", "official_web")) -> HistoryFrames:
    """[d0, d1] のレースを読み込む（None は無制限）。"""
    where = []
    params: dict = {}
    if d0:
        where.append("race_date >= :d0"); params["d0"] = str(d0)
    if d1:
        where.append("race_date <= :d1"); params["d1"] = str(d1)
    w = ("WHERE " + " AND ".join(where)) if where else ""
    races = _read(f"SELECT id, race_date, stadium_code, race_no, closed_at, grade, race_type, distance_m, day_no, status FROM races {w}", params)
    races["race_date"] = pd.to_datetime(races["race_date"])
    races["closed_at"] = pd.to_datetime(races["closed_at"])
    ids = "SELECT id FROM races " + w
    entries = _read(f"SELECT * FROM entries WHERE race_id IN ({ids})", params)
    # 締切以前に取得した最新の直前情報（レース×艇ごと）
    src_list = ",".join(f"'{s}'" for s in preview_sources)
    previews = _read(f"""
        SELECT p.race_id, p.lane, p.course, p.st_exh, p.exhibition_time, p.tilt, p.weight AS pv_weight,
               p.weight_adj, p.fetched_at, p.source
        FROM preview_snapshots p JOIN races r ON r.id = p.race_id
        WHERE p.race_id IN ({ids}) AND p.source IN ({src_list})
          AND (r.closed_at IS NULL OR p.fetched_at <= r.closed_at)
        ORDER BY p.race_id, p.lane, p.fetched_at""", params)
    if len(previews):
        previews = previews.groupby(["race_id", "lane"], as_index=False).tail(1)
    conditions = _read(f"""
        SELECT c.race_id, c.weather, c.temp_c, c.water_temp_c, c.wind_dir, c.wind_speed_m, c.wave_cm, c.observed_at
        FROM race_conditions c JOIN races r ON r.id = c.race_id
        WHERE c.race_id IN ({ids}) AND c.phase='preview' AND c.source IN ({src_list})
          AND (r.closed_at IS NULL OR c.observed_at <= r.closed_at)
        ORDER BY c.race_id, c.observed_at""", params)
    if len(conditions):
        conditions = conditions.groupby("race_id", as_index=False).tail(1)
    results = _read(f"SELECT race_id, trifecta, trifecta_payout, kimarite, is_irregular, refunds, payouts FROM results WHERE race_id IN ({ids})", params)
    result_entries = _read(f"SELECT race_id, lane, regno, finish_pos, course, st, abnormal FROM result_entries WHERE race_id IN ({ids})", params)
    return HistoryFrames(races, entries, previews, conditions, results, result_entries)
