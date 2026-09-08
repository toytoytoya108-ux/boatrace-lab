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


def _read(sql: str, params: dict | None = None, chunksize: int = 100_000) -> pd.DataFrame:
    # チャンク読みでピークメモリを抑える（一括だと行タプルの中間リストで最終フレームの数倍を使う）
    parts = list(pd.read_sql_query(text(sql), get_engine(), params=params or {}, chunksize=chunksize))
    if not parts:
        return pd.read_sql_query(text(sql + " LIMIT 0"), get_engine(), params=params or {})
    return pd.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0]


ENTRY_SLIM_COLS = ("race_id", "lane", "regno", "age", "weight", "klass", "f_count", "l_count", "avg_st",
                   "nat_win_rate", "nat_rate2", "nat_rate3", "loc_win_rate", "loc_rate2", "loc_rate3",
                   "motor_no", "motor_rate2", "motor_rate3", "boat_no", "boat_rate2", "boat_rate3", "is_absent")


def load_history(d0: date | None = None, d1: date | None = None,
                 preview_sources: tuple[str, ...] = ("openapi_v3_hist", "turnmark_hist", "openapi_api", "official_web"),
                 slim: bool = False) -> HistoryFrames:
    """[d0, d1] のレースを読み込む（None は無制限）。
    slim=True: 特徴量計算に使わない列（選手名・支部・節間成績・払戻JSON）を読まない。当日運用の3年履歴キャッシュ用
    （2GB サーバーでのメモリ対策。特徴量の値は変わらない）。"""
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
    ecols = ", ".join(ENTRY_SLIM_COLS) if slim else "*"
    entries = _read(f"SELECT {ecols} FROM entries WHERE race_id IN ({ids})", params)
    # 締切以前に取得した最新の直前情報（レース×艇ごと）
    src_list = ",".join(f"'{s}'" for s in preview_sources)
    # 当日運用では同じレースの直前情報が数分おきに追記されるため、最新1件だけを SQL 側で選ぶ
    # （全スナップショットを pandas に読むとメモリが数倍に膨らみ、2GB サーバーでは落ちる）
    previews = _read(f"""
        SELECT race_id, lane, course, st_exh, exhibition_time, tilt, pv_weight, weight_adj, parts, fetched_at, source FROM (
          SELECT p.race_id, p.lane, p.course, p.st_exh, p.exhibition_time, p.tilt, p.weight AS pv_weight,
                 p.weight_adj, p.parts, p.fetched_at, p.source,
                 ROW_NUMBER() OVER (PARTITION BY p.race_id, p.lane ORDER BY p.fetched_at DESC, p.id DESC) AS rn
          FROM preview_snapshots p JOIN races r ON r.id = p.race_id
          WHERE p.race_id IN ({ids}) AND p.source IN ({src_list})
            AND (r.closed_at IS NULL OR p.fetched_at <= r.closed_at)
        ) WHERE rn = 1 ORDER BY race_id, lane""", params)
    conditions = _read(f"""
        SELECT race_id, weather, temp_c, water_temp_c, wind_dir, wind_speed_m, wave_cm, observed_at FROM (
          SELECT c.race_id, c.weather, c.temp_c, c.water_temp_c, c.wind_dir, c.wind_speed_m, c.wave_cm, c.observed_at,
                 ROW_NUMBER() OVER (PARTITION BY c.race_id ORDER BY c.observed_at DESC, c.id DESC) AS rn
          FROM race_conditions c JOIN races r ON r.id = c.race_id
          WHERE c.race_id IN ({ids}) AND c.phase='preview' AND c.source IN ({src_list})
            AND (r.closed_at IS NULL OR c.observed_at <= r.closed_at)
        ) WHERE rn = 1 ORDER BY race_id""", params)
    rcols = "race_id, trifecta, trifecta_payout, kimarite, is_irregular, refunds" + ("" if slim else ", payouts")
    results = _read(f"SELECT {rcols} FROM results WHERE race_id IN ({ids})", params)
    result_entries = _read(f"SELECT race_id, lane, regno, finish_pos, course, st, abnormal FROM result_entries WHERE race_id IN ({ids})", params)
    return HistoryFrames(races, entries, previews, conditions, results, result_entries)
