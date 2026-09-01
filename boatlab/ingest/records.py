"""正規化レコード（取得層 → 保存層の受け渡し形式）。

ここに無い項目はシステムに存在しない。推測値で埋めない（欠損は None）。
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Any

from pydantic import BaseModel


def make_race_id(d: date, stadium: int, race_no: int) -> int:
    """決定的な race_id: YYYYMMDD*10000 + 場コード*100 + R"""
    return int(d.strftime("%Y%m%d")) * 10000 + stadium * 100 + race_no


class RaceRec(BaseModel):
    race_id: int
    race_date: date
    stadium_code: int
    race_no: int
    closed_at: datetime | None = None  # 締切予定（JST, naive）
    grade: str | None = None            # SG/G1/G2/G3/一般
    title: str | None = None
    race_type: str | None = None        # 予選/準優勝戦/優勝戦 等（subtitle）
    distance_m: int | None = None
    day_no: int | None = None
    source: str


class EntryRec(BaseModel):
    race_id: int
    lane: int
    regno: int | None = None
    name: str | None = None
    age: int | None = None
    weight: float | None = None
    branch: int | None = None
    birthplace: int | None = None
    klass: str | None = None
    f_count: int | None = None
    l_count: int | None = None
    avg_st: float | None = None
    nat_win_rate: float | None = None
    nat_rate2: float | None = None
    nat_rate3: float | None = None
    loc_win_rate: float | None = None
    loc_rate2: float | None = None
    loc_rate3: float | None = None
    motor_no: int | None = None
    motor_rate2: float | None = None
    motor_rate3: float | None = None
    boat_no: int | None = None
    boat_rate2: float | None = None
    boat_rate3: float | None = None
    source: str


class PreviewRec(BaseModel):
    race_id: int
    lane: int
    fetched_at: datetime
    source: str
    course: int | None = None          # スタート展示の進入
    st_exh: float | None = None        # 展示ST
    weight: float | None = None
    weight_adj: float | None = None
    exhibition_time: float | None = None
    tilt: float | None = None
    propeller: str | None = None
    parts: list[dict[str, Any]] | None = None


class ConditionRec(BaseModel):
    race_id: int
    source: str
    observed_at: datetime
    phase: str                          # 'preview' / 'result'
    weather: str | None = None
    temp_c: float | None = None
    water_temp_c: float | None = None
    wind_dir: str | None = None
    wind_speed_m: float | None = None
    wave_cm: float | None = None


class ResultRec(BaseModel):
    race_id: int
    trifecta: str | None = None
    trifecta_payout: int | None = None
    kimarite: str | None = None
    payouts: dict[str, list[dict[str, Any]]] = {}
    refunds: list[int] = []
    is_irregular: bool = False
    irregular_note: str | None = None
    is_cancelled: bool = False
    source: str
    fetched_at: datetime


class ResultEntryRec(BaseModel):
    race_id: int
    lane: int
    regno: int | None = None
    finish_pos: int | None = None      # 1..6、失格等は None
    course: int | None = None
    st: float | None = None
    abnormal: str | None = None        # 妨/エ/転/落/沈/不/失/F/L/欠/他


class OddsRec(BaseModel):
    race_id: int
    bet_type: str                      # '3t','3f','2t','2f','wide','win','place'
    captured_at: datetime
    source: str
    odds: dict[str, Any]               # {'1-2-3': 5.6, ...}  拡連複/複勝は {'lo':..,'hi':..}


class DayBundle(BaseModel):
    """1日分の正規化結果。"""
    races: list[RaceRec] = []
    entries: list[EntryRec] = []
    previews: list[PreviewRec] = []
    conditions: list[ConditionRec] = []
    results: list[ResultRec] = []
    result_entries: list[ResultEntryRec] = []
    odds: list[OddsRec] = []
