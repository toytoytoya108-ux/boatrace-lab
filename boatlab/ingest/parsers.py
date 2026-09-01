"""JSON → 正規化レコード。

2つの書式を扱う:
- v3 flat: BoatraceOpenAPI/{programs,previews,results} v3（{"programs":[...]} など、2018〜）
- v1 nested: turnmark/api v1 と BoatraceOpenAPI/api v1（programs.stadiums.{s}.races.{r} 配下に racers/preview/odds/result）

時刻はすべて JST naive datetime。
過去一括取込の previews / odds は取得時刻が不明なため、fetched_at/captured_at に締切時刻を入れ、
source に '_hist' / '_final' を付けて「事後取込」であることを明示する（docs/03 §5、docs/04 §15）。
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

from boatlab.config import GRADE, PLACE_CODE, RACER_CLASS, TECHNIQUE, WEATHER, WIND_DIR
from boatlab.ingest.records import (
    ConditionRec, DayBundle, EntryRec, OddsRec, PreviewRec, RaceRec, ResultEntryRec, ResultRec, make_race_id,
)

BET_KEYS = {
    "trifecta": "3t", "trio": "3f", "exacta": "2t", "quinella": "2f",
    "quinella_place": "wide", "win": "win", "place": "place",
}


def _dt(s: str | None) -> datetime | None:
    if not s:
        return None
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")


def _f(x) -> float | None:
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _i(x) -> int | None:
    if x is None:
        return None
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def _boats(o: dict) -> list[dict]:
    """boats は list（旧）と lane キーの dict（新）の両方がある。"""
    b = o.get("boats")
    if isinstance(b, dict):
        return [v if isinstance(v, dict) else {} for _, v in sorted(b.items(), key=lambda kv: int(kv[0]))]
    return list(b or [])


def _place(code) -> tuple[int | None, str | None]:
    code = _i(code)
    if code is None:
        return None, None
    if 1 <= code <= 6:
        return code, None
    return None, PLACE_CODE.get(code, str(code))


def _payouts_summary(payouts: dict[str, list[dict[str, Any]]]):
    """3連単の的中組み合わせ・払戻、異常（特払/不成立/票なし）を要約。"""
    tri = payouts.get("trifecta") or []
    trifecta = None
    payout = None
    irregular = False
    notes: list[str] = []
    for p in tri:
        if p.get("combination") and p.get("amount") is not None:
            if trifecta is None:  # 同着で複数ある場合は先頭を代表にし、全体は payouts に残す
                trifecta, payout = p["combination"], int(p["amount"])
            else:
                irregular = True
                notes.append("同着")
        elif p.get("label"):
            irregular = True
            notes.append(str(p["label"]))
        elif p.get("combination") and p.get("amount") is None:
            irregular = True
            notes.append("票なし")
    for bt, lst in payouts.items():
        for p in lst or []:
            if p.get("label"):
                irregular = True
                notes.append(f"{bt}:{p['label']}")
    cancelled = all(not (payouts.get(k) or []) for k in BET_KEYS) if payouts else True
    return trifecta, payout, irregular, ("; ".join(sorted(set(notes))) or None), cancelled


# ---------------------------------------------------------------- v3 flat
def parse_v3_day(d: date, programs: dict | None, previews: dict | None, results: dict | None,
                 source_prefix: str = "openapi_v3") -> DayBundle:
    b = DayBundle()
    closed: dict[int, datetime] = {}

    for p in (programs or {}).get("programs", []):
        rid = make_race_id(d, int(p["stadium_number"]), int(p["number"]))
        ca = _dt(p.get("closed_at"))
        if ca:
            closed[rid] = ca
        b.races.append(RaceRec(
            race_id=rid, race_date=d, stadium_code=int(p["stadium_number"]), race_no=int(p["number"]),
            closed_at=ca, grade=GRADE.get(_i(p.get("grade_number")), p.get("grade_label")),
            title=p.get("title"), race_type=p.get("subtitle"), distance_m=_i(p.get("distance")),
            day_no=_day_no(p.get("day_label")), source=source_prefix,
        ))
        for bt in _boats(p):
            b.entries.append(EntryRec(
                race_id=rid, lane=int(bt["racer_boat_number"]), regno=_i(bt.get("racer_number")),
                name=bt.get("racer_name"), age=_i(bt.get("racer_age")), weight=_f(bt.get("racer_weight")),
                branch=_i(bt.get("racer_branch_number")), birthplace=_i(bt.get("racer_birthplace_number")),
                klass=RACER_CLASS.get(_i(bt.get("racer_class_number"))),
                f_count=_i(bt.get("racer_flying_count")), l_count=_i(bt.get("racer_late_count")),
                avg_st=_f(bt.get("racer_average_start_timing")),
                nat_win_rate=_f(bt.get("racer_national_top_1_percent")),
                nat_rate2=_f(bt.get("racer_national_top_2_percent")),
                nat_rate3=_f(bt.get("racer_national_top_3_percent")),
                loc_win_rate=_f(bt.get("racer_local_top_1_percent")),
                loc_rate2=_f(bt.get("racer_local_top_2_percent")),
                loc_rate3=_f(bt.get("racer_local_top_3_percent")),
                motor_no=_i(bt.get("racer_assigned_motor_number")),
                motor_rate2=_f(bt.get("racer_assigned_motor_top_2_percent")),
                motor_rate3=_f(bt.get("racer_assigned_motor_top_3_percent")),
                boat_no=_i(bt.get("racer_assigned_boat_number")),
                boat_rate2=_f(bt.get("racer_assigned_boat_top_2_percent")),
                boat_rate3=_f(bt.get("racer_assigned_boat_top_3_percent")),
                source=source_prefix,
            ))

    for pv in (previews or {}).get("previews", []):
        rid = make_race_id(d, int(pv["stadium_number"]), int(pv["number"]))
        ts = closed.get(rid) or datetime.combine(d, datetime.min.time()) + timedelta(hours=23, minutes=59)
        src = f"{source_prefix}_hist"
        b.conditions.append(_cond(rid, src, ts, "preview", pv))
        for bt in _boats(pv):
            b.previews.append(PreviewRec(
                race_id=rid, lane=int(bt["racer_boat_number"]), fetched_at=ts, source=src,
                course=_i(bt.get("racer_course_number")), st_exh=_f(bt.get("racer_start_timing")),
                weight=_f(bt.get("racer_weight")), weight_adj=_f(bt.get("racer_weight_adjustment")),
                exhibition_time=_f(bt.get("racer_exhibition_time")), tilt=_f(bt.get("racer_tilt_adjustment")),
            ))

    for rs in (results or {}).get("results", []):
        rid = make_race_id(d, int(rs["stadium_number"]), int(rs["number"]))
        ts = (closed.get(rid) or datetime.combine(d, datetime.min.time()) + timedelta(hours=23, minutes=59))
        _add_result(b, rid, rs, _boats(rs), key_boat="racer_boat_number", key_regno="racer_number",
                    key_course="racer_course_number", key_st="racer_start_timing", key_place="racer_place_number",
                    payouts=rs.get("payouts") or {}, refunds=None, remarks=None,
                    source=f"{source_prefix}_hist", fetched_at=ts + timedelta(minutes=30))
    return b


def _day_no(label: str | None) -> int | None:
    if not label:
        return None
    if "初日" in label:
        return 1
    digits = "".join(ch for ch in label if ch.isdigit())
    z = label.translate(str.maketrans("０１２３４５６７８９", "0123456789"))
    digits = "".join(ch for ch in z if ch.isdigit())
    return int(digits) if digits else None


def _cond(rid: int, src: str, ts: datetime, phase: str, o: dict) -> ConditionRec:
    return ConditionRec(
        race_id=rid, source=src, observed_at=ts, phase=phase,
        weather=WEATHER.get(_i(o.get("weather_number"))), temp_c=_f(o.get("air_temperature")),
        water_temp_c=_f(o.get("water_temperature")), wind_dir=WIND_DIR.get(_i(o.get("wind_direction_number"))),
        wind_speed_m=_f(o.get("wind_speed")), wave_cm=_f(o.get("wave_height")),
    )


def _add_result(b: DayBundle, rid: int, rs: dict, boats, *, key_boat, key_regno, key_course, key_st, key_place,
                payouts, refunds, remarks, source, fetched_at, pending_ok: bool = False):
    trifecta, payout, irregular, note, cancelled = _payouts_summary(payouts)
    if pending_ok and cancelled:
        # 当日ライブ取得では「result ブロックはあるが払戻が空」＝未確定レース。
        # 着順が1つも無ければ結果として扱わない（中止判定は翌日の turnmark 照合で行う）。
        if not any(_place(bt.get(key_place))[0] for bt in boats):
            return
    derived_refunds: list[int] = []
    for bt in boats:
        lane = int(bt[key_boat])
        pos, abn = _place(bt.get(key_place))
        if abn in ("F", "L", "欠"):
            derived_refunds.append(lane)
        b.result_entries.append(ResultEntryRec(
            race_id=rid, lane=lane, regno=_i(bt.get(key_regno)), finish_pos=pos,
            course=_i(bt.get(key_course)), st=_f(bt.get(key_st)), abnormal=abn,
        ))
    ref = list(refunds) if refunds is not None else derived_refunds
    if ref and not irregular:
        irregular = True
        note = (note + "; " if note else "") + "返還艇あり"
    if remarks and not note:
        note = remarks
    b.results.append(ResultRec(
        race_id=rid, trifecta=trifecta, trifecta_payout=payout,
        kimarite=TECHNIQUE.get(_i(rs.get("technique_number"))), payouts=payouts, refunds=ref,
        is_irregular=irregular, irregular_note=note, is_cancelled=cancelled, source=source, fetched_at=fetched_at,
    ))
    b.conditions.append(_cond(rid, source, fetched_at, "result", rs))


# ---------------------------------------------------------------- v1 nested
def _flatten_odds(bt: str, node) -> dict[str, Any]:
    """turnmark の入れ子オッズを {'1-2-3': 5.6} 形式にする。"""
    out: dict[str, Any] = {}
    sep = "=" if bt in ("3f", "2f", "wide") else "-"

    def rec(n, prefix: list[str]):
        if isinstance(n, dict) and ("lower_limit" in n or "upper_limit" in n):
            out[sep.join(prefix)] = {"lo": _f(n.get("lower_limit")), "hi": _f(n.get("upper_limit"))}
            return
        if isinstance(n, dict):
            for k, v in n.items():
                rec(v, prefix + [str(k)])
            return
        out[sep.join(prefix)] = _f(n)

    rec(node, [])
    return out


def parse_v1_day(d: date, doc: dict, source_prefix: str, fetched_at: datetime | None = None,
                 odds_source: str | None = None) -> DayBundle:
    """turnmark/api v1 および BoatraceOpenAPI/api v1。

    fetched_at: 当日取得の場合はその時刻。None なら過去一括（締切時刻を代入し、source に _hist を付ける）。
    """
    b = DayBundle()
    hist = fetched_at is None
    stadiums = ((doc or {}).get("programs") or {}).get("stadiums") or {}
    for s_key, s in stadiums.items():
        for r_key, r in (s.get("races") or {}).items():
            stadium = int(r.get("stadium_number") or s_key)
            rno = int(r.get("race_number") or r_key)
            # race_id は「ファイル内の日付」を使う。today.json が前日分のまま（未更新）でも
            # 前日の結果を当日の race_id に書いてしまわないため（実地テストで検出したバグの修正）
            rd = d
            if r.get("date"):
                try:
                    rd = date.fromisoformat(str(r["date"]))
                except ValueError:
                    pass
            rid = make_race_id(rd, stadium, rno)
            ca = _dt(r.get("closed_at"))
            b.races.append(RaceRec(
                race_id=rid, race_date=rd, stadium_code=stadium, race_no=rno, closed_at=ca,
                grade=GRADE.get(_i(r.get("grade_number"))), title=r.get("title"), race_type=r.get("subtitle"),
                distance_m=_i(r.get("distance")), day_no=_i(r.get("day_number")), source=source_prefix,
            ))
            for lane_key, rc in (r.get("racers") or {}).items():
                b.entries.append(EntryRec(
                    race_id=rid, lane=int(rc.get("entry_number") or lane_key), regno=_i(rc.get("number")),
                    name=rc.get("name"), age=_i(rc.get("age")), weight=_f(rc.get("weight")),
                    branch=_i(rc.get("branch_number")), birthplace=_i(rc.get("birthplace_number")),
                    klass=RACER_CLASS.get(_i(rc.get("rank_number"))),
                    f_count=_i(rc.get("flying_count")), l_count=_i(rc.get("late_count")),
                    avg_st=_f(rc.get("average_start_timing")),
                    nat_win_rate=_f(rc.get("national_win_rate")), nat_rate2=_f(rc.get("national_top_2_percent")),
                    nat_rate3=_f(rc.get("national_top_3_percent")), loc_win_rate=_f(rc.get("local_win_rate")),
                    loc_rate2=_f(rc.get("local_top_2_percent")), loc_rate3=_f(rc.get("local_top_3_percent")),
                    motor_no=_i(rc.get("motor_number")), motor_rate2=_f(rc.get("motor_top_2_percent")),
                    motor_rate3=_f(rc.get("motor_top_3_percent")), boat_no=_i(rc.get("boat_number")),
                    boat_rate2=_f(rc.get("boat_top_2_percent")), boat_rate3=_f(rc.get("boat_top_3_percent")),
                    source=source_prefix,
                ))
            ts_pre = fetched_at or ca or datetime.combine(rd, datetime.min.time()) + timedelta(hours=23, minutes=59)
            pv = r.get("preview")
            if pv and pv.get("racers"):
                src = f"{source_prefix}_hist" if hist else source_prefix
                b.conditions.append(_cond(rid, src, ts_pre, "preview", pv))
                for lane_key, rc in pv["racers"].items():
                    b.previews.append(PreviewRec(
                        race_id=rid, lane=int(rc.get("entry_number") or lane_key), fetched_at=ts_pre, source=src,
                        course=_i(rc.get("course_number")), st_exh=_f(rc.get("start_timing")),
                        weight=_f(rc.get("weight")), weight_adj=_f(rc.get("weight_adjustment")),
                        exhibition_time=_f(rc.get("exhibition_time")), tilt=_f(rc.get("tilt_adjustment")),
                        propeller=rc.get("propeller"), parts=rc.get("parts"),
                    ))
            od = r.get("odds")
            if od:
                osrc = odds_source or (f"{source_prefix}_final" if hist else source_prefix)
                for k, bt in BET_KEYS.items():
                    if od.get(k):
                        flat = _flatten_odds(bt, od[k])
                        if any(v is not None for v in flat.values()):
                            b.odds.append(OddsRec(race_id=rid, bet_type=bt, captured_at=ts_pre, source=osrc, odds=flat))
            rs = r.get("result")
            if rs and rs.get("racers"):
                boats = [dict(v, _lane=int(v.get("entry_number") or k)) for k, v in rs["racers"].items()]
                _add_result(b, rid, rs, boats, key_boat="_lane", key_regno="number", key_course="course_number",
                            key_st="start_timing", key_place="place_number", payouts=rs.get("payouts") or {},
                            refunds=rs.get("refunds"), remarks=rs.get("remarks"),
                            source=f"{source_prefix}_hist" if hist else source_prefix,
                            fetched_at=fetched_at or ts_pre + timedelta(minutes=30), pending_ok=not hist)
    return b
