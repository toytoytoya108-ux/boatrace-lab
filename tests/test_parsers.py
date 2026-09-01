import json
from datetime import date
from pathlib import Path

from boatlab.ingest.parsers import parse_v1_day, parse_v3_day
from boatlab.ingest.records import make_race_id

FX = Path(__file__).parent / "fixtures"


def load(name):
    return json.load(open(FX / name, encoding="utf-8"))


def test_race_id_is_deterministic():
    assert make_race_id(date(2026, 5, 1), 24, 12) == 202605012412


def test_parse_v3_programs():
    b = parse_v3_day(date(2021, 9, 1), load("programs_20210901.json"), None, None)
    assert len(b.races) == 120 and len(b.entries) == 720
    r = b.races[0]
    assert r.stadium_code == 1 and r.race_no == 1 and r.closed_at is not None and r.grade == "一般"
    e = b.entries[0]
    assert e.regno == 4746 and e.klass == "B1" and e.nat_win_rate == 5.25 and e.motor_no == 36 and e.avg_st == 0.18


def test_parse_v3_previews_and_results():
    b = parse_v3_day(date(2023, 5, 1), None, load("previews_20230501.json"), None)
    assert len(b.previews) == 216 * 6
    p = b.previews[0]
    assert p.course == 1 and p.exhibition_time == 6.8 and p.tilt == -0.5 and p.source.endswith("_hist")
    c = [x for x in b.conditions if x.phase == "preview"][0]
    assert c.weather == "雨" and c.wind_dir == "無風" and c.water_temp_c == 17

    b = parse_v3_day(date(2024, 5, 1), None, None, load("results_20240501.json"))
    assert len(b.results) == 156
    r = b.results[0]
    assert r.trifecta == "6-1-4" and r.trifecta_payout == 10410 and r.kimarite == "抜き" and not r.is_cancelled
    re = [x for x in b.result_entries if x.race_id == r.race_id]
    assert sorted(x.finish_pos for x in re) == [1, 2, 3, 4, 5, 6]
    # 異常コードの検出（F/L/欠 → 返還）
    abn = [x for x in b.result_entries if x.abnormal]
    assert all(x.finish_pos is None for x in abn)


def test_parse_v1_turnmark_with_odds():
    b = parse_v1_day(date(2026, 5, 1), load("turnmark_20260501_s1.json"), source_prefix="turnmark")
    assert len(b.races) == 12 and len(b.entries) == 72
    odds3t = [o for o in b.odds if o.bet_type == "3t"]
    assert len(odds3t) == 12
    o = odds3t[0].odds
    assert len(o) == 120 and o["1-2-3"] == 37.8 and odds3t[0].source == "turnmark_final"
    wide = [o for o in b.odds if o.bet_type == "wide"][0].odds
    assert "1=2" in wide and set(wide["1=2"]) == {"lo", "hi"}
    r = b.results[0]
    assert r.trifecta == "4-3-1" and r.trifecta_payout == 3040
    pv = [p for p in b.previews if p.race_id == r.race_id]
    assert len(pv) == 6 and pv[0].st_exh == 0.18
