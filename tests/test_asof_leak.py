"""as-of 特徴量に未来・同日の情報が混入しないことを検証する。"""
from datetime import date

import numpy as np
import pandas as pd

from boatlab.features.build import build_features
from boatlab.features.history import HistoryFrames


def _mk_hist(rows):
    """rows: (race_id, date, stadium, regno, lane, finish_pos)"""
    races = pd.DataFrame([{"id": r[0], "race_date": pd.Timestamp(r[1]), "stadium_code": r[2], "race_no": 1,
                           "closed_at": pd.Timestamp(r[1]) + pd.Timedelta(hours=12), "grade": "一般", "race_type": "予選",
                           "distance_m": 1800, "day_no": 1, "status": "finished"} for r in rows]).drop_duplicates("id")
    entries = pd.DataFrame([{"race_id": r[0], "lane": r[4], "regno": r[3], "motor_no": 10 + r[4], "boat_no": 1,
                             "klass": "A1", "age": 30, "weight": 52.0, "f_count": 0, "l_count": 0, "avg_st": 0.15,
                             "nat_win_rate": 6.0, "nat_rate2": 40.0, "nat_rate3": 55.0, "loc_win_rate": 6.0,
                             "loc_rate2": 40.0, "loc_rate3": 55.0, "motor_rate2": 35.0, "motor_rate3": 50.0,
                             "boat_rate2": 30.0, "boat_rate3": 45.0} for r in rows])
    res = pd.DataFrame([{"race_id": r[0], "lane": r[4], "regno": r[3], "finish_pos": r[5], "course": r[4],
                         "st": 0.1, "abnormal": None} for r in rows])
    empty_prev = pd.DataFrame(columns=["race_id", "lane", "course", "st_exh", "exhibition_time", "tilt", "pv_weight", "weight_adj", "fetched_at", "source"])
    empty_cond = pd.DataFrame(columns=["race_id", "weather", "temp_c", "water_temp_c", "wind_dir", "wind_speed_m", "wave_cm", "observed_at"])
    return HistoryFrames(races, entries, empty_prev, empty_cond, pd.DataFrame(), res)


def _target(rid, d, stadium):
    rows = [(rid, d, stadium, 100 + i, i, None) for i in range(1, 7)]
    h = _mk_hist(rows)
    return HistoryFrames(h.races, h.entries, h.previews, h.conditions, pd.DataFrame(), pd.DataFrame(columns=h.result_entries.columns))


def test_future_and_same_day_rows_do_not_change_features():
    past = []
    rid = 1
    for day in range(1, 20):
        for i in range(1, 7):
            past.append((rid, date(2020, 1, day), 1, 100 + i, i, (i % 6) + 1))
        rid += 1
    hist = _mk_hist(past)
    target = _target(999, date(2020, 1, 25), 1)
    f0 = build_features(target, hist)

    # 同日（2020-01-25）と未来（2020-02-xx）の行を注入：選手101が全勝
    extra = list(past)
    rid = 5000
    for day in [25, 26, 27, 28, 29]:
        m = 1 if day == 25 else 2
        for i in range(1, 7):
            extra.append((rid, date(2020, m, day), 1, 100 + i, i, 1 if i == 1 else i))
        rid += 1
    hist2 = _mk_hist(extra)
    f1 = build_features(target, hist2)
    cols = [c for c in f0.columns if c.startswith(("r_", "rc_", "rs_", "rsc_", "sc_", "ms_", "g_"))]
    pd.testing.assert_frame_equal(f0[cols].reset_index(drop=True), f1[cols].reset_index(drop=True))
    # 過去実績は効いている（ゼロではない）
    assert (f0["r_all_n"] > 0).all()
    # 前日までしか使わない：初日の対象は全ゼロ
    t0 = _target(998, date(2020, 1, 1), 1)
    fz = build_features(t0, hist)
    assert (fz["r_all_n"] == 0).all() and fz["r_all_win_rate"].isna().all()
