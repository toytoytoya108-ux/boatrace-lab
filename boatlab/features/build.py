"""特徴量セット v1（docs/04 §2）。

build_features(target_races, hist) は「対象レース群」と「履歴」を受け取り、艇（entry）単位の特徴量を返す。
当日予想でもバックテストでも同じ関数を使う。履歴側は race_date < 対象日 の行だけが効く（asof.py）。
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from boatlab.features.asof import AsofTables, attach_asof, cumulative_table, perf_log, shrink
from boatlab.features.history import HistoryFrames

FEATURE_SET_VERSION = "fs1"

RACE_TYPE_MAP = {"予選": "yosen", "一般": "ippan", "準優勝戦": "junyu", "優勝戦": "yusho", "特別選抜": "tokusen",
                 "選抜": "senbatsu", "特選": "tokusen"}
KLASS_ORD = {"A1": 4, "A2": 3, "B1": 2, "B2": 1}
KEY_FIELDS_FOR_COMPLETENESS = ["exhibition_time", "st_exh", "course_pred", "wind_speed_m", "nat_win_rate", "motor_rate2", "avg_st"]
SHRINK_K = 25.0


def _race_type(s) -> str:
    if s is None or (isinstance(s, float) and np.isnan(s)) or not isinstance(s, str) or not s:
        return "unknown"
    for k, v in RACE_TYPE_MAP.items():
        if k in s:
            return v
    return "other"


def base_frame(target: HistoryFrames) -> pd.DataFrame:
    """対象レースの entries ⨝ races ⨝ previews ⨝ conditions（結果は含めない）。"""
    r = target.races.rename(columns={"id": "race_id"})
    x = target.entries.merge(r[["race_id", "race_date", "stadium_code", "race_no", "closed_at", "grade", "race_type",
                                "distance_m", "day_no"]], on="race_id", how="inner")
    if len(target.previews):
        x = x.merge(target.previews.drop(columns=["source"], errors="ignore"), on=["race_id", "lane"], how="left")
    else:
        for c in ["course", "st_exh", "exhibition_time", "tilt", "pv_weight", "weight_adj", "fetched_at"]:
            x[c] = np.nan
    if len(target.conditions):
        x = x.merge(target.conditions.drop(columns=["observed_at"], errors="ignore"), on="race_id", how="left")
    else:
        for c in ["weather", "temp_c", "water_temp_c", "wind_dir", "wind_speed_m", "wave_cm"]:
            x[c] = np.nan
    x["race_date"] = pd.to_datetime(x["race_date"])
    x = x.rename(columns={"course": "course_exh"})
    # 予想進入コース：スタート展示の進入。無ければ艇番（docs/04 §2）
    x["course_pred"] = x["course_exh"].fillna(x["lane"]).astype(int)
    x["race_type_cat"] = x["race_type"].map(_race_type)
    x["klass_ord"] = x["klass"].map(KLASS_ORD)
    x["month"] = x["race_date"].dt.month
    return x


def add_asof_features(x: pd.DataFrame, hist: HistoryFrames | AsofTables) -> pd.DataFrame:
    tables = hist if isinstance(hist, AsofTables) else AsofTables(perf_log(hist.races, hist.entries, hist.result_entries))
    x = x.copy()
    x["course"] = x["course_pred"].astype("Int64")  # as-of 結合用のキー（予想進入コース）
    x = attach_asof(x, tables.course, ["course"], "g_course")
    x = attach_asof(x, tables.racer, ["regno"], "r_all")
    x = attach_asof(x, tables.racer, ["regno"], "r_1y", window_days=365)
    x = attach_asof(x, tables.racer, ["regno"], "r_30d", window_days=30)
    x = attach_asof(x, tables.rc, ["regno", "course"], "rc_2y", window_days=730)
    x = attach_asof(x, tables.rs, ["regno", "stadium_code"], "rs_all")
    x = attach_asof(x, tables.rsc, ["regno", "stadium_code", "course"], "rsc_all")
    x = attach_asof(x, tables.sc, ["stadium_code", "course"], "sc_1y", window_days=365)
    xm = x
    xm["motor_no"] = xm["motor_no"].astype("Int64")
    x = attach_asof(xm, tables.ms, ["stadium_code", "motor_no"], "ms_90d", window_days=90)

    # 縮約（少サンプルの暴れを抑える）。事前分布 = コース別全体1着率（as-of）
    prior_win = x["g_course_win_rate"].fillna(0.17)
    prior_top2 = x["g_course_top2_rate"].fillna(0.33)
    prior_top3 = x["g_course_top3_rate"].fillna(0.5)
    for p in ["rc_2y", "rsc_all", "rs_all", "ms_90d"]:
        x[f"{p}_win_shr"] = shrink(x[f"{p}_win_rate"], x[f"{p}_n"], prior_win, SHRINK_K)
        x[f"{p}_top2_shr"] = shrink(x[f"{p}_top2_rate"], x[f"{p}_n"], prior_top2, SHRINK_K)
        x[f"{p}_top3_shr"] = shrink(x[f"{p}_top3_rate"], x[f"{p}_n"], prior_top3, SHRINK_K)
    return x.drop(columns=["course"])


def add_relative_features(x: pd.DataFrame) -> pd.DataFrame:
    g = x.groupby("race_id")
    for c in ["nat_win_rate", "motor_rate2", "rc_2y_win_shr", "r_1y_win_rate"]:
        x[f"{c}_rank"] = g[c].rank(ascending=False, method="min")
        x[f"{c}_dmean"] = x[c] - g[c].transform("mean")
    for c in ["avg_st", "exhibition_time", "st_exh"]:
        x[f"{c}_rank"] = g[c].rank(ascending=True, method="min")
        x[f"{c}_dmin"] = x[c] - g[c].transform("min")
    x["exh_time_dmean"] = x["exhibition_time"] - g["exhibition_time"].transform("mean")
    # 1号艇との差（本命度）
    lane1 = x[x["lane"] == 1].set_index("race_id")["nat_win_rate"]
    x["nat_win_vs_lane1"] = x["nat_win_rate"] - x["race_id"].map(lane1)
    return x


def completeness(x: pd.DataFrame) -> pd.Series:
    return x[KEY_FIELDS_FOR_COMPLETENESS].notna().mean(axis=1)


def build_features(target: HistoryFrames, hist: HistoryFrames) -> pd.DataFrame:
    """対象レース群の艇別特徴量。hist は target の各レース日より前の実績のみが効く。"""
    x = base_frame(target)
    x = add_asof_features(x, hist)
    x = add_relative_features(x)
    x["completeness"] = completeness(x).groupby(x["race_id"]).transform("mean")
    return x


def attach_labels(x: pd.DataFrame, result_entries: pd.DataFrame) -> pd.DataFrame:
    y = result_entries[["race_id", "lane", "finish_pos", "course", "st", "abnormal"]].rename(
        columns={"course": "course_actual", "st": "st_actual"})
    x = x.merge(y, on=["race_id", "lane"], how="left")
    x["y_win"] = (x["finish_pos"] == 1).astype(float)
    x["y_top2"] = (x["finish_pos"] <= 2).astype(float)
    x["y_top3"] = (x["finish_pos"] <= 3).astype(float)
    return x


NUMERIC_FEATURES = [
    "lane", "course_pred", "klass_ord", "age", "weight", "f_count", "l_count", "avg_st",
    "nat_win_rate", "nat_rate2", "nat_rate3", "loc_win_rate", "loc_rate2", "loc_rate3",
    "motor_rate2", "motor_rate3", "boat_rate2", "boat_rate3",
    "exhibition_time", "st_exh", "tilt", "weight_adj",
    "temp_c", "water_temp_c", "wind_speed_m", "wave_cm",
    "race_no", "day_no", "distance_m", "month",
    "g_course_win_rate", "g_course_top2_rate", "g_course_top3_rate",
    "r_all_n", "r_all_win_rate", "r_all_top2_rate", "r_all_top3_rate", "r_all_avg_st",
    "r_1y_n", "r_1y_win_rate", "r_1y_top2_rate", "r_1y_top3_rate", "r_1y_avg_st", "r_1y_avg_fin",
    "r_30d_n", "r_30d_avg_fin", "r_30d_win_rate",
    "rc_2y_n", "rc_2y_win_shr", "rc_2y_top2_shr", "rc_2y_top3_shr", "rc_2y_avg_st",
    "rs_all_n", "rs_all_win_shr", "rs_all_top2_shr",
    "rsc_all_n", "rsc_all_win_shr", "rsc_all_top2_shr",
    "sc_1y_win_rate", "sc_1y_top2_rate", "sc_1y_top3_rate",
    "ms_90d_n", "ms_90d_win_shr", "ms_90d_top2_shr",
    "nat_win_rate_rank", "nat_win_rate_dmean", "motor_rate2_rank", "motor_rate2_dmean",
    "rc_2y_win_shr_rank", "rc_2y_win_shr_dmean", "r_1y_win_rate_rank", "r_1y_win_rate_dmean",
    "avg_st_rank", "avg_st_dmin", "exhibition_time_rank", "exhibition_time_dmin", "exh_time_dmean",
    "st_exh_rank", "st_exh_dmin", "nat_win_vs_lane1",
]
CATEGORICAL_FEATURES = ["stadium_code", "grade", "race_type_cat", "weather", "wind_dir"]


def feature_matrix(x: pd.DataFrame) -> pd.DataFrame:
    m = x[NUMERIC_FEATURES].astype("float32").copy()
    for c in CATEGORICAL_FEATURES:
        m[c] = x[c].astype("category")
    return m


def feature_hash(row: pd.Series) -> str:
    payload = json.dumps({k: (None if pd.isna(v) else (float(v) if isinstance(v, (int, float, np.floating)) else str(v)))
                          for k, v in row.items()}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]
