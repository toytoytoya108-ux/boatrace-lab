"""特徴量セット v1（docs/04 §2）。

build_features(target_races, hist) は「対象レース群」と「履歴」を受け取り、艇（entry）単位の特徴量を返す。
当日予想でもバックテストでも同じ関数を使う。履歴側は race_date < 対象日 の行だけが効く（asof.py）。
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from boatlab.features.asof import AsofTables, attach_asof, attach_form, cumulative_table, perf_log, perf_log_ext, shrink, wind_bucket
from boatlab.features.history import HistoryFrames

FEATURE_SET_VERSION = "fs2"

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
    # 部品交換（直前情報。2026年〜のみ値がある）
    if "parts" in x.columns:
        def _pn(v):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return np.nan
            try:
                lst = json.loads(v) if isinstance(v, str) else v
                return float(len(lst)) if isinstance(lst, list) else np.nan
            except Exception:
                return np.nan
        x["parts_n"] = x["parts"].map(_pn)
        major = ("ピストン", "リング", "シリンダ", "クランク", "ギヤ", "キャブ", "電気")
        x["parts_major"] = x["parts"].map(lambda v: float(any(m in str(v) for m in major)) if isinstance(v, str) and v not in ("", "null", "[]") else (np.nan if v is None or (isinstance(v, float) and np.isnan(v)) else 0.0))
    else:
        x["parts_n"] = np.nan
        x["parts_major"] = np.nan
    x["race_type_cat"] = x["race_type"].map(_race_type)
    x["klass_ord"] = x["klass"].map(KLASS_ORD)
    x["month"] = x["race_date"].dt.month
    return x


def add_asof_features(x: pd.DataFrame, hist: HistoryFrames | AsofTables) -> pd.DataFrame:
    if isinstance(hist, AsofTables):
        tables = hist
    else:
        tables = AsofTables(perf_log_ext(hist.races, hist.entries, hist.result_entries,
                                         getattr(hist, "results", None), getattr(hist, "previews", None),
                                         getattr(hist, "conditions", None)))
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

    if getattr(tables, "ext", False):
        from boatlab.features.asof import CK_COLS, ENTRY_COLS, EXT_RACER_COLS, K_COLS, WX_COLS
        x = attach_asof(x, tables.rext, ["regno"], "rx", stat_cols=EXT_RACER_COLS)
        n2 = x["rx_n2"].replace(0, np.nan)
        x["r_mz_rate"] = x["rx_mz"] / n2
        x["r_wn_rate"] = x["rx_wn"] / n2
        x["r_dlane"] = x["rx_dl_sum"] / x["rx_dl_n"].replace(0, np.nan)
        stn = x["rx_st_n"].replace(0, np.nan)
        mean_st = x["rx_st_sum"] / stn
        x["r_st_std"] = np.sqrt(np.clip(x["rx_st2_sum"] / stn - np.square(mean_st), 0, None))
        x["r_f_rate"] = x["rx_f_n"] / n2
        wn_ = x["rx_w_n"].replace(0, np.nan)
        x["r_w_sashi_shr"] = (x["rx_w_sashi"] + x["rx_w_makz"]) / wn_
        x["r_w_mak_shr"] = x["rx_w_mak"] / wn_
        x["r_e2_rate"] = x["rx_e2_top2"] / x["rx_e2_n"].replace(0, np.nan)
        x["r_e2_n"] = x["rx_e2_n"]
        x = x.drop(columns=[f"rx_{c}" for c in EXT_RACER_COLS])
        # 選手×枠の進入（全期間）
        x = attach_asof(x, tables.rl, ["regno", "lane"], "rl", stat_cols=ENTRY_COLS)
        rln = x["rl_n2"].replace(0, np.nan)
        x["rl_mz_rate"] = x["rl_mz"] / rln
        x["rl_dlane"] = x["rl_dl_sum"] / x["rl_dl_n"].replace(0, np.nan)
        x["rl_n"] = x["rl_n2"]
        x = x.drop(columns=[f"rl_{c}" for c in ENTRY_COLS])
        # 選手×コース：ST分散＋決まり手（2年）
        x = attach_asof(x, tables.rck, ["regno", "course"], "rk", window_days=730, stat_cols=CK_COLS)
        stn = x["rk_st_n"].replace(0, np.nan)
        mean_st = x["rk_st_sum"] / stn
        x["rc_st_std"] = np.sqrt(np.clip(x["rk_st2_sum"] / stn - np.square(mean_st), 0, None))
        wn_ = x["rk_w_n"].replace(0, np.nan)
        x["rc_w_nige_shr"] = x["rk_w_nige"] / wn_
        x["rc_w_sashi_shr"] = (x["rk_w_sashi"] + x["rk_w_makz"]) / wn_
        x["rc_w_mak_shr"] = x["rk_w_mak"] / wn_
        x["rc_w_n"] = x["rk_w_n"]
        x = x.drop(columns=[f"rk_{c}" for c in CK_COLS])
        # 場×コースの決まり手率（1年）
        x = attach_asof(x, tables.sck, ["stadium_code", "course"], "sk", window_days=365, stat_cols=K_COLS)
        wn_ = x["sk_w_n"].replace(0, np.nan)
        x["sck_nige_rate"] = x["sk_w_nige"] / wn_
        x["sck_sashi_rate"] = (x["sk_w_sashi"] + x["sk_w_makz"]) / wn_
        x["sck_mak_rate"] = x["sk_w_mak"] / wn_
        x = x.drop(columns=[f"sk_{c}" for c in K_COLS])
        # 場×コース×風（3年）
        sec, stg = wind_bucket(x.get("wind_dir"), x.get("wind_speed_m"))
        x["wsec"] = sec.values
        x["wstr"] = stg.values
        x = attach_asof(x, tables.scw, ["stadium_code", "course", "wsec", "wstr"], "swc", window_days=1095, stat_cols=WX_COLS)
        swn = x["swc_n"].replace(0, np.nan)
        x["swc_win_rate"] = x["swc_win"] / swn
        x["swc_top2_rate"] = x["swc_top2"] / swn
        x["swc_win_delta"] = x["swc_win_rate"] - x["sc_1y_win_rate"]
        x = x.drop(columns=["swc_win", "swc_top2", "swc_top3", "wsec", "wstr"])
        # 直近5走・10走（走数ベース）
        x = attach_form(x, tables.form, "r")
        # 今節（同一場×直近10日）
        x = attach_asof(x, tables.rs, ["regno", "stadium_code"], "rs10", window_days=10)

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
    # 進入関連の相対量（fs2）
    if "r_mz_rate" in x.columns:
        x["course_pred_dlane"] = x["course_pred"] - x["lane"]
        mz = x["r_mz_rate"].fillna(0.0)
        tot = mz.groupby(x["race_id"]).transform("sum")
        x["mz_others_sum"] = tot - mz
        x["mz_race_max"] = mz.groupby(x["race_id"]).transform("max")
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

# fs2 追加グループ（1グループずつ効果検証する。docs/04 §15）
FEATURE_GROUPS = {
    "entry":    ["r_mz_rate", "r_wn_rate", "r_dlane", "rl_mz_rate", "rl_dlane", "rl_n",
                 "course_pred_dlane", "mz_others_sum", "mz_race_max"],
    "st":       ["r_st_std", "r_f_rate", "rc_st_std"],
    "kimarite": ["r_w_sashi_shr", "r_w_mak_shr", "rc_w_nige_shr", "rc_w_sashi_shr", "rc_w_mak_shr", "rc_w_n",
                 "sck_nige_rate", "sck_sashi_rate", "sck_mak_rate"],
    "weather":  ["swc_n", "swc_win_rate", "swc_top2_rate", "swc_win_delta"],
    "form":     ["r_l5_win_rate", "r_l5_top3_rate", "r_l5_avg_fin", "r_l5_avg_st",
                 "r_l10_win_rate", "r_l10_top3_rate", "r_l10_avg_fin", "r_l10_avg_st",
                 "rs10_n", "rs10_win_rate", "rs10_top2_rate", "rs10_avg_fin", "rs10_avg_st"],
    "exh_trust": ["r_e2_rate", "r_e2_n"],
    "parts":    ["parts_n", "parts_major"],
}
ALL_GROUP_FEATURES = [f for cols in FEATURE_GROUPS.values() for f in cols]


def feature_matrix(x: pd.DataFrame, numeric: list[str] | None = None, categorical: list[str] | None = None) -> pd.DataFrame:
    num = list(numeric or NUMERIC_FEATURES)
    cat = list(categorical or CATEGORICAL_FEATURES)
    m = x[num].astype("float32").copy()
    for c in cat:
        m[c] = x[c].astype("category")
    return m


def feature_hash(row: pd.Series) -> str:
    payload = json.dumps({k: (None if pd.isna(v) else (float(v) if isinstance(v, (int, float, np.floating)) else str(v)))
                          for k, v in row.items()}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]
