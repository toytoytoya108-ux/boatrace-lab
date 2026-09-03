"""as-of（時点固定）集計。

原則：対象レースの race_date より **前の日** までの実績だけを使う（同日の先行レースは使わない。docs/03 §5-5）。
実装：日次集計 → キー内で累積 → merge_asof(allow_exact_matches=False) で「その日より前」の累積値を引く。
未来の行を混ぜても、対象日より前の累積値は変わらないので出力は変わらない（tests/test_asof_leak.py）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

STAT_COLS = ["n", "win", "top2", "top3", "st_sum", "st_n", "fin_sum", "fin_n"]


def perf_log(races: pd.DataFrame, entries: pd.DataFrame, result_entries: pd.DataFrame) -> pd.DataFrame:
    """選手×レースの実績ログ（結果があるレースのみ）。"""
    r = races[["id", "race_date", "stadium_code"]].rename(columns={"id": "race_id"})
    x = result_entries.merge(r, on="race_id", how="inner")
    x = x.merge(entries[["race_id", "lane", "motor_no", "boat_no"]], on=["race_id", "lane"], how="left")
    x = x[x["regno"].notna()].copy()
    x["regno"] = x["regno"].astype(int)
    fin = x["finish_pos"]
    x["win"] = (fin == 1).astype(float)
    x["top2"] = (fin <= 2).astype(float)
    x["top3"] = (fin <= 3).astype(float)
    started = fin.notna() | x["abnormal"].isin(["妨", "エ", "転", "落", "沈", "不", "失", "F", "L"])
    x["n"] = started.astype(float)
    st_ok = x["st"].notna() & (x["st"] >= 0) & (x["st"] < 1)  # F は負値 → 平均STから除外
    x["st_sum"] = np.where(st_ok, x["st"], 0.0)
    x["st_n"] = st_ok.astype(float)
    x["fin_sum"] = np.where(fin.notna(), fin, 0.0)
    x["fin_n"] = fin.notna().astype(float)
    x["course"] = x["course"].astype("Int64")
    return x


def cumulative_table(log: pd.DataFrame, keys: list[str], stat_cols: list[str] | None = None) -> pd.DataFrame:
    """keys × race_date の日次集計を、キー内で累積したテーブル（その日を含む累積）。"""
    cols = list(stat_cols or STAT_COLS)
    keys = list(keys)
    log = log.assign(race_date=pd.to_datetime(log["race_date"]).astype("datetime64[ns]"))
    g = log.dropna(subset=keys).groupby(keys + ["race_date"], as_index=False)[cols].sum()
    g = g.sort_values(keys + ["race_date"])
    g[cols] = g.groupby(keys)[cols].cumsum()
    return g


def attach_asof(targets: pd.DataFrame, cum: pd.DataFrame, keys: list[str], prefix: str,
                window_days: int | None = None, date_col: str = "race_date",
                stat_cols: list[str] | None = None) -> pd.DataFrame:
    """targets に「date_col より前」の累積統計を付ける。window_days を指定すると直近ウィンドウの値。

    stat_cols を指定した場合は派生率を作らず、生の累積和を <prefix>_<col> として付ける。
    """
    STAT = list(stat_cols or STAT_COLS)
    keys = list(keys)
    t = targets.copy()
    t["_order"] = np.arange(len(t))
    t["_d"] = pd.to_datetime(t[date_col]).astype("datetime64[ns]")
    valid = t.dropna(subset=keys)
    cum_sorted = cum.rename(columns={"race_date": "_d"}).copy()
    cum_sorted["_d"] = pd.to_datetime(cum_sorted["_d"]).astype("datetime64[ns]")
    cum_sorted = cum_sorted.sort_values("_d")
    for k in keys:
        cum_sorted[k] = cum_sorted[k].astype(valid[k].dtype) if len(valid) else cum_sorted[k]

    def _asof(frame: pd.DataFrame, shift_days: int = 0) -> pd.DataFrame:
        f = frame[["_order", "_d"] + keys].copy()
        if shift_days:
            f["_d"] = f["_d"] - pd.Timedelta(days=shift_days)
        f = f.sort_values("_d")
        m = pd.merge_asof(f, cum_sorted[["_d"] + keys + STAT], on="_d", by=keys,
                          allow_exact_matches=False, direction="backward")
        return m.set_index("_order")[STAT].fillna(0.0)

    before = _asof(valid)
    if window_days:
        before = before - _asof(valid, window_days)
    out = pd.DataFrame(0.0, index=t["_order"], columns=STAT)
    out.loc[before.index] = before.values
    if stat_cols is not None:
        feats = pd.DataFrame({f"{prefix}_{c}": out[c].values for c in STAT}, index=t.index)
        return pd.concat([t.drop(columns=["_order", "_d"]), feats], axis=1)
    n = out["n"].replace(0, np.nan)
    feats = pd.DataFrame({
        f"{prefix}_n": out["n"].values,
        f"{prefix}_win_rate": (out["win"] / n).values,
        f"{prefix}_top2_rate": (out["top2"] / n).values,
        f"{prefix}_top3_rate": (out["top3"] / n).values,
        f"{prefix}_avg_st": (out["st_sum"] / out["st_n"].replace(0, np.nan)).values,
        f"{prefix}_avg_fin": (out["fin_sum"] / out["fin_n"].replace(0, np.nan)).values,
    }, index=t.index)
    return pd.concat([t.drop(columns=["_order", "_d"]), feats], axis=1)


KIMARITE_MAP = {"逃げ": "w_nige", "差し": "w_sashi", "まくり": "w_mak", "まくり差し": "w_makz"}
ENTRY_COLS = ["n2", "mz", "wn", "dl_sum", "dl_n"]
EXT_RACER_COLS = ENTRY_COLS + ["st2_sum", "st_sum", "st_n", "f_n",
                               "w_nige", "w_sashi", "w_mak", "w_makz", "w_oth", "w_n",
                               "e2_top2", "e2_n"]
CK_COLS = ["st2_sum", "st_sum", "st_n", "w_nige", "w_sashi", "w_mak", "w_makz", "w_oth", "w_n"]
K_COLS = ["w_nige", "w_sashi", "w_mak", "w_makz", "w_oth", "w_n", "n"]
WX_COLS = ["n", "win", "top2", "top3"]


def wind_bucket(wind_dir, wind_speed) -> tuple[pd.Series, pd.Series]:
    """風向16方位→4セクター（0-3）、無風/欠損→4。風速→0:〜2m, 1:3〜5m, 2:6m〜（欠損→-1）。"""
    d = pd.to_numeric(wind_dir, errors="coerce")
    sp = pd.to_numeric(wind_speed, errors="coerce")
    sec = ((d - 1) // 4).where(d.between(1, 16), 4)
    sec = sec.where(~(sp < 1), 4).fillna(4).astype(int)
    stg = pd.cut(sp, [-0.1, 2, 5, 99], labels=[0, 1, 2]).astype("float").fillna(-1).astype(int)
    return sec, stg


def perf_log_ext(races, entries, result_entries, results=None, previews=None, conditions=None) -> pd.DataFrame:
    """perf_log ＋ 進入・実ST分散・F・決まり手・展示対応・風バケット（fs2 用）。無い入力の列は0/NaN。"""
    x = perf_log(races, entries, result_entries)
    started = x["n"] > 0
    both = x["course"].notna() & x["lane"].notna() & started
    x["n2"] = both.astype(float)
    x["mz"] = (both & (x["course"] < x["lane"])).astype(float)
    x["wn"] = (both & (x["course"] == x["lane"])).astype(float)
    x["dl_sum"] = np.where(both, (x["course"].astype(float) - x["lane"]), 0.0)
    x["dl_n"] = both.astype(float)
    st_ok = x["st_n"] > 0
    x["st2_sum"] = np.where(st_ok, np.square(np.where(st_ok, x["st"], 0.0)), 0.0)
    x["f_n"] = ((x["abnormal"] == "F") | (x["st"].fillna(1) < 0)).astype(float)
    for c in ["w_nige", "w_sashi", "w_mak", "w_makz", "w_oth", "w_n"]:
        x[c] = 0.0
    if results is not None and len(results) and "kimarite" in results:
        km = results[["race_id", "kimarite"]].drop_duplicates("race_id")
        x = x.merge(km, on="race_id", how="left")
        winner = (x["win"] > 0) & x["kimarite"].notna()
        x.loc[winner, "w_n"] = 1.0
        mapped = x["kimarite"].map(KIMARITE_MAP)
        for col in set(KIMARITE_MAP.values()):
            x.loc[winner & (mapped == col), col] = 1.0
        x.loc[winner & mapped.isna(), "w_oth"] = 1.0
        x = x.drop(columns=["kimarite"])
    x["e2_top2"] = 0.0
    x["e2_n"] = 0.0
    if previews is not None and len(previews) and "exhibition_time" in previews:
        pv = previews[["race_id", "lane", "exhibition_time"]].dropna(subset=["exhibition_time"])
        pv = pv.groupby(["race_id", "lane"], as_index=False)["exhibition_time"].last()
        pv["e_rank"] = pv.groupby("race_id")["exhibition_time"].rank(method="min")
        x = x.merge(pv[["race_id", "lane", "e_rank"]], on=["race_id", "lane"], how="left")
        e2 = (x["e_rank"] <= 2) & (x["fin_n"] > 0)
        x["e2_n"] = e2.astype(float)
        x["e2_top2"] = (e2 & (x["top2"] > 0)).astype(float)
        x = x.drop(columns=["e_rank"])
    x["wsec"] = 4
    x["wstr"] = -1
    if conditions is not None and len(conditions) and "wind_dir" in conditions:
        cd = conditions[["race_id", "wind_dir", "wind_speed_m"]].drop_duplicates("race_id")
        x = x.merge(cd, on="race_id", how="left", suffixes=("", "_c"))
        sec, stg = wind_bucket(x["wind_dir"], x["wind_speed_m"])
        x["wsec"] = sec
        x["wstr"] = stg
        x = x.drop(columns=[c for c in ["wind_dir", "wind_speed_m"] if c in x])
    # メモリ対策：統計列を float32 に
    for c in x.columns:
        if x[c].dtype == "float64":
            x[c] = x[c].astype("float32")
    return x


def form_table(log: pd.DataFrame, windows=(5, 10)) -> pd.DataFrame:
    """走数ベースの直近フォーム。(regno, race_date) ごとに「その日より前の直近N走」の統計。
    実装：レース順に rolling → 同一(選手, 日)の最終行 → 利用側は merge_asof(allow_exact_matches=False)。"""
    f = log[log["n"] > 0][["regno", "race_date", "race_id", "win", "top3", "fin_sum", "fin_n", "st_sum", "st_n"]].copy()
    f = f.sort_values(["regno", "race_date", "race_id"]).reset_index(drop=True)
    g = f.groupby("regno")
    out = f[["regno", "race_date"]].copy()
    for w in windows:
        r = g[["win", "top3", "fin_sum", "fin_n", "st_sum", "st_n"]].rolling(w, min_periods=1).sum().reset_index(drop=True)
        out[f"l{w}_n"] = g.cumcount().values + 1
        out[f"l{w}_n"] = np.minimum(out[f"l{w}_n"], w)
        out[f"l{w}_win_rate"] = (r["win"] / out[f"l{w}_n"]).values
        out[f"l{w}_top3_rate"] = (r["top3"] / out[f"l{w}_n"]).values
        out[f"l{w}_avg_fin"] = (r["fin_sum"] / r["fin_n"].replace(0, np.nan)).values
        out[f"l{w}_avg_st"] = (r["st_sum"] / r["st_n"].replace(0, np.nan)).values
    # 同一（選手×日）の最終値のみ残す（利用側で「前日以前」の行を引く）
    out = out.groupby(["regno", "race_date"], as_index=False).tail(1)
    out["race_date"] = pd.to_datetime(out["race_date"]).astype("datetime64[ns]")
    return out.sort_values("race_date")


def attach_form(targets: pd.DataFrame, form: pd.DataFrame, prefix: str = "r") -> pd.DataFrame:
    t = targets.copy()
    t["_order"] = np.arange(len(t))
    t["_d"] = pd.to_datetime(t["race_date"]).astype("datetime64[ns]")
    valid = t.dropna(subset=["regno"]).copy()
    valid["regno"] = valid["regno"].astype(form["regno"].dtype)
    cols = [c for c in form.columns if c.startswith("l")]
    m = pd.merge_asof(valid[["_order", "_d", "regno"]].sort_values("_d"),
                      form.rename(columns={"race_date": "_d"}), on="_d", by="regno",
                      allow_exact_matches=False, direction="backward")
    m = m.set_index("_order")[cols]
    feats = pd.DataFrame(index=t["_order"], columns=cols, dtype=float)
    feats.loc[m.index] = m.values
    feats.columns = [f"{prefix}_{c}" for c in cols]
    feats.index = t.index
    return pd.concat([t.drop(columns=["_order", "_d"]), feats], axis=1)


class AsofTables:
    """全履歴から累積テーブルを1回だけ作り、対象チャンクごとに使い回す（メモリ対策）。"""

    def __init__(self, log: pd.DataFrame):
        self.course = cumulative_table(log, ["course"])
        self.racer = cumulative_table(log, ["regno"])
        self.rc = cumulative_table(log, ["regno", "course"])
        self.rs = cumulative_table(log, ["regno", "stadium_code"])
        self.rsc = cumulative_table(log, ["regno", "stadium_code", "course"])
        self.sc = cumulative_table(log, ["stadium_code", "course"])
        log_m = log.dropna(subset=["motor_no"])
        log_m = log_m.assign(motor_no=log_m["motor_no"].astype(int))
        self.ms = cumulative_table(log_m, ["stadium_code", "motor_no"])
        # fs2 拡張（perf_log_ext の列がある場合のみ）
        self.ext = "mz" in log.columns
        if self.ext:
            self.rext = cumulative_table(log, ["regno"], EXT_RACER_COLS)
            self.rl = cumulative_table(log, ["regno", "lane"], ENTRY_COLS)
            self.rck = cumulative_table(log, ["regno", "course"], CK_COLS)
            self.sck = cumulative_table(log, ["stadium_code", "course"], K_COLS)
            wl = log[log["wstr"] >= 0]
            self.scw = cumulative_table(wl, ["stadium_code", "course", "wsec", "wstr"], WX_COLS)
            self.form = form_table(log)


def shrink(rate: pd.Series, n: pd.Series, prior: pd.Series | float, k: float) -> pd.Series:
    """Empirical Bayes 縮約: (成功数 + k·prior) / (n + k)"""
    rate = rate.fillna(0.0)
    n = n.fillna(0.0)
    return (rate * n + k * prior) / (n + k)
