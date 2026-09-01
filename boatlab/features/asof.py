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


def cumulative_table(log: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """keys × race_date の日次集計を、キー内で累積したテーブル（その日を含む累積）。"""
    keys = list(keys)
    log = log.assign(race_date=pd.to_datetime(log["race_date"]).astype("datetime64[ns]"))
    g = log.dropna(subset=keys).groupby(keys + ["race_date"], as_index=False)[STAT_COLS].sum()
    g = g.sort_values(keys + ["race_date"])
    g[STAT_COLS] = g.groupby(keys)[STAT_COLS].cumsum()
    return g


def attach_asof(targets: pd.DataFrame, cum: pd.DataFrame, keys: list[str], prefix: str,
                window_days: int | None = None, date_col: str = "race_date") -> pd.DataFrame:
    """targets に「date_col より前」の累積統計を付ける。window_days を指定すると直近ウィンドウの値。"""
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
        m = pd.merge_asof(f, cum_sorted[["_d"] + keys + STAT_COLS], on="_d", by=keys,
                          allow_exact_matches=False, direction="backward")
        return m.set_index("_order")[STAT_COLS].fillna(0.0)

    before = _asof(valid)
    if window_days:
        before = before - _asof(valid, window_days)
    out = pd.DataFrame(0.0, index=t["_order"], columns=STAT_COLS)
    out.loc[before.index] = before.values
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


def shrink(rate: pd.Series, n: pd.Series, prior: pd.Series | float, k: float) -> pd.Series:
    """Empirical Bayes 縮約: (成功数 + k·prior) / (n + k)"""
    rate = rate.fillna(0.0)
    n = n.fillna(0.0)
    return (rate * n + k * prior) / (n + k)
