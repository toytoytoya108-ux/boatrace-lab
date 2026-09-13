"""地方競馬（NAR）公式CSVの読み込みと、単勝・複勝の市場を測る道具。

データ出所: 地方競馬情報サイト「月別開催日程」の レース情報 / オッズ情報 ZIP（人が画面から取得したもの）。
  YYYYMM_racelist.csv  レース単位（頭数・距離・馬場・天候…）
  YYYYMM_horselist.csv 馬単位（馬番・着順・人気・騎手…）
  YYYYMM_payback.csv   レース単位の払戻金（同着はレースが複数行になる）
  YYYYMM_0x_odds.csv   賭式ごとのオッズ（単勝は オッズ、複勝は オッズ=下限 / オッズ（最大）=上限）

検証済み（2026-03、`reports/research/nar_market.md` §0）:
  単勝オッズ × 100 == 単勝払戻金 が 99.5%で一致。残り0.46%は同着（払戻が分割される）。
  → このオッズは**確定オッズ**。人気も払戻の人気と100%一致。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

KEY = ["競馬場", "競走年月日", "レース番号"]
ENC = "utf-8-sig"
RATE = 0.80          # 単勝・複勝の払戻率（船橋の公表値。実測で検算する）
NEED = 1.0 / RATE    # 必要な 実勝率÷市場確率 = 1.250


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding=ENC, dtype=str)


def load_month(d: Path, ym: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(馬単位, レース単位) を返す。馬単位は 単勝オッズ・複勝オッズ・着順・実払戻を持つ。"""
    odds = pd.concat([_read(p) for p in sorted(d.glob(f"{ym}_*_odds.csv"))], ignore_index=True)
    horse = _read(d / f"{ym}_horselist.csv")
    pay = _read(d / f"{ym}_payback.csv")
    race = _read(d / f"{ym}_racelist.csv")

    win = odds[odds["賭式"] == "単勝"][KEY + ["番号1", "オッズ", "人気"]].copy()
    win.columns = KEY + ["馬番", "win_odds", "pop"]
    pla = odds[odds["賭式"] == "複勝"][KEY + ["番号1", "オッズ", "オッズ（最大）"]].copy()
    pla.columns = KEY + ["馬番", "pla_odds_lo", "pla_odds_hi"]

    h = horse[KEY + ["馬番", "枠番", "着順", "人気", "馬体重"]].copy()
    h = h.rename(columns={"人気": "pop_h"})
    df = win.merge(pla, on=KEY + ["馬番"], how="left").merge(h, on=KEY + ["馬番"], how="left")

    # 実際の払戻（同着はレースが複数行になるので melt して馬番ごとに集約）
    wp = pay[KEY + ["単勝組番", "単勝払戻金（円）"]].rename(columns={"単勝組番": "馬番", "単勝払戻金（円）": "win_pay"})
    wp = wp.dropna(subset=["馬番"]).groupby(KEY + ["馬番"], as_index=False)["win_pay"].max()
    pp = []
    for i in (1, 2, 3):
        c = pay[KEY + [f"複勝組番{i}", f"複勝払戻金{i}（円）"]].copy()
        c.columns = KEY + ["馬番", "pla_pay"]
        pp.append(c.dropna(subset=["馬番"]))
    pp = pd.concat(pp, ignore_index=True).groupby(KEY + ["馬番"], as_index=False)["pla_pay"].max()
    df = df.merge(wp, on=KEY + ["馬番"], how="left").merge(pp, on=KEY + ["馬番"], how="left")

    for c in ("win_odds", "pla_odds_lo", "pla_odds_hi", "pop", "着順", "win_pay", "pla_pay", "馬番", "枠番"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["win_pay"] = df["win_pay"].fillna(0.0)
    df["pla_pay"] = df["pla_pay"].fillna(0.0)
    df["dnf"] = df["着順"].isna()
    df["won"] = (df["着順"] == 1).astype(int)
    df["placed"] = (df["pla_pay"] > 0).astype(int)
    df["race_id"] = df["競馬場"] + "_" + df["競走年月日"] + "_" + df["レース番号"].str.zfill(2)
    df["date"] = pd.to_datetime(df["競走年月日"], format="%Y%m%d")
    df["banei"] = (df["競馬場"] == "帯広ば").astype(int)

    # 市場確率（単勝プール）: レース内で 1/オッズ を正規化。
    # 出走取消・競走除外は**オッズ行そのものが無い**（＝返還）ので自動的に外れる。
    # 競走中止・失格はオッズも人気もあり、馬券は外れになるので**残す**（着順NaNのまま負け扱い）。
    live = df["win_odds"].gt(0)
    df["inv"] = np.where(live, 1.0 / df["win_odds"], np.nan)
    g = df.groupby("race_id")["inv"]
    df["inv_sum"] = g.transform("sum")
    df["q"] = df["inv"] / df["inv_sum"]
    df["book_rate"] = 1.0 / df["inv_sum"]          # 実測の払戻率
    df["n_live"] = g.transform("count")
    df = df[live].copy()
    df["q_rank"] = df.groupby("race_id")["q"].rank(ascending=False, method="first").astype(int)

    r = race[KEY + ["頭数", "距離", "天候", "馬場", "芝ダート区分", "競走種類名称"]].copy()
    r["race_id"] = r["競馬場"] + "_" + r["競走年月日"] + "_" + r["レース番号"].str.zfill(2)
    return df, r


def roi_ci(stake: np.ndarray, ret: np.ndarray, n_boot: int = 400, seed: int = 0) -> tuple[float, float, float]:
    """回収率と95%区間（レース単位でなく買い目単位のブートストラップ）。"""
    if stake.sum() <= 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    n = len(stake)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = ret[idx].sum(1) / np.maximum(stake[idx].sum(1), 1e-9)
    return (ret.sum() / stake.sum(), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975)))


def summarize(hit: np.ndarray, pay: np.ndarray, seed: int = 0) -> dict:
    """100円均等で買ったときの成績。pay は当たったときの払戻（円）、外れは0。"""
    stake = np.full(len(pay), 100.0)
    roi, lo, hi = roi_ci(stake, pay, seed=seed)
    return dict(n=len(pay), hit=float(hit.mean()) if len(hit) else np.nan, roi=roi, lo=lo, hi=hi,
                avg=float(pay[pay > 0].mean()) if (pay > 0).any() else 0.0,
                ratio=roi / RATE if roi == roi else np.nan)
