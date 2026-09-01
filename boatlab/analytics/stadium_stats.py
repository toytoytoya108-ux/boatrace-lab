"""場別・コース別・決まり手・配当・気象条件の統計（docs/04 §13 / Phase 3）。

すべて結果テーブルの事後集計（分析表示用）。予想特徴量は features/ の as-of 集計を使う。
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sqlalchemy import text

from boatlab.store.db import get_engine


def _df(sql: str, params=None) -> pd.DataFrame:
    return pd.read_sql_query(text(sql), get_engine(), params=params or {})


def course_stats(d0: str | None = None, d1: str | None = None) -> pd.DataFrame:
    """場×進入コースの 1着率・2連対率・3連対率・平均ST。"""
    w = []
    p = {}
    if d0: w.append("r.race_date >= :d0"); p["d0"] = d0
    if d1: w.append("r.race_date <= :d1"); p["d1"] = d1
    where = ("AND " + " AND ".join(w)) if w else ""
    return _df(f"""
        SELECT r.stadium_code, s.name AS stadium, x.course, COUNT(*) AS n,
               ROUND(AVG(CASE WHEN x.finish_pos=1 THEN 1.0 ELSE 0 END),4) AS win_rate,
               ROUND(AVG(CASE WHEN x.finish_pos<=2 THEN 1.0 ELSE 0 END),4) AS top2_rate,
               ROUND(AVG(CASE WHEN x.finish_pos<=3 THEN 1.0 ELSE 0 END),4) AS top3_rate,
               ROUND(AVG(CASE WHEN x.st>=0 THEN x.st END),3) AS avg_st
        FROM result_entries x JOIN races r ON r.id=x.race_id JOIN stadiums s ON s.code=r.stadium_code
        WHERE x.course BETWEEN 1 AND 6 AND r.status='finished' {where}
        GROUP BY 1,2,3 ORDER BY 1,3""", p)


def kimarite_stats() -> pd.DataFrame:
    """場別の決まり手率（逃げ・差し・まくり・まくり差し・抜き・恵まれ）。"""
    df = _df("""
        SELECT r.stadium_code, s.name AS stadium, res.kimarite, COUNT(*) AS n
        FROM results res JOIN races r ON r.id=res.race_id JOIN stadiums s ON s.code=r.stadium_code
        WHERE res.kimarite IS NOT NULL GROUP BY 1,2,3""")
    piv = df.pivot_table(index=["stadium_code", "stadium"], columns="kimarite", values="n", fill_value=0)
    return (piv.div(piv.sum(axis=1), axis=0)).round(4).reset_index()


def payout_stats() -> pd.DataFrame:
    """場別の3連単平均配当・中央値・高配当率（5,000円以上）・万舟率。"""
    df = _df("""
        SELECT r.stadium_code, s.name AS stadium, res.trifecta_payout AS payout
        FROM results res JOIN races r ON r.id=res.race_id JOIN stadiums s ON s.code=r.stadium_code
        WHERE res.trifecta_payout IS NOT NULL""")
    g = df.groupby(["stadium_code", "stadium"])["payout"]
    return pd.DataFrame({
        "n": g.size(), "avg_payout": g.mean().round(0), "median_payout": g.median(),
        "over5000_rate": g.apply(lambda s: (s >= 5000).mean()).round(4),
        "manshu_rate": g.apply(lambda s: (s >= 10000).mean()).round(4),
    }).reset_index()


def wind_wave_stats() -> pd.DataFrame:
    """場×風向×風速帯×波高帯 の 1コース1着率・平均配当（直前情報の気象を使用）。"""
    df = _df("""
        SELECT r.stadium_code, s.name AS stadium, c.wind_dir, c.wind_speed_m, c.wave_cm,
               res.trifecta_payout AS payout,
               (SELECT finish_pos FROM result_entries x WHERE x.race_id=r.id AND x.course=1) AS c1_pos
        FROM races r JOIN stadiums s ON s.code=r.stadium_code
        JOIN race_conditions c ON c.race_id=r.id AND c.phase='preview'
        JOIN results res ON res.race_id=r.id
        WHERE r.status='finished' AND c.wind_speed_m IS NOT NULL""")
    df["wind_band"] = pd.cut(df["wind_speed_m"], [-1, 0, 2, 4, 6, 99], labels=["0", "1-2", "3-4", "5-6", "7+"])
    df["wave_band"] = pd.cut(df["wave_cm"].fillna(0), [-1, 0, 3, 6, 10, 999], labels=["0", "1-3", "4-6", "7-10", "11+"])
    df["c1_win"] = (df["c1_pos"] == 1).astype(float)
    out = (df.groupby(["stadium_code", "stadium", "wind_dir", "wind_band", "wave_band"], observed=True)
             .agg(n=("c1_win", "size"), c1_win_rate=("c1_win", "mean"), avg_payout=("payout", "mean"))
             .reset_index())
    out["c1_win_rate"] = out["c1_win_rate"].round(4)
    out["avg_payout"] = out["avg_payout"].round(0)
    return out[out["n"] >= 30]


def write_all(out_dir: str = "reports/stats") -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, fn in [("course_stats", course_stats), ("kimarite_stats", kimarite_stats),
                     ("payout_stats", payout_stats), ("wind_wave_stats", wind_wave_stats)]:
        p = out / f"{name}.csv"
        fn().to_csv(p, index=False)
        paths.append(p)
    return paths
