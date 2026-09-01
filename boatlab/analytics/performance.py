"""実運用（ペーパートレード）成績と実戦投入判定（docs/19〜21, 06 §2.5〜2.6）。

母集団：role='active' AND stage='final' AND scoring.valid（モデル版をまたいで累計）。
見送りは category で別集計（skip_would_hit / skip_correct）。
"""
from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import pandas as pd
from sqlalchemy import text

from boatlab.backtest.metrics import max_drawdown, max_losing_streak, roi_bootstrap, wilson
from boatlab.store.db import get_engine


def load_scored(model_version: str | None = None, role: str = "active", stage: str = "final") -> pd.DataFrame:
    sql = """
        SELECT sc.*, p.model_version, p.decision, p.confidence, p.expected_return, p.flags, p.created_at,
               r.race_date, r.stadium_code, r.grade, r.race_no, s.name AS stadium,
               (SELECT AVG(ps.odds_at_pred) FROM prediction_selections ps WHERE ps.prediction_id = p.id) AS avg_odds,
               (SELECT AVG(ps.odds_at_pred) FROM prediction_selections ps WHERE ps.prediction_id = p.id AND ps.kind='hole') AS hole_avg_odds
        FROM scoring sc JOIN predictions p ON p.id = sc.prediction_id
        JOIN races r ON r.id = sc.race_id JOIN stadiums s ON s.code = r.stadium_code
        WHERE p.role = :role AND p.stage = :stage AND sc.valid = 1"""
    params = {"role": role, "stage": stage}
    if model_version:
        sql += " AND p.model_version = :mv"
        params["mv"] = model_version
    df = pd.read_sql_query(text(sql + " ORDER BY r.race_date, r.closed_at, sc.race_id"), get_engine(), params=params)
    df["race_date"] = pd.to_datetime(df["race_date"])
    df["hit"] = df["hit"].astype(bool)
    return df


def _block(v: pd.DataFrame) -> dict:
    n = len(v)
    k = int(v["hit"].sum()) if n else 0
    stake = v["stake_total"].fillna(0).values.astype(float)
    pay = v["payout_total"].fillna(0).values.astype(float)
    d = {"n": n, "hits": k, "hit_rate": (k / n) if n else None, "hit_rate_ci": wilson(k, n) if n else None,
         "stake": float(stake.sum()), "payout": float(pay.sum()), "pnl": float(pay.sum() - stake.sum()),
         "roi": (float(pay.sum() / stake.sum()) if stake.sum() else None),
         "roi_ci": roi_bootstrap(stake, pay) if n else None,
         "avg_payout_on_hit": (float(v.loc[v["hit"], "payout_total"].mean()) if k else None),
         "avg_odds": (float(v["avg_odds"].mean()) if n else None),
         "main_hits": int((v["hit_kind"] == "main").sum()), "hole_hits": int((v["hit_kind"] == "hole").sum()),
         "max_losing_streak": max_losing_streak(v["hit"].values) if n else 0,
         "max_drawdown": max_drawdown(v["pnl"].fillna(0).values.astype(float)) if n else 0}
    if n:
        d["main_hit_rate"] = d["main_hits"] / n
        d["hole_hit_rate"] = d["hole_hits"] / n
        d["hole_roi"] = float(v.loc[v["hit_kind"] == "hole", "payout_total"].sum()) / (n * 5 * 200)
        d["main_roi"] = float(v.loc[v["hit_kind"] == "main", "payout_total"].sum()) / (n * 10 * 200)
    return d


def summary(df: pd.DataFrame, since: date | None = None, until: date | None = None) -> dict:
    v = df
    if since is not None:
        v = v[v["race_date"] >= pd.Timestamp(since)]
    if until is not None:
        v = v[v["race_date"] <= pd.Timestamp(until)]
    buy = v[v["decision"] == "buy"]
    skip = v[v["decision"] == "skip"]
    out = {"all_virtual": _block(v), "buy": _block(buy), "skip_virtual": _block(skip),
           "n_total": int(len(v)), "n_buy": int(len(buy)), "n_skip": int(len(skip)),
           "skip_would_hit": int((skip["hit"]).sum()), "skip_correct": int((~skip["hit"]).sum())}
    for n in (50, 100, 200, 500):
        out[f"recent{n}_buy"] = _block(buy.tail(n))
    return out


def breakdown(df: pd.DataFrame, by: str, decision: str | None = "buy") -> pd.DataFrame:
    v = df if decision is None else df[df["decision"] == decision]
    v = v.copy()
    if by == "odds_band":
        v["odds_band"] = pd.cut(v["avg_odds"], [0, 10, 20, 40, 80, 1e9], labels=["<10", "10-20", "20-40", "40-80", "80+"])
    elif by == "month":
        v["month"] = v["race_date"].dt.to_period("M").astype(str)
    elif by == "confidence_band":
        v["confidence_band"] = pd.cut(v["confidence"], [0, 0.5, 0.6, 0.7, 0.75, 0.8, 1.0])
    elif by == "kind":
        return pd.DataFrame([{"kind": "main", **{k: _block(v)[k] for k in ("n", "main_hits", "main_hit_rate", "main_roi")}},
                             {"kind": "hole", **{k: _block(v)[k] for k in ("n", "hole_hits", "hole_hit_rate", "hole_roi")}}]) if len(v) else pd.DataFrame()
    g = v.groupby(by, observed=True)
    rows = []
    for key, sub in g:
        b = _block(sub)
        rows.append({by: key, "n": b["n"], "hit_rate": b["hit_rate"], "roi": b["roi"], "pnl": b["pnl"],
                     "hole_hits": b["hole_hits"], "max_losing_streak": b["max_losing_streak"]})
    return pd.DataFrame(rows)


def readiness(df: pd.DataFrame, thresholds: dict) -> dict:
    """実戦投入判定（docs 21）。自動購入はしない。警告を付ける。"""
    buy = df[df["decision"] == "buy"]
    b = _block(buy)
    r100 = _block(buy.tail(int(thresholds.get("recent_n", 100))))
    checks = [
        {"name": "累計レース", "value": b["n"], "target": thresholds["min_races"], "ok": b["n"] >= thresholds["min_races"], "fmt": "int"},
        {"name": "累計的中率", "value": b["hit_rate"], "target": thresholds["hit_rate"], "ok": (b["hit_rate"] or 0) >= thresholds["hit_rate"], "fmt": "pct"},
        {"name": "累計回収率", "value": b["roi"], "target": thresholds["roi"], "ok": (b["roi"] or 0) >= thresholds["roi"], "fmt": "pct"},
        {"name": f"直近{thresholds.get('recent_n', 100)}レース的中率", "value": r100["hit_rate"], "target": thresholds["recent_hit_rate"],
         "ok": (r100["hit_rate"] or 0) >= thresholds["recent_hit_rate"] and r100["n"] >= thresholds.get("recent_n", 100), "fmt": "pct"},
        {"name": "目標回収率", "value": b["roi"], "target": thresholds["target_roi"], "ok": (b["roi"] or 0) >= thresholds["target_roi"], "fmt": "pct"},
    ]
    warnings = []
    if b["n"] and b["hit_rate_ci"] and b["hit_rate_ci"][0] < thresholds["hit_rate"] <= (b["hit_rate"] or 0):
        warnings.append("的中率は条件を満たしていますが、95%信頼区間の下限は条件未満です（サンプル不足の可能性）")
    if b["n"] and b["roi_ci"] and b["roi_ci"][0] < 1.0 <= (b["roi"] or 0):
        warnings.append("回収率100%超は統計的に有意ではありません（区間下限が100%未満）")
    if len(buy):
        by_st = buy.groupby("stadium")["pnl"].sum()
        pos = by_st[by_st > 0].sum()
        if pos > 0 and by_st.max() / pos > 0.5:
            warnings.append(f"損益の50%超が特定の場（{by_st.idxmax()}）に依存しています")
        est = buy["flags"].map(lambda f: bool((f or {}).get("odds_estimated")) if isinstance(f, dict) else False).mean()
        if est > 0.2:
            warnings.append(f"推定オッズに依存した予想が{est:.0%}あります（実オッズ未取得）")
        if (b["max_losing_streak"] or 0) >= 10:
            warnings.append(f"最大連敗 {b['max_losing_streak']} 回")
        recent = _block(buy.tail(50))
        if recent["n"] >= 50 and (recent["roi"] or 0) < 0.8:
            warnings.append("直近50レースの回収率が80%未満です（成績悪化）")
    return {"checks": checks, "passed": all(c["ok"] for c in checks), "warnings": warnings, "n_buy": b["n"],
            "note": "最終的な購入判断はユーザーが行ってください。本システムは自動購入を行いません。"}


def calibration_check(df: pd.DataFrame) -> pd.DataFrame:
    v = df.copy()
    v["band"] = pd.cut(v["confidence"], [0, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0])
    g = v.groupby("band", observed=True)
    return pd.DataFrame({"n": g.size(), "conf_mean": g["confidence"].mean(), "hit_rate": g["hit"].mean()}).reset_index()


def miss_analysis(df: pd.DataFrame) -> dict:
    """外れレースの傾向（購入候補）。"""
    buy = df[df["decision"] == "buy"]
    if not len(buy):
        return {}
    miss = buy[~buy["hit"]]
    return {
        "n_miss": int(len(miss)),
        "by_stadium": miss.groupby("stadium").size().sort_values(ascending=False).head(10).to_dict(),
        "by_grade": miss.groupby("grade").size().to_dict(),
        "avg_actual_payout_on_miss": float(miss["actual_payout"].mean()) if len(miss) else None,
        "manshu_share_on_miss": float((miss["actual_payout"] >= 10000).mean()) if len(miss) else None,
        "avg_confidence_on_miss": float(miss["confidence"].mean()) if len(miss) else None,
    }
