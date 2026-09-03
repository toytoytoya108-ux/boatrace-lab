"""採点と成績指標（docs/05 §3, docs/07 §5）。

records: 1行=1レースの予想結果。必要列:
  race_date, race_id, decision, hit, hit_kind, stake_total, payout_total, pnl, main_hit, hole_hit, valid,
  confidence, expected_return, S, odds_source
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from boatlab.model.trifecta import PERMS


def score_race(sel_idx: list[int], main_idx: list[int], tri_idx: int, payout: int | None, refund_lanes: list[int],
               stake: int = 200, cancelled: bool = False, stakes: list[int] | None = None) -> dict:
    """15点を採点する。返還艇を含む買い目は投資から除外。払戻は100円あたり → 賭け金/100 倍。

    stakes: 点ごとの賭け金（sel_idx と同順）。None なら全点 stake 円。
    """
    if cancelled or tri_idx is None or tri_idx < 0:
        return dict(valid=False, hit=None, hit_kind=None, stake_total=0, payout_total=0, pnl=0, refunded_points=0, refunded_stake=0)
    st = {i: (int(stakes[k]) if stakes is not None else int(stake)) for k, i in enumerate(sel_idx)}
    refund = set(int(l) - 1 for l in (refund_lanes or []))
    refunded = [i for i in sel_idx if set(PERMS[i]) & refund]
    live = [i for i in sel_idx if i not in refunded]
    stake_total = sum(st[i] for i in live)
    hit = tri_idx in live
    payout_total = int(round((payout or 0) * st[tri_idx] / 100)) if hit else 0
    return dict(valid=True, hit=hit, hit_kind=("main" if hit and tri_idx in main_idx else ("hole" if hit else None)),
                stake_total=stake_total, payout_total=payout_total, pnl=payout_total - stake_total,
                refunded_points=len(refunded), refunded_stake=sum(st[i] for i in refunded))


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def roi_bootstrap(stake: np.ndarray, payout: np.ndarray, n_boot: int = 500, seed: int = 0) -> tuple[float, float]:
    if len(stake) == 0 or stake.sum() == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(stake), size=(n_boot, len(stake)))
    rois = payout[idx].sum(axis=1) / np.clip(stake[idx].sum(axis=1), 1, None)
    return (float(np.percentile(rois, 2.5)), float(np.percentile(rois, 97.5)))


def max_losing_streak(hit: np.ndarray) -> int:
    best = cur = 0
    for h in hit:
        cur = 0 if h else cur + 1
        best = max(best, cur)
    return int(best)


def max_drawdown(pnl: np.ndarray) -> int:
    cum = np.cumsum(pnl)
    peak = np.maximum.accumulate(np.concatenate([[0], cum]))[1:]
    return int((peak - cum).max()) if len(cum) else 0


def summarize(rec: pd.DataFrame, label: str = "") -> dict:
    """有効レースのみ。decision 別に集計。"""
    v = rec[rec["valid"] == True].sort_values(["race_date", "race_id"])  # noqa: E712
    out = {"label": label, "n_all": int(len(rec)), "n_valid": int(len(v))}
    for name, sub in [("all", v), ("buy", v[v["decision"] == "buy"]), ("skip", v[v["decision"] == "skip"])]:
        n = len(sub)
        k = int(sub["hit"].sum()) if n else 0
        stake = sub["stake_total"].values.astype(float)
        pay = sub["payout_total"].values.astype(float)
        d = {
            "n": n, "hits": k, "hit_rate": (k / n if n else float("nan")),
            "hit_rate_ci": wilson(k, n),
            "stake": float(stake.sum()), "payout": float(pay.sum()), "pnl": float(pay.sum() - stake.sum()),
            "roi": (pay.sum() / stake.sum() if stake.sum() else float("nan")),
            "roi_ci": roi_bootstrap(stake, pay),
            "avg_payout_on_hit": (float(sub.loc[sub["hit"] == True, "payout_total"].mean()) if k else float("nan")),  # noqa: E712
            "main_hits": int((sub["hit_kind"] == "main").sum()), "hole_hits": int((sub["hit_kind"] == "hole").sum()),
            "max_losing_streak": max_losing_streak(sub["hit"].values.astype(bool)) if n else 0,
            "max_drawdown": max_drawdown(sub["pnl"].values.astype(float)) if n else 0,
            "avg_confidence": float(sub["confidence"].mean()) if n else float("nan"),
            "avg_expected_return": float(sub["expected_return"].mean()) if n else float("nan"),
        }
        if n:
            # 点ごとの配分がある場合は実際の本線/穴投資額で割る（無ければ均等200円と仮定）
            hole_stake = float(sub["hole_stake"].sum()) if "hole_stake" in sub else n * 5 * 200.0
            main_stake = float(sub["main_stake"].sum()) if "main_stake" in sub else n * 10 * 200.0
            hole_pay = float(sub.loc[sub["hit_kind"] == "hole", "payout_total"].sum())
            d["hole_roi"] = hole_pay / max(hole_stake, 1)
            d["main_roi"] = float(sub.loc[sub["hit_kind"] == "main", "payout_total"].sum()) / max(main_stake, 1)
        out[name] = d
    out["buy_rate"] = (out["buy"]["n"] / out["all"]["n"]) if out["all"]["n"] else float("nan")
    out["skip_would_hit"] = int(v[(v["decision"] == "skip") & (v["hit"] == True)].shape[0])  # noqa: E712
    return out


def monthly(rec: pd.DataFrame, decision: str | None = "buy") -> pd.DataFrame:
    v = rec[rec["valid"] == True]  # noqa: E712
    if decision:
        v = v[v["decision"] == decision]
    g = v.groupby(v["race_date"].dt.to_period("M"))
    return pd.DataFrame({"n": g.size(), "hit_rate": g["hit"].mean(), "stake": g["stake_total"].sum(),
                         "payout": g["payout_total"].sum()}).assign(roi=lambda d: d["payout"] / d["stake"].replace(0, np.nan))


def recent(rec: pd.DataFrame, n: int, decision: str = "buy") -> dict:
    v = rec[(rec["valid"] == True) & (rec["decision"] == decision)].sort_values(["race_date", "race_id"]).tail(n)  # noqa: E712
    if not len(v):
        return {"n": 0}
    return {"n": int(len(v)), "hit_rate": float(v["hit"].mean()),
            "roi": float(v["payout_total"].sum() / max(v["stake_total"].sum(), 1))}


def set_calibration_table(rec: pd.DataFrame) -> pd.DataFrame:
    """信頼度帯ごとの実セット的中率（校正の検証）。"""
    v = rec[rec["valid"] == True].copy()  # noqa: E712
    v["band"] = pd.cut(v["confidence"], [0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0])
    g = v.groupby("band", observed=True)
    return pd.DataFrame({"n": g.size(), "conf_mean": g["confidence"].mean(), "hit_rate": g["hit"].mean()}).reset_index()
