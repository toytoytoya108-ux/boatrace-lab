"""単勝・複勝プールの値付けの遅れを拾う買い方（2026-09-09 追加、観測フェーズ）。

考え方（`reports/backtest/pool_arbitrage.md`）:
  競艇の売上は3連単に約93%が集中し、単勝は約0.08%・複勝は約0.07%しかない。
  薄いプールはオッズが荒く、3連単プールがすでに織り込んでいることを反映しきれていない。
  そこで「3連単オッズが示す確率」を基準に、「単勝（複勝）オッズが示す確率」が大きく低い艇を買う。

  2026年1〜8月の確定オッズでの実測: 単勝 ×2.0・20倍以下で 4,571本・回収率184%（全8か月プラス）、
  複勝 ×1.5・20倍以下で 26,711本・回収率142%。

**重要**: 上の実測は確定オッズ同士の比較で、実際に買えるのは締切前。締切前オッズでも同じ歪みが
見えるかは未検証。このモジュールは当面 **観測（仮想の記録）専用** で、購入判断には使わない。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from boatlab.model.trifecta import PERMS

FIRST = np.array([p[0] for p in PERMS])
SECOND = np.array([p[1] for p in PERMS])
PARAMS_VERSION = "pg1"


@dataclass
class PoolGapParams:
    """既定値は2026年1〜8月の確定オッズ検証で決めたもの（探索/確認で分けて確認済み）。"""
    win_ratio_min: float = 2.0     # 3連単基準の勝率 ÷ 単勝オッズ基準の勝率
    place_ratio_min: float = 1.5
    odds_lo: float = 1.1           # 元返し（1.0）は買っても増えない
    odds_hi: float = 20.0
    stake: int = 100               # 観測フェーズは仮想100円固定
    enabled: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


def ref_probs(odds3t: dict) -> tuple[np.ndarray, np.ndarray] | None:
    """3連単オッズ → (1着確率6, 2着以内確率6)。控除率は正規化で消える。"""
    inv = np.zeros(120)
    for k, v in (odds3t or {}).items():
        try:
            a, b, c = (int(x) - 1 for x in k.split("-"))
        except Exception:
            continue
        if v and float(v) > 0:
            inv[PERMS.index((a, b, c))] = 1.0 / float(v)
    if (inv > 0).sum() < 100:
        return None
    q = inv / inv.sum()
    win = np.array([q[FIRST == a].sum() for a in range(6)])
    plc = np.array([q[(FIRST == a) | (SECOND == a)].sum() for a in range(6)])
    return win, plc


def _pool_prob(vals: list[float | None], total: float) -> np.ndarray | None:
    v = np.array([np.nan if not x or float(x) <= 0 else float(x) for x in vals], float)
    if not np.isfinite(v).all():
        return None
    inv = 1.0 / v
    return inv / inv.sum() * total


def pool_probs(win_odds: dict | None, place_odds: dict | None) -> tuple[np.ndarray | None, np.ndarray | None]:
    """単勝・複勝オッズ → そのプールが示す確率。複勝は下限側（lo）を使う＝候補判定は厳しめになる。"""
    w = _pool_prob([(win_odds or {}).get(str(i + 1)) for i in range(6)], 1.0) if win_odds else None
    pl = None
    if place_odds:
        lo = [(place_odds.get(str(i + 1)) or {}).get("lo") if isinstance(place_odds.get(str(i + 1)), dict)
              else place_odds.get(str(i + 1)) for i in range(6)]
        pl = _pool_prob(lo, 2.0)
    return w, pl


def _odds_of(d: dict | None, lane: int) -> float | None:
    v = (d or {}).get(str(lane + 1))
    if isinstance(v, dict):
        v = v.get("lo")
    return float(v) if v else None


def find_picks(odds3t: dict, win_odds: dict | None, place_odds: dict | None,
               prm: PoolGapParams | None = None) -> list[dict]:
    """買い候補。各要素 {bet_type, lane, odds, p_pool, p_ref, ratio}。"""
    prm = prm or PoolGapParams()
    if not prm.enabled:
        return []
    ref = ref_probs(odds3t)
    if ref is None:
        return []
    ref_win, ref_plc = ref
    pool_win, pool_plc = pool_probs(win_odds, place_odds)
    out: list[dict] = []
    for bt, pool, refp, rmin, src in (("win", pool_win, ref_win, prm.win_ratio_min, win_odds),
                                      ("place", pool_plc, ref_plc, prm.place_ratio_min, place_odds)):
        if pool is None:
            continue
        for lane in range(6):
            o = _odds_of(src, lane)
            if o is None or not (prm.odds_lo <= o <= prm.odds_hi) or pool[lane] <= 0:
                continue
            ratio = float(refp[lane] / pool[lane])
            if ratio >= rmin:
                out.append(dict(bet_type=bt, lane=lane + 1, odds=round(o, 2),
                                p_pool=round(float(pool[lane]), 5), p_ref=round(float(refp[lane]), 5),
                                ratio=round(ratio, 3)))
    return out


# ---------------------------------------------------------------- 記録の集計（読み取り時に採点）
def _payout_of(payouts: dict | None, bet_type: str, lane: int) -> float:
    """公式の確定配当から、その艇の払戻（100円あたり）。当たっていなければ 0。"""
    for e in (payouts or {}).get("win" if bet_type == "win" else "place", []) or []:
        try:
            if int(str(e["combination"]).strip()) == lane:
                return float(e["amount"] or 0)
        except Exception:
            continue
    return 0.0


def report(day: str | None = None) -> dict:
    """観測フェーズの成績（仮想）と、締切前→確定のオッズの動き。"""
    import json as _json

    import pandas as pd
    from sqlalchemy import text

    from boatlab.store.db import get_engine
    eng = get_engine()
    df = pd.read_sql_query(text("""
        SELECT g.*, r.race_date, r.stadium_code, r.race_no, res.payouts,
               COALESCE(st.name, CAST(r.stadium_code AS TEXT)) AS stadium,
               (SELECT o.odds FROM odds_snapshots o WHERE o.race_id = g.race_id AND o.bet_type = g.bet_type
                  AND o.source = 'turnmark_final' LIMIT 1) AS final_odds
        FROM pool_gap_picks g JOIN races r ON r.id = g.race_id
        LEFT JOIN stadiums st ON st.code = r.stadium_code
        LEFT JOIN results res ON res.race_id = g.race_id
        ORDER BY r.race_date DESC, g.race_id DESC, g.id DESC"""), eng)
    if not len(df):
        return dict(n=0, today=[], summary=[], drift=None, params_version=PARAMS_VERSION)

    def _load(v):
        if isinstance(v, str):
            try:
                v = _json.loads(v)
            except Exception:
                return None
        return v if isinstance(v, dict) else None      # 未確定のレースは NULL（NaN で来る）

    df["payouts"] = df["payouts"].map(_load)
    df["final_odds"] = df["final_odds"].map(_load)
    df["ret"] = [(_payout_of(p, bt, ln) / 100.0 * st) if p else np.nan
                 for p, bt, ln, st in zip(df["payouts"], df["bet_type"], df["lane"], df["stake"])]
    df["hit"] = df["ret"] > 0
    fo = []
    for od, bt, ln in zip(df["final_odds"], df["bet_type"], df["lane"]):
        v = (od or {}).get(str(ln))
        if isinstance(v, dict):
            v = v.get("lo")
        fo.append(float(v) if v else np.nan)
    df["odds_final"] = fo
    df["drift"] = df["odds_final"] / df["odds_seen"] - 1.0

    done = df[df["ret"].notna()]
    summary = []
    for (bt, stage), g in done.groupby(["bet_type", "stage"]):
        stake = float(g["stake"].sum())
        summary.append(dict(bet_type=bt, stage=stage, n=int(len(g)), hit=round(float(g["hit"].mean()), 4),
                            roi=round(float(g["ret"].sum() / stake), 4) if stake else None,
                            pnl=int(g["ret"].sum() - stake)))
    d = df["drift"].dropna()
    drift = None
    if len(d):
        drift = dict(n=int(len(d)), median=round(float(d.median()), 4),
                     p10=round(float(d.quantile(0.1)), 4), p90=round(float(d.quantile(0.9)), 4),
                     within10=round(float((d.abs() <= 0.10).mean()), 4))
    day = day or (str(df["race_date"].iloc[0])[:10] if len(df) else None)
    today = df[df["race_date"].astype(str).str[:10] == str(day)].head(60)
    cols = ["race_id", "stadium", "race_no", "bet_type", "lane", "stage", "odds_seen", "ratio",
            "p_pool", "p_ref", "odds_final", "ret", "minutes_before"]
    rows = [{k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in r.items()}
            for r in today[cols].to_dict("records")]
    return dict(n=int(len(df)), n_scored=int(len(done)), day=str(day), params_version=PARAMS_VERSION,
                today=rows,
                summary=sorted(summary, key=lambda x: (x["bet_type"], x["stage"])), drift=drift)
