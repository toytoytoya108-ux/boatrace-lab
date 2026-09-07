"""市場一致フィルタの検証（絞り込み型 × 実オッズ、2026-01〜08）。

絞り込み型（FocusedParams）に「市場との乖離が大きい買い目を外す」条件を足したとき、
回収率が上がるかを実オッズ期間で検証する。
  探索期間: 2026-01-01〜2026-05-31（しきい値を選ぶ）
  確認期間: 2026-06-01〜2026-08-30（探索で選んだ規則を1回だけ評価）
変種:
  pq_max : P/Q（モデル確率÷市場含意確率）が pq_max を超える点を外す
  beta   : 市場確率への縮約（p_hat = (1-β)p + βq）で期待値を計算
  odds帯 : 5〜50倍 → 別の帯
  race gate: モデル1位＝市場1位のレースだけ買う
使い方: python scripts/run_market_filter.py [probstore名=test_lgb]
出力: reports/backtest/market_filter.{csv,md}
"""
from __future__ import annotations
import sys
from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from boatlab.backtest.dataset import build_race_dataset
from boatlab.backtest.metrics import roi_bootstrap
from boatlab.backtest.walkforward import ProbStore
from boatlab.model.selection import FocusedParams, implied_q, select_focused

OUT = Path("reports/backtest")


def load(store_name: str):
    st = ProbStore.load(f"data/probstores/{store_name}.pkl")
    ids = np.concatenate([np.array(p.test_ids) for p in st.periods])
    P = np.concatenate([p.test_p for p in st.periods]).astype(np.float64)
    T = np.concatenate([p.test_tri for p in st.periods])
    R = build_race_dataset(date(2026, 1, 1), date(2026, 8, 30)).set_index("race_id")
    keep = np.isin(ids, R.index.values)
    ids, P, T = ids[keep], P[keep], T[keep]
    R = R.loc[ids]
    real = R["real_odds"].values
    good = np.array([isinstance(o, np.ndarray) and np.isfinite(o).sum() >= 100 for o in real]) & (T >= 0) & (R["status"] != "cancelled").values
    ids, P, T, R = ids[good], P[good], T[good], R[good]
    OD = np.vstack([np.asarray(o, dtype=float) for o in R["real_odds"].values])
    OD = np.where(np.isfinite(OD) & (OD > 0), OD, np.nan)
    P = P / P.sum(1, keepdims=True)
    PAY = R["trifecta_payout"].fillna(0).values.astype(float)
    dates = pd.to_datetime(R["race_date"]).values
    expl = dates < np.datetime64("2026-06-01")
    return ids, P, T, OD, PAY, expl, R


def simulate(P, T, OD, PAY, prm: FocusedParams, pq_max: float | None = None, gate_top1: bool = False):
    """各レースの (stake, payout, hit) を返す（買わないレースは stake=0）。"""
    N = len(T)
    stake = np.zeros(N); pay = np.zeros(N); hit = np.zeros(N, bool); npts = np.zeros(N, int)
    for i in range(N):
        odds = OD[i]
        if pq_max is not None:
            q = implied_q(odds)
            odds = np.where(P[i] / np.clip(q, 1e-9, None) <= pq_max, odds, np.nan)
        if gate_top1 and int(np.argmax(P[i])) != int(np.nanargmin(np.where(np.isfinite(OD[i]), OD[i], np.inf))):
            continue
        f = select_focused(P[i], odds, prm)
        if f.decision != "buy":
            continue
        stake[i] = sum(f.stakes); npts[i] = len(f.points)
        if T[i] in f.points:
            hit[i] = True
            pay[i] = PAY[i] * f.stakes[f.points.index(int(T[i]))] / 100
    return stake, pay, hit, npts


def summarize(stake, pay, hit, npts, sel):
    m = (stake > 0) & sel
    n = int(m.sum())
    if n == 0:
        return dict(n=0)
    lo, hi = roi_bootstrap(stake[m], pay[m])
    pnl = pay[m] - stake[m]
    cum = np.cumsum(pnl); dd = float((np.maximum.accumulate(cum) - cum).max())
    return dict(n=n, share=round(n / max(int(sel.sum()), 1), 3), pts=round(float(npts[m].mean()), 2), hit=round(float(hit[m].mean()), 4),
                roi=round(float(pay[m].sum() / stake[m].sum()), 4), ci_lo=round(lo, 3), ci_hi=round(hi, 3),
                pnl=int(pnl.sum()), avg_stake=int(stake[m].mean()), max_dd=int(dd))


VARIANTS = {
    "base(現行)": dict(),
    "P/Q≤1.5": dict(pq_max=1.5),
    "P/Q≤2.0": dict(pq_max=2.0),
    "P/Q≤2.5": dict(pq_max=2.5),
    "P/Q≤3.0": dict(pq_max=3.0),
    "β=0.2": dict(prm=dict(beta=0.2)),
    "β=0.4": dict(prm=dict(beta=0.4)),
    "β=0.6": dict(prm=dict(beta=0.6)),
    "odds5-20": dict(prm=dict(odds_hi=20)),
    "odds5-30": dict(prm=dict(odds_hi=30)),
    "odds3-50": dict(prm=dict(odds_lo=3)),
    "odds10-50": dict(prm=dict(odds_lo=10)),
    "gate:1位一致": dict(gate_top1=True),
    "EV≥0.9": dict(prm=dict(ev_min=0.9)),
    "EV≥1.1": dict(prm=dict(ev_min=1.1)),
    "S15≥0.78": dict(prm=dict(s15_min=0.78)),
    "max3点": dict(prm=dict(max_points=3)),
    # --- 組合せ（探索で安定していた odds5-20 を軸に） ---
    "odds5-15": dict(prm=dict(odds_hi=15)),
    "odds5-10": dict(prm=dict(odds_hi=10)),
    "odds5-20+EV0.9": dict(prm=dict(odds_hi=20, ev_min=0.9)),
    "odds5-20+EV0.8": dict(prm=dict(odds_hi=20, ev_min=0.8)),
    "odds5-20+EV0.7": dict(prm=dict(odds_hi=20, ev_min=0.7)),
    "odds5-20+EV0.8+max3": dict(prm=dict(odds_hi=20, ev_min=0.8, max_points=3)),
    "odds5-20+EV0.9+gate": dict(prm=dict(odds_hi=20, ev_min=0.9), gate_top1=True),
    "odds5-20+EV0.9+S15≥0.70": dict(prm=dict(odds_hi=20, ev_min=0.9, s15_min=0.70)),
    "odds5-20+EV0.9+P/Q≤2.5": dict(prm=dict(odds_hi=20, ev_min=0.9), pq_max=2.5),
}


def main(store_name="test_lgb", extra: dict | None = None):
    ids, P, T, OD, PAY, expl, R = load(store_name)
    print(f"実オッズありレース {len(T):,}（探索 {expl.sum():,} / 確認 {(~expl).sum():,}）", flush=True)
    variants = dict(VARIANTS)
    if extra:
        variants.update(extra)
    rows = []
    for name, v in variants.items():
        prm = replace(FocusedParams(), **v.get("prm", {}))
        s, p, h, n = simulate(P, T, OD, PAY, prm, v.get("pq_max"), v.get("gate_top1", False))
        a = summarize(s, p, h, n, expl); b = summarize(s, p, h, n, ~expl)
        rows.append(dict(variant=name, **{f"探索_{k}": x for k, x in a.items()}, **{f"確認_{k}": x for k, x in b.items()}))
        print(name, "探索", a, "確認", b, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"market_filter_{store_name}.csv", index=False)
    md = [f"# 市場一致フィルタの検証（{store_name}、実オッズ 2026-01〜08、探索1〜5月 / 確認6〜8月）", "",
          "| 変種 | 探索N | 対象率 | 平均点 | 的中率 | 回収率 | 95%CI | 損益 | 確認N | 的中率 | 回収率 | 95%CI | 損益 | 最大DD |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r.get("探索_n", 0) == 0 or r.get("確認_n", 0) == 0:
            md.append(f"| {r['variant']} | {r.get('探索_n', 0)} | – | – | – | – | – | – | {r.get('確認_n', 0)} | – | – | – | – | – |")
            continue
        md.append(f"| {r['variant']} | {r['探索_n']} | {r['探索_share']*100:.0f}% | {r['探索_pts']} | {r['探索_hit']*100:.1f}% | {r['探索_roi']*100:.1f}% | "
                  f"{r['探索_ci_lo']*100:.0f}–{r['探索_ci_hi']*100:.0f} | {r['探索_pnl']:+,} | {r['確認_n']} | {r['確認_hit']*100:.1f}% | {r['確認_roi']*100:.1f}% | "
                  f"{r['確認_ci_lo']*100:.0f}–{r['確認_ci_hi']*100:.0f} | {r['確認_pnl']:+,} | {r['確認_max_dd']:,} |")
    (OUT / f"market_filter_{store_name}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "test_lgb")
