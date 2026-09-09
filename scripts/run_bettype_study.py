"""券種別（3連単・3連複・2連単・2連複・ワイド・単勝・複勝）の回収率検証。

2026-09-03 の検証は「推定オッズ」で行ったため、ここでは **実際の払戻金** だけを使う。
モデルの3連単120通り確率を各券種に周辺化し、確率上位k点を100円ずつ買った場合の
回収率を、results.payouts（公式の確定配当）で計算する。オッズ推定は一切入らない。

  探索期間 2024-01〜2025-06（84,804R） … ここで「どの券種・何点・どのゲート」が良いか探す
  封印テスト 2025-07〜2026-08（65,292R） … 探索で選んだものを1回だけ当てる

出力: reports/backtest/bettype_study.{csv,md}
"""
from __future__ import annotations

import json
import sqlite3
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from boatlab.backtest.metrics import roi_bootstrap
from boatlab.config import ROOT
from boatlab.model.trifecta import PERMS

OUTDIR = Path(ROOT) / "reports" / "backtest"
# 公式の払戻率: 競艇は全券種一律75%（控除率25%）。競馬と違い券種で変わらない。
# 実測（Σ(1/確定オッズ)の逆数）でも 単勝73.6% / 3連単74.8% とほぼ一致する。
RATE = {k: .75 for k in ("3連単", "3連複", "2連単", "2連複", "ワイド", "単勝", "複勝")}
DB = str(Path(ROOT) / "data" / "lab.db")
CACHE = Path("/tmp/claude-0")

LANES = range(6)
EXACTA = [(a, b) for a in LANES for b in LANES if a != b]                       # 30
QUINELLA = list(combinations(LANES, 2))                                        # 15
TRIO = list(combinations(LANES, 3))                                            # 20

# 券種ごとの (点数, DBのキー, 組番文字列→index, 確率の周辺化行列を作る関数)
def _key_win(c):  return int(c) - 1
def _key_ord2(c): a, b = c.split("-"); return EXACTA.index((int(a) - 1, int(b) - 1))
def _key_un2(c):  return QUINELLA.index(tuple(sorted(int(x) - 1 for x in c.split("="))))
def _key_un3(c):  return TRIO.index(tuple(sorted(int(x) - 1 for x in c.split("="))))
def _key_tri(c):  return PERMS.index(tuple(int(x) - 1 for x in c.split("-")))

BETS = {
    "3連単":  dict(n=120, key="trifecta",       parse=_key_tri),
    "3連複":  dict(n=20,  key="trio",           parse=_key_un3),
    "2連単":  dict(n=30,  key="exacta",         parse=_key_ord2),
    "2連複":  dict(n=15,  key="quinella",       parse=_key_un2),
    "ワイド": dict(n=15,  key="quinella_place", parse=_key_un2),
    "単勝":   dict(n=6,   key="win",            parse=_key_win),
    "複勝":   dict(n=6,   key="place",          parse=_key_win),
}


def marginals(P: np.ndarray) -> dict[str, np.ndarray]:
    """3連単120通りの確率から各券種の確率を作る（周辺化。合計は必ず1になる）。"""
    A = np.array([p[0] for p in PERMS]); B = np.array([p[1] for p in PERMS]); C = np.array([p[2] for p in PERMS])
    out = {"3連単": P}
    out["単勝"] = np.stack([P[:, A == a].sum(1) for a in LANES], 1)
    out["複勝"] = np.stack([P[:, (A == a) | (B == a)].sum(1) for a in LANES], 1)     # 1着か2着
    out["2連単"] = np.stack([P[:, (A == a) & (B == b)].sum(1) for a, b in EXACTA], 1)
    out["2連複"] = np.stack([P[:, ((A == a) & (B == b)) | ((A == b) & (B == a))].sum(1) for a, b in QUINELLA], 1)
    out["ワイド"] = np.stack([P[:, ((A == a) | (B == a) | (C == a)) & ((A == b) | (B == b) | (C == b))].sum(1)
                            for a, b in QUINELLA], 1)
    out["3連複"] = np.stack([P[:, ((A == a) | (B == a) | (C == a)) & ((A == b) | (B == b) | (C == b))
                              & ((A == c) | (B == c) | (C == c))].sum(1) for a, b, c in TRIO], 1)
    return out


def payout_tables(ids: np.ndarray) -> dict[str, np.ndarray]:
    """race_id × 買い目 の実払戻（100円あたり、円）。当たっていない買い目は0。"""
    con = sqlite3.connect(DB)
    pos = {int(r): i for i, r in enumerate(ids)}
    tabs = {b: np.zeros((len(ids), v["n"]), np.float32) for b, v in BETS.items()}
    ok = np.zeros(len(ids), bool)
    q = "SELECT race_id, payouts, is_irregular FROM results WHERE payouts IS NOT NULL"
    for rid, js, irr in con.execute(q):
        i = pos.get(int(rid))
        if i is None or irr:
            continue
        try:
            d = json.loads(js)
        except Exception:
            continue
        good = True
        for b, v in BETS.items():
            for e in d.get(v["key"]) or []:
                try:
                    tabs[b][i, v["parse"](e["combination"])] = float(e["amount"] or 0)
                except Exception:
                    good = False
        ok[i] = good and bool(d.get("trifecta"))
    con.close()
    return tabs, ok


def roi_topk(prob: np.ndarray, pay: np.ndarray, k: int, sel: np.ndarray) -> dict:
    """確率上位k点を100円ずつ。sel=買う対象レース。"""
    idx = np.argsort(-prob[sel], axis=1)[:, :k]
    got = np.take_along_axis(pay[sel], idx, axis=1)
    stake = np.full(len(got), 100.0 * k)
    ret = got.sum(1)
    lo, hi = roi_bootstrap(stake, ret, n_boot=400)
    return dict(n=int(sel.sum()), k=k, hit=round(float((got > 0).any(1).mean()), 4),
                roi=round(float(ret.sum() / stake.sum()), 4), lo=round(lo, 3), hi=round(hi, 3),
                pnl=int(ret.sum() - stake.sum()))


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    rows = []
    data = {}
    for tag, span in (("探索", "valid_lgb_hNone"), ("封印テスト", "test_lgb")):
        z = np.load(CACHE / f"{span}_probs.npz")
        ids, P = z["ids"], z["P"].astype(np.float64)
        tabs, ok = payout_tables(ids)
        M = marginals(P)
        data[tag] = (M, tabs, ok, ids)
        for b in BETS:
            for k in range(1, BETS[b]["n"] + 1):
                if k > 20 and b == "3連単" and k % 5:
                    continue
                r = roi_topk(M[b], tabs[b], k, ok)
                rows.append(dict(period=tag, bet=b, **r))
        print(tag, "races", int(ok.sum()))
    df = pd.DataFrame(rows)
    df.to_csv(OUTDIR / "bettype_study.csv", index=False)

    ex = df[df.period == "探索"]
    te = df[df.period == "封印テスト"].set_index(["bet", "k"])
    L = ["# 券種別の回収率（実際の払戻金のみ・オッズ推定なし）\n",
         f"探索 2024-01〜2025-06（{int(data['探索'][2].sum()):,}R） / 封印テスト 2025-07〜2026-08（{int(data['封印テスト'][2].sum()):,}R）",
         "確率上位k点を100円ずつ購入。払戻は公式の確定配当。\n",
         "## 0. 出発点（券種ごとの払戻率）と、モデルがどこまで縮めたか\n",
         "競艇の払戻率は全券種一律75%（控除率25%）。全通り均等買いはそれより低くなる",
         "（人気薄の方が割高＝favorite-longshot bias のため）。\n",
         "| 券種 | 公式の払戻率 | 全通り均等買い | モデル最良（テスト） | 100%まで |",
         "|---|---:|---:|---:|---:|"]
    for b in BETS:
        n = BETS[b]["n"]
        base = te.loc[(b, n)].roi
        bestk = te.loc[b].roi.max()
        L.append(f"| {b} | {RATE[b]*100:.0f}% | {base*100:.1f}% | {bestk*100:.1f}% | −{(1-bestk)*100:.1f}pt |")
    L += ["\n## 1. 券種ごとの最良（探索期間で点数を選び、封印テストで確認）\n",
         "| 券種 | 探索の最良点数 | 探索回収 | テストn | テスト的中 | テスト回収 | 95%区間 |",
         "|---|---:|---:|---:|---:|---:|---|"]
    for b in BETS:
        s = ex[ex.bet == b].sort_values("roi", ascending=False).iloc[0]
        t = te.loc[(b, int(s.k))]
        L.append(f"| {b} | {int(s.k)}点 | {s.roi*100:.1f}% | {t.n} | {t.hit*100:.1f}% | {t.roi*100:.1f}% | {t.lo*100:.0f}〜{t.hi*100:.0f}% |")
    L.append("\n## 2. 点数ごとの回収率（封印テスト）\n")
    L.append("| 券種 | " + " | ".join(f"{k}点" for k in (1, 2, 3, 5, 10, 15)) + " |")
    L.append("|---" * 7 + "|")
    for b in BETS:
        cells = []
        for k in (1, 2, 3, 5, 10, 15):
            cells.append(f"{te.loc[(b,k)].roi*100:.1f}%" if (b, k) in te.index else "-")
        L.append(f"| {b} | " + " | ".join(cells) + " |")
    # 3. 自信で絞る: 買い目の確率が高いレースだけ
    L.append("\n## 3. 自信の高いレースだけ買う（上位k点の確率和で上位x%のレース）\n")
    L.append("| 券種 | 点数 | 対象 | 探索回収 | テストn | テスト的中 | テスト回収 | 95%区間 |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---|")
    best = []
    for b in BETS:
        k = int(ex[ex.bet == b].sort_values("roi", ascending=False).iloc[0].k)
        for frac in (0.20, 0.05, 0.01):
            out = {}
            for tag in ("探索", "封印テスト"):
                M, tabs, ok, _ = data[tag]
                s = np.sort(M[b], axis=1)[:, -k:].sum(1)
                th = np.quantile(s[ok], 1 - frac)
                out[tag] = roi_topk(M[b], tabs[b], k, ok & (s >= th))
            e, t = out["探索"], out["封印テスト"]
            L.append(f"| {b} | {k} | 上位{frac*100:.0f}% | {e['roi']*100:.1f}% | {t['n']} | {t['hit']*100:.1f}% | "
                     f"{t['roi']*100:.1f}% | {t['lo']*100:.0f}〜{t['hi']*100:.0f}% |")
            best.append((e["roi"], t["roi"], b, k, frac))
    be = max(best)
    L.append(f"\n探索期間で最も良かった絞り方は {be[2]}・{be[3]}点・上位{be[4]*100:.0f}%（探索 {be[0]*100:.1f}%）→ "
             f"封印テストでは {be[1]*100:.1f}%。\n")
    (OUTDIR / "bettype_study.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
