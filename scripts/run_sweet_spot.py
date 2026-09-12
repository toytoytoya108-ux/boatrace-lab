"""「確率の高さ」と「払戻が元返しに当たらないこと」の積が最大になる券種はどこか。

これまでに分かった緊張関係:
  - 人気側の歪み（favorite-longshot bias）は確率が高いほど大きい
  - しかし確率が高いほど払戻は100円（元返し）に張り付き、回収率は100%に頭打ちになる
  複勝はこの2つが 99.1% で交差した。では確率が高いまま配当が下限に当たらない券種はないか。

方法: 選定は**市場（3連単オッズ）**に任せる（モデルより正確なため）。各券種で市場がいちばん
確からしいと見る1点を買い、市場の確信が高いレースだけに絞る。払戻は公式の確定配当。
探索1〜5月で絞り方を決め、確認6〜8月で1回評価する。

出力: reports/backtest/sweet_spot.md
"""
from __future__ import annotations

import json
import sqlite3
from itertools import combinations, permutations
from pathlib import Path

import numpy as np

from boatlab.backtest.metrics import roi_bootstrap
from boatlab.config import ROOT
from boatlab.model.trifecta import PERM_LABELS, PERMS

OUT = Path(ROOT) / "reports" / "backtest" / "sweet_spot.md"
CACHE = Path("/tmp/claude-0")
DB = str(Path(ROOT) / "data" / "lab.db")
EXPLORE_END = "2026-05-31"
A = np.array([p[0] for p in PERMS])
B = np.array([p[1] for p in PERMS])
C = np.array([p[2] for p in PERMS])

EX2 = list(permutations(range(6), 2))          # 2連単 30
QU = list(combinations(range(6), 2))           # 2連複・ワイド 15
TRIO = list(combinations(range(6), 3))         # 3連複 20


def masks():
    """各券種の各買い目が「当たり」になる3連単120通りの集合。"""
    m = {}
    m["単勝"] = [(A == a) for a in range(6)]
    m["複勝"] = [((A == a) | (B == a)) for a in range(6)]
    m["2連単"] = [((A == a) & (B == b)) for a, b in EX2]
    m["2連複"] = [(((A == a) & (B == b)) | ((A == b) & (B == a))) for a, b in QU]
    m["ワイド"] = [(((A == a) | (B == a) | (C == a)) & ((A == b) | (B == b) | (C == b))) for a, b in QU]
    m["3連複"] = [(((A == a) | (B == a) | (C == a)) & ((A == b) | (B == b) | (C == b))
                  & ((A == c) | (B == c) | (C == c))) for a, b, c in TRIO]
    m["3連単"] = [(np.arange(120) == i) for i in range(120)]
    return m


KEY = {"単勝": ("win", lambda i: str(i + 1)), "複勝": ("place", lambda i: str(i + 1)),
       "2連単": ("exacta", lambda i: f"{EX2[i][0]+1}-{EX2[i][1]+1}"),
       "2連複": ("quinella", lambda i: f"{QU[i][0]+1}={QU[i][1]+1}"),
       "ワイド": ("quinella_place", lambda i: f"{QU[i][0]+1}={QU[i][1]+1}"),
       "3連複": ("trio", lambda i: "=".join(str(x + 1) for x in TRIO[i])),
       "3連単": ("trifecta", lambda i: PERM_LABELS[i])}


def load():
    z = np.load(CACHE / "test_lgb_probs.npz")
    ids = z["ids"]
    n = len(ids)
    pos = {int(r): i for i, r in enumerate(ids)}
    lab2 = {l: i for i, l in enumerate(PERM_LABELS)}
    O3 = np.full((n, 120), np.nan)
    date = np.empty(n, dtype="U10")
    raw = [None] * n
    con = sqlite3.connect(DB)
    for rid, js in con.execute("SELECT race_id, odds FROM odds_snapshots WHERE bet_type='3t' AND source='turnmark_final'"):
        i = pos.get(int(rid))
        if i is None:
            continue
        for k, v in (json.loads(js) if isinstance(js, str) else js).items():
            j = lab2.get(k)
            if j is not None and v:
                O3[i, j] = float(v)
    for rid, js, rd, irr in con.execute(
            "SELECT res.race_id, res.payouts, r.race_date, res.is_irregular FROM results res "
            "JOIN races r ON r.id=res.race_id WHERE res.payouts IS NOT NULL"):
        i = pos.get(int(rid))
        if i is None or irr:
            continue
        d = json.loads(js) if isinstance(js, str) else js
        if isinstance(d, dict):
            raw[i] = d
            date[i] = str(rd)[:10]
    con.close()
    ok = np.array([(np.isfinite(O3[i]).sum() >= 110) and date[i] != "" and raw[i] is not None for i in range(n)])
    idx = np.flatnonzero(ok)
    inv = np.where(np.isfinite(O3[idx]), 1.0 / O3[idx], 0.0)
    return inv / inv.sum(1, keepdims=True), [raw[i] for i in idx], date[idx]


def payout_of(d, bt, combo):
    key, fmt = KEY[bt]
    want = fmt(combo)
    ws = frozenset(want.replace("=", "-").split("-"))
    for e in (d or {}).get(key) or []:
        got = str(e.get("combination", "")).strip()
        if got == want or ("=" in want and frozenset(got.replace("=", "-").split("-")) == ws):
            return float(e.get("amount") or 0)
    return 0.0


def main():
    q3, raws, date = load()
    M = masks()
    e, c = date <= EXPLORE_END, date > EXPLORE_END
    L = [f"# 確率の高さと配当の下限、積が最大になるのはどこか（{len(date):,}R・2026年）\n",
         "選定は市場（3連単オッズ）。各券種で市場がいちばん確からしいと見る1点を100円。",
         f"探索 〜{EXPLORE_END} で絞り方を決め、確認期間で1回評価。\n",
         "| 券種 | 対象 | 確認n | 市場の確率 | 的中 | 平均払戻 | 元返し | 回収率 | 95%区間 |",
         "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    best = []
    for bt, ms in M.items():
        P = np.stack([q3[:, m].sum(1) for m in ms], 1)
        conf, sel = P.max(1), P.argmax(1)
        for frac in (1.0, 0.3, 0.1, 0.05):
            th = np.quantile(conf[e], 1 - frac) if frac < 1 else -1
            s = c & (conf >= th)
            n = int(s.sum())
            if n < 100:
                continue
            ret = np.array([payout_of(raws[i], bt, int(sel[i])) for i in np.flatnonzero(s)])
            stake = np.full(n, 100.0)
            lo, hi = roi_bootstrap(stake, ret, n_boot=400)
            roi = ret.sum() / stake.sum()
            lab = "全レース" if frac == 1 else f"上位{frac*100:g}%"
            L.append(f"| {bt} | {lab} | {n:,} | {conf[s].mean()*100:.1f}% | {(ret>0).mean()*100:.1f}% | "
                     f"{ret[ret>0].mean():.0f}円 | {(ret==100).mean()*100:.1f}% | **{roi*100:.1f}%** | "
                     f"{lo*100:.0f}〜{hi*100:.0f}% |")
            best.append((roi, bt, lab, n, lo, hi))
    best.sort(reverse=True)
    L.append("\n## 上位5つ\n")
    for roi, bt, lab, n, lo, hi in best[:5]:
        L.append(f"- **{bt} {lab}**: 回収率 {roi*100:.1f}%（n={n:,}、95%区間 {lo*100:.0f}〜{hi*100:.0f}%）")
    top = best[0]
    L.append(f"\n最高は **{top[1]} {top[2]} の {top[0]*100:.1f}%**。" +
             ("**100%を超えた。**次は締切前オッズで同じ選定ができるかを確かめる。"
              if top[4] > 1.0 else "区間の下限が100%を超えたものは無く、やはり壁は越えていない。"))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
