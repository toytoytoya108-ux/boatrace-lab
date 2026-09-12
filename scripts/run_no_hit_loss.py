"""3連単（堅い予想）モードで「当たったのにマイナス」をゼロにする設計の実測。

的中でも赤字になるのは、払戻 < 投資合計 のとき。払戻均等配分（賭け金 ∝ 1/オッズ）にすると
どの点が当たっても払戻は同額 K になり、K ≥ 投資合計 ⟺ 選んだ点の Σ(1/オッズ) ≤ 1。
そこで **本線をモデル確率順に、Σ(1/オッズ) が上限を超える手前まで採用**する。
1点目からすでに超えるレース（本命が10倍未満）は見送り。

比較（2026年・実オッズの封印テスト記録、本線10点はテスト時の選定をそのまま使う）:
  A. 10点固定・100円均等                     （今の形）
  B. 10点固定・払戻均等
  C. Σ(1/オッズ) ≤ 上限 まで削る・払戻均等   （提案）

出力: reports/backtest/no_hit_loss.md
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from boatlab.backtest.metrics import roi_bootstrap
from boatlab.config import ROOT
from boatlab.model.trifecta import PERM_LABELS

OUT = Path(ROOT) / "reports" / "backtest" / "no_hit_loss.md"
DB = str(Path(ROOT) / "data" / "lab.db")
UNIT = 100


def guaranteed(odds: np.ndarray, budget: int):
    """後ろから削りながら、Σstake ≤ budget かつ 全点の払戻 ≥ budget を満たす最大の点数を探す。
    返り値 (採用点数, 各点の賭け金)。成立しなければ (0, None)。"""
    for k in range(len(odds), 0, -1):
        st = np.ceil(budget / odds[:k] / UNIT) * UNIT
        if st.sum() <= budget:
            return k, st.astype(int)
    return 0, None


def stakes_payout_equal(odds: np.ndarray, total: int) -> np.ndarray:
    """賭け金 ∝ 1/オッズ、100円単位、合計 total。端数は最大剰余法。"""
    w = 1.0 / odds
    n_units = total // UNIT
    raw = w / w.sum() * n_units
    base = np.floor(raw).astype(int)
    rem = n_units - base.sum()
    if rem > 0:
        base[np.argsort(-(raw - base), kind="stable")[:rem]] += 1
    base = np.maximum(base, 1)                      # 各点 100円以上
    return base * UNIT


def main():
    rec = pd.read_parquet(Path(ROOT) / "reports" / "backtest" / "test_records.parquet")
    rec = rec[rec["valid"] & (rec["odds_source"] == "real")].copy()
    con = sqlite3.connect(DB)
    lab2 = {l: i for i, l in enumerate(PERM_LABELS)}
    O = {}
    for rid, js in con.execute(
            "SELECT race_id, odds FROM odds_snapshots WHERE bet_type='3t' AND source='turnmark_final'"):
        d = json.loads(js) if isinstance(js, str) else js
        arr = np.full(120, np.nan)
        for k, v in d.items():
            j = lab2.get(k)
            if j is not None and v and float(v) > 0:
                arr[j] = float(v)
        O[int(rid)] = arr
    con.close()
    rec = rec[rec["race_id"].isin(O)].copy()

    def run(sub, cap, label):
        out = []
        for r in sub.itertuples():
            od = O[int(r.race_id)]
            main = [int(i) for i in r.main][:10]
            odds = np.array([od[i] for i in main])
            okm = np.isfinite(odds) & (odds > 0)
            main = [m for m, o in zip(main, okm) if o]
            odds = odds[okm]
            if len(main) == 0:
                continue
            act = int(r.actual_idx) if pd.notna(r.actual_idx) else -1
            pay = float(r.actual_payout) if pd.notna(r.actual_payout) else 0.0
            if cap is None:                                   # A/B: 10点固定
                keep = np.arange(len(main))
                o = odds[keep]
                st = np.full(len(o), 100) if label.startswith("A") else stakes_payout_equal(o, 3000)
            else:                                             # C: 保証つき（cap = 1レース予算）
                k, st = guaranteed(odds, int(cap))
                if k == 0:
                    out.append(dict(skip=True, k=0, stake=0, ret=0.0, hit=False))
                    continue
                keep = np.arange(k)
                o = odds[keep]
            hit_pos = [j for j, m in enumerate(np.array(main)[keep]) if m == act]
            ret = st[hit_pos[0]] * pay / 100.0 if hit_pos else 0.0
            out.append(dict(skip=False, k=len(o), stake=int(st.sum()), ret=float(ret), hit=bool(hit_pos)))
        return pd.DataFrame(out)

    L = [f"# 「当たったのにマイナス」をゼロにする設計（{len(rec):,}R・2026年・実オッズ）\n",
         "本線10点はテスト記録の選定（モデル確率順）をそのまま使う。払戻は公式の確定配当。",
         "払戻均等＝賭け金を 1/オッズ に比例させ、どの点が当たっても払戻がほぼ同額になる配分。\n",
         "| 対象 | 方式 | 見送り | 平均点数 | 1レース投資 | 的中率 | 的中でも赤字 | 的中時払戻 中央値 | 回収率 | 95%区間 |",
         "|---|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for tgt, sub in (("全レース", rec), ("信頼度0.70以上（購入判定）", rec[rec["decision"] == "buy"])):
        for cap, label in ((None, "A. 10点・100円均等（1,000円）"), (None, "B. 10点・払戻均等（3,000円）"),
                           (2000, "C. 保証つき・予算2,000円"), (3000, "C. 保証つき・予算3,000円"),
                           (5000, "C. 保証つき・予算5,000円")):
            d = run(sub, cap, label)
            b = d[~d["skip"]]
            if len(b) < 50:
                continue
            lo, hi = roi_bootstrap(b["stake"].values.astype(float), b["ret"].values, n_boot=400)
            hits = b[b["hit"]]
            loss_on_hit = float((hits["ret"] < hits["stake"]).mean()) if len(hits) else float("nan")
            L.append(f"| {tgt} | {label} | {d['skip'].mean()*100:.1f}% | {b['k'].mean():.1f}点 | "
                     f"{b['stake'].mean():,.0f}円 | {b['hit'].mean()*100:.1f}% | **{loss_on_hit*100:.1f}%** | "
                     f"{hits['ret'].median():,.0f}円 | **{b['ret'].sum()/b['stake'].sum()*100:.1f}%** | "
                     f"{lo*100:.0f}〜{hi*100:.0f}% |")
    L += ["",
          "## 読み方\n",
          "- 回収率は配分では動かない（選定が同じなので）。動くのは**的中でも赤字の割合**と的中率。",
          "- C は各点を `ceil(予算 ÷ オッズ)` で買う。Σ賭け金 ≤ 予算 が成り立つ点数まで後ろから削る。",
          "  **どの点が当たっても払戻 ≥ 予算 ≥ 投資** が構造で成り立つので、的中でも赤字はゼロになる。",
          "  代わりに点数が減って的中率が下がり、本命が安すぎて1点も成立しないレースは見送りになる。",
          "- 予算が大きいほど100円単位の丸めの影響が小さくなり、点数を多く残せる。",
          "",
          "※ 削る判断に確定オッズを使っているので、締切前オッズでは点数が前後する。締切前→確定で",
          "  本命のオッズは下がる傾向（人気が集まる）なので、締切前に Σ ≤ 1 でも確定で超えることがある。",
          "  実運用では上限に余裕（C'）を持たせるのが安全。"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
