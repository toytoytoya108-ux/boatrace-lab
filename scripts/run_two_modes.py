"""2モード表示ツールの設計材料（穴狙い ／ 的中率・回収率重視）。

## 前提（すでに実測済み）
- `longshot.md`: **いちばん人気がない1点**を買うのは最悪。3連単27.6%・2連単19.5%・単勝34.1%。
  一方、3連単を人気順で見ると回収率は 人気2番目 79.7% → 40番目 68.4% → 120番目 27.6%。
  **「大きい配当が欲しい」なら最不人気ではなく、万舟圏に最も安く届く帯を買うべき。**
- `manshu.md`: 万舟は1日24.8本・6Rに1回。市場は万舟を17%過大評価。荒れやすい条件は実在するが
  市場はすべて知っている（どの区分も実測÷市場 < 1.00）。
- `favorite_edge.md`: 複勝・市場確信度上位10%で回収率99.1%（元返しが天井）。

## ここで測ること
穴モードの設計空間を総当たりする。選定は市場（3連単確定オッズ）の人気順。

  1. **どの人気帯を買うか**（連続する順位の窓）
  2. **何点買うか**
  3. **レースを絞るか**（市場が示す万舟確率／荒れやすい条件）
  4. 回収率だけでなく **的中間隔・連敗・月次の振れ** まで出す（穴モードは体感がすべてなので）

出力: reports/backtest/two_modes.md
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

from run_sweet_spot import load, masks, payout_of  # noqa: E402

OUT = Path(ROOT) / "reports" / "backtest" / "two_modes.md"
DB = str(Path(ROOT) / "data" / "lab.db")
Q_MAN = 0.0075


def streaks(hit: np.ndarray) -> int:
    """最長連敗（的中0の連続本数）。"""
    best = cur = 0
    for h in hit:
        cur = 0 if h else cur + 1
        best = max(best, cur)
    return best


def evaluate(raws, ord3, sel_races, lo, hi, label, date):
    """人気 lo〜hi 番目（0起点、hi は含まない）を各レース100円ずつ。"""
    idx = np.flatnonzero(sel_races)
    k = hi - lo
    n = len(idx)
    if n < 100:
        return None
    ret = np.zeros(n)
    hits = np.zeros(n)
    for a, i in enumerate(idx):
        got = 0.0
        for j in ord3[i, lo:hi]:
            p = payout_of(raws[i], "3連単", int(j))
            if p > 0:
                got += p
        ret[a] = got
        hits[a] = got > 0
    stake = np.full(n, 100.0 * k)
    r_lo, r_hi = roi_bootstrap(stake, ret, n_boot=400)
    pnl = ret - stake
    mon = pd.Series(pnl).groupby(pd.Series(date[idx]).str[:7]).sum()
    return dict(label=label, k=k, n=n, hit=hits.mean(), nhit=int(hits.sum()),
                roi=ret.sum() / stake.sum(), lo=r_lo, hi=r_hi,
                avg=(ret[ret > 0].mean() if hits.any() else 0.0), mx=ret.max(),
                per_month=hits.sum() / max(len(mon), 1), maxlose=streaks(hits > 0),
                m_best=mon.max(), m_worst=mon.min(), m_plus=int((mon > 0).sum()), m_n=len(mon),
                stake_day=100.0 * k * n / max(pd.Series(date[idx]).nunique(), 1))


def main():
    q3, raws, date = load()
    n = len(date)
    ord3 = np.argsort(-q3, 1)
    q_man = np.where(q3 <= Q_MAN, q3, 0.0).sum(1)          # 市場が示す万舟確率

    L = [f"# 2モード表示ツールの設計材料（{n:,}R・2026年）\n",
         "選定はすべて市場（3連単の確定オッズ）。モデルは市場を超えないことが実測済みなので、",
         "選定に使わない（`market_combine.md`: 合成の優位は控除率の56分の1）。\n",
         "## A. 穴モード — どの人気帯を買うのがいちばん安いか\n",
         "各レースで人気 lo〜hi 番目の3連単を100円ずつ。全レース対象。\n",
         "| 買う帯 | 点数 | 1日の投資 | 的中率 | 的中数 | 月あたり的中 | 最長連敗 | 平均払戻 | 最高払戻 | 回収率 | 95%区間 |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    allr = np.ones(n, bool)
    rows = []
    for lo, hi, lab in ((0, 5, "人気1〜5"), (5, 15, "人気6〜15"), (15, 30, "人気16〜30"),
                        (29, 50, "人気30〜50"), (49, 70, "人気50〜70"), (69, 90, "人気70〜90"),
                        (89, 110, "人気90〜110"), (109, 120, "人気110〜120"),
                        (19, 40, "人気20〜40"), (29, 60, "人気30〜60")):
        r = evaluate(raws, ord3, allr, lo, hi, lab, date)
        if r:
            rows.append(r)
            L.append(f"| {r['label']} | {r['k']} | {r['stake_day']:,.0f}円 | {r['hit']*100:.1f}% | "
                     f"{r['nhit']:,} | {r['per_month']:.0f}本 | {r['maxlose']}R | {r['avg']:,.0f}円 | "
                     f"{r['mx']:,.0f}円 | **{r['roi']*100:.1f}%** | {r['lo']*100:.0f}〜{r['hi']*100:.0f}% |")

    best = max(rows, key=lambda x: x["roi"])
    man_rows = [r for r in rows if r["avg"] >= 10000]
    L += ["", f"回収率が最大なのは **{best['label']}（{best['roi']*100:.1f}%）**。",
          "ただしこれは穴ではない。**平均払戻が万舟級（1万円以上）になる帯**のうち、",
          f"いちばん回収率が高いのは **{max(man_rows, key=lambda x: x['roi'])['label']}"
          f"（{max(man_rows, key=lambda x: x['roi'])['roi']*100:.1f}%、"
          f"平均払戻 {max(man_rows, key=lambda x: x['roi'])['avg']:,.0f}円）**。",
          f"最不人気1点の 27.6% と比べて **{max(man_rows, key=lambda x: x['roi'])['roi']/0.276:.1f}倍**。",
          "",
          "### 読み方: 何を払って何を買うか\n",
          "| 選択 | 回収率 | 平均払戻 | 月あたり的中 |", "|---|---:|---:|---:|"]
    for r in rows:
        if r["label"] in ("人気1〜5", "人気16〜30", "人気30〜50", "人気50〜70", "人気110〜120"):
            L.append(f"| {r['label']} | {r['roi']*100:.1f}% | {r['avg']:,.0f}円 | {r['per_month']:.0f}本 |")

    # ---- レース選択は効くか
    L += ["\n## B. 穴モード — レースを絞ると良くなるか\n",
          "`manshu.md` のとおり、荒れやすい条件は実在するが市場はすべて知っている。実際に効くか測る。\n",
          "| レースの絞り方 | 買う帯 | n | 的中率 | 平均払戻 | 回収率 | 95%区間 |",
          "|---|---|---:|---:|---:|---:|---|"]
    for th, lab in ((0.0, "絞らない"), (0.70, "市場の万舟確率 上位30%"), (0.90, "上位10%"),
                    (-0.70, "市場の万舟確率 下位30%（堅いレース）")):
        if th == 0.0:
            sel = allr
        elif th > 0:
            sel = q_man >= np.quantile(q_man, th)
        else:
            sel = q_man <= np.quantile(q_man, -th)
        for lo, hi, bl in ((29, 50, "人気30〜50"), (19, 40, "人気20〜40")):
            r = evaluate(raws, ord3, sel, lo, hi, lab, date)
            if r:
                L.append(f"| {lab} | {bl} | {r['n']:,} | {r['hit']*100:.1f}% | {r['avg']:,.0f}円 | "
                         f"**{r['roi']*100:.1f}%** | {r['lo']*100:.0f}〜{r['hi']*100:.0f}% |")

    # ---- 月次の振れ
    L += ["\n## C. 穴モードの月次の振れ（体感の目安）\n",
          "100円×点数×全レース。2026年1〜8月の8か月。\n",
          "| 買う帯 | 1日の投資 | 月次プラスの回数 | 最良の月 | 最悪の月 | 最長連敗 |",
          "|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        if r["label"] in ("人気1〜5", "人気16〜30", "人気30〜50", "人気50〜70", "人気110〜120"):
            L.append(f"| {r['label']} | {r['stake_day']:,.0f}円 | {r['m_plus']}/{r['m_n']}か月 | "
                     f"{r['m_best']:+,.0f}円 | {r['m_worst']:+,.0f}円 | {r['maxlose']}R |")

    # ---- 的中率・回収率重視モード
    L += ["\n## D. 的中率・回収率重視モード（再掲＋点数の比較）\n",
          "選定は市場。市場がいちばん確信している艇を複勝で買い、市場確信度の高いレースだけ。\n",
          "| 買い方 | 対象 | n | 的中率 | 平均払戻 | 元返し | 回収率 | 95%区間 |",
          "|---|---|---:|---:|---:|---:|---:|---|"]
    M = masks()
    for bt in ("複勝", "ワイド", "単勝"):
        P = np.stack([q3[:, m].sum(1) for m in M[bt]], 1)
        conf, sel = P.max(1), P.argmax(1)
        for frac in (1.0, 0.3, 0.1, 0.05):
            th = np.quantile(conf, 1 - frac) if frac < 1 else -1
            s = conf >= th
            idx = np.flatnonzero(s)
            ret = np.array([payout_of(raws[i], bt, int(sel[i])) for i in idx])
            lo_, hi_ = roi_bootstrap(np.full(len(idx), 100.0), ret, n_boot=400)
            lab = "全レース" if frac == 1 else f"確信度 上位{frac*100:g}%"
            L.append(f"| {bt}1点 | {lab} | {len(idx):,} | {(ret>0).mean()*100:.1f}% | "
                     f"{ret[ret>0].mean():.0f}円 | {(ret==100).mean()*100:.1f}% | "
                     f"**{ret.sum()/(100*len(idx))*100:.1f}%** | {lo_*100:.0f}〜{hi_*100:.0f}% |")
    # ---- 現実的な運用設定に落とす
    L += ["\n## E. 現実的な設定例（1日に買う本数を絞る）\n",
          "上の表は全レース購入なので1日の投資が非現実的。**1日あたり上位K レースだけ**に絞る。",
          "穴モードの選定は「市場が示す万舟確率が高い順」、つまり市場自身が荒れると見ているレース。\n",
          "| モード | 設定 | 1日の投資 | n | 的中率 | 月あたり的中 | 最長連敗 | 平均払戻 | 回収率 | 95%区間 |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    day = pd.Series(date)
    rank_in_day = day.groupby(day).cumcount() * 0        # placeholder
    dfq = pd.DataFrame({"d": date, "q": q_man, "i": np.arange(n)})
    dfq["rk"] = dfq.groupby("d")["q"].rank(ascending=False, method="first")
    for K in (3, 5, 10):
        for lo, hi, bl in ((19, 30, "人気20〜30（11点）"), (19, 40, "人気20〜40（21点）")):
            sel = np.zeros(n, bool)
            sel[dfq.loc[dfq["rk"] <= K, "i"].values] = True
            r = evaluate(raws, ord3, sel, lo, hi, bl, date)
            if r:
                L.append(f"| 穴 | 1日{K}R × {bl} | {r['stake_day']:,.0f}円 | {r['n']:,} | "
                         f"{r['hit']*100:.1f}% | {r['per_month']:.0f}本 | {r['maxlose']}R | "
                         f"{r['avg']:,.0f}円 | **{r['roi']*100:.1f}%** | {r['lo']*100:.0f}〜{r['hi']*100:.0f}% |")
    # 的中率重視モードの現実設定
    Pp = np.stack([q3[:, m].sum(1) for m in M["複勝"]], 1)
    confp, selp = Pp.max(1), Pp.argmax(1)
    dfp = pd.DataFrame({"d": date, "q": confp, "i": np.arange(n)})
    dfp["rk"] = dfp.groupby("d")["q"].rank(ascending=False, method="first")
    for K in (3, 5, 10, 15):
        idx = dfp.loc[dfp["rk"] <= K, "i"].values
        ret = np.array([payout_of(raws[i], "複勝", int(selp[i])) for i in idx])
        lo_, hi_ = roi_bootstrap(np.full(len(idx), 100.0), ret, n_boot=400)
        days_n = pd.Series(date[idx]).nunique()
        mon = pd.Series(ret - 100).groupby(pd.Series(date[idx]).str[:7]).sum()
        L.append(f"| 的中率 | 1日{K}R × 複勝1点 | {100*len(idx)/days_n:,.0f}円 | {len(idx):,} | "
                 f"{(ret>0).mean()*100:.1f}% | — | {streaks(ret>0)}R | {ret[ret>0].mean():.0f}円 | "
                 f"**{ret.sum()/(100*len(idx))*100:.1f}%** | {lo_*100:.0f}〜{hi_*100:.0f}% |")
    L += ["",
          "※ すべて**確定オッズ**での測定。締切前オッズで同じ選定ができるかは別問題で、",
          "9/11から貯めている締切前3連単オッズで確認する必要がある",
          "（`pool_arbitrage.md` の観測フェーズでは、オッズの水準そのものを使う選定は締切前に崩れた。",
          "ただし今回の選定はオッズの**散らばりの形**なので、水準ほど動かない可能性が高い）。"]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
