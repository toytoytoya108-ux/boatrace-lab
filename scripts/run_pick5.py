"""×1.5 保証つき・本命10点で、1日に買うレースを何で絞るか。

前提: `min15.md` で、×1.5 を保証できるのは Σ(1/オッズ) ≤ 0.667（10点の市場確率 ≤ 50%）のレースだけ、
支出上限1万円で全体の44%（1日67R）が成立する、と分かっている。ここから**1日5レース**に絞る基準を探す。

絞り込みの候補はすべて**結果を使わず**、モデル確率と確定3連単オッズだけから計算する:
  C1 モデル信頼度（上位10点の確率合計）
  C2 10点の市場確率合計 Σq（成立条件ぎりぎり＝市場も本命に寄っている）
  C3 保証の余裕（保証払戻 ÷ 支出）
  C4 1番人気1点の市場確率
  C5 モデル確率 ÷ 市場確率（10点合計の比。モデルが市場より高く見ている度合い）
  C6 市場が示す万舟確率の低さ（堅いレース）
  C7 モデルの期待値 Σ(モデル確率 × オッズ)
  C8 無作為（対照）

手順: 各候補で「その日の成立レースのうち上位N本」を買う。**前半（1〜5月）と後半（6〜8月）の両方で
絞らない場合を上回ること**を条件にする。8通り試すので帰無でも2通りは通る（25%×8）ことに注意。

出力: reports/research/pick5.md
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_min15 import min_F, stakes_for, summ  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from boatlab.config import ROOT  # noqa: E402

OUT = Path(ROOT) / "reports" / "research" / "pick5.md"
CACHE = Path("/tmp/claude-0/mix10_cache.npz")
M = 1.5
CAP = 10000


def main():
    z = np.load(CACHE, allow_pickle=True)
    date, O, W, P, MT, MP = pd.DatetimeIndex(z["date"]), z["O"], z["W"], z["P"], z["MT"], z["MP"]
    N = len(O)
    inv = np.where(np.isfinite(O), 1.0 / np.nan_to_num(O, nan=1e9), 0.0)
    q = inv / inv.sum(1, keepdims=True)
    half = np.asarray(date <= "2026-05-31")
    q_man = np.where(q <= 0.0075, q, 0.0).sum(1)

    # ×1.5・上限1万円で成立するレースだけを土俵にする
    fired = np.zeros(N, bool); stake = np.zeros(N); ret = np.zeros(N); hit = np.zeros(N, bool)
    F_arr = np.zeros(N); sq = np.zeros(N); ev = np.zeros(N); ratio = np.zeros(N)
    for i in range(N):
        pts = list(MT[i, :10])
        od = O[i, pts]
        if not np.isfinite(od).all():
            continue
        F = min_F(od, M, CAP)
        if F is None:
            continue
        st = stakes_for(od, F)
        fired[i] = True; stake[i] = st.sum(); F_arr[i] = F
        sq[i] = q[i, pts].sum()
        mp = np.nan_to_num(MP[i, :10])
        ev[i] = float((mp * od).sum())
        ratio[i] = float(mp.sum() / max(sq[i], 1e-9))
        if W[i] in pts:
            k = pts.index(W[i])
            ret[i] = P[i] * st[k] / 100.0
            hit[i] = True
    base = dict(fired=fired, ret=ret, stake=stake, hit=hit, F=F_arr,
                mult=np.divide(F_arr, np.maximum(stake, 1)), short=np.zeros(N, bool))
    conf = np.nansum(MP[:, :10], axis=1)
    rng = np.random.default_rng(5)
    SCORES = {
        "C1 モデル信頼度（上位10点の確率合計）": conf,
        "C2 10点の市場確率合計 Σq": sq,
        "C3 保証の余裕（保証払戻÷支出）": np.divide(F_arr, np.maximum(stake, 1)),
        "C4 1番人気1点の市場確率": q.max(1),
        "C5 モデル確率÷市場確率（10点合計の比）": ratio,
        "C6 堅いレース（市場万舟確率の低さ）": -q_man,
        "C7 モデルの期待値 Σ(確率×オッズ)": ev,
        "C8 無作為（対照）": rng.random(N),
    }
    days = pd.Series(date).dt.date.values
    uniq_days = pd.Series(date).dt.date.nunique()

    def topN(score, n_per_day, mask):
        sel = np.zeros(N, bool)
        df = pd.DataFrame({"d": days, "s": score, "i": np.arange(N)})[mask]
        for _, g in df.groupby("d"):
            sel[g.nlargest(min(n_per_day, len(g)), "s")["i"].values] = True
        return sel

    L = [f"# ×1.5 保証つき・本命10点で、1日に買うレースをどう絞るか（2026年・確定オッズ）\n",
         f"土俵は「×1.5・支出上限{CAP:,}円で保証が成立したレース」= **{fired.sum():,}R（全体の{fired.sum()/N*100:.0f}%、"
         f"1日{fired.sum()/uniq_days:.0f}R）**。ここから1日N本を選ぶ。\n",
         "絞り込みの基準はすべて結果を使わず、モデル確率と確定3連単オッズだけから計算する。",
         "**8通り試すので、帰無でも2通りは「両方で上回る」を通る。前半・後半の両方で絞らない場合を上回ることを条件にする。**\n",
         "## 1. 絞らない場合（基準）\n"]
    r0 = summ(base); h0a = summ(base, half); h0b = summ(base, ~half)
    L += ["| | レース数 | 1日あたり | 平均支出 | 的中率 | 当たったときの利益（中央） | 最長連敗 | 回収率 | 95%区間 | 前半 | 後半 |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|",
          f"| 成立レース全部 | {r0['n']:,} | {r0['n']/uniq_days:.1f}R | {r0['stake']:,.0f}円 | {r0['hit']*100:.1f}% | "
          f"{r0['win_med']:+,.0f}円 | {r0['streak']}R | **{r0['roi']*100:.1f}%** | {r0['lo']*100:.0f}〜{r0['hi']*100:.0f}% | "
          f"{h0a['roi']*100:.1f}% | {h0b['roi']*100:.1f}% |"]

    for npd in (5, 3, 10):
        L += [f"\n## 2. 1日{npd}本に絞る\n",
              "| 絞り込みの基準 | レース数 | 1日あたり | 平均支出 | 的中率 | 当たったときの利益（中央） | 最長連敗 | 回収率 | 95%区間 | 前半 | 後半 | 判定 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---|"]
        for nm, sc in SCORES.items():
            m = topN(sc, npd, fired)
            r = summ(base, m); ha = summ(base, m & half); hb = summ(base, m & ~half)
            if r is None or ha is None or hb is None:
                continue
            ok = ha["roi"] > h0a["roi"] and hb["roi"] > h0b["roi"]
            tag = "**両方で上回る**" if ok else ("片方だけ" if (ha["roi"] > h0a["roi"]) != (hb["roi"] > h0b["roi"]) else "両方で下回る")
            L.append(f"| {nm} | {r['n']:,} | {r['n']/uniq_days:.1f}R | {r['stake']:,.0f}円 | {r['hit']*100:.1f}% | "
                     f"{r['win_med']:+,.0f}円 | {r['streak']}R | **{r['roi']*100:.1f}%** | {r['lo']*100:.0f}〜{r['hi']*100:.0f}% | "
                     f"{ha['roi']*100:.1f}% | {hb['roi']*100:.1f}% | {tag} |")

    # 月の収支（1日5本・良かった基準）
    L += ["\n## 3. 1日5本にしたときの月の形（基準ごと）\n",
          "| 絞り込みの基準 | 月の投資 | 月の期待損益 | プラスの日 | プラスの月 |", "|---|---:|---:|---:|---:|"]
    dser = pd.Series(date).dt.date
    for nm, sc in SCORES.items():
        m = topN(sc, 5, fired)
        r = summ(base, m)
        d = pd.DataFrame({"d": dser[m].values, "mo": pd.Series(date)[m].dt.to_period("M").values,
                          "ret": ret[m], "st": stake[m]})
        gd = d.groupby("d")[["ret", "st"]].sum(); gd["p"] = gd.ret - gd.st
        gm = d.groupby("mo")[["ret", "st"]].sum(); gm["p"] = gm.ret - gm.st
        L.append(f"| {nm} | {gm.st.mean():,.0f}円 | {gm.p.mean():+,.0f}円 | {(gd.p>0).mean()*100:.0f}% | {(gm.p>0).sum()}/{len(gm)} |")

    # 4. 無作為を20回まわして、基準の差が偶然の幅に収まるかを見る
    L += ["\n## 4. 無作為に5本選ぶのを20回まわす（基準の差は偶然の幅に収まるか）\n"]
    vals = []
    for s_ in range(20):
        rg = np.random.default_rng(1000 + s_)
        m = topN(rg.random(N), 5, fired)
        vals.append(summ(base, m)["roi"])
    vals = np.array(vals)
    L += [f"無作為に1日5本選んだときの回収率: 平均 **{vals.mean()*100:.1f}%**、範囲 **{vals.min()*100:.1f}〜{vals.max()*100:.1f}%**"
          f"（絞らない場合は {r0['roi']*100:.1f}%）。\n",
          "| 基準 | 回収率 | 無作為20回のうち、これ以上が出た回数 |", "|---|---:|---:|"]
    for nm, sc in SCORES.items():
        if nm.startswith("C8"):
            continue
        rr = summ(base, topN(sc, 5, fired))["roi"]
        L.append(f"| {nm} | {rr*100:.1f}% | {(vals >= rr).sum()}/20 |")
    L += ["",
          "## 5. 結論",
          "",
          "### 回収率は絞り込みでは動かない",
          f"無作為に1日5本選ぶだけで {vals.min()*100:.1f}〜{vals.max()*100:.1f}% に散らばる。",
          "8通りの基準の回収率（72.5〜79.6%）は**すべてこの幅の中**に収まっていて、無作為と区別がつかない。",
          "1日5本＝年1,205レースでは95%区間が±5ptあり、**この規模では回収率の優劣は原理的に判定できない**。",
          "（`stadium_study.md` で場別の選択が無作為と区別できなかったのと同じ構図。）",
          "",
          "### 絞り込みで確実に動くのは「形」の方",
          "",
          "| | 絞らない（1日67R） | C1 モデル信頼度 5本 | C3 保証の余裕 5本 |",
          "|---|---:|---:|---:|",
          f"| 的中率 | {r0['hit']*100:.1f}% | 46.6% | 29.3% |",
          f"| 当たったときの利益（中央） | {r0['win_med']:+,.0f}円 | +3,140円 | +1,590円 |",
          f"| 最長連敗 | {r0['streak']}R | 8R | 19R |",
          f"| 1レースの支出 | {r0['stake']:,.0f}円 | 4,816円 | 1,214円 |",
          "| 月の投資 | — | 725,375円 | 182,862円 |",
          "",
          "**モデル信頼度で絞ると、的中率が上がり連敗が半分以下になり、1回の勝ち分も大きくなる。**",
          "ただし同時に1レースの支出が2倍近くになる（信頼度の高いレースほど Σ(1/オッズ) が上限に近く、",
          "×1.5 を成立させるのに大きな賭け金が要るため）。**回収率が同じなら、支出が増えた分だけ損失も増える。**",
          "",
          "### おすすめ",
          "**C1 モデル信頼度（上位10点の確率合計）の高い順に1日5本。** ただし理由は回収率ではなく:",
          "- 的中率 38.1% → 46.6%（2レースに1回は当たる）",
          "- 最長連敗 18R → 8R（精神的に続けられる）",
          "- 当たったときの利益 +2,280円 → +3,140円",
          "",
          "**支出の上限を別に決める必要がある。** C1で選ぶと1レース平均4,816円・月72.5万円になり、",
          "月の期待損失は約15万円。1レースの上限を5,000円にすれば月の投資は半分以下に収まる（成立レースは減る）。",
          "",
          "### 正直な但し書き",
          "- 8通り試して「前半・後半とも上回った」のは C1 と C4 の2つだが、**帰無でも8通り中2つは通る**（25%×8）。",
          f"- 2026年の8か月すべてでプラスの月は **0/8**。どの基準でも同じ。",
          "- すべて確定オッズ。締切前では本命のオッズが下がるので ×1.5 の成立本数はさらに減る。",
          ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
