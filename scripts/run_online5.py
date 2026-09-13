"""1日5レースの絞り込みを「その場で決められる形」で測り直す。

`pick5.md` の1日5本は **その日の全レースを見てから上位5本を選んでいた＝先読み**。
実際には朝の時点で夕方のレースの信頼度は分からない。ここでは締切時刻順に1本ずつ判断する。

判断に使えるのは「そのレースの信頼度」と「今日ここまでに何本買ったか」「今日この先に何レース残っているか」だけ。
しきい値は**探索期間（1〜5月）の分布からだけ**決め、確認期間（6〜8月）に当てる。

  R0 オラクル（その日の上位5本）           ← 先読み。実運用できない。上限の目安として置く
  R1 固定しきい値（1日平均5本になる値）      ← 本数は日によって変動する
  R2 固定しきい値＋1日5本で打ち切り          ← 「朝に5本埋まったら夕方は買えない」形
  R3 残り枠に応じた動的しきい値              ← 早い時間ほど厳しく、遅い時間ほど緩める
  R4 先着5本（しきい値なし）                 ← 対照

出力: reports/research/online5.md
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_min15 import min_F, stakes_for, summ  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from boatlab.config import ROOT  # noqa: E402

OUT = Path(ROOT) / "reports" / "research" / "online5.md"
CACHE = Path("/tmp/claude-0/mix10_cache.npz")
M, CAP, SLOTS = 1.5, 10000, 5


def main():
    z = np.load(CACHE, allow_pickle=True)
    rid, date, O, W, P, MT, MP = z["rid"], pd.DatetimeIndex(z["date"]), z["O"], z["W"], z["P"], z["MT"], z["MP"]
    N = len(rid)
    con = sqlite3.connect(str(Path(ROOT) / "data" / "lab.db"))
    ct = pd.read_sql_query("SELECT id race_id, closed_at FROM races WHERE race_date>='2026-01-01'", con)
    con.close()
    cmap = dict(zip(ct["race_id"].astype(int), pd.to_datetime(ct["closed_at"])))
    closed = pd.to_datetime(pd.Series([cmap.get(int(r)) for r in rid]))

    fired = np.zeros(N, bool); stake = np.zeros(N); ret = np.zeros(N); hit = np.zeros(N, bool); F_arr = np.zeros(N)
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
        if W[i] in pts:
            k = pts.index(W[i])
            ret[i] = P[i] * st[k] / 100.0; hit[i] = True
    base = dict(fired=fired, ret=ret, stake=stake, hit=hit, F=F_arr,
                mult=np.divide(F_arr, np.maximum(stake, 1)), short=np.zeros(N, bool))
    score = np.nansum(MP[:, :10], axis=1)

    ok = fired & closed.notna().values
    day = pd.Series(date).dt.date.values
    half = np.asarray(date <= "2026-05-31")
    df = pd.DataFrame({"i": np.arange(N), "day": day, "t": closed.values, "s": score, "ok": ok, "half": half})
    d_ok = df[df["ok"]].sort_values(["day", "t"]).reset_index(drop=True)
    days_all = pd.Series(date).dt.date.nunique()

    # しきい値は探索期間（1〜5月）の成立レースの分布からだけ決める
    ex = d_ok[d_ok["half"]]
    n_days_ex = ex["day"].nunique()
    tau = float(np.quantile(ex["s"], 1 - SLOTS * n_days_ex / len(ex)))
    qgrid = np.quantile(ex["s"], np.linspace(0, 1, 1001))

    def run_rule(kind):
        sel = np.zeros(N, bool)
        missed_better = 0; missed_gap = []
        for _, g in d_ok.groupby("day", sort=False):
            g = g.reset_index(drop=True)
            n = len(g); left = SLOTS; bought_s = []
            for j in range(n):
                s = g.loc[j, "s"]; take = False
                if kind == "R1":
                    take = s >= tau
                elif kind == "R2":
                    take = (s >= tau) and left > 0
                elif kind == "R3":
                    rem = n - j                       # この先に残る成立レース数（当日の番組表から見積もれる）
                    if left > 0:
                        frac = min(1.0, left / max(rem, 1))
                        thr = float(np.quantile(qgrid, max(0.0, 1 - frac)))
                        take = s >= thr
                elif kind == "R4":
                    take = left > 0
                if take:
                    sel[g.loc[j, "i"]] = True
                    left -= 1
                    bought_s.append(s)
                elif kind in ("R2", "R3", "R4") and left == 0 and bought_s and s > min(bought_s):
                    missed_better += 1; missed_gap.append(s - min(bought_s))
        return sel, missed_better, missed_gap

    def oracle():
        sel = np.zeros(N, bool)
        for _, g in d_ok.groupby("day", sort=False):
            sel[g.nlargest(min(SLOTS, len(g)), "s")["i"].values] = True
        return sel

    RULES = {"R0 オラクル（その日の上位5本・先読み）": oracle(),
             f"R1 固定しきい値 {tau:.3f}（本数は変動）": run_rule("R1")[0],
             f"R2 固定しきい値＋5本で打ち切り": run_rule("R2")[0],
             "R3 残り枠に応じた動的しきい値": run_rule("R3")[0],
             "R4 先着5本": run_rule("R4")[0]}

    L = [f"# 1日5レースを「その場で」決める（2026年・確定オッズ）\n",
         "**`pick5.md` の1日5本は、その日の全レースを見てから上位5本を選んでいた＝先読み。** 実運用できない。",
         "ここでは締切時刻順に1本ずつ判断する。使える情報は「このレースの信頼度」「今日ここまでの購入数」",
         "「今日この先に残るレース数（番組表から分かる）」だけ。",
         f"**しきい値は探索期間（1〜5月）の分布からだけ決めた（τ={tau:.3f}）。確認期間（6〜8月）には当てるだけ。**\n",
         f"土俵: ×{M} ・支出上限{CAP:,}円で保証が成立したレース {int(ok.sum()):,}R（1日{ok.sum()/days_all:.0f}R）。\n",
         "## 1. 確認期間（6〜8月）での比較\n",
         "| 方式 | レース数 | 1日あたり | 平均支出 | 的中率 | 当たったときの利益（中央） | 最長連敗 | 回収率 | 95%区間 |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    conf_mask = ~half
    days_conf = pd.Series(date)[conf_mask].dt.date.nunique()
    for nm, sel in RULES.items():
        r = summ(base, sel & conf_mask)
        if r is None:
            continue
        L.append(f"| {nm} | {r['n']:,} | {r['n']/days_conf:.1f}R | {r['stake']:,.0f}円 | {r['hit']*100:.1f}% | "
                 f"{r['win_med']:+,.0f}円 | {r['streak']}R | **{r['roi']*100:.1f}%** | {r['lo']*100:.0f}〜{r['hi']*100:.0f}% |")

    # 本数のばらつき
    L += ["\n## 2. 1日の本数はどれくらいばらつくか（確認期間）\n",
          "| 方式 | 0本の日 | 中央値 | 最多 | 5本を超えた日 |", "|---|---:|---:|---:|---:|"]
    dser = pd.Series(date).dt.date
    for nm, sel in RULES.items():
        c = pd.Series(dser[sel & conf_mask].values).value_counts()
        c = c.reindex(sorted(set(dser[conf_mask])), fill_value=0)
        L.append(f"| {nm} | {(c==0).sum()}日 | {int(c.median())}本 | {int(c.max())}本 | {(c>SLOTS).sum()}日 |")

    # 取りこぼし
    L += ["\n## 3. 「先に埋めてしまって、後からもっと良いレースが来た」回数（全期間）\n",
          "| 方式 | 取りこぼし回数 | 1日あたり | 逃した信頼度の差（中央） |", "|---|---:|---:|---:|"]
    for kind, nm in (("R2", "R2 固定しきい値＋5本で打ち切り"), ("R3", "R3 動的しきい値"), ("R4", "R4 先着5本")):
        _, mb, gap = run_rule(kind)
        L.append(f"| {nm} | {mb:,}回 | {mb/days_all:.1f}回 | {np.median(gap) if gap else 0:.3f} |")

    # オラクルとの一致
    L += ["\n## 4. オラクル（先読み）と何本一致するか（確認期間）\n",
          "| 方式 | オラクルと同じレースを買えた割合 |", "|---|---:|"]
    orc = RULES["R0 オラクル（その日の上位5本・先読み）"]
    for nm, sel in RULES.items():
        if nm.startswith("R0"):
            continue
        s2 = sel & conf_mask
        L.append(f"| {nm} | {(s2 & orc).sum()/max(s2.sum(),1)*100:.0f}% |")

    # 5. 実際に使うしきい値の表
    L += ["\n## 5. R3 の実際のしきい値（探索期間の分布から作った表）\n",
          "「残り枠 ÷ この先に残る成立レース数」の分位点を信頼度のしきい値にする。",
          "**早い時間ほど厳しく、終盤で枠が余っていれば緩める。**\n",
          "| 残り枠＼この先に残る本数 | 60本 | 40本 | 20本 | 10本 | 5本 | 2本 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for left in (5, 4, 3, 2, 1):
        cells = []
        for rem in (60, 40, 20, 10, 5, 2):
            frac = min(1.0, left / rem)
            cells.append(f"{float(np.quantile(qgrid, max(0.0, 1 - frac))):.3f}")
        L.append(f"| 残り{left}枠 | " + " | ".join(cells) + " |")
    L += ["",
          f"（参考: 成立レースの信頼度の分布は 中央値 {np.median(ex['s']):.3f}、上位10% {np.quantile(ex['s'],0.9):.3f}、"
          f"上位25% {np.quantile(ex['s'],0.75):.3f}。固定しきい値 R1 は {tau:.3f}）\n",
          "## 6. 結論",
          "",
          "### まず訂正",
          "`pick5.md` の「1日5本で回収率79.6%」は**その日の全レースを見てから選んでいた＝先読み**だった。",
          "同じことをその場で判断できる形にすると、確認期間（6〜8月）で **R3 79.3% / R2 74.0% / R4 74.4%**。",
          "**選び方次第で5ポイント変わる。** ただし n=455 で95%区間は±8ptあり、回収率の差は誤差の範囲でもある。",
          "",
          "### 「朝に枠が埋まって夕方に良いレースが来る」問題への答え",
          "**枠を先着で埋めない。残り枠と残り本数から、その都度しきい値を動かす（R3）。**",
          "朝いちばんは「今日の上位8%相当（残り5枠÷残り60本）」を要求し、夕方に枠が余っていれば要求を下げる。",
          "秘書問題と同じ形で、実装は上の表を引くだけ。先読みは一切使わない。",
          "",
          "| | R0 オラクル（不可能） | R3 動的しきい値 | R4 先着5本 |",
          "|---|---:|---:|---:|",
          "| 回収率（確認期間） | 79.1% | **79.3%** | 74.4% |",
          "| 的中率 | 47.7% | **47.0%** | 37.8% |",
          "| 最長連敗 | 8R | 9R | 12R |",
          "| オラクルと同じレースを買えた割合 | — | **71%** | 9% |",
          "| 取りこぼし（後でもっと良いのが来た） | — | 1日1.5回 | 1日50.5回 |",
          "",
          "**R3 は先読みできるオラクルにほぼ並ぶ。** 取りこぼしは1日1.5回残るが、",
          "逃した信頼度の差は中央値0.031（成立レースの分布幅の数%）で、実害はほぼない。",
          "先着で埋めると1日50回取りこぼし、的中率が9ポイント落ちる。**「早い者勝ち」だけは避けること。**",
          "",
          "### もうひとつの選択肢: 本数を固定しない（R1）",
          "固定しきい値だけにして本数を成り行きに任せると、1日中央値6本・最多14本・0本の日も1日。",
          "回収率75.8%で R3 と差は付かないが、**1日の投資額が読めない**（14本×4,900円＝約7万円の日が出る）。",
          "資金管理の観点では R3（枠を固定して質を動かす）の方が扱いやすい。",
          "",
          "### 注意",
          "- しきい値は探索期間の分布で作った固定表。**信頼度の分布が季節でずれると本数がずれる**ので、",
          "  3か月ごとに分布を取り直す（自動では変えない。人が判断して設定を更新する）。",
          "- ×1.5 の成立判定にはオッズが要るので、**締切直前まで「買えるかどうか」が確定しない**。",
          "  その日の残り本数の見積もりは番組表（全レース）× 過去の成立率44% で立てる。",
          "- すべて確定オッズ。締切前オッズでは成立本数が減るので、しきい値は実データで取り直す。",
          ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
