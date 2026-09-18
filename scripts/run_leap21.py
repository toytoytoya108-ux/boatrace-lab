"""飛躍21: 券種の総当たりで「どの市場が間違っているか」を突き止める（2026-09-18）。

飛躍17 で「SG・G1 の 1〜11R で 1-2-3 が 93.8%」が出た。飛躍19 で「観測量は1号艇を絞るのに効く」が出た。
**同じ発見が券種をまたいでどう現れるかを見れば、市場のどこが間違っているかが分かる。**

全7券種の確定配当（`results.payouts`）と結果だけで計算する。オッズ不要・477,679レース。
物差しは `回収率 = 的中率 × 的中時の平均配当 ÷ 100` と、その比の分解
`積 = 的中率比 × 配当比`（1.00 なら市場が完全に織り込んでいる）。

出力: reports/research/leaps10.md
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from boatlab.config import ROOT  # noqa: E402

OUT = Path(ROOT) / "reports" / "research" / "leaps10.md"
C = Path("/tmp/claude-0")
RULES = [("複勝", "place", "1", {1}), ("複勝", "place", "2", {2}), ("単勝", "win", "1", {1}),
         ("拡連複", "quinella_place", "1=2", {1, 2}), ("拡連複", "quinella_place", "1=3", {1, 3}),
         ("拡連複", "quinella_place", "2=3", {2, 3}),
         ("2連複", "quinella", "1=2", {1, 2}), ("2連複", "quinella", "1=3", {1, 3}),
         ("2連単", "exacta", "1-2", {1, 2}), ("2連単", "exacta", "1-3", {1, 3}),
         ("3連複", "trio", "1=2=3", {1, 2, 3}), ("3連複", "trio", "1=2=4", {1, 2, 4}),
         ("3連単", "trifecta", "1-2-3", {1, 2, 3}), ("3連単", "trifecta", "1-3-2", {1, 3, 2})]


def build():
    cache = C / "bt_rules.pkl"
    if cache.exists():
        return pickle.load(open(cache, "rb"))
    pay = pd.read_parquet(C / "allpay.parquet")
    rf = pd.read_parquet(C / "refund.parquet")
    races = pd.read_parquet(C / "re.parquet")[["race_id", "year"]].drop_duplicates()
    valid = set(pay.loc[pay["bt"] == "place", "race_id"].unique())
    races = races[races["race_id"].isin(valid)]
    reflanes = rf.groupby("race_id")["lane"].apply(set)
    res = {}
    for nm, bt, combo, lanes in RULES:
        p = pay[(pay["bt"] == bt) & (pay["combo"] == combo)][["race_id", "amt"]]
        d = races.merge(p, on="race_id", how="left")
        d["amt"] = d["amt"].fillna(0.0)
        bad = d["race_id"].map(reflanes).apply(
            lambda s: bool(s) and bool(s & lanes) if isinstance(s, set) else False)
        res[(nm, combo)] = d[~bad]
    pickle.dump(res, open(cache, "wb"))
    return res


def ci(x):
    x = np.asarray(x, float)
    return len(x), x.mean(), 1.96 * x.std() / np.sqrt(max(len(x), 1))


def main():
    res = build()
    man = pd.read_parquet(Path(ROOT) / "reports" / "research" / "manshu_features.parquet")[
        ["race_id", "grade", "race_no"]].set_index("race_id")
    BIG = set(man[man["grade"].isin(["SG", "G1"]) & (man["race_no"] <= 11)].index)
    GEN = set(man[man["grade"].eq("一般") & (man["race_no"] <= 11)].index)
    x = pd.read_parquet(C / "ent.parquet")
    r = x.copy()
    for c, asc in (("exhibition_time", True), ("motor_rate2", False)):
        r[f"rk_{c}"] = r.groupby("race_id")[c].rank(ascending=asc, method="min")
    f = r[r["lane"] == 1].set_index("race_id")[["rk_exhibition_time", "rk_motor_rate2"]]
    FILT = set(f[(f["rk_motor_rate2"] == 1) & (f["rk_exhibition_time"] == 1)].index)

    L = ["# 飛躍21: 券種の総当たりで「どの市場が間違っているか」を突き止める（2018〜2026）\n",
         "全7券種の確定配当と結果だけで計算（**オッズ不要・約47万レース**）。",
         "物差しは `回収率 = 的中率 × 的中時の平均配当 ÷ 100` と、その比の分解 `積 = 的中率比 × 配当比`",
         "（**1.00 なら市場が完全に織り込んでいる。>1 は取りこぼし、<1 は過剰反応**）。\n",
         "## 1. 素の固定ルール — 順位は元返し率の順位\n",
         "| 券種 | 買い目 | レース数 | 的中率 | 的中時平均 | 元返しの割合 | 回収率 | ±95% |",
         "|---|---|---:|---:|---:|---:|---:|---:|"]
    moto, roi = [], []
    for (nm, combo), d in res.items():
        n, mu, h = ci(d["amt"].values)
        w = d.loc[d["amt"] > 0, "amt"]
        moto.append((w == 100).mean()); roi.append(mu)
        L.append(f"| {nm} | {combo} | {n:,} | {(d['amt']>0).mean()*100:.1f}% | {w.mean():.0f}円 | "
                 f"{(w==100).mean()*100:.1f}% | **{mu:.1f}%** | ±{h:.2f} |")
    L += ["",
          f"**素の回収率と元返し率の相関 = {np.corrcoef(moto, roi)[0,1]:+.3f}。**",
          "`sweet_spot.md` の「元返し（100円下限）は胴元からの補助金」を、14の固定ルールで定量化した。",
          "**券種の優劣はほぼ元返し率で決まる**（複勝1号艇 46.3% → 94.5%、3連単 0% → 82.6%）。\n",
          "## 2. 飛躍19のフィルタ（1号艇がモーター2連率1位かつ展示タイム1位）を全券種にかける\n",
          "| 券種 | 買い目 | 素 | フィルタ後 | 差 | ±95% |", "|---|---|---:|---:|---:|---:|"]
    gains = []
    for (nm, combo), d in res.items():
        g = d[d["race_id"].isin(FILT)]
        n, mu, h = ci(g["amt"].values)
        if n < 3000:
            continue
        gains.append((np.mean((d.loc[d["amt"] > 0, "amt"] == 100)), mu - d["amt"].mean()))
        L.append(f"| {nm} | {combo} | {d['amt'].mean():.1f}% | **{mu:.1f}%** | "
                 f"{mu-d['amt'].mean():+.1f}pt | ±{h:.2f} |")
    ga = np.array(gains)
    L += ["",
          f"**上げ幅と元返し率の相関 = {np.corrcoef(ga[:,0], ga[:,1])[0,1]:+.3f}（ほぼ無関係）。**",
          "予想と違い、フィルタの効きは元返しの領域に集中していない。全券種に +1.6〜+8.1pt で広く効く。",
          "**最良は 複勝1号艇 98.2%（±0.82）と 単勝1号艇 98.2%（±1.26）で並ぶ。**",
          "単勝は素で 91.1% だったのが +7.1pt 上がる（元返しが 17.3%→22.9% に増える寄与もある）。\n",
          "## 3. 飛躍17の SG・G1 を全券種にかける — **ここで市場が割れた**\n",
          "| 券種 | 買い目 | 一般戦 1〜11R | SG・G1 1〜11R | 的中率比 | 配当比 | 積 | 判定 |",
          "|---|---|---:|---:|---:|---:|---:|---|"]
    for (nm, combo), d in res.items():
        a = d[d["race_id"].isin(GEN)]; b = d[d["race_id"].isin(BIG)]
        if len(b) < 3000:
            continue
        ha = (a["amt"] > 0).mean(); hb = (b["amt"] > 0).mean()
        pa = a.loc[a["amt"] > 0, "amt"].mean(); pb = b.loc[b["amt"] > 0, "amt"].mean()
        prod = hb / ha * pb / pa
        v = "**過剰反応**" if prod < 0.99 else ("**取りこぼし**" if prod > 1.01 else "織り込み済み")
        L.append(f"| {nm} | {combo} | {a['amt'].mean():.1f}% | **{b['amt'].mean():.1f}%** | "
                 f"{hb/ha:.3f} | {pb/pa:.3f} | **{prod:.3f}** | {v} |")
    L += ["",
          "**1号艇の単独成績を買う券種（複勝・単勝・拡連複1=X）は全部「過剰反応」（積 0.96〜0.97）。**",
          "**並びを当てる券種（3連単・3連複・2連単）は全部「取りこぼし」（積 1.01〜1.15）。**",
          "同じレース・同じ情報なのに、券種によって市場の誤りの符号が逆になる。\n",
          "### 「番組の席次が並びに出る」という説明は誤りだった\n"]
    re_ = pd.read_parquet(C / "re.parquet")
    p = re_.pivot_table(index="race_id", columns="lane", values="finish_pos")
    L += ["| 区分 | 1号艇が1着 | その中で2号艇が2着 | さらに3号艇が3着 |", "|---|---:|---:|---:|"]
    for nm, S in (("一般戦 1〜11R", GEN), ("SG・G1 1〜11R", BIG)):
        q = p[p.index.isin(S)]; w1 = q[q[1] == 1]
        L.append(f"| {nm} | {len(w1)/len(q)*100:.1f}% | {(w1[2]==2).mean()*100:.1f}% | "
                 f"{((w1[2]==2)&(w1[3]==3)).mean()*100:.1f}% |")
    L += ["",
          "**1号艇が勝った条件つきでの並びは、一般戦とSG・G1でまったく同じ**（34.1%→34.5%、12.9%→12.9%）。",
          "1-2-3 の出現率の上昇（×1.159）は **ほぼ全部 1着率の上昇（×1.154）で説明できる**。",
          "つまり SG・G1 に特別な「並びの構造」は無い。\n",
          "### では何が起きているのか — **市場の内部矛盾**\n",
          "同じ「1号艇の1着率が ×1.15 になる」という一つの事実に対し、市場の反応が券種で食い違う。\n",
          "| 券種 | 的中率比 | 配当比 | どう反応したか |", "|---|---:|---:|---|",
          "| 単勝 1号艇 | 1.154 | **0.830** | 1着率の上昇より**大きく**配当を下げた（過剰） |",
          "| 3連単 1-2-3 | 1.159 | **0.991** | 1着率が15%上がったのに配当を**1%しか**下げていない |",
          "",
          "**配当比の食い違いは 0.991 ÷ 0.830 = 1.19。約20%の内部矛盾。**",
          "群衆は「SG・G1 では1号艇が強い」を単勝の値段には入れ（むしろ入れすぎ）、",
          "**3連単の値段には伝えていない。** 同じ情報が、同じレースの別の窓口で違う値段になっている。\n",
          "`leap_self.md`（飛躍1）は「市場の内部矛盾」を PL 構造で探して**否定された**（矛盾は雑音でなく知識だった）。",
          "**ここで初めて、条件つきで本物の内部矛盾が見つかった。** そして飛躍15（誤りは後ろの位置ほど大きい）と",
          "同じ向き: 市場は1着の値付けは上手いが、順序付きの買い目にそれを反映しきれない。\n",
          "## まとめ: 飛躍21\n",
          f"1. **券種の優劣は元返し率で決まる**（相関 {np.corrcoef(moto, roi)[0,1]:+.3f}）。"
          "`sweet_spot.md` の補助金の話を14の固定ルールで定量化した。",
          "2. **飛躍19のフィルタは全券種に広く効く**（+1.6〜+8.1pt、元返し率とは無関係）。"
          "**単勝1号艇が 91.1%→98.2% で複勝に並ぶ**のが新しい。",
          "3. **SG・G1 では市場の誤りの符号が券種で逆になる。** 1号艇単独は過剰反応（積0.96〜0.97）、"
          "並びは取りこぼし（積1.01〜1.15）。",
          "4. **正体は『席次が並びに出る』ではなく『市場の内部矛盾』。** 1着率の上昇は単勝に伝わって"
          "いるが3連単に伝わっていない（配当比の食い違い 1.19）。**飛躍1が探して見つからなかった矛盾が、"
          "条件つきで見つかった。**",
          "5. **それでも最良は 98.2%**（複勝／単勝1号艇＋フィルタ）で100%に届かない。"
          "SG・G1 の 3連単1-2-3 は 93.8%。**矛盾は実在するが、取り出せる量は控除率に足りない。**"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[-12:]))


if __name__ == "__main__":
    main()
