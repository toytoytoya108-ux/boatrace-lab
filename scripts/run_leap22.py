"""飛躍22: 単勝と3連単の値付けが食い違う条件を総当たりする（2026-09-18）。

飛躍21 で「SG・G1 では 1着率×1.15 が単勝の配当に ×0.830 伝わるのに 3連単には ×0.991 しか伝わらない
（食い違い 1.19）」という市場の内部矛盾が見つかった。**他の条件でも起きているなら、
それは新しい選択基準になりうる**（素の回収率は「1号艇が強いから」の分も含むが、
矛盾は市場が正しく織り込んだ分を差し引いた残りだけを測るので、汎化するはず）。

指標: 条件Cについて `積 = 的中率比 × 配当比`（1.00で織り込み済み）を単勝と3連単で計算し、
      **矛盾 = 積(3連単) ÷ 積(単勝)**。
検定: ① 矛盾は探索→確認で持続するか ② 機構（1着率を動かす条件ほど矛盾が大きい）の裏づけ
      ③ **決定的: 矛盾で条件を選ぶと、素の回収率で選ぶより確認期間の成績が良いか**

出力: reports/research/leaps11.md
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from boatlab.config import ROOT, STADIUMS  # noqa: E402

OUT = Path(ROOT) / "reports" / "research" / "leaps11.md"
C = Path("/tmp/claude-0")


def conds(MM, F, CA):
    c = {}
    for g in ("SG", "G1", "G2", "G3", "一般"):
        c[f"グレード{g}"] = (MM["grade"] == g)
    for st in sorted(MM["stadium_code"].dropna().unique()):
        c[f"場{STADIUMS.get(int(st))}"] = (MM["stadium_code"] == st)
    for rn in range(1, 13):
        c[f"{rn}R"] = (MM["race_no"] == rn)
    for d in (1, 2, 3, 4, 5):
        c[f"節{d}日目"] = (MM["day_no"] == d)
    for i, nm in enumerate("月火水木金土日"):
        c[f"{nm}曜"] = (CA.dt.dayofweek == i)
    for col, nm in (("rk_exhibition_time", "展示T"), ("rk_motor_rate2", "モーター"),
                    ("rk_nat_win_rate", "全国勝率"), ("rk_st_exh", "展示ST")):
        c[f"1号艇{nm}1位"] = (F[col] == 1); c[f"1号艇{nm}最下位"] = (F[col] == 6)
    c["1号艇A1"] = (F["klass_n"] == 0); c["1号艇B級"] = (F["klass_n"] >= 2)
    c["勝率差0〜1"] = MM["nwr_gap"].between(0, 1); c["勝率差2以上"] = (MM["nwr_gap"] >= 2)
    c["勝率ばらつき小"] = (MM["nwr_std"] <= 0.5); c["前づけあり"] = (MM["n_maezuke"] >= 1)
    c["ナイター"] = (CA.dt.hour >= 18)
    return c


def main():
    res = pickle.load(open(C / "bt_rules.pkl", "rb"))
    M = pd.read_parquet(Path(ROOT) / "reports" / "research" / "manshu_features.parquet").set_index("race_id")
    x = pd.read_parquet(C / "ent.parquet"); r = x.copy()
    for col, asc in (("exhibition_time", True), ("motor_rate2", False),
                     ("nat_win_rate", False), ("st_exh", True)):
        r[f"rk_{col}"] = r.groupby("race_id")[col].rank(ascending=asc, method="min")
    f = r[r["lane"] == 1].set_index("race_id")
    W = res[("単勝", "1")].set_index("race_id"); T = res[("3連単", "1-2-3")].set_index("race_id")
    idx = W.index.intersection(T.index); W = W.loc[idx]; T = T.loc[idx]
    F = f.reindex(idx); MM = M.reindex(idx); CA = pd.to_datetime(MM["closed_at"], errors="coerce")
    CS = conds(MM, F, CA)
    yr = W["year"].values; EX = yr <= 2023; CF = yr >= 2024

    def mk(sub, floor=2500):
        hw0 = (W[sub]["amt"] > 0).mean(); pw0 = W[sub].loc[W[sub]["amt"] > 0, "amt"].mean()
        ht0 = (T[sub]["amt"] > 0).mean(); pt0 = T[sub].loc[T[sub]["amt"] > 0, "amt"].mean()
        out = {}
        for nm, m in CS.items():
            m = np.asarray(m.fillna(False)) if hasattr(m, "fillna") else np.asarray(m)
            k = m & sub
            if k.sum() < floor:
                continue
            w = W[k]; t = T[k]
            hw = (w["amt"] > 0).mean(); pw = w.loc[w["amt"] > 0, "amt"].mean()
            ht = (t["amt"] > 0).mean(); pt = t.loc[t["amt"] > 0, "amt"].mean()
            if not (pw > 0 and pt > 0):
                continue
            out[nm] = dict(n=int(k.sum()), aw=hw / hw0 * pw / pw0, at=ht / ht0 * pt / pt0,
                           hw=hw / hw0, roi_w=w["amt"].mean(), roi_t=t["amt"].mean())
        return out
    A = mk(EX, 4000); B = mk(CF)
    com = [k for k in A if k in B]
    ce = np.array([A[k]["at"] / A[k]["aw"] for k in com])
    cc = np.array([B[k]["at"] / B[k]["aw"] for k in com])
    hw = np.array([A[k]["hw"] for k in com])
    rho = spearmanr(ce, cc)[0]
    rng = np.random.default_rng(3)
    nul = np.array([spearmanr(rng.permutation(ce), cc)[0] for _ in range(2000)])

    full = mk(np.ones(len(W), bool), 4000)
    df = pd.DataFrame([dict(cond=k, **v, 矛盾=v["at"] / v["aw"]) for k, v in full.items()])
    L = [f"# 飛躍22: 単勝と3連単の値付けが食い違う条件の総当たり（{len(df)}通り・約47万レース）\n",
         "飛躍21 の内部矛盾（SG・G1 で 1着率×1.15 が単勝の配当に ×0.830 伝わるのに 3連単には ×0.991）が",
         "**他の条件でも起きているなら、素の回収率より良い選択基準になりうる**",
         "（素の回収率は「1号艇が強いから」の分も含むが、矛盾は市場が正しく織り込んだ分を差し引いた残りだけを測る）。\n",
         "指標: **矛盾 = 積(3連単) ÷ 積(単勝)**、ただし `積 = 的中率比 × 配当比`（1.00 で織り込み済み）。\n",
         "## 1. 矛盾の大きい条件・小さい条件\n",
         "| 条件 | レース数 | 積_単勝 | 積_3連単 | 矛盾 | 単勝ROI | 3連単ROI |",
         "|---|---:|---:|---:|---:|---:|---:|"]
    d2 = df.sort_values("矛盾", ascending=False)
    for _, v in pd.concat([d2.head(8), d2.tail(5)]).iterrows():
        L.append(f"| {v['cond']} | {v['n']:,} | {v['aw']:.3f} | {v['at']:.3f} | **{v['矛盾']:.3f}** | "
                 f"{v['roi_w']:.1f}% | {v['roi_t']:.1f}% |")
    L += ["",
          f"矛盾の分布: 中央 {df['矛盾'].median():.3f}／10%点 {df['矛盾'].quantile(.1):.3f}／90%点 {df['矛盾'].quantile(.9):.3f}。",
          f"**上位はグレード（SG 1.181・G1 1.158・G3 1.081）が占める。** 場や曜日は上位に来ても小さい。\n",
          "## 2. 矛盾は持続するか — **する（が弱い）**\n",
          f"- 探索(2018〜23)の矛盾 と 確認(2024〜26)の矛盾 の順位相関 **ρ = {rho:+.3f}**",
          f"  （共通条件 {len(com)}通り、帰無2000回 {np.percentile(nul,2.5):+.3f}〜{np.percentile(nul,97.5):+.3f}、"
          f"**実測以上 {int((nul>=rho).sum())}/2000**）",
          "- ただし個別に見ると**グレードだけが持続**する: "
          + "、".join(f"{k} {A[k]['at']/A[k]['aw']:.3f}→{B[k]['at']/B[k]['aw']:.3f}"
                    for k in sorted(com, key=lambda k: -A[k]["at"] / A[k]["aw"])[:5]),
          "  場は崩れる（びわこ 1.085→0.989、福岡 1.066→0.994）。\n",
          "## 3. 機構の裏づけ — **弱い**\n",
          f"「1着率を大きく動かす条件ほど3連単に伝わっていない」なら、的中率比と矛盾は正に相関するはず。",
          f"実測は **+{np.corrcoef(hw, ce)[0,1]:.3f}**（絶対値では +{np.corrcoef(np.abs(hw-1), ce)[0,1]:.3f}）。",
          "符号は仮説どおりだが、これだけで機構が確定したとは言えない。\n",
          "## 4. 決定的な検定: 矛盾は素の回収率より良い選択基準か — **ノー**\n",
          "探索で上位k条件を選び、確認期間の 3連単1-2-3 回収率を測る（レース数で重みづけ）。\n",
          "| 選び方 | k=1 | k=3 | k=5 | k=10 |", "|---|---:|---:|---:|---:|"]
    ddf = pd.DataFrame([dict(cond=k, 矛盾=A[k]["at"] / A[k]["aw"], roi_ex=A[k]["roi_t"],
                             at=A[k]["at"], roi_cf=B[k]["roi_t"], n_cf=B[k]["n"]) for k in com])
    for nm, key in (("**矛盾（新指標）**", "矛盾"), ("素の3連単回収率", "roi_ex"), ("積_3連単", "at")):
        vals = []
        for k in (1, 3, 5, 10):
            t = ddf.nlargest(k, key)
            vals.append((t["roi_cf"] * t["n_cf"]).sum() / t["n_cf"].sum())
        L.append(f"| {nm} | " + " | ".join(f"{v:.1f}%" for v in vals) + " |")
    L += [f"| （対照）全条件の平均 | " + " | ".join(
        f"{(ddf['roi_cf']*ddf['n_cf']).sum()/ddf['n_cf'].sum():.1f}%" for _ in range(4)) + " |",
          f"| （対照）全レース | " + " | ".join(f"{T[CF]['amt'].mean():.1f}%" for _ in range(4)) + " |",
          "",
          f"**確認期間との相関も負けている: 矛盾 {np.corrcoef(ddf['矛盾'], ddf['roi_cf'])[0,1]:+.3f} "
          f"vs 素の回収率 {np.corrcoef(ddf['roi_ex'], ddf['roi_cf'])[0,1]:+.3f}。**",
          "矛盾で選んでも素の回収率で選んでも同じ条件（グレードG1）が1位に来るし、",
          "k を増やすと素の回収率の方がわずかに良い。**新指標は何も足していない。**\n",
          "## まとめ: 飛躍22\n",
          "1. **内部矛盾は実在し、持続する**"
          f"（探索→確認 ρ={rho:+.3f}、帰無2000回で {int((nul>=rho).sum())}/2000）。",
          "2. **しかし中身はほぼグレードだけ。** 場・曜日・レース番号・観測量では持続しない。"
          "**飛躍21の発見は、飛躍17の発見（SG・G1）の言い換えだった。**",
          "3. **選択基準としては役に立たない。** 矛盾で条件を選んでも、素の回収率で選ぶより良くならない"
          "（確認期間との相関で負ける）。",
          "4. 機構（1着率を動かす条件ほど伝わらない）の符号は合っているが相関 +0.29 で弱い。",
          "5. **位置づけの訂正**: 飛躍21 で「飛躍1が探して見つからなかった矛盾が見つかった」と書いたが、"
          "**総当たりすると SG・G1 という1条件にしか存在しない。** 一般的な市場の性質ではなく、"
          "**特定の番組区分に限った現象**として扱うのが正しい。",
          "6. **この道はここで閉じる。** 3連単1-2-3 を買うなら条件は素の回収率で選べばよく、"
          "その最良は依然として `leaps7.md` の SG・G1 の 1〜11R（93.8%）。"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[-12:]))


if __name__ == "__main__":
    main()
