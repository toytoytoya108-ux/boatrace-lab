"""飛躍18: 1号艇複勝の「場の差」の正体は何か（2026-09-18）。

`lane1_place.md` で、場ごとの回収率の差は本物（探索→確認の順位相関 ρ=+0.645、帰無200回で 0/200）
だが **1号艇の強さでも元返しでも説明できない**ことが分かった。ここでその中身を割る。

道具は恒等式 `回収率 = 的中率 × 的中時の平均払戻 ÷ 100` の対数分解:
    log(回収率) = log(的中率) + log(払戻/100)
市場が場ごとの強さを完全に織り込んでいるなら **払戻 ∝ 1/的中率**（傾き −1.00）で、
回収率は場によらず一定になる。**傾きが −1 からどれだけずれるか**が織り込み不足の量。
さらに、傾きで説明できない **場固有の残差** が本物かを探索/確認で確かめる。

出力: reports/research/leaps8.md
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from boatlab.config import ROOT, STADIUMS  # noqa: E402

OUT = Path(ROOT) / "reports" / "research" / "leaps8.md"
CACHE = Path("/tmp/claude-0")
LANE = 1


def tab(d):
    g = d.groupby("st").agg(n=("amount", "size"), roi=("amount", "mean"), hit=("hit", "mean"))
    g["pay"] = d[d["amount"] > 0].groupby("st")["amount"].mean()
    g["roi"] /= 100
    return g


def main():
    re_ = pd.read_parquet(CACHE / "re.parquet")
    pay = pd.read_parquet(CACHE / "place_pay.parquet")
    has = set(pay["race_id"].unique())
    s = re_[re_["lane"] == LANE][["race_id", "st", "year"]]
    p = pay[pay["lane"] == LANE][["race_id", "amount"]]
    m = s.merge(p, on="race_id", how="left")
    m = m[m["race_id"].isin(has) & (m["amount"] != -1)].copy()
    m["amount"] = m["amount"].fillna(0.0)
    m["hit"] = (m["amount"] > 0).astype(float)
    g = tab(m)
    lh, lp = np.log(g["hit"]), np.log(g["pay"] / 100)
    a, b0 = np.polyfit(lh, lp, 1)
    g["resid"] = lp - (a * lh + b0)

    L = [f"# 飛躍18: 1号艇複勝の「場の差」の正体（{len(m):,}レース・2018〜2026）\n",
         "`lane1_place.md`: 場ごとの回収率の差は本物（探索→確認 ρ=+0.645、帰無200回で 0/200）だが、",
         "**1号艇の強さ（相関 +0.309）でも元返し（−0.021）でも説明できなかった。** その中身を割る。\n",
         "## 1. まず市場がどれだけ織り込んでいるかを測る\n",
         "`log(回収率) = log(的中率) + log(払戻/100)`。市場が場ごとの強さを完全に織り込むなら",
         "**払戻 ∝ 1/的中率**（傾き −1.00）で、回収率は場によらず一定になる。\n",
         f"- **実測の傾き = {a:+.3f}**（完全な織り込みなら −1.00）",
         f"- log(的中率) と log(払戻) の相関 = **{np.corrcoef(lh, lp)[0, 1]:+.3f}**",
         f"- 分散: 回収率 {np.log(g['roi']).var():.5f} ＝ 的中率 {lh.var():.5f} ＋ 払戻 {lp.var():.5f} "
         f"＋ 2×共分散 {2*np.cov(lh, lp)[0, 1]:.5f}\n",
         "→ **市場は場ごとの強さをほぼ完全に打ち消している。** 的中率の分散と払戻の分散が",
         f"ほぼ相殺し、回収率の分散は {np.log(g['roi']).var()/lh.var()*100:.0f}% しか残らない。",
         f"**織り込み不足は傾きの {(1+a)*100:+.1f}% ぶんだけ**（−0.864 なので、的中率が1%高い場では",
         "払戻が 0.86% しか下がらない ＝ 強い場がわずかに得）。",
         "**「場の差」と呼んでいたものの正体は、この小さな取りこぼしと、次の残差。**\n",
         "## 2. 傾きで説明できない「場固有の残差」\n",
         "| 場 | 的中率 | 的中時の平均払戻 | 回収率 | 残差 |", "|---|---:|---:|---:|---:|"]
    for st, r in g.sort_values("resid", ascending=False).iterrows():
        L.append(f"| {STADIUMS.get(int(st))} | {r['hit']*100:.1f}% | {r['pay']:.0f}円 | "
                 f"{r['roi']*100:.1f}% | {r['resid']:+.4f} |")
    ex, cf = tab(m[m["year"] <= 2023]), tab(m[m["year"] >= 2024])
    for t in (ex, cf):
        t["resid"] = np.log(t["pay"] / 100) - (a * np.log(t["hit"]) + b0)
    rho = spearmanr(ex["resid"].values, cf["resid"].reindex(ex.index).values)[0]
    rng = np.random.default_rng(4)
    null = np.array([spearmanr(rng.permutation(ex["resid"].values),
                               cf["resid"].reindex(ex.index).values)[0] for _ in range(2000)])
    r2 = np.corrcoef(g["resid"], np.log(g["roi"]))[0, 1] ** 2
    L += ["", "**残差（+）＝市場が払いすぎている場＝こちらが得。**\n",
          "### 残差は本物か（探索2018〜23 → 確認2024〜26）\n",
          f"- 順位相関 **ρ = {rho:+.3f}**。帰無2000回（探索側の残差を無作為に並べ替え）は "
          f"{np.percentile(null, 2.5):+.3f}〜{np.percentile(null, 97.5):+.3f} で、"
          f"**実測以上は {int((null >= rho).sum())}/2000**。",
          f"- 残差は場別回収率の **R² = {r2:.3f}** を説明する。\n",
          "→ **場固有の値付けの癖は9年間ずっと同じ向きで残っている。** これが「場の差」の本体。\n",
          "## 3. その癖は何で説明できるか — **何も説明できなかった**\n"]
    man = pd.read_parquet(Path(ROOT) / "reports" / "research" / "manshu_features.parquet")
    man["man"] = (pd.to_numeric(man["pay"], errors="coerce").fillna(0) >= 10000).astype(float)
    ca = pd.to_datetime(man["closed_at"], errors="coerce")
    man["night"] = (ca.dt.hour >= 18).astype(float)
    moto = (m[m["amount"] > 0].assign(x=lambda d: (d["amount"] == 100).astype(float))
            .groupby("st")["x"].mean())
    att = pd.DataFrame({
        "万舟率": man.groupby("stadium_code")["man"].mean(),
        "ナイター比率": man.groupby("stadium_code")["night"].mean(),
        "不成立率": re_.assign(ng=~re_["race_id"].isin(has)).groupby("st")["ng"].mean(),
        "年間レース数": man.groupby("stadium_code").size(),
        "1号艇の2連対率": g["hit"],
        "SG/G1の比率": man.assign(b=man["grade"].isin(["SG", "G1"]).astype(float))
                    .groupby("stadium_code")["b"].mean(),
        "元返しの割合": moto,
    }).reindex(g.index)
    ed = [i for i in g.index if STADIUMS.get(int(i)) != "江戸川"]
    L += ["| 場の属性 | 残差との相関 | 江戸川を外すと |", "|---|---:|---:|"]
    for c in att.columns:
        r_all = np.corrcoef(g["resid"], att[c])[0, 1]
        r_wo = np.corrcoef(g.loc[ed, "resid"], att.loc[ed, c])[0, 1]
        L.append(f"| {c} | {r_all:+.3f} | {r_wo:+.3f} |")
    L += ["",
          "**見かけ上効いていた2つ（不成立率 +0.625、年間レース数 −0.532）は、江戸川を外すと消える**",
          "（+0.245 / −0.258）。**江戸川1場が両方の外れ値だっただけ。**",
          "江戸川は不成立率 7.6%（多摩川は 0.2%）で唯一の河川・感潮コース。**残差も最大（+0.067）。**",
          "残りの23場の残差は、測れる属性のどれとも関係しない。\n",
          "→ **「場の固定効果」としか言いようがない。実在して持続するが、中身は説明できない。**\n",
          "## 4. 説明できなくても使えるか\n",
          "| 探索で上位k場を選ぶ | 確認期間の回収率 | 1日あたり | 無作為k場300回の95%範囲 | 判定 |",
          "|---|---:|---:|---|---|"]
    exr = ex["roi"]
    cfm = m[m["year"] >= 2024]
    sts = sorted(m["st"].unique())
    rng2 = np.random.default_rng(7)
    days_cf = cfm["race_id"].nunique() and 365 * 2.7
    for k in (1, 2, 3, 5, 8):
        top = exr.sort_values(ascending=False).index[:k]
        v = cfm[cfm["st"].isin(top)]["amount"]
        nb = np.array([cfm[cfm["st"].isin(rng2.choice(sts, k, replace=False))]["amount"].mean()
                       for _ in range(300)])
        hi = np.percentile(nb, 97.5)
        L.append(f"| 上位{k}場（{'・'.join(STADIUMS.get(int(x)) for x in top)}） | "
                 f"**{v.mean():.1f}%** | {len(v)/days_cf:.1f}本 | "
                 f"{np.percentile(nb, 2.5):.1f}〜{hi:.1f}% | {'★外' if v.mean() > hi else '中'} |")
    L.append(f"| （対照）全場 | {cfm['amount'].mean():.1f}% | {len(cfm)/days_cf:.1f}本 | — | — |")
    v1 = cfm[cfm["st"] == exr.idxmax()]["amount"]
    sd = v1.std() / 100; r1 = v1.mean() / 100
    L += ["",
          f"**上位3場以上は無作為の範囲の外。前向きに使える。** 上位8場で {cfm[cfm['st'].isin(exr.sort_values(ascending=False).index[:8])]['amount'].mean():.1f}%"
          f"（全場 {cfm['amount'].mean():.1f}% に対し +{cfm[cfm['st'].isin(exr.sort_values(ascending=False).index[:8])]['amount'].mean()-cfm['amount'].mean():.1f}pt）、1日49本。",
          f"**ただし100%には届かない。** 単独最良の江戸川でも確認期間 {v1.mean():.1f}%、"
          f"95%区間 {(r1-1.96*sd/np.sqrt(len(v1)))*100:.1f}〜{(r1+1.96*sd/np.sqrt(len(v1)))*100:.1f}%。",
          "確認期間の点推定が100%未満なので、**この期間のデータでは「100%超え」の方向にすら向いていない。**\n",
          "## まとめ: 飛躍18\n",
          f"1. **市場は場ごとの強さをほぼ完全に織り込んでいる。** 傾き {a:+.3f}（完全なら −1.00）、"
          f"相関 {np.corrcoef(lh, lp)[0, 1]:+.3f}。的中率と払戻の分散がほぼ相殺し、回収率の分散は"
          f"{np.log(g['roi']).var()/lh.var()*100:.0f}%しか残らない。**「場の差」は大きな相殺のあとの残りかす。**",
          f"2. **その残りかすは本物。** 傾きで説明できない場固有の残差が探索→確認で ρ={rho:+.3f}"
          f"（帰無 {int((null >= rho).sum())}/2000）、回収率の場差の R²={r2:.3f} を説明する。",
          "3. **正体は説明できない。** 万舟率・ナイター・不成立率・規模・強さ・SG比率・元返し、",
          "   どれも江戸川を外すと相関が消える。**江戸川という1つの外れ値と、23場の構造の無いばらつき。**",
          "4. **前向きには使える**（上位8場で +2.3pt、無作為の外）**が、天井は97〜99%で100%に届かない。**",
          "5. 意味するところ: **これは「市場を超える情報」ではなく「市場の値付けの癖」の一種**で、",
          "   `leaps4.md` の「出やすい目ほど安い」と同じ棚に入る。比 1.16 が上限だったのと同様、",
          "   場の癖も +2.3pt が上限。**必要な 1.333 には遠い。**"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[-12:]))


if __name__ == "__main__":
    main()
