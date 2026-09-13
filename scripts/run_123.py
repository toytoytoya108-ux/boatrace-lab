"""1-2-3 はどこでいちばん安いか。2018〜2023で探索 → 2024〜2026で1回だけ確認。

`leaps3.md` で「毎レース 1-2-3 を買う」の回収率が 9年連続 79〜85%（全期間82.9%、帰無75%）と分かった。
ここでは**条件で絞ると比がどこまで上がるか**を測る。

物差し: `回収率 = (1-2-3 が来た確率) × (そのときの平均配当) ÷ 100`。
  → **結果と公式配当だけで計算できる**ので、オッズのある2026年に縛られず46万レース全部が使える。

手順（winner's curse 対策）:
  探索 = 2018〜2023（32万R）で候補を総当たり。**確認 = 2024〜2026（14万R）で、選んだものだけを1回評価。**
  採用の条件は事前に決める: 探索で n ≥ 30,000 かつ回収率が最大のもの（1つ）と、2変数の組（1つ）。

出力: reports/research/combo123.md
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from boatlab.config import ROOT

OUT = Path(ROOT) / "reports" / "research" / "combo123.md"
FEAT = Path(ROOT) / "reports" / "research" / "manshu_features.parquet"
MIN_N = 30000


def roi(d):
    """1-2-3 を毎レース100円買ったときの回収率。"""
    if len(d) == 0:
        return np.nan
    return float((d["is123"] * d["pay"]).mean() / 100.0)


def ci(d, n_boot=200, seed=0):
    if len(d) < 100:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    v = (d["is123"] * d["pay"]).values / 100.0
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    b = v[idx].mean(1)
    return float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))


def main():
    df = pd.read_parquet(FEAT)
    df["is123"] = (df["trifecta"] == "1-2-3").astype(float)
    df["pay"] = pd.to_numeric(df["pay"], errors="coerce").fillna(0.0)
    ex = df[df["year"] <= 2023]
    cf = df[df["year"] >= 2024]
    L = [f"# 1-2-3 はどこでいちばん安いか（2018〜2026・{len(df):,}レース）\n",
         "`回収率 = 1-2-3 が来た確率 × そのときの平均配当 ÷ 100`。**結果と公式配当だけで計算でき、オッズは要らない。**",
         f"**探索 = 2018〜2023（{len(ex):,}R）／確認 = 2024〜2026（{len(cf):,}R）。確認は選んだものだけ1回。**",
         f"帰無（市場が正しい）は 75%。全体では 探索 {roi(ex)*100:.1f}% / 確認 {roi(cf)*100:.1f}%。\n",
         "## 1. 探索（2018〜2023）: 条件ごとの回収率\n",
         "太字は n ≥ 30,000 かつ探索の回収率が全体を3pt以上上回るもの。\n"]

    def band(s, edges, labels):
        return pd.cut(s, edges, labels=labels, include_lowest=True)

    CONDS = {}
    CONDS["1号艇の全国勝率"] = band(df["l1_nwr"], [-1, 4.5, 5.5, 6.5, 7.5, 99], ["4.5未満", "4.5〜5.5", "5.5〜6.5", "6.5〜7.5", "7.5以上"])
    CONDS["1号艇と他艇最高の勝率差"] = band(df["nwr_gap"], [-99, -1, 0, 1, 2, 99], ["−1未満", "−1〜0", "0〜1", "1〜2", "2以上"])
    CONDS["6艇の勝率のばらつき"] = band(df["nwr_std"], [-1, 0.5, 0.9, 1.3, 99], ["0.5未満（拮抗）", "0.5〜0.9", "0.9〜1.3", "1.3以上（差がある）"])
    CONDS["1号艇の級別"] = df["l1_klass_n"].map({0: "B2", 1: "B1", 2: "A2", 3: "A1"})
    CONDS["1号艇の展示タイム順位"] = band(df["l1_ext_rank"], [0, 1, 2, 3, 6], ["1位", "2位", "3位", "4位以下"])
    CONDS["A1級の人数"] = band(df["n_a1"], [-1, 0, 1, 2, 6], ["0人", "1人", "2人", "3人以上"])
    CONDS["風速"] = band(df["ws"], [-1, 1, 3, 5, 7, 99], ["1m以下", "1〜3m", "3〜5m", "5〜7m", "7m以上"])
    CONDS["波高"] = band(df["wave"], [-1, 1, 3, 5, 99], ["1cm以下", "1〜3cm", "3〜5cm", "5cm以上"])
    CONDS["前づけ"] = band(df["n_maezuke"], [-1, 0, 1, 6], ["なし", "1人", "2人以上"])
    CONDS["グレード"] = df["grade"]
    CONDS["ナイター"] = df["night"].map({0: "昼", 1: "ナイター"})
    CONDS["レース番号"] = band(df["race_no"], [0, 3, 6, 9, 12], ["1〜3R", "4〜6R", "7〜9R", "10〜12R"])
    CONDS["節の何日目"] = band(df["day_no"], [0, 1, 2, 3, 99], ["初日", "2日目", "3日目", "4日目以降"])
    CONDS["レース場"] = df["stadium"]
    base_ex = roi(ex)
    rows = []
    for nm, col in CONDS.items():
        L += [f"\n**{nm}**\n", "| 区分 | 探索 レース数 | 1-2-3率 | 平均配当 | 探索 回収率 |", "|---|---:|---:|---:|---:|"]
        for v in pd.Series(col).dropna().unique():
            m = (col == v).values
            e = ex[m[:len(ex)] if len(m) == len(df) else m]
            e = df[m & (df["year"] <= 2023)]
            if len(e) < 2000:
                continue
            r = roi(e)
            good = len(e) >= MIN_N and r >= base_ex + 0.03
            rows.append((f"{nm}: {v}", m & (df["year"] <= 2023).values, m, len(e), r))
            L.append(f"| {v} | {len(e):,} | {e['is123'].mean()*100:.2f}% | {e.loc[e['is123']>0,'pay'].mean():,.0f}円 | "
                     f"{'**' if good else ''}{r*100:.1f}%{'**' if good else ''} |")

    # 事前に決めた採用条件: 探索で n ≥ 30,000 かつ回収率最大
    cands = [(nm, mall, n, r) for nm, mex, mall, n, r in rows if n >= MIN_N]
    cands.sort(key=lambda x: -x[3])
    L += ["\n## 2. 探索で選ばれた条件（n ≥ 30,000 の中で回収率が高い順・上位10）\n",
          "| 条件 | 探索 レース数 | 探索 回収率 |", "|---|---:|---:|"]
    for nm, _, n, r in cands[:10]:
        L.append(f"| {nm} | {n:,} | **{r*100:.1f}%** |")
    best_nm, best_m, best_n, best_r = cands[0]

    # 2変数の組（上位5条件の総当たり）
    top5 = cands[:5]
    pairs = []
    for i in range(len(top5)):
        for j in range(i + 1, len(top5)):
            m = top5[i][1] & top5[j][1]
            e = df[m & (df["year"] <= 2023).values]
            if len(e) >= MIN_N:
                pairs.append((f"{top5[i][0]} かつ {top5[j][0]}", m, len(e), roi(e)))
    pairs.sort(key=lambda x: -x[3])
    L += ["\n### 2変数の組（上位5条件の総当たり、n ≥ 30,000）\n",
          "| 条件 | 探索 レース数 | 探索 回収率 |", "|---|---:|---:|"]
    for nm, _, n, r in pairs[:8]:
        L.append(f"| {nm} | {n:,} | **{r*100:.1f}%** |")

    # ---- 確認（1回だけ）
    L += ["\n## 3. 確認（2024〜2026）: 選んだ2つだけを1回評価\n",
          "**探索で決めた条件を、触らずにそのまま当てる。**\n",
          "| 買い方 | 確認 レース数 | 1日あたり | 1-2-3率 | 平均配当 | 確認 回収率 | 95%区間 | 探索との差 |",
          "|---|---:|---:|---:|---:|---:|---|---:|"]
    days_cf = cf["race_date"].nunique()
    picks = [("全レースで 1-2-3 を買う（基準）", np.ones(len(df), bool), base_ex),
             (best_nm, best_m, best_r)]
    if pairs:
        picks.append((pairs[0][0], pairs[0][1], pairs[0][3]))
    for nm, m, r_ex in picks:
        c = df[m & (df["year"] >= 2024).values]
        r = roi(c); lo, hi = ci(c)
        L.append(f"| {nm} | {len(c):,} | {len(c)/days_cf:.1f}R | {c['is123'].mean()*100:.2f}% | "
                 f"{c.loc[c['is123']>0,'pay'].mean():,.0f}円 | **{r*100:.1f}%** | {lo*100:.1f}〜{hi*100:.1f}% | "
                 f"{(r - r_ex)*100:+.1f}pt |")
    # 年別の安定性
    L += ["\n### 選んだ条件の年別（探索期間も含めて全部出す。確認は2024以降）\n",
          "| 年 | " + " | ".join(nm for nm, _, _ in picks) + " |", "|---|" + "---:|" * len(picks)]
    for y in range(2018, 2027):
        cells = []
        for nm, m, _ in picks:
            d = df[m & (df["year"] == y).values]
            cells.append(f"{roi(d)*100:.1f}%" if len(d) > 500 else "—")
        L.append(f"| {y}{'（確認）' if y >= 2024 else ''} | " + " | ".join(cells) + " |")

    # ---- 2026年のオッズを足すと、帯は条件の上にさらに効くか
    import numpy as _np
    from boatlab.model.trifecta import PERM_LABELS as _PLB
    cache = Path("/tmp/claude-0/mix10_cache.npz")
    if cache.exists():
        zz = _np.load(cache, allow_pickle=True)
        rid26, O26, W26, P26 = zz["rid"], zz["O"], zz["W"], zz["P"]
        i123 = _PLB.index("1-2-3")
        od123 = O26[:, i123]
        hit26 = (W26 == i123)
        key = pd.Series(rid26.astype(np.int64))
        cond_map = {}
        for nm, m, _ in picks[1:]:
            cond_map[nm] = set(df.loc[m & (df["year"] == 2026).values, "race_id"].astype(np.int64))
        L += ["\n## 4. 2026年のオッズを足す: 『4〜15倍の帯』は条件の上にさらに効くか\n",
              "条件（出走表・展示だけで決まる）と帯（オッズが要る）は重なっているかもしれない。2026年で確かめる。\n",
              "| 条件 | 帯 | レース数 | 的中率 | 平均配当 | 回収率 | 比 |", "|---|---|---:|---:|---:|---:|---:|"]
        for nm, cset in [("条件なし", None)] + [(k, v) for k, v in cond_map.items()]:
            base_mask = _np.isfinite(od123) if cset is None else _np.array([int(r) in cset for r in rid26]) & _np.isfinite(od123)
            for bl, bh, blab in ((0, 1e9, "帯なし"), (4, 15, "4〜15倍")):
                m = base_mask & (od123 >= bl) & (od123 < bh)
                if m.sum() < 1000:
                    continue
                r = _np.where(hit26[m], P26[m], 0.0).sum() / (100.0 * m.sum())
                L.append(f"| {nm} | {blab} | {int(m.sum()):,} | {hit26[m].mean()*100:.2f}% | "
                         f"{P26[m][hit26[m]].mean():,.0f}円 | **{r*100:.1f}%** | {r/0.75:.3f} |")

    L += ["\n## 5. 結論",
          "",
          "### 絞れた。ただし上がり幅は +5pt、天井は 86%",
          "探索（2018〜2023）で n≥30,000 の条件を総当たりし、上位を確認（2024〜2026）で1回だけ評価した。",
          "",
          "| 買い方 | 確認の回収率 | 95%区間 | 1日あたり | 基準との差 |",
          "|---|---:|---|---:|---:|",
          "| 全レースで 1-2-3 | 80.9% | 78.7〜82.8% | 147.6R | — |",
          "| ＋1号艇と他艇最高の勝率差 0〜1 | **85.4%** | 81.7〜89.5% | 28.5R | +4.5pt |",
          "| ＋1号艇の展示タイム1位 かつ A1級 | **86.0%** | 81.4〜90.5% | 16.2R | +5.1pt |",
          "",
          "**探索→確認で全部が約3pt下がったが、基準も同じだけ下がっている**（83.8→80.9）。",
          "つまり**条件の上乗せ分（+5pt前後）はそのまま残った**。下がったのは水準の方で、これは年の違い。",
          "",
          "### 効く条件の形は「1号艇が強いが、圧倒的ではない」",
          "探索の上位は 勝率差0〜1（88.9%）、全国勝率6.5〜7.5（88.6%）、展示タイム1位（88.0%）、",
          "風1m以下（87.3%）、A1級（87.1%）。**1号艇が明確に上だが独走ではない、静かな水面のレース**。",
          "逆に勝率差2以上（圧倒的）は伸びない。圧倒的なレースでは市場もちゃんと値を付けている。",
          "",
          "### それでも 86% で、100%には届かない",
          "必要な比 1.333 に対して **1.147**。1日16レース × 100円 = 1,600円で、月の期待損失は約6,700円。",
          "**これは「いちばん負けにくい3連単の買い方」であって、勝てる買い方ではない。**",
          "",
          "### 年の傾向についての訂正",
          "`leaps3.md` で「2018年84.7% → 2025年79.1% と緩やかに低下」と書いたが、**2026年は82.2%に戻っている**。",
          "2024〜2025が低く2026が戻った形で、**低下傾向と言い切れる形ではない**。9点では判定できない。",
          ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[-40:]))


if __name__ == "__main__":
    main()
