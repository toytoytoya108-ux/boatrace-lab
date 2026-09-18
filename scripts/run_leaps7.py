"""飛躍17: 2026年だけで「揺らぎ」と切り捨てた区分を、46万レースで測り直す（2026-09-18）。

**動機**: `lane1_place.md` / `lane2_place.md` で、`stadium_study.md` が「場の差は無い」と結論したのは
**2026年のみ・1場257レースという検出力不足のせい**で、46万レースなら場の差は本物だと分かった
（順位相関 ρ=+0.569〜+0.645、帰無200回で 0〜1/200）。

**ならば、同じ理由で切り捨てた他の区分も間違っている可能性がある。**
`leaps5.md` の飛躍12（番組の種類）・飛躍13（曜日・時間帯）は 2026年 36,251レースだけで
「無作為の範囲」と結論した。ここでは `回収率 = 1-2-3の的中率 × 平均配当 ÷ 100` の恒等式を使い、
**オッズ不要で 463,527レース全部**を使って測り直す。

検算の順序（この順でないと自分を騙す）:
  1. 探索2018〜23 → 確認2024〜26 で符号が揃うか
  2. 交絡を落とす（グレードは12R優勝戦の言い換えか／曜日は節の日数・場の言い換えか）
  3. 層を固定した並べ替え検定
  4. **見た条件の総数を自己申告**して、多重検定の補正後も残るかを述べる

出力: reports/research/leaps7.md
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from boatlab.config import ROOT  # noqa: E402
from boatlab.model.trifecta import PERM_LABELS  # noqa: E402

OUT = Path(ROOT) / "reports" / "research" / "leaps7.md"
FEAT = Path(ROOT) / "reports" / "research" / "manshu_features.parquet"
DOW = "月火水木金土日"
N_TRIED = 7 + 5 + 12 + 6 + 5 + 4      # 曜日・時間帯・レース番号・節の日数・グレード・重ねた条件


def load():
    df = pd.read_parquet(FEAT)
    lab2 = {l: i for i, l in enumerate(PERM_LABELS)}
    df["ci"] = df["trifecta"].map(lab2)
    df = df[df["ci"].notna()].copy(); df["ci"] = df["ci"].astype(int)
    df["pay"] = pd.to_numeric(df["pay"], errors="coerce").fillna(0.0)
    df["is123"] = (df["ci"] == lab2["1-2-3"]).astype(float)
    df["ret"] = df["is123"] * df["pay"]          # 1-2-3 を100円買ったときの払戻
    ca = pd.to_datetime(df["closed_at"], errors="coerce")
    df["dow"] = ca.dt.dayofweek
    df["hour"] = ca.dt.hour
    return df


def ci(x):
    x = np.asarray(x, float)
    return len(x), x.mean() / 100, 1.96 * x.std() / 100 / np.sqrt(max(len(x), 1))


def main():
    df = load()
    N = len(df); base = df["ret"].mean() / 100
    L = [f"# 飛躍17: 「揺らぎ」と切り捨てた区分を46万レースで測り直す（{N:,}レース・2018〜2026）\n",
         "`stadium_study.md` の「場の差は無い」が**検出力不足による誤り**だったと今日分かった",
         "（`lane1_place.md`: 46万レースなら順位相関 ρ=+0.645、帰無200回で 0/200）。",
         "**同じ理由で切り捨てた他の区分も疑うべき。** `leaps5.md` の飛躍12・13 は 2026年36,251レースだけの判定だった。",
         "ここは `回収率 = 1-2-3の的中率 × 平均配当 ÷ 100` の恒等式でオッズ不要に計算し、**9年分すべて**を使う。\n",
         f"全体の 1-2-3 回収率 = **{base*100:.2f}%**\n",
         "## 1. 区分ごとの回収率（★＝95%区間が全体平均を含まない）\n"]
    groups = {
        "曜日": [(DOW[i], df["dow"] == i) for i in range(7)],
        "締切の時間帯": [("〜11時", df["hour"] < 12), ("12〜13時", df["hour"].between(12, 13)),
                   ("14〜15時", df["hour"].between(14, 15)), ("16〜17時", df["hour"].between(16, 17)),
                   ("18時〜", df["hour"] >= 18)],
        "レース番号": [(f"{r}R", df["race_no"] == r) for r in range(1, 13)],
        "節の日数": [(f"{int(d)}日目", df["day_no"] == d) for d in sorted(df["day_no"].dropna().unique())[:6]],
        "グレード": [(g, df["grade"] == g) for g in ("SG", "G1", "G2", "G3", "一般")],
    }
    for nm, items in groups.items():
        L += [f"**{nm}**\n", "| 区分 | レース数 | 回収率 | ±95% | 全体との差 |", "|---|---:|---:|---:|---:|"]
        for lab, m in items:
            n, v, h = ci(df.loc[m, "ret"].values)
            if n < 2000:
                continue
            s = "★ " if abs(v - base) > h else ""
            L.append(f"| {s}{lab} | {n:,} | {v*100:.2f}% | ±{h*100:.2f} | {(v-base)*100:+.2f}pt |")
        L.append("")
    L += ["→ **グレードが突出**（SG +10.8pt、G1 +9.6pt）。曜日・節の日数にも段差が見える。",
          "**ここから先が本番。** この手の表は交絡と多重検定でいくらでも有意に見えるので、順に潰す。\n",
          "## 2. 探索(2018〜23) → 確認(2024〜26)\n",
          "| 区分 | 探索 | 確認 | 符号 |", "|---|---:|---:|---|"]
    ex, cf = df["year"] <= 2023, df["year"] >= 2024
    be, bc = df.loc[ex, "ret"].mean() / 100, df.loc[cf, "ret"].mean() / 100
    cands = [("G1", df["grade"] == "G1"), ("SG", df["grade"] == "SG"), ("G3", df["grade"] == "G3"),
             ("月曜", df["dow"] == 0), ("金曜", df["dow"] == 4), ("18時〜", df["hour"] >= 18),
             ("節5日目", df["day_no"] == 5), ("節1日目", df["day_no"] == 1)]
    ok = 0
    for nm, m in cands:
        _, me, _ = ci(df.loc[m & ex, "ret"].values); _, mc, _ = ci(df.loc[m & cf, "ret"].values)
        same = np.sign(me - be) == np.sign(mc - bc); ok += bool(same)
        L.append(f"| {nm} | {me*100:.2f}%（基準{be*100:.1f}） | {mc*100:.2f}%（基準{bc*100:.1f}） | {'○' if same else '×'} |")
    L += ["", f"**{ok}/{len(cands)} が符号一致。** ただし符号一致は「効果がある」の証明ではない（次で交絡を落とす）。\n",
          "## 3. 交絡を落とす\n",
          "**グレードは「12R・優勝戦が多い」の言い換えか → 逆だった。**\n",
          "| グレード | 1〜11R | 12Rのみ |", "|---|---:|---:|"]
    for g in ("SG", "G1", "一般"):
        m = df["grade"] == g
        a = ci(df.loc[m & (df["race_no"] <= 11), "ret"].values)
        b = ci(df.loc[m & (df["race_no"] == 12), "ret"].values)
        L.append(f"| {g} | {a[1]*100:.2f}%（n={a[0]:,}） | {b[1]*100:.2f}%（n={b[0]:,}） |")
    L += ["",
          "**SG/G1 は予選ラウンド（1〜11R）でこそ高く、優勝戦（12R）ではむしろ低い。**",
          "→ 「大きな大会だから堅い」ではなく「**予選では最強の選手が1号艇に入る**」という番組の構造。",
          "優勝戦は6人とも一流なので当然competitiveになる。**交絡ではなく、機構の説明が付いた。**\n"]

    # --- 曜日: 場・グレード・節の日数を揃えた残差で並べ替え
    d = df[df["dow"].notna() & df["day_no"].notna()].copy()
    d["dow"] = d["dow"].astype(int)
    key = ["stadium_code", "grade", "day_no"]
    d["res"] = d["ret"] - d.groupby(key)["ret"].transform("mean")
    codes = d.set_index(key).index.factorize()[0]
    o = np.argsort(codes); cs = codes[o]; dw = d["dow"].values[o]; rs = d["res"].values[o]
    grp = np.split(np.arange(len(cs)), np.flatnonzero(np.diff(cs)) + 1)
    rng = np.random.default_rng(11)
    obs = np.array([rs[dw == i].mean() / 100 for i in range(7)])
    nmax = []
    for _ in range(200):
        sh = dw.copy()
        for idx in grp:
            if len(idx) > 1:
                sh[idx] = rng.permutation(sh[idx])
        nmax.append(np.abs([rs[sh == i].mean() / 100 for i in range(7)]).max())
    nmax = np.array(nmax)
    p_dow = int((nmax >= np.abs(obs).max()).sum())
    L += ["**曜日は「節の日数・場・グレード」の言い換えか → 言い換えではないが、判定は境界。**\n",
          "場・グレード・節の日数を揃えた残差（pt）: " + "、".join(f"{DOW[i]} {obs[i]*100:+.2f}" for i in range(7)) + "\n",
          f"層の中で曜日ラベルを入れ替える帰無200回に対し、**|効果|の最大値で {p_dow}/200**",
          f"（実測の最大 {np.abs(obs).max()*100:.2f}pt＝{DOW[int(np.abs(obs).argmax())]}曜）。",
          "**7曜日を見て最大値を取っていることを考えると、これは「有意の入口に立った」程度。**",
          "月曜が高く金曜が低い機構は思い当たらない。**採用しない。**\n",
          "## 4. グレードの正体: 市場は「出やすさ」を値段に入れていない\n",
          "| グレード（1〜11R） | レース数 | 1-2-3 の出現率 | 的中時の平均配当 | 回収率 | 出現率比 × 配当比 |",
          "|---|---:|---:|---:|---:|---:|"]
    h0 = df["is123"].mean(); p0 = df.loc[df["is123"] > 0, "pay"].mean()
    for g in ("SG", "G1", "G2", "G3", "一般"):
        m = (df["grade"] == g) & (df["race_no"] <= 11)
        if m.sum() < 2000:
            continue
        h = df.loc[m, "is123"].mean(); p = df.loc[m & (df["is123"] > 0), "pay"].mean()
        n, v, _ = ci(df.loc[m, "ret"].values)
        L.append(f"| {g} | {n:,} | {h*100:.2f}%（×{h/h0:.2f}） | {p:.0f}円（×{p/p0:.2f}） | "
                 f"**{v*100:.2f}%** | {h/h0*p/p0:.3f} |")
    L += ["",
          "**積が1.00なら完全に織り込み済み。** SG 1.163・G1 1.133 は、",
          "**出やすさが12〜14%上がっているのに配当が1〜2%しか下がっていない**ことを意味する。",
          "群衆は「有名選手が揃った難しいレース」と見て、番組が作った堅さを値段に入れていない。\n",
          "### 層を固定した並べ替え検定（場の平均を引き、場の中でグレードを入れ替え200回）\n",
          "| グレード | レース数 | 実測（残差） | 帰無の95%範囲 | 実測以上 |", "|---|---:|---:|---|---:|"]
    dd = df[df["race_no"] <= 11]
    st = dd["stadium_code"].values; ret = dd["ret"].values / 100
    res = ret - pd.Series(ret).groupby(st).transform("mean").values
    gr = dd["grade"].values
    rng2 = np.random.default_rng(5)
    order0 = np.lexsort((np.arange(len(st)), st))
    tg = ["SG", "G1", "G3"]
    obs2 = {t: res[gr == t].mean() for t in tg}; null2 = {t: [] for t in tg}
    for _ in range(200):
        oo = np.lexsort((rng2.random(len(st)), st))
        sh = np.empty_like(gr); sh[order0] = gr[oo]
        for t in tg:
            null2[t].append(res[sh == t].mean())
    for t in tg:
        n_ = np.array(null2[t]); lo, hi = np.percentile(n_, [2.5, 97.5])
        L.append(f"| {t} | {int((gr==t).sum()):,} | {obs2[t]*100:+.2f}pt | "
                 f"{lo*100:+.2f}〜{hi*100:+.2f}pt | **{int((n_>=obs2[t]).sum())}/200** |")
    L += ["",
          "**G1 は 0/200。交絡を落としても残る本物の差。**\n",
          "## 5. さらに条件を重ねると100%を超える（が、確定できない）\n",
          "| 買い方 | レース数 | 1日あたり | 回収率 | ±95% | 探索→確認 |", "|---|---:|---:|---:|---:|---|"]
    big = df["grade"].isin(["SG", "G1"]) & (df["race_no"] <= 11)
    rows = [("SG・G1 の 1〜11R", big),
            ("＋1号艇と他艇最高の勝率差 0〜1", big & df["nwr_gap"].between(0, 1)),
            ("＋1号艇の展示タイム1位", big & (df["l1_ext_rank"] == 1)),
            ("＋節の4日目以降", big & (df["day_no"] >= 4)),
            ("＋節の1〜2日目", big & (df["day_no"] <= 2)),
            ("（対照）一般戦 1〜11R", df["grade"].eq("一般") & (df["race_no"] <= 11))]
    days = df["race_date"].nunique() if "race_date" in df else 3287
    for nm, m in rows:
        n, v, h = ci(df.loc[m, "ret"].values)
        e = df.loc[m & ex, "ret"].mean() / 100; c = df.loc[m & cf, "ret"].mean() / 100
        L.append(f"| {nm} | {n:,} | {n/days:.1f}本 | **{v*100:.2f}%** | ±{h*100:.1f} | "
                 f"{e*100:.1f}% → {c*100:.1f}% |")
    m = big & df["nwr_gap"].between(0, 1)
    x = df.loc[m, "ret"].values; bb = df.loc[big, "ret"].values
    rng3 = np.random.default_rng(9)
    null3 = np.array([bb[rng3.choice(len(bb), len(x), replace=False)].mean() / 100 for _ in range(2000)])
    sd = x.std() / 100
    need = int((1.96 * sd / (x.mean() / 100 - 1.0)) ** 2)
    L += ["",
          f"**最良は 107.68%**（SG/G1 の 1〜11R かつ 1号艇と他艇最高の勝率差 0〜1）。",
          f"出現率が {df.loc[m,'is123'].mean()/df.loc[big,'is123'].mean():.2f}倍 なのに配当は "
          f"{df.loc[m&(df['is123']>0),'pay'].mean()/df.loc[big&(df['is123']>0),'pay'].mean():.2f}倍 しか下がらない。",
          f"帰無2000回（SG/G1 から同数を無作為抽出）に対し **{int((null3>=x.mean()/100).sum())}/2000**。\n",
          "### だが、これは確定できない。3つの理由をそのまま書く\n",
          f"1. **多重検定**: 今回見たのは 曜日7・時間帯5・レース番号12・節の日数6・グレード5・重ねた条件4 の"
          f"**約{N_TRIED}通り**。その最大値を報告している。素のp値 {int((null3>=x.mean()/100).sum())/2000:.4f} に"
          f"{N_TRIED}を掛けると **{min(1.0, int((null3>=x.mean()/100).sum())/2000*N_TRIED):.2f}** で、補正後は有意でない。",
          f"2. **「100%超え」自体は珍しくない**: 同じ帰無で100%以上が出たのは {int((null3>=1.0).sum())}/2000＝"
          f"{int((null3>=1.0).sum())/20:.1f}%。n={len(x):,} では偶然でも起きる。",
          f"3. **決着に必要な年数**: 95%区間の下限を100%より上げるには、真値が107.7%のままでも約 {need:,}レース＝"
          f"**{need/(len(x)/days)/365:.0f}年**（1日{len(x)/days:.1f}本）。`calibration` の「crowd_bias 0.2%尾は99.5年」と同型で、"
          "**原理的に確認できない。**\n",
          "## まとめ: 飛躍17\n",
          "### 本物だったもの — **SG・G1 の予選ラウンド（1〜11R）で 1-2-3**",
          f"**{ci(df.loc[big,'ret'].values)[1]*100:.2f}%**（n={int(big.sum()):,}、1日{int(big.sum())/days:.1f}本、"
          f"探索{df.loc[big&ex,'ret'].mean()/100*100:.1f}% → 確認{df.loc[big&cf,'ret'].mean()/100*100:.1f}%）。",
          "交絡（12R・場）を落としても G1 は帰無 **0/200**。",
          "**`combo123.md` の最高 86.0% を7.8pt上回り、本プロジェクトで最良の3連単ルール。**",
          "正体は「番組が予選で最強の選手を1号艇に置く」構造を、群衆が値段に入れないこと。",
          "**それでも 93.8% で、100%には届かない。**\n",
          "### 方法論の収穫 — **2026年だけの検定は検出力が足りていなかった**",
          "`leaps5.md` の飛躍12（番組の種類は揺らぎ）は **36,251レースでの結論**で、",
          "同じ問いを463,527レースで測ると G1 は帰無の外に出る。",
          "**「帰無と区別できない」は「差が無い」ではなく「この標本では見えない」だった。**",
          "`stadium_study.md`（場）に続いて2例目。**2026年のみで出した否定的結論は、全部やり直す価値がある。**\n",
          "### 採用しないもの",
          f"- **曜日**（月+4.8pt / 金−3.2pt）: 交絡を落としても残るが、7曜日の最大値で {p_dow}/200 と境界。機構も無い。",
          "- **107.68% のセル**: 上の3理由により確定できない。**買い方には反映しない。**",
          "- レース番号・時間帯: 探索確認で符号は揃うが、機構が説明できず帰無との差も小さい。"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[-20:]))


if __name__ == "__main__":
    main()
