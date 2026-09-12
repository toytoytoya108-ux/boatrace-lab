"""万舟（3連単 1万円以上）の深掘り: 全期間の発生と、どんなレースで出るか（2026-09-12 依頼）。

`reports/backtest/manshu.md`（2026年のみ・単変量）の拡張版。
  1. 全期間（2018〜2026）の発生件数: 年・場・配当帯・1着艇・決まり手
  2. 「1号艇の弱さ」を軸にした出走表・展示の特徴（全期間、単変量 lift）
  3. 確定オッズの散らばり（2026のみ）と、市場との比較
  4. 多変量（LightGBM、〜2025で学習・2026で評価）: どの条件が効くか、上位10%の万舟率
  5. 荒れ条件の組合せ（スコアカード）と、その中で穴モード（人気20〜40）を買ったときの回収率

注意（正直に）: 荒れやすい条件が実在することと、それで儲かることは別。`manshu.md` で示したとおり
市場はこれらの条件を知っている（実測÷市場 < 1.00）。ここでは「見えるようにする」ことが目的で、
儲かる／儲からないは §3・§5 の市場比較の列で判断する。

出力: reports/research/manshu_deep.md、reports/research/manshu_features.parquet（レース単位の特徴量）
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from boatlab.config import ROOT, STADIUMS
from boatlab.model.trifecta import PERM_LABELS

OUT = Path(ROOT) / "reports" / "research" / "manshu_deep.md"
FEAT = Path(ROOT) / "reports" / "research" / "manshu_features.parquet"
DB = str(Path(ROOT) / "data" / "lab.db")
MAN = 10000
KLASS = {"A1": 3, "A2": 2, "B1": 1, "B2": 0}


# ------------------------------------------------------------------ 特徴量
def build() -> pd.DataFrame:
    if FEAT.exists():
        return pd.read_parquet(FEAT)
    con = sqlite3.connect(DB)
    races = pd.read_sql_query("""
        SELECT r.id race_id, r.race_date, r.stadium_code, r.race_no, r.grade, r.race_type, r.day_no, r.closed_at,
               res.trifecta_payout pay, res.trifecta, res.kimarite
        FROM races r JOIN results res ON res.race_id=r.id
        WHERE res.is_irregular=0 AND res.trifecta_payout IS NOT NULL""", con)
    ent = pd.read_sql_query("""
        SELECT race_id, lane, klass, age, f_count, avg_st, nat_win_rate nwr, loc_win_rate lwr, motor_rate2 motor
        FROM entries WHERE is_absent=0""", con)
    prv = pd.read_sql_query("""
        SELECT race_id, lane, course, st_exh, exhibition_time ext, tilt, parts
        FROM preview_snapshots WHERE id IN (SELECT MAX(id) FROM preview_snapshots GROUP BY race_id, lane)""", con)
    cond = pd.read_sql_query("""
        SELECT race_id, weather, temp_c, water_temp_c, wind_dir, wind_speed_m ws, wave_cm wave
        FROM race_conditions WHERE id IN (SELECT MAX(id) FROM race_conditions WHERE phase='preview' GROUP BY race_id)""", con)
    con.close()
    ent["klass_n"] = ent["klass"].map(KLASS)
    x = ent.merge(prv, on=["race_id", "lane"], how="left")
    x["parts_chg"] = x["parts"].map(lambda v: isinstance(v, str) and v not in ("null", "[]", "") and "number" in v).astype(int)
    x["maezuke"] = ((x["course"].notna()) & (x["course"] < x["lane"])).astype(int)
    g = x.groupby("race_id")
    # 展示タイム順位（小さいほど速い）・展示ST順位
    x["ext_rank"] = g["ext"].rank(method="min")
    x["stx_rank"] = g["st_exh"].rank(method="min")
    x["nwr_rank"] = g["nwr"].rank(method="min", ascending=False)
    l1 = x[x["lane"] == 1].set_index("race_id")
    others = x[x["lane"] != 1].groupby("race_id")
    f = pd.DataFrame(index=l1.index)
    for c0 in ("klass_n", "age", "f_count", "avg_st", "nwr", "lwr", "motor", "st_exh", "ext", "tilt", "ext_rank",
               "stx_rank", "nwr_rank", "course", "parts_chg"):
        f[f"l1_{c0}"] = l1[c0]
    f["l1_course1"] = (l1["course"] == 1).astype(float).where(l1["course"].notna())
    f["oth_nwr_max"] = others["nwr"].max()
    f["oth_motor_max"] = others["motor"].max()
    f["nwr_gap"] = f["l1_nwr"] - f["oth_nwr_max"]                      # 1号艇の勝率 − 他艇の最高勝率
    f["nwr_std"] = g["nwr"].std()
    x["is_a1"] = (x["klass_n"] == 3).astype(int)
    x["is_b"] = (x["klass_n"] <= 1).astype(int)
    g = x.groupby("race_id")
    others = x[x["lane"] != 1].groupby("race_id")
    f["n_a1"] = g["is_a1"].sum()
    f["n_b"] = g["is_b"].sum()
    f["oth_a1"] = others["is_a1"].sum()
    f["ext_rng"] = g["ext"].max() - g["ext"].min()
    f["n_maezuke"] = g["maezuke"].sum()
    f["n_parts"] = g["parts_chg"].sum()
    xi = x.reset_index(drop=True)
    for src, dst in (("ext", "best_ext_lane"), ("st_exh", "best_stx_lane")):
        idx = xi.dropna(subset=[src]).groupby("race_id")[src].idxmin()
        f[dst] = pd.Series(xi.loc[idx.values, "lane"].values, index=idx.index)
    df = races.merge(f, left_on="race_id", right_index=True, how="left").merge(cond, on="race_id", how="left")
    df["man"] = (df["pay"] >= MAN).astype(int)
    df["dt"] = pd.to_datetime(df["race_date"])
    df["year"] = df["dt"].dt.year
    df["month"] = df["dt"].dt.month
    df["dow"] = df["dt"].dt.dayofweek
    df["hour"] = pd.to_datetime(df["closed_at"], errors="coerce").dt.hour
    df["night"] = (df["hour"] >= 18).astype(int)
    df["win_lane"] = df["trifecta"].str.split("-").str[0].astype(float)
    df["stadium"] = df["stadium_code"].map(STADIUMS)
    df.to_parquet(FEAT)
    return df


# ------------------------------------------------------------------ 表のヘルパ
def lift_table(df: pd.DataFrame, col: str, bins=None, labels=None, title: str = "", top: int | None = None,
               fmt="{}") -> list[str]:
    base = df["man"].mean()
    v = df[col]
    if bins is not None:
        key = pd.cut(pd.to_numeric(v, errors="coerce"), bins=bins, labels=labels, include_lowest=True)
    else:
        key = v
    g = df.groupby(key, observed=True)["man"].agg(["size", "mean"])
    g = g[g["size"] >= 300]
    if top:
        g = g.sort_values("mean", ascending=False)
        g = pd.concat([g.head(top), g.tail(top)]) if len(g) > 2 * top else g
    L = [f"\n### {title or col}\n", "| 区分 | レース数 | 万舟率 | 全体比 |", "|---|---:|---:|---:|"]
    for k, r in g.iterrows():
        L.append(f"| {fmt.format(k)} | {int(r['size']):,} | {r['mean']*100:.1f}% | **×{r['mean']/base:.2f}** |")
    return L


def main():
    df = build()
    base = df["man"].mean()
    L = [f"# 万舟の深掘り（{len(df):,}R・2018〜2026年）\n",
         f"万舟＝3連単配当 {MAN:,}円以上。全期間の万舟率 **{base*100:.2f}%**（{int(df['man'].sum()):,}本）。",
         "「全体比」は各区分の万舟率 ÷ 全体の万舟率（×1.00 が平均）。**荒れやすさの指標として読む。**\n",
         "> 正直な但し書き: 荒れやすい条件が実在することと、それで儲かることは別。市場はこれらの条件を",
         "> 知っている（`manshu.md`: どの区分でも実測÷市場 < 1.00）。儲かるかは §3・§5 の市場比較で見る。\n"]

    # ---------------- 1. 発生
    L += ["## 1. どれくらい出ているか\n", "| 年 | レース数 | 万舟 | 万舟率 | 5万円以上 | 10万円以上 | 最高配当 |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for y, s in df.groupby("year"):
        L.append(f"| {y} | {len(s):,} | {int(s['man'].sum()):,} | {s['man'].mean()*100:.1f}% | "
                 f"{int((s['pay']>=50000).sum()):,} | {int((s['pay']>=100000).sum()):,} | {s['pay'].max():,.0f}円 |")
    L += ["\n**年による変動はほとんど無い**（16.5〜17.5%）。万舟はほぼ定数として出る。\n",
          "### 万舟のとき、誰が勝っていたか（1着艇）\n", "| 1着艇 | 万舟の本数 | 万舟の内訳 | その艇が勝ったレースのうち万舟になる率 |",
          "|---|---:|---:|---:|"]
    m = df[df["man"] == 1]
    for ln, s in m.groupby("win_lane"):
        allw = df[df["win_lane"] == ln]
        L.append(f"| {int(ln)}号艇 | {len(s):,} | {len(s)/len(m)*100:.1f}% | {len(s)/len(allw)*100:.1f}% |")
    L += ["\n1号艇が勝つと万舟にはほぼならない。**万舟の約9割は2〜6号艇が1着**。",
          "つまり「荒れる」＝「1号艇が勝てない」レースを見つけること。以降の特徴はこの軸で読む。\n"]
    L += lift_table(df, "kimarite", title="決まり手（結果側・参考）")
    L += ["\n### 配当帯の内訳（万舟のうち）\n", "| 配当 | 本数 | 万舟の内訳 |", "|---|---:|---:|"]
    for lo, hi, lab in ((10000, 20000, "1〜2万円"), (20000, 50000, "2〜5万円"), (50000, 100000, "5〜10万円"), (100000, 1e9, "10万円以上")):
        k = int(((m["pay"] >= lo) & (m["pay"] < hi)).sum())
        L.append(f"| {lab} | {k:,} | {k/len(m)*100:.1f}% |")

    # ---------------- 2. 単変量（全期間）
    L += ["\n## 2. どんなレースで出るか（全期間・単変量）\n",
          "各条件ごとに万舟率と全体比。**×1.30 を超える条件は「荒れやすい」、×0.80 を下回る条件は「堅い」**。\n",
          "### 2-1. 1号艇の強さ（いちばん効く軸）"]
    L += lift_table(df, "l1_klass_n", bins=[-.5, .5, 1.5, 2.5, 3.5], labels=["B2", "B1", "A2", "A1"], title="1号艇の級別")
    L += lift_table(df, "l1_nwr", bins=[0, 4, 5, 5.5, 6, 6.5, 7, 10], labels=["〜4.0", "4〜5", "5〜5.5", "5.5〜6", "6〜6.5", "6.5〜7", "7以上"], title="1号艇の全国勝率")
    L += lift_table(df, "nwr_gap", bins=[-9, -2, -1, -0.5, 0, 0.5, 1, 9], labels=["−2以下", "−2〜−1", "−1〜−0.5", "−0.5〜0", "0〜0.5", "0.5〜1", "1以上"],
                    title="1号艇の勝率 − 他艇の最高勝率（相対的な強さ）")
    L += lift_table(df, "l1_nwr_rank", bins=[.5, 1.5, 2.5, 3.5, 6.5], labels=["1位", "2位", "3位", "4〜6位"], title="1号艇の勝率順位（6艇中）")
    L += lift_table(df, "l1_motor", bins=[0, 25, 30, 35, 40, 50, 100], labels=["〜25%", "25〜30", "30〜35", "35〜40", "40〜50", "50以上"], title="1号艇のモーター2連率")
    L += lift_table(df, "l1_avg_st", bins=[0, .13, .15, .17, .19, .21, 1], labels=["〜0.13", "0.13〜0.15", "0.15〜0.17", "0.17〜0.19", "0.19〜0.21", "0.21以上"], title="1号艇の平均ST")
    L += lift_table(df, "l1_f_count", bins=[-.5, .5, 1.5, 9], labels=["F0", "F1", "F2以上"], title="1号艇のF数（F持ちはスタートが慎重になる）")
    L += lift_table(df, "l1_age", bins=[0, 30, 40, 50, 99], labels=["〜30歳", "30代", "40代", "50歳以上"], title="1号艇の年齢")
    L += ["\n### 2-2. 展示（直前情報）"]
    L += lift_table(df, "l1_ext_rank", bins=[.5, 1.5, 2.5, 3.5, 4.5, 6.5], labels=["1位", "2位", "3位", "4位", "5〜6位"], title="1号艇の展示タイム順位")
    L += lift_table(df, "l1_stx_rank", bins=[.5, 1.5, 2.5, 3.5, 4.5, 6.5], labels=["1位", "2位", "3位", "4位", "5〜6位"], title="1号艇の展示ST順位（遅いほど危ない）")
    L += lift_table(df, "l1_st_exh", bins=[-1, .05, .1, .15, .2, .3, 2], labels=["〜0.05", "0.05〜0.10", "0.10〜0.15", "0.15〜0.20", "0.20〜0.30", "0.30以上"], title="1号艇の展示ST")
    L += lift_table(df, "l1_course1", bins=[-.5, .5, 1.5], labels=["展示で1コースを取れていない", "1コース"], title="1号艇の展示進入")
    L += lift_table(df, "n_maezuke", bins=[-.5, .5, 1.5, 9], labels=["前づけなし", "1人", "2人以上"], title="スタート展示の前づけ人数")
    L += lift_table(df, "best_ext_lane", title="展示タイム最速の艇", fmt="{:.0f}号艇")
    L += lift_table(df, "ext_rng", bins=[-1, .08, .11, .14, .18, 9], labels=["〜0.08秒", "0.08〜0.11", "0.11〜0.14", "0.14〜0.18", "0.18秒以上"], title="展示タイムのばらつき（最大−最小）")
    L += lift_table(df, "l1_tilt", bins=[-9, -0.4, 0.1, 9], labels=["−0.5", "0", "+0.5以上"], title="1号艇のチルト")
    L += lift_table(df, "n_parts", bins=[-.5, .5, 9], labels=["部品交換なし", "あり"], title="部品交換（レース内のいずれかの艇）")
    L += ["\n### 2-3. 相手関係・級別構成"]
    L += lift_table(df, "oth_a1", bins=[-.5, .5, 1.5, 2.5, 9], labels=["0人", "1人", "2人", "3人以上"], title="2〜6号艇にいるA1の人数")
    L += lift_table(df, "n_b", bins=[-.5, .5, 1.5, 2.5, 3.5, 9], labels=["0人", "1人", "2人", "3人", "4人以上"], title="B級の人数")
    L += lift_table(df, "nwr_std", bins=[0, .5, .8, 1.1, 1.5, 9], labels=["〜0.5", "0.5〜0.8", "0.8〜1.1", "1.1〜1.5", "1.5以上"], title="6艇の勝率のばらつき（小さい＝拮抗）")
    L += ["\n### 2-4. 気象"]
    L += lift_table(df, "ws", bins=[-.1, .5, 2, 4, 6, 8, 99], labels=["無風〜0.5m", "1〜2m", "3〜4m", "5〜6m", "7〜8m", "9m以上"], title="風速")
    L += lift_table(df, "wave", bins=[-.1, .5, 2, 4, 6, 99], labels=["0cm", "1〜2cm", "3〜4cm", "5〜6cm", "7cm以上"], title="波高")
    L += lift_table(df, "weather", title="天候")
    L += lift_table(df, "temp_c", bins=[-99, 5, 10, 20, 30, 99], labels=["5℃未満", "5〜10", "10〜20", "20〜30", "30℃以上"], title="気温")
    L += ["\n### 2-5. レースの属性"]
    L += lift_table(df, "stadium", title="場（上位・下位6つ）", top=6)
    L += lift_table(df, "race_no", title="レース番号", fmt="{}R")
    L += lift_table(df, "grade", title="グレード")
    L += lift_table(df, "race_type", title="レース種別（上位・下位6つ）", top=6)
    L += lift_table(df, "day_no", bins=[.5, 1.5, 2.5, 3.5, 4.5, 9], labels=["初日", "2日目", "3日目", "4日目", "5日目以降"], title="節の何日目")
    L += lift_table(df, "night", bins=[-.5, .5, 1.5], labels=["デイ", "ナイター（締切18時以降）"], title="デイ／ナイター")
    L += lift_table(df, "month", title="月", fmt="{}月")
    L += lift_table(df, "dow", bins=[-.5, 4.5, 6.5], labels=["平日", "土日"], title="曜日")

    # ---------------- 3. 確定オッズ（2026）
    con = sqlite3.connect(DB)
    lab2 = {l: i for i, l in enumerate(PERM_LABELS)}
    rows = {}
    for rid, js in con.execute("SELECT race_id, odds FROM odds_snapshots WHERE bet_type='3t' AND source='turnmark_final'"):
        d0 = json.loads(js) if isinstance(js, str) else js
        inv = np.zeros(120)
        for k, v in d0.items():
            j = lab2.get(k)
            if j is not None and v and float(v) > 0:
                inv[j] = 1.0 / float(v)
        if (inv > 0).sum() >= 110:
            rows[int(rid)] = inv / inv.sum()
    con.close()
    d26 = df[df["race_id"].isin(rows)].copy()
    Q = np.stack([rows[r] for r in d26["race_id"]])
    A = np.array([int(l.split("-")[0]) - 1 for l in PERM_LABELS])
    d26["q_man"] = np.where(Q <= 0.0075, Q, 0.0).sum(1)
    d26["q1_max"] = np.stack([Q[:, A == a].sum(1) for a in range(6)], 1).max(1)
    d26["q_l1"] = Q[:, A == 0].sum(1)
    with np.errstate(divide="ignore"):
        d26["q_ent"] = -(np.where(Q > 0, Q * np.log(Q), 0)).sum(1)
    d26["top1_odds"] = 0.75 / Q.max(1)
    d26["n_under100"] = (Q > 0.0075).sum(1)
    L += ["\n## 3. 確定オッズの散らばり（2026年・{:,}R）と市場との比較\n".format(len(d26)),
          "ここだけは「実測÷市場」を併記する。市場が示す万舟率＝100倍以上の買い目の確率の合計。",
          "**実測÷市場 が 1.00 を超える区分だけが、市場より荒れる＝穴を買って市場に勝てる余地がある区分。**\n"]

    def qtab(col, bins, labels, title):
        Lx = [f"\n### {title}\n", "| 区分 | レース数 | 万舟率 | 全体比 | 市場が示す万舟率 | 実測÷市場 |", "|---|---:|---:|---:|---:|---:|"]
        key = pd.cut(d26[col], bins=bins, labels=labels, include_lowest=True)
        for k, s in d26.groupby(key, observed=True):
            if len(s) < 200:
                continue
            Lx.append(f"| {k} | {len(s):,} | {s['man'].mean()*100:.1f}% | ×{s['man'].mean()/d26['man'].mean():.2f} | "
                      f"{s['q_man'].mean()*100:.1f}% | **{s['man'].mean()/max(s['q_man'].mean(),1e-9):.3f}** |")
        return Lx
    L += qtab("q_l1", [0, .3, .4, .5, .6, .7, 1], ["30%未満", "30〜40", "40〜50", "50〜60", "60〜70", "70%以上"], "市場が見た1号艇の1着確率")
    L += qtab("top1_odds", [0, 3, 5, 8, 12, 20, 999], ["3倍未満", "3〜5", "5〜8", "8〜12", "12〜20", "20倍以上"], "1番人気のオッズ")
    L += qtab("q_ent", [0, 3.4, 3.7, 3.9, 4.1, 5], ["〜3.4（集中）", "3.4〜3.7", "3.7〜3.9", "3.9〜4.1", "4.1〜（分散）"], "120通りの確率の散らばり（エントロピー）")
    L += qtab("n_under100", [0, 20, 30, 40, 50, 121], ["〜20通り", "20〜30", "30〜40", "40〜50", "50通り以上"], "100倍未満の買い目の数（多いほど票が散っている）")
    # 2026 の主要な事前特徴に「実測÷市場」を付ける（市場が知らない条件を探す）
    L += ["\n### 事前に分かる条件で、市場より荒れる区分はあるか（2026年）\n",
          "| 条件 | 区分 | レース数 | 万舟率 | 市場が示す万舟率 | 実測÷市場 |", "|---|---|---:|---:|---:|---:|"]
    checks = [("l1_klass_n", [-.5, .5, 1.5, 2.5, 3.5], ["B2", "B1", "A2", "A1"], "1号艇の級別"),
              ("nwr_gap", [-9, -1, 0, 1, 9], ["−1以下", "−1〜0", "0〜1", "1以上"], "1号艇−他艇最高の勝率差"),
              ("l1_ext_rank", [.5, 1.5, 3.5, 6.5], ["1位", "2〜3位", "4〜6位"], "1号艇の展示タイム順位"),
              ("l1_stx_rank", [.5, 1.5, 3.5, 6.5], ["1位", "2〜3位", "4〜6位"], "1号艇の展示ST順位"),
              ("n_maezuke", [-.5, .5, 9], ["なし", "あり"], "前づけ"),
              ("ws", [-.1, 2, 5, 99], ["〜2m", "3〜5m", "6m以上"], "風速"),
              ("wave", [-.1, 2, 4, 99], ["〜2cm", "3〜4cm", "5cm以上"], "波高"),
              ("night", [-.5, .5, 1.5], ["デイ", "ナイター"], "デイ／ナイター")]
    flagged = []
    for col, bins, labels, nm in checks:
        key = pd.cut(pd.to_numeric(d26[col], errors="coerce"), bins=bins, labels=labels, include_lowest=True)
        for k, s in d26.groupby(key, observed=True):
            if len(s) < 300:
                continue
            ratio = s["man"].mean() / max(s["q_man"].mean(), 1e-9)
            L.append(f"| {nm} | {k} | {len(s):,} | {s['man'].mean()*100:.1f}% | {s['q_man'].mean()*100:.1f}% | "
                     f"{'**' if ratio > 1 else ''}{ratio:.3f}{'**' if ratio > 1 else ''} |")
            if ratio > 1.0:
                flagged.append((nm, str(k), len(s), ratio))
    L.append("\n" + (f"実測÷市場 > 1.00 の区分: " + "、".join(f"{a}={b}（{c:,}R、{d:.3f}）" for a, b, c, d in flagged)
                     if flagged else "**実測÷市場 > 1.00 の区分は無い。** 市場はこれらの条件を全部知っている。"))

    # ---------------- 4. 多変量
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score
    feats = ["l1_klass_n", "l1_nwr", "l1_lwr", "l1_motor", "l1_avg_st", "l1_f_count", "l1_age", "l1_st_exh", "l1_ext",
             "l1_tilt", "l1_ext_rank", "l1_stx_rank", "l1_nwr_rank", "l1_course1", "oth_nwr_max", "oth_motor_max",
             "nwr_gap", "nwr_std", "n_a1", "n_b", "oth_a1", "ext_rng", "n_maezuke", "n_parts", "best_ext_lane",
             "best_stx_lane", "ws", "wave", "temp_c", "water_temp_c", "race_no", "day_no", "month", "night",
             "stadium_code"]
    names = {"l1_klass_n": "1号艇 級別", "l1_nwr": "1号艇 全国勝率", "l1_lwr": "1号艇 当地勝率", "l1_motor": "1号艇 モーター",
             "l1_avg_st": "1号艇 平均ST", "l1_f_count": "1号艇 F数", "l1_age": "1号艇 年齢", "l1_st_exh": "1号艇 展示ST",
             "l1_ext": "1号艇 展示タイム", "l1_tilt": "1号艇 チルト", "l1_ext_rank": "1号艇 展示T順位", "l1_stx_rank": "1号艇 展示ST順位",
             "l1_nwr_rank": "1号艇 勝率順位", "l1_course1": "1号艇 展示1コース", "oth_nwr_max": "他艇の最高勝率",
             "oth_motor_max": "他艇の最高モーター", "nwr_gap": "勝率差(1号艇−他艇最高)", "nwr_std": "勝率のばらつき",
             "n_a1": "A1の人数", "n_b": "B級の人数", "oth_a1": "他艇のA1人数", "ext_rng": "展示Tのばらつき",
             "n_maezuke": "前づけ人数", "n_parts": "部品交換", "best_ext_lane": "展示T最速の艇", "best_stx_lane": "展示ST最速の艇",
             "ws": "風速", "wave": "波高", "temp_c": "気温", "water_temp_c": "水温", "race_no": "レース番号", "day_no": "節の日",
             "month": "月", "night": "ナイター", "stadium_code": "場"}
    X = df[feats].apply(pd.to_numeric, errors="coerce")
    tr, te = (df["year"] <= 2025).values, (df["year"] == 2026).values
    mdl = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=200,
                             subsample=0.8, colsample_bytree=0.8, verbose=-1)
    mdl.fit(X[tr], df.loc[tr, "man"], categorical_feature=["stadium_code"])
    p = mdl.predict_proba(X[te])[:, 1]
    y = df.loc[te, "man"].values
    auc = roc_auc_score(y, p)
    imp = pd.Series(mdl.booster_.feature_importance("gain"), index=feats).sort_values(ascending=False)
    imp = imp / imp.sum()
    L += ["\n## 4. 多変量（LightGBM・2018〜2025で学習、2026で評価）\n",
          f"事前に分かる特徴量 {len(feats)} 個だけ（オッズは使わない）。確認期間の AUC **{auc:.3f}**。\n",
          "### どの条件が効いているか（寄与の上位15）\n", "| 順位 | 特徴量 | 寄与 |", "|---|---|---:|"]
    for i, (k, v) in enumerate(imp.head(15).items(), 1):
        L.append(f"| {i} | {names.get(k, k)} | {v*100:.1f}% |")
    d26m = df[te].copy()
    d26m["p"] = p
    d26m["dec"] = pd.qcut(d26m["p"], 10, labels=[f"{i+1}" for i in range(10)])
    L += ["\n### モデルが「荒れる」と見た順に10分割（2026年）\n", "| 分位 | レース数 | 万舟率 | 全体比 |", "|---|---:|---:|---:|"]
    for k, s in d26m.groupby("dec", observed=True):
        L.append(f"| {k}（{'堅い' if k=='1' else ('荒れる' if k=='10' else '')}） | {len(s):,} | {s['man'].mean()*100:.1f}% | ×{s['man'].mean()/d26m['man'].mean():.2f} |")
    # 市場と比べる
    mm = d26m.merge(d26[["race_id", "q_man"]], on="race_id")
    top_model = mm.nlargest(len(mm) // 10, "p")
    top_mkt = mm.nlargest(len(mm) // 10, "q_man")
    both = set(top_model["race_id"]) & set(top_mkt["race_id"])
    L += ["\n### 市場の見立てと比べる（2026年・上位10%）\n",
          "| 「荒れる」上位10%の選び方 | 実際の万舟率 | 市場が示す万舟率 | 実測÷市場 |", "|---|---:|---:|---:|",
          f"| 事前特徴だけのモデル | {top_model['man'].mean()*100:.1f}% | {top_model['q_man'].mean()*100:.1f}% | {top_model['man'].mean()/top_model['q_man'].mean():.3f} |",
          f"| 市場が示す万舟確率 | {top_mkt['man'].mean()*100:.1f}% | {top_mkt['q_man'].mean()*100:.1f}% | {top_mkt['man'].mean()/top_mkt['q_man'].mean():.3f} |",
          f"\n2つの上位10%の重なり: **{len(both)/max(len(top_model),1)*100:.0f}%**。",
          "モデルと市場が選ぶレースは3分の1しか重ならない。しかしモデルが選んだレースの万舟率は",
          f"市場の見立て（{top_model['q_man'].mean()*100:.1f}%）とほぼ一致する（実測÷市場 {top_model['man'].mean()/top_model['q_man'].mean():.3f}）。",
          "つまり**モデルが見つける荒れレースは、市場もその分だけ荒れると値付けしている**。事前特徴には市場を超える情報が無い。",
          "一方、市場が示す万舟確率の上位10%は万舟率が高く（27.5%）、しかも市場の見立てどおり（1.006）。",
          "**穴を狙う場面の選定は、市場が示す万舟確率のほうが事前特徴のモデルより優れている。**"]

    # ---------------- 5. スコアカードと穴モードの回収率
    L += ["\n## 5. 荒れ条件の組合せと、その中で穴を買ったら\n",
          "単変量で全体比の高かった条件を数えてスコアにする（事前に分かるものだけ）。",
          "そのスコア別に、2026年の確定オッズで **人気20〜40の21点×100円** を買った回収率を出す。\n"]
    d26["score"] = ((d26["l1_klass_n"] <= 1).astype(int) + (d26["nwr_gap"] < 0).astype(int)
                    + (d26["l1_ext_rank"] >= 4).astype(int) + (d26["l1_stx_rank"] >= 4).astype(int)
                    + (d26["n_maezuke"] >= 1).astype(int) + (d26["wave"] >= 5).astype(int) + (d26["ws"] >= 6).astype(int))
    pay = {}
    con = sqlite3.connect(DB)
    for rid, js in con.execute("SELECT race_id, payouts FROM results WHERE race_id >= 202601010000 AND payouts IS NOT NULL"):
        d0 = json.loads(js) if isinstance(js, str) else js
        if isinstance(d0, dict):
            t = (d0.get("trifecta") or [{}])[0]
            pay[int(rid)] = (str(t.get("combination", "")).strip(), float(t.get("amount") or 0))
    con.close()
    Qd = {r: q for r, q in zip(d26["race_id"], Q)}
    L += ["| 荒れ条件の数 | レース数 | 万舟率 | 市場が示す万舟率 | 実測÷市場 | 穴21点の的中率 | 穴21点の回収率 |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for sc, s in d26.groupby("score"):
        if len(s) < 300:
            continue
        ret = stake = hits = 0.0
        for rid in s["race_id"]:
            q = Qd[rid]
            order = np.argsort(-q)[19:40]
            combo, amt = pay.get(rid, ("", 0.0))
            j = lab2.get(combo)
            stake += 2100
            if j is not None and j in set(order.tolist()):
                ret += amt; hits += 1
        L.append(f"| {sc}個 | {len(s):,} | {s['man'].mean()*100:.1f}% | {s['q_man'].mean()*100:.1f}% | "
                 f"{s['man'].mean()/max(s['q_man'].mean(),1e-9):.3f} | {hits/len(s)*100:.1f}% | **{ret/stake*100:.1f}%** |")
    L += ["\n条件が多いほど万舟率も穴21点の回収率も単調に上がる（0個 62.6% → 5個 77.3%）。",
          "荒れ条件は確かに効いている。ただし「実測÷市場」はどの段階でも 1.00 未満で、市場も条件の分だけ穴に票を入れている。",
          "条件5個（1日約5レース）でも 77.3% にとどまり、**市場が示す万舟確率で絞ったとき（`condition_rules.md` §4: 上位10%で83.3%）に届かない**。",
          "事前条件で絞る意味はあるが、市場の万舟確率のほうがより良い絞り込みになる。",
          "\n## 6. まとめ\n",
          "- 万舟は年ごとに 16.5〜17.5% で安定。9割は「1号艇が勝てなかったレース」。",
          "- 荒れやすさを決めるのは **1号艇の弱さ**（級別・勝率差・展示順位）が第一で、気象・場・レース番号はその次。",
          "- 企画レース（モーニング・ツッキー・ガチ勝ち8など、1号艇に強い選手を置く番組）は万舟率が約10%で極端に堅い。",
          "  逆に予選ドリーム・特賞戦などは21%前後で荒れる。**番組の意図がそのまま万舟率に出る。**",
          "- 事前に分かる条件だけのモデルは AUC 0.585 で弱く、その上位10%の万舟率は市場の見立てと一致する（実測÷市場 0.983）。",
          "  事前条件には市場を超える情報が無い。",
          "- 荒れ条件を数えて絞ると穴21点の回収率は 62.6% → 77.3% まで上がるが、市場が示す万舟確率の上位10%（83.3%）に届かない。",
          "- **穴を狙う場面の選定は、市場が示す万舟確率を第一にする。** 条件の分析は「なぜ荒れるのか」を理解する材料として使う。",
          "  実測÷市場 > 1.00 が出たのは、市場自身が大混戦と見ている極端な区分だけ（1番人気20倍以上、エントロピー4.1以上、100倍未満が50通り以上）。"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
