"""万舟券（3連単配当1万円以上）はどれくらい出て、どんなレースで出るのか。

## 測ること
1. 発生頻度（1日あたり・全体・月別）
2. **市場は万舟の確率を正しく見ているか**
   3連単の確定オッズが100倍以上の買い目の確率を足せば、市場が示す万舟確率になる
   （配当1万円 ⟺ オッズ100倍 ⟺ 市場確率 q ≤ 0.0075）。実際の発生率と比べる。
3. 場別・レース番号別・グレード別・節日別
4. 万舟レースの特徴（気象・級別構成・市場の集中度・展示進入）
5. **市場を超えて万舟を予測できるか**
   市場が示す万舟確率をオフセットに置いたロジスティック回帰。γ=0 なら市場がすでに持っている情報。
6. 目的2への応用（複勝の見送り判断に使えるか）

注意: `longshot.md` で、穴を買う戦略は回収率19〜46%と最悪であることを実測済み。
**「万舟を当てる」方向に使う話ではない。** 荒れを読むことは「勝ち馬を読む」ことと別の問題で、
使いどころは「荒れそうなレースを避ける」側にある。

出力: reports/backtest/manshu.md
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit

from boatlab.config import ROOT
from boatlab.model.trifecta import PERM_LABELS

OUT = Path(ROOT) / "reports" / "backtest" / "manshu.md"
DB = str(Path(ROOT) / "data" / "lab.db")
EXPLORE_END = "2026-05-31"
MAN = 10000.0            # 万舟の定義（3連単配当）
Q_MAN = 0.75 / 100.0     # オッズ100倍に対応する市場確率 0.0075
RNG = np.random.default_rng(20260912)


def load() -> pd.DataFrame:
    con = sqlite3.connect(DB)
    df = pd.read_sql_query("""
      SELECT r.id race_id, r.race_date, r.stadium_code, r.race_no, r.grade, r.race_type, r.day_no,
             res.trifecta, res.trifecta_payout pay, res.trifecta_popularity pop, res.kimarite,
             res.payouts payouts_js,
             COALESCE(st.name, CAST(r.stadium_code AS TEXT)) stadium,
             (SELECT w.wind_speed_m FROM race_conditions w WHERE w.race_id=r.id AND w.phase='preview' LIMIT 1) ws,
             (SELECT w.wave_cm     FROM race_conditions w WHERE w.race_id=r.id AND w.phase='preview' LIMIT 1) wave,
             (SELECT w.weather     FROM race_conditions w WHERE w.race_id=r.id AND w.phase='preview' LIMIT 1) weather,
             (SELECT COUNT(*) FROM entries e WHERE e.race_id=r.id AND e.klass='A1' AND e.is_absent=0) n_a1,
             (SELECT COUNT(*) FROM entries e WHERE e.race_id=r.id AND e.klass IN ('B1','B2') AND e.is_absent=0) n_b,
             (SELECT MIN(p.course - p.lane) FROM preview_snapshots p
                WHERE p.race_id=r.id AND p.course IS NOT NULL) mz_exh,
             (SELECT AVG(p.exhibition_time) FROM preview_snapshots p WHERE p.race_id=r.id) ex_mean,
             (SELECT MAX(p.exhibition_time)-MIN(p.exhibition_time) FROM preview_snapshots p
                WHERE p.race_id=r.id) ex_rng
      FROM races r JOIN results res ON res.race_id=r.id
      LEFT JOIN stadiums st ON st.code=r.stadium_code
      WHERE r.race_date>='2026-01-01' AND res.is_irregular=0 AND res.trifecta_payout IS NOT NULL""", con)
    lab2 = {l: i for i, l in enumerate(PERM_LABELS)}
    rows = {}
    for rid, js in con.execute(
            "SELECT race_id, odds FROM odds_snapshots WHERE bet_type='3t' AND source='turnmark_final'"):
        d = json.loads(js) if isinstance(js, str) else js
        inv = np.zeros(120)
        for k, v in d.items():
            j = lab2.get(k)
            if j is not None and v and float(v) > 0:
                inv[j] = 1.0 / float(v)
        if (inv > 0).sum() >= 110:
            rows[int(rid)] = inv / inv.sum()
    con.close()
    df["has_q"] = df["race_id"].isin(rows)
    Q = np.full((len(df), 120), np.nan)
    for i, rid in enumerate(df["race_id"].values):
        if rid in rows:
            Q[i] = rows[rid]
    df["q_man"] = np.nansum(np.where(Q <= Q_MAN, Q, 0.0), 1)     # 市場が示す万舟確率
    A = np.array([int(l.split("-")[0]) - 1 for l in PERM_LABELS])
    B2 = np.array([int(l.split("-")[1]) - 1 for l in PERM_LABELS])
    q1 = np.stack([np.nansum(np.where(A == a, Q, 0.0), 1) for a in range(6)], 1)
    q2 = np.stack([np.nansum(np.where((A == a) | (B2 == a), Q, 0.0), 1) for a in range(6)], 1)
    df["q1_max"] = q1.max(1)                    # 市場がいちばん高く見ている1着確率
    df["q2_max"] = q2.max(1)                    # 同じく2着以内確率（複勝の対象）
    df["q2_arg"] = q2.argmax(1)
    df["q_max"] = np.nanmax(Q, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        df["q_ent"] = -np.nansum(np.where(Q > 0, Q * np.log(Q), 0.0), 1)   # 市場の散らばり
    df["man"] = (df["pay"] >= MAN).astype(int)
    df["dt"] = pd.to_datetime(df["race_date"])
    return df


def fit_logit_offset(Z, off, y, l2=1e-4):
    def f(w):
        z = off + Z @ w
        p = expit(z)
        loss = -np.mean(y * np.log(np.clip(p, 1e-12, None)) + (1 - y) * np.log(np.clip(1 - p, 1e-12, None))) \
            + l2 * float(w @ w)
        g = Z.T @ (p - y) / len(y) + 2 * l2 * w
        return loss, g
    return minimize(f, np.zeros(Z.shape[1]), jac=True, method="L-BFGS-B", options={"maxiter": 800}).x


def bll(p, y):
    return float(-np.mean(y * np.log(np.clip(p, 1e-12, None)) + (1 - y) * np.log(np.clip(1 - p, 1e-12, None))))


def band(df, col, bins, labels, L, title, fmt="{:.0f}"):
    L += [f"\n### {title}\n", "| 区分 | レース数 | 万舟率 | 市場が示す万舟率 | 実測÷市場 | 平均配当 |",
          "|---|---:|---:|---:|---:|---:|"]
    g = pd.cut(df[col], bins=bins, labels=labels, include_lowest=True)
    for k, s in df.groupby(g, observed=True):
        if len(s) < 100:
            continue
        a, m = s["man"].mean(), s["q_man"].mean()
        L.append(f"| {k} | {len(s):,} | {a*100:.1f}% | {m*100:.1f}% | {a/max(m,1e-9):.3f} | "
                 f"{s['pay'].mean():,.0f}円 |")


def main():
    df = load()
    d = df[df["has_q"]].copy()
    days = df["dt"].dt.date.nunique()
    L = [f"# 万舟券はどれくらい出て、どんなレースで出るのか（{len(df):,}R・2026年）\n",
         "万舟＝3連単配当 1万円以上。配当1万円は**オッズ100倍**＝市場確率 0.0075 に対応する。",
         "市場が示す万舟確率は「確定オッズが100倍以上の買い目の確率の合計」で計算できる。\n",
         "## 1. どれくらい出ているか\n",
         "| 区分 | 本数 | 発生率 | 1日あたり | 1レースに1回の頻度 |", "|---|---:|---:|---:|---:|"]
    for lab, th in (("万舟（1万円以上）", 10000), ("5万円以上", 50000), ("10万円以上", 100000),
                    ("20万円以上", 200000)):
        n = int((df["pay"] >= th).sum())
        L.append(f"| {lab} | {n:,} | {n/len(df)*100:.2f}% | {n/days:.1f}本 | {len(df)/max(n,1):.0f}Rに1回 |")
    L.append(f"\n対象 {days} 日、1日平均 {len(df)/days:.1f} レース。"
             f"配当の最高は **{df['pay'].max():,.0f}円**、平均 {df['pay'].mean():,.0f}円、"
             f"中央値 {df['pay'].median():,.0f}円。\n")

    L += ["### 月別\n", "| 月 | レース数 | 万舟率 | 1日あたり | 平均配当 |", "|---|---:|---:|---:|---:|"]
    for m, s in df.groupby(df["dt"].dt.to_period("M")):
        dd = s["dt"].dt.date.nunique()
        L.append(f"| {m} | {len(s):,} | {s['man'].mean()*100:.1f}% | {s['man'].sum()/dd:.1f}本 | "
                 f"{s['pay'].mean():,.0f}円 |")

    # ---------------- 2. 市場は万舟の確率を正しく見ているか
    L += ["\n## 2. 市場は万舟の確率を正しく見ているか\n",
          f"- 実際の万舟率: **{d['man'].mean()*100:.2f}%**",
          f"- 市場が示す万舟率: **{d['q_man'].mean()*100:.2f}%**",
          f"- 実測 ÷ 市場 = **{d['man'].mean()/d['q_man'].mean():.3f}**\n",
          "1.00 なら市場は正しい。1.00 を下回れば市場は万舟を**過大評価**している"
          "（＝穴を買うと損をする）。`longshot.md` の実測と同じ向きになるはず。\n",
          "市場が示す万舟確率で10分割して、実際の発生率と並べる（校正の確認）。\n",
          "| 市場が示す万舟率（10分割） | レース数 | 市場の平均 | 実測 | 実測÷市場 | 平均配当 |",
          "|---|---:|---:|---:|---:|---:|"]
    d["qb"] = pd.qcut(d["q_man"], 10, duplicates="drop")
    for k, s in d.groupby("qb", observed=True):
        a, m = s["man"].mean(), s["q_man"].mean()
        L.append(f"| {k} | {len(s):,} | {m*100:.1f}% | {a*100:.1f}% | {a/max(m,1e-9):.3f} | "
                 f"{s['pay'].mean():,.0f}円 |")

    # ---------------- 3-4. 特徴
    L += ["\n## 3. 万舟が出やすいレースの特徴\n",
          "各区分で「実測÷市場」も併記する。1.00 から離れていれば、市場がその区分を読み違えている。"]
    band(d, "ws", [-.1, .5, 2, 4, 6, 99], ["無風〜0.5m", "1〜2m", "3〜4m", "5〜6m", "7m以上"], L, "風速")
    band(d, "wave", [-.1, .5, 2, 4, 99], ["0cm", "1〜2cm", "3〜4cm", "5cm以上"], L, "波高")
    band(d, "n_a1", [-.1, .5, 1.5, 2.5, 6], ["A1が0人", "1人", "2人", "3人以上"], L, "A1級の人数")
    band(d, "q1_max", [0, .4, .5, .6, .7, 1], ["40%未満", "40〜50%", "50〜60%", "60〜70%", "70%以上"], L,
         "市場のいちばん高い1着確率（＝本命の堅さ）")
    L += ["\n### 展示タイムのばらつき（最大−最小）\n",
          "| 区分 | レース数 | 万舟率 | 市場が示す万舟率 | 実測÷市場 | 平均配当 |",
          "|---|---:|---:|---:|---:|---:|"]
    for k, s5 in d.groupby(pd.qcut(pd.to_numeric(d["ex_rng"], errors="coerce"), 5,
                                   duplicates="drop"), observed=True):
        a, m = s5["man"].mean(), s5["q_man"].mean()
        L.append(f"| {k}秒 | {len(s5):,} | {a*100:.1f}% | {m*100:.1f}% | {a/max(m,1e-9):.3f} | "
                 f"{s5['pay'].mean():,.0f}円 |")
    band(d, "mz_exh", [-9, -1.5, -.5, .5], ["展示で2人以上が前づけ", "1人が前づけ", "枠なり"], L,
         "スタート展示の進入")
    band(d, "race_no", [0, 3, 6, 9, 12], ["1〜3R", "4〜6R", "7〜9R", "10〜12R"], L, "レース番号")
    band(d, "day_no", [0, 1.5, 3.5, 9], ["初日〜2日目", "3〜4日目", "5日目以降"], L, "節の何日目")

    for col, title in (("grade", "グレード"), ("weather", "天候"), ("kimarite", "決まり手（結果・参考）")):
        L += [f"\n### {title}\n", "| 区分 | レース数 | 万舟率 | 市場が示す万舟率 | 実測÷市場 | 平均配当 |",
              "|---|---:|---:|---:|---:|---:|"]
        for k, s in d.groupby(col, observed=True):
            if len(s) < 200:
                continue
            a, m = s["man"].mean(), s["q_man"].mean()
            L.append(f"| {k} | {len(s):,} | {a*100:.1f}% | {m*100:.1f}% | {a/max(m,1e-9):.3f} | "
                     f"{s['pay'].mean():,.0f}円 |")

    L += ["\n### 場別（万舟率の高い順・上位と下位）\n",
          "| 場 | レース数 | 万舟率 | 市場が示す万舟率 | 実測÷市場 | 平均配当 |", "|---|---:|---:|---:|---:|---:|"]
    st = d.groupby("stadium").agg(n=("man", "size"), a=("man", "mean"), m=("q_man", "mean"),
                                  p=("pay", "mean")).sort_values("a", ascending=False)
    st = st[st["n"] >= 200]
    for k, r in pd.concat([st.head(6), st.tail(6)]).iterrows():
        L.append(f"| {k} | {r['n']:,.0f} | {r['a']*100:.1f}% | {r['m']*100:.1f}% | "
                 f"{r['a']/max(r['m'],1e-9):.3f} | {r['p']:,.0f}円 |")

    # ---------------- 5. 市場を超えて万舟を予測できるか
    FE = [("ws", "風速"), ("wave", "波高"), ("n_a1", "A1の人数"), ("n_b", "B級の人数"),
          ("ex_rng", "展示タイムのばらつき"), ("mz_exh", "展示の前づけ"), ("race_no", "レース番号"),
          ("day_no", "節の日数")]
    w = d.copy()
    for c, _ in FE:
        w[c] = pd.to_numeric(w[c], errors="coerce")
    w[[c for c, _ in FE]] = w[[c for c, _ in FE]].fillna(w[[c for c, _ in FE]].median())
    Zr = w[[c for c, _ in FE]].values.astype(float)
    Zr = (Zr - Zr.mean(0)) / (Zr.std(0) + 1e-9)
    Z = np.column_stack([np.ones(len(w)), Zr])
    off = logit(np.clip(w["q_man"].values, 1e-6, 1 - 1e-6))
    y = w["man"].values.astype(float)
    e = (w["race_date"] <= EXPLORE_END).values
    c = ~e
    gam = fit_logit_offset(Z[e], off[e], y[e])
    p_mkt, p_mod = expit(off), expit(off + Z @ gam)
    L += ["\n## 5. 市場を超えて万舟を予測できるか\n",
          "市場が示す万舟確率をオフセットに置いたロジスティック回帰。γ=0 なら市場がすでに持っている情報。",
          f"探索 〜{EXPLORE_END}（{int(e.sum()):,}R）で推定、確認期間（{int(c.sum()):,}R）で1回評価。\n",
          "| 予測 | 確認 対数損失 |", "|---|---:|",
          f"| 市場が示す万舟確率のみ | {bll(p_mkt[c], y[c]):.4f} |",
          f"| ＋レースの特徴 | {bll(p_mod[c], y[c]):.4f} |",
          f"| 差 | **{bll(p_mkt[c], y[c]) - bll(p_mod[c], y[c]):+.4f}** |",
          "", "| 特徴量 | 係数（正＝市場より万舟が出やすい） |", "|---|---:|",
          f"| 切片 | {gam[0]:+.4f} |"]
    for i, (_, nm) in enumerate(FE):
        L.append(f"| {nm} | {gam[i+1]:+.4f} |")
    hi = p_mod[c] >= np.quantile(p_mod[e], 0.9)
    L += ["", f"モデルが「荒れる」と見た上位10%（確認 n={int(hi.sum()):,}）の実際の万舟率: "
          f"**{y[c][hi].mean()*100:.1f}%**（全体 {y[c].mean()*100:.1f}%、"
          f"市場が示す値 {p_mkt[c][hi].mean()*100:.1f}%）"]

    # ---------------- 6. 目的2への応用を実測する
    def place_pay(js, lane):
        d0 = json.loads(js) if isinstance(js, str) else js
        for x in (d0 or {}).get("place") or []:
            try:
                if int(str(x.get("combination", "")).strip()) == lane:
                    return float(x.get("amount") or 0)
            except Exception:
                continue
        return 0.0

    w["pl_pay"] = [place_pay(js, int(a) + 1) for js, a in zip(w["payouts_js"], w["q2_arg"])]
    # 目的2の買い方: 市場がいちばん確信している艇の複勝1点、市場確信度が探索期間の上位10%のレース
    th_conf = np.quantile(w.loc[e, "q2_max"], 0.90)
    base = c & (w["q2_max"].values >= th_conf)
    gz = (Z @ gam) - gam[0]          # 市場を超えた「荒れ」信号だけ（切片を除く）
    L += ["\n## 6. 目的2（負けないツール）への応用を実測する\n",
          "**万舟を当てる方向には使えない。** `longshot.md` の実測で穴買いの回収率は",
          "単勝34.1%／2連単19.5%／3連単27.6%。市場は万舟を過大評価しているので、買うほど損をする。\n",
          "使いどころは逆側。複勝（市場がいちばん確信している艇1点・市場確信度の上位10%）の",
          "負けは荒れたレースから来る。**市場を超えた荒れ信号（γ·z）が高いレースを追加で見送ると",
          "回収率は上がるか。** 実測する。\n",
          "| 追加の見送り | n | 的中 | 平均払戻 | 回収率 |", "|---|---:|---:|---:|---:|"]
    for cut, lab in ((1.01, "見送らない"), (0.90, "γ·z 上位10%を見送る"),
                     (0.75, "上位25%を見送る"), (0.50, "上位50%を見送る")):
        if cut > 1:
            sel = base
        else:
            sel = base & (gz <= np.quantile(gz[e], cut))
        n = int(sel.sum())
        if n < 100:
            continue
        ret = w["pl_pay"].values[sel]
        L.append(f"| {lab} | {n:,} | {(ret>0).mean()*100:.1f}% | {ret[ret>0].mean():.0f}円 | "
                 f"**{ret.sum()/(100*n)*100:.1f}%** |")
    L += ["",
          "**上がらない。** 理由は、我々の荒れ信号が市場を超えて持っている情報が +0.0023 nats しかなく、",
          "しかも複勝の天井は元返し（100円下限）で決まっていて荒れの読みでは動かないため。",
          "見送りを増やすと的中率は上がるが平均払戻が100円に近づき、積は下がる"
          "（`favorite_edge.md` の「99.1%が天井」と同じ構造）。\n",
          "## 7. 分かったこと\n",
          f"- 万舟は **1日 {int((df['pay']>=10000).sum())/days:.1f}本**、**6レースに1回**。",
          f"  5万円以上が1日 {int((df['pay']>=50000).sum())/days:.1f}本、10万円以上が {int((df['pay']>=100000).sum())/days:.1f}本。",
          f"- **市場は万舟を17%ほど過大評価している**（実測 {d['man'].mean()*100:.2f}% / 市場 {d['q_man'].mean()*100:.2f}%、比 {d['man'].mean()/d['q_man'].mean():.3f}）。",
          "- **過大評価は「市場が堅いと見たレース」でいちばん大きい**（比 0.736）。",
          "  市場が荒れると見たレースではほぼ正確（比 1.006）。",
          "  つまり群衆は「堅いレースにも万が一がある」を買いすぎている。",
          "- 万舟が出やすい条件は実在する（波5cm以上 18.4%、風7m以上 18.1%、展示で2人以上の前づけ 19.5%、",
          "  節の初日〜2日目 18.2%、G2 20.2%、江戸川 19.6% / 福岡 12.1%）。",
          "  **ただしどの区分でも「実測÷市場」は1.00未満で、市場はこれらをすべて知っている。**",
          "- 市場を超えた予測力は +0.0023 nats。荒れの読みでは複勝の回収率は上がらなかった。",
          "",
          "決まり手の内訳が万舟の正体を一言で表している: **逃げ 3.6% / まくり差し 36.5%**。",
          "万舟とは「1号艇が逃げられなかったレース」のこと。1号艇の1着率は約55%で、",
          "市場はそれを正確に値付けしている。"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
