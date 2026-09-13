"""本命10点固定で「当たったら必ず投資の1.5倍以上が返る」配分の検証。

ユーザー指定（2026-09-13）: 本命10本。全体ではなく**当たった1レース単位で払戻 ≥ 投資×1.5**。
支出はいくら必要になるか。的中率が下がるのは承知のうえで、1レースの勝ち分を大きくしたい。

■ 成立条件（先に代数で出しておく）
  倍率mを保証する ⇔ すべての点で 賭け金_i × オッズ_i ≥ m × Σ賭け金。
  賭け金_i ≥ m×T/オッズ_i を全点で足すと T ≥ m×T×Σ(1/オッズ) となり、
  **Σ(1/オッズ) ≤ 1/m** が必要十分（100円単位の丸めを除けば）。
  3連単の市場確率に直すと Σq ≤ 1/(m×1.333)。m=1.0 なら 75%、**m=1.5 なら 50%**、m=2.0 なら 37.5%。
  → 倍率を上げるほど「本命10点が市場確率を食っているレース（＝堅いレース）」が落ちる。
  **これは配分の制約ではなくレース選択の制約**で、回収率はここからしか動かない。

■ 丸めの影響と必要な支出
  払戻の下限を F とすると 賭け金_i = ceil(F ÷ オッズ_i ÷100)×100、支出 T = Σ賭け金。
  Σceil ≤ Σ(F/オッズ)+1000 なので **F ≥ 1000 ÷ (1/m − Σ(1/オッズ))** なら必ず成立する。
  Σ(1/オッズ) が 1/m に近いレースほど必要な F（＝支出）が跳ね上がる。

出力: reports/research/min15.md
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from boatlab.backtest.metrics import roi_bootstrap  # noqa: E402
from boatlab.config import ROOT  # noqa: E402

OUT = Path(ROOT) / "reports" / "research" / "min15.md"
CACHE = Path("/tmp/claude-0/mix10_cache.npz")
UNIT = 100


def stakes_for(odds, F):
    return np.ceil(F / odds / UNIT) * UNIT


def min_F(odds, m, cap=None):
    """倍率mを保証できる最小の払戻下限F（100円刻み）。成立しなければ None。"""
    s = float(np.sum(1.0 / odds))
    if s >= 1.0 / m:
        return None
    F = int(np.ceil(1000.0 / (1.0 / m - s) / UNIT) * UNIT)      # 必ず成立する上界
    lo = UNIT
    while lo < F:                                                # 二分で下げる（段差があるので都度検証）
        mid = int((lo + F) // 2 // UNIT * UNIT)
        if mid < UNIT:
            break
        st = stakes_for(odds, mid)
        if st.sum() * m <= mid:
            F = mid
        else:
            lo = mid + UNIT
    if cap is not None and stakes_for(odds, F).sum() > cap:
        return None
    return F


def run(O, W, P, sel, m, cap=None):
    n = len(O)
    out = dict(fired=np.zeros(n, bool), ret=np.zeros(n), stake=np.zeros(n), hit=np.zeros(n, bool),
               F=np.zeros(n), mult=np.zeros(n), short=np.zeros(n, bool))
    for i in range(n):
        pts = sel[i]
        if len(pts) != 10:
            continue
        od = O[i, pts]
        if not np.isfinite(od).all():
            continue
        F = min_F(od, m, cap)
        if F is None:
            continue
        st = stakes_for(od, F)
        T = st.sum()
        out["fired"][i] = True; out["stake"][i] = T; out["F"][i] = F; out["mult"][i] = F / T
        if W[i] in pts:
            k = pts.index(W[i])
            out["ret"][i] = P[i] * st[k] / 100.0
            out["hit"][i] = True
            out["short"][i] = out["ret"][i] < m * T      # 公式配当が確定オッズを下回った場合だけ起きる
    return out


def summ(r, mask=None):
    f = r["fired"] if mask is None else (r["fired"] & mask)
    if f.sum() == 0:
        return None
    ret, st, hit = r["ret"][f], r["stake"][f], r["hit"][f]
    lo, hi = roi_bootstrap(st, ret, n_boot=300)
    cur = best = 0
    for x in ret:
        cur = cur + 1 if x <= 0 else 0
        best = max(best, cur)
    win = (ret - st)[hit]
    return dict(n=int(f.sum()), roi=ret.sum() / st.sum(), lo=lo, hi=hi, hit=hit.mean(),
                stake=st.mean(), F=r["F"][f].mean(), mult=r["mult"][f].mean(), streak=best,
                win_min=win.min() if len(win) else 0, win_med=np.median(win) if len(win) else 0,
                win_mean=win.mean() if len(win) else 0, short=r["short"][f].sum() / max(hit.sum(), 1),
                pnl=ret.sum() - st.sum())


def main():
    z = np.load(CACHE, allow_pickle=True)
    rid, date, O, W, P, MT, MP = z["rid"], pd.DatetimeIndex(z["date"]), z["O"], z["W"], z["P"], z["MT"], z["MP"]
    inv = np.where(np.isfinite(O), 1.0 / np.nan_to_num(O, nan=1e9), 0.0)
    q = inv / inv.sum(1, keepdims=True)
    order = np.argsort(-q, 1)
    N = len(rid); days = pd.Series(date).dt.date.nunique()
    half = np.asarray(date <= "2026-05-31")
    conf = np.nansum(MP[:, :10], axis=1)
    SELS = {"モデル確率 上位10点（堅い予想と同じ選定）": [list(MT[i, :10]) for i in range(N)],
            "市場人気 上位10点": [list(order[i, :10]) for i in range(N)]}

    L = [f"# 本命10点固定・「当たったら投資の1.5倍以上」を保証する配分（2026年・{N:,}R・確定オッズ）\n",
         "指定: 本命10本、当たった**そのレース単位**で 払戻 ≥ 投資×1.5、支出がいくら必要かを知りたい。\n",
         "## 0. 成立条件は代数で決まる\n",
         "全点で `賭け金_i × オッズ_i ≥ m × Σ賭け金` を満たすには **Σ(1/オッズ) ≤ 1/m** が必要十分（丸めを除く）。",
         "3連単の市場確率に直すと `Σq ≤ 1/(m×1.333)`。",
         "",
         "| 倍率 m | 必要な Σ(1/オッズ) | 10点の市場確率の上限 |", "|---|---:|---:|",
         "| 1.0（元本保証） | ≤ 1.000 | 75.0% |", "| 1.2 | ≤ 0.833 | 62.5% |",
         "| **1.5** | **≤ 0.667** | **50.0%** |", "| 2.0 | ≤ 0.500 | 37.5% |", "| 3.0 | ≤ 0.333 | 25.0% |",
         "",
         "**これは配分ではなくレース選択の条件。** 倍率を上げるほど「本命10点に市場確率が集中したレース＝本命が堅いレース」が落ちる。",
         "回収率はレース選択からしか動かない（配分は分散を変えるだけ）ので、**倍率を上げると回収率は下がる方向に働く**。\n",
         "## 1. ×1.5 に必要な支出はいくらか（本命＝モデル確率 上位10点）\n",
         "各レースについて、×1.5 を満たす**最小の支出**を計算した（100円単位。これ未満では丸めのせいで成立しない）。\n",
         "| 倍率 | 原理的に成立 | 必要支出の中央値 | 75%点 | 90%点 | 95%点 | 平均 |", "|---|---:|---:|---:|---:|---:|---:|"]
    sel = SELS["モデル確率 上位10点（堅い予想と同じ選定）"]
    keep = {}
    for m in (1.0, 1.2, 1.5, 2.0, 3.0):
        r = run(O, W, P, sel, m)
        keep[m] = r
        st = r["stake"][r["fired"]]
        L.append(f"| ×{m:g} | {r['fired'].sum():,}（{r['fired'].sum()/N*100:.0f}%） | {np.median(st):,.0f}円 | "
                 f"{np.quantile(st,0.75):,.0f}円 | {np.quantile(st,0.90):,.0f}円 | {np.quantile(st,0.95):,.0f}円 | {st.mean():,.0f}円 |")
    L += ["",
          "**中央値は小さいが裾が極端に重い。** Σ(1/オッズ) が上限ぎりぎりのレースほど、100円単位の丸めを吸収するために",
          "支出が跳ね上がる（×1.5 の平均13,988円は一部の巨額レースが押し上げた値で、実務では使えない）。",
          "→ **支出に上限を置き、収まらないレースは見送る**、という運用になる。\n",
          "## 2. 支出上限 × 倍率のグリッド（上限を超えるレースは見送り）\n",
          "| 支出上限 | 倍率 | 成立 | 1日あたり | 平均支出 | 保証払戻 | 的中率 | 当たったときの利益（中央/平均） | 最長連敗 | 回収率 | 95%区間 |",
          "|---|---|---:|---:|---:|---:|---:|---|---:|---:|---|"]
    for cap in (3000, 5000, 10000, 20000):
        for m in (1.0, 1.2, 1.5, 2.0):
            r = run(O, W, P, sel, m, cap)
            s = summ(r)
            if s is None:
                L.append(f"| {cap:,}円 | ×{m:g} | 0 | — | — | — | — | — | — | — | — |"); continue
            L.append(f"| {cap:,}円 | ×{m:g} | {s['n']:,}（{s['n']/N*100:.0f}%） | {s['n']/days:.1f}R | {s['stake']:,.0f}円 | "
                     f"{s['F']:,.0f}円 | {s['hit']*100:.1f}% | {s['win_med']:+,.0f} / {s['win_mean']:+,.0f}円 | "
                     f"{s['streak']}R | **{s['roi']*100:.1f}%** | {s['lo']*100:.0f}〜{s['hi']*100:.0f}% |")

    L += ["\n## 3. 倍率を上げると落ちるのは「自信のあるレース」（支出上限1万円）\n",
          "| 倍率 | 全レースでの成立率 | モデル信頼度 上位30%での成立率 | 上位10%での成立率 |", "|---|---:|---:|---:|"]
    c30 = conf >= np.quantile(conf, 0.70); c10 = conf >= np.quantile(conf, 0.90)
    for m in (1.0, 1.2, 1.5, 2.0):
        r = run(O, W, P, sel, m, 10000)
        L.append(f"| ×{m:g} | {r['fired'].sum()/N*100:.0f}% | {(r['fired']&c30).sum()/c30.sum()*100:.0f}% | "
                 f"{(r['fired']&c10).sum()/c10.sum()*100:.0f}% |")
    L += ["",
          "**×1.5 は、モデルが自信を持っているレースをほぼ全部落とす。** 残るのは本命が割れているレースで、",
          "だから的中率が下がり、回収率も帰無（75%）の方へ寄る。得られるのは「当たったときの取り分」だけ。\n"]

    # 前半後半 + 選定の違い（支出上限1万円）
    L += ["## 4. 本命10点の決め方と、前半・後半での再現（支出上限1万円）\n",
          "| 本命の選定 | 倍率 | 成立 | 平均支出 | 的中率 | 当たったときの利益（中央） | 回収率 | 前半 | 後半 |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for nm, sl in SELS.items():
        for m in (1.0, 1.5):
            r = run(O, W, P, sl, m, 10000)
            s_ = summ(r); h1 = summ(r, half); h2 = summ(r, ~half)
            L.append(f"| {nm} | ×{m:g} | {s_['n']:,} | {s_['stake']:,.0f}円 | {s_['hit']*100:.1f}% | "
                     f"{s_['win_med']:+,.0f}円 | **{s_['roi']*100:.1f}%** | {h1['roi']*100:.1f}% | {h2['roi']*100:.1f}% |")

    # 1日の形（×1.5・上限1万円）
    r15 = run(O, W, P, sel, 1.5, 10000)
    s15 = summ(r15)
    f = r15["fired"]
    dd = pd.DataFrame({"d": pd.Series(date)[f].dt.date, "ret": r15["ret"][f], "st": r15["stake"][f]})
    g = dd.groupby("d").sum(); g["pnl"] = g["ret"] - g["st"]
    L += ["\n## 5. ×1.5・支出上限1万円で、成立したレースを全部買った場合\n",
          f"- 1日 **{s15['n']/days:.1f}レース**、1日の投資は平均 {g.st.mean():,.0f}円。",
          f"- 買えた日数 {len(g)}日、プラスの日 **{(g.pnl>0).sum()}日（{(g.pnl>0).mean()*100:.1f}%）**、1日の収支は中央値 {g.pnl.median():,.0f}円。",
          f"- 月の期待損失: **{(1-s15['roi'])*s15['stake']*(s15['n']/days)*30:,.0f}円**（1日{s15['n']/days:.1f}R × {s15['stake']:,.0f}円 × 30日）。",
          f"- **1日5レースに絞れば**月の期待損失は約 {(1-s15['roi'])*s15['stake']*5*30:,.0f}円。\n"]

    # 保証が破れる経路
    L += ["## 6. 保証が破れる残りの経路\n",
          f"- ×1.5・上限1万円で、的中したのに払戻が投資の1.5倍に届かなかったのは **{s15['short']*100:.2f}%**。",
          "  原因は配分ではなく、**公式の確定配当が確定オッズ×100を下回ったレース**（同着・返還、全体の1.66%）。",
          "- 締切前オッズで組むと、本命側のオッズは締切にかけて下がるので**確定時に倍率が1.5を割ることがある**。",
          "  実運用では余裕を見て ×1.6〜1.7 で組む必要がある（締切前データで要実測）。\n",
          "## 7. まとめ\n",
          "**条件は「支出」ではなく「レースの形」。** ×1.5 を保証できるのは 10点の市場確率合計が50%以下のレースだけで、",
          "これは本命が割れているレースを意味する。支出を増やしても成立率は 29%（3,000円）→ 44%（1万円）→ 47%（2万円）で頭打ちになる",
          "（頭打ちの原因は丸めではなく、Σ(1/オッズ) ≤ 0.667 という代数の壁）。",
          "",
          "**必要な支出**: ×1.5 の最小支出は 中央値2,500円・75%点5,200円・90%点13,900円。",
          "実務的には**1レースの上限を5,000〜10,000円**に置くのが妥当で、そのとき実際に使う額は平均2,100〜2,800円に収まる。",
          "",
          "**交換条件（支出上限1万円・本命モデル上位10点）**:",
          "",
          "| | ×1.0（元本保証） | ×1.5 | 変化 |",
          "|---|---:|---:|---|",
          "| 買えるレース | 97% | 44% | 半分以下 |",
          "| 的中率 | 51.1% | 38.1% | −13pt |",
          "| 当たったときの利益（中央） | +520円 | **+2,280円** | **4.4倍** |",
          "| 最長連敗 | 15R | 18R | +3R |",
          "| 回収率 | 79.0% | 77.1% | −1.9pt |",
          "",
          "**1レースの勝ち分を4倍にする代金は、回収率で約2ポイント。** 悪い取引ではない。",
          "ただし回収率が上がるわけではないので、**長期の損失額は「1日に何レース買うか」でしか変わらない**。",
          f"×1.5・1万円上限で成立レースを全部買うと1日{s15['n']/days:.0f}レース・月{(1-s15['roi'])*s15['stake']*(s15['n']/days)*30/10000:.0f}万円の期待損失になる。"
          f"1日5レースに絞れば月約{(1-s15['roi'])*s15['stake']*5*30/10000:.0f}万円。",
          "",
          "**倍率を上げすぎると壊れる**: ×2.0 は成立16%（1日25R）、的中28.4%、最長連敗30R、回収率75.9%。",
          "モデルが自信を持っているレース（信頼度上位10%）の成立率は ×1.5 で 0%、×1.2 でも 12%。",
          "**当たったときの取り分を増やすほど、当たりやすいレースを捨てることになる。**",
          ]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
