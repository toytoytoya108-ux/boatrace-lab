"""飛躍8〜10。1-2-3 の発見を一般化し、その裏にある機構（決まり手）と市場の厚さを測る。

飛躍8: **1-2-3 は特別なのか、それとも「整い方」の端なのか。**
  着順が艇番順にどれだけ近いかを転倒数（0〜3）で測り、**120通り全部の「毎レース買う回収率」**を出す。
  これは結果と公式配当だけで計算でき、46万レース全数・選択の余地なし。
  勾配があるなら「整った決着ほど安い」という一般法則で、1-2-3 はその端にすぎない。

飛躍9: **`leap_self.md` の診断（PLは決まり手を書けない）を直接測る。**
  決まり手を知ったら120通りの不確かさはどれだけ減るか（相互情報量）。
  そしてレース前の情報から決まり手はどれだけ当てられるか。**積が「決まり手経由で取れる上限」。**

飛躍10: **市場が薄いレースほど歪みは大きいのでは。**
  Σ(1/オッズ) は丸めのぶん 1.333 より大きくなる。**薄いプールほど丸めが効く**ので、この値は厚さの代理になる。
  厚さで層別して歪み（実測÷市場）を見る。

出力: reports/research/leaps4.md
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from boatlab.config import ROOT  # noqa: E402
from boatlab.model.trifecta import PERM_LABELS, PERMS  # noqa: E402

OUT = Path(ROOT) / "reports" / "research" / "leaps4.md"
FEAT = Path(ROOT) / "reports" / "research" / "manshu_features.parquet"
CACHE = Path("/tmp/claude-0/mix10_cache.npz")
_A = np.array([p[0] for p in PERMS]) + 1
_B = np.array([p[1] for p in PERMS]) + 1
_C = np.array([p[2] for p in PERMS]) + 1
INV = ((_A > _B).astype(int) + (_A > _C).astype(int) + (_B > _C).astype(int))   # 転倒数 0〜3


def main():
    df = pd.read_parquet(FEAT)
    df["pay"] = pd.to_numeric(df["pay"], errors="coerce").fillna(0.0)
    lab2 = {l: i for i, l in enumerate(PERM_LABELS)}
    df["ci"] = df["trifecta"].map(lab2)
    d = df[df["ci"].notna()].copy()
    d["ci"] = d["ci"].astype(int)
    N = len(d)
    L = [f"# 飛躍8〜10（2018〜2026・{N:,}レース）\n",
         "## 飛躍8: 1-2-3 は特別か、それとも『整い方』の端か\n",
         "着順が艇番順からどれだけ崩れているかを**転倒数**で測る（1-2-3 は0、3-2-1 は3）。",
         "**120通りすべてについて「毎レース買ったときの回収率」**を出す。",
         "`回収率 = その目が来る確率 × そのときの平均配当 ÷100` で、結果と公式配当だけから計算できる。",
         "46万レース全数なので選択の余地がない。**帰無（市場が正しい）なら全部75%になるはず。**\n"]
    cnt = np.bincount(d["ci"].values, minlength=120)
    paysum = np.bincount(d["ci"].values, weights=d["pay"].values, minlength=120)
    roi120 = paysum / (100.0 * N)                       # 毎レースその目を買ったときの回収率
    L += ["| 転倒数 | 目の数 | 出現率の合計 | 平均回収率 | 最小 | 最大 | 例 |", "|---|---:|---:|---:|---:|---:|---|"]
    for k in range(4):
        m = INV == k
        ex = ", ".join(PERM_LABELS[i] for i in np.flatnonzero(m)[:3])
        L.append(f"| {k}（{'艇番順そのまま' if k == 0 else '崩れ' + str(k)}） | {int(m.sum())} | "
                 f"{cnt[m].sum()/N*100:.1f}% | **{roi120[m].mean()*100:.1f}%** | {roi120[m].min()*100:.1f}% | "
                 f"{roi120[m].max()*100:.1f}% | {ex} |")
    L += ["", "**回収率の高い順 上位12（毎レース買った場合）**", "",
          "| 順 | 目 | 転倒数 | 出現率 | 平均配当 | 回収率 |", "|---|---|---:|---:|---:|---:|"]
    order = np.argsort(-roi120)
    for r, i in enumerate(order[:12], 1):
        L.append(f"| {r} | {PERM_LABELS[i]} | {INV[i]} | {cnt[i]/N*100:.2f}% | "
                 f"{paysum[i]/max(cnt[i],1):,.0f}円 | **{roi120[i]*100:.1f}%** |")
    L += ["", "**回収率の低い順 下位6**", "", "| 目 | 転倒数 | 出現率 | 回収率 |", "|---|---:|---:|---:|"]
    for i in order[-6:]:
        L.append(f"| {PERM_LABELS[i]} | {INV[i]} | {cnt[i]/N*100:.3f}% | {roi120[i]*100:.1f}% |")
    # 「1号艇が1着」で分けたときの転倒数の効果
    L += ["\n### 1号艇が1着のときに限ると（転倒数の効果は艇番の効果と別か）\n",
          "| 転倒数 | 目の数 | 平均回収率 | 例 |", "|---|---:|---:|---|"]
    for k in range(3):
        m = (INV == k) & (_A == 1)
        if m.sum() == 0:
            continue
        L.append(f"| {k} | {int(m.sum())} | **{roi120[m].mean()*100:.1f}%** | "
                 f"{', '.join(PERM_LABELS[i] for i in np.flatnonzero(m)[:4])} |")

    # 飛躍8の本当の軸は「整い方」ではなく「出やすさ」ではないか
    L += ["\n### 本当の軸は『整い方』ではなく『出やすさ』ではないか\n",
          "120通りを**出現率**で並べ直す。回収率が出現率に沿って上がるなら、1-2-3 は",
          "「いちばん出やすい目」だから安いだけで、整列は関係ないことになる。\n",
          "| 出現率の帯 | 目の数 | 平均出現率 | 平均配当 | 平均回収率 | 転倒数の平均 |", "|---|---:|---:|---:|---:|---:|"]
    fr = cnt / N
    qs8 = np.quantile(fr, [0.2, 0.4, 0.6, 0.8])
    e8 = [-1] + list(qs8) + [1]
    for k in range(5):
        m = (fr > e8[k]) & (fr <= e8[k + 1])
        if m.sum() == 0:
            continue
        L.append(f"| {e8[k]*100:.2f}%〜{e8[k+1]*100:.2f}% | {int(m.sum())} | {fr[m].mean()*100:.2f}% | "
                 f"{(paysum[m] / np.maximum(cnt[m], 1)).mean():,.0f}円 | **{roi120[m].mean()*100:.1f}%** | {INV[m].mean():.2f} |")
    cr = np.corrcoef(np.log(np.clip(fr, 1e-9, None)), roi120)[0, 1]
    L += ["", f"出現率（log）と回収率の相関 **{cr:+.3f}**。**出やすい目ほど安い**という関係がはっきりある。",
          f"転倒数との相関は {np.corrcoef(INV, roi120)[0, 1]:+.3f} で、出現率を通した見かけの関係。",
          "→ **1-2-3 が特別なのではなく、『いちばん出やすい目』が最も過小評価される**",
          "（人気薄バイアスの、目のレベルでの現れ方）。ただし 1-2-3 は出現率も回収率も単独首位で、",
          "2位の 1-3-2（5.24%・80.7%）を出現率で1.9pt・回収率で2.2pt上回る。\n"]

    # ================= 飛躍9: 決まり手
    L += ["\n## 飛躍9: 決まり手を知ったら、どれだけ分かるか\n",
          "`leap_self.md` の診断は「PL は『誰が1着かで2着の顔ぶれが変わる』を書けず 0.0385 nats 失う」だった。",
          "その正体が決まり手なら、**決まり手を知ると120通りの不確かさが大きく減る**はず。まず上限を測る。\n"]
    km = d["kimarite"].fillna("不明")
    top_km = km.value_counts()
    L += ["| 決まり手 | 割合 | そのときの1-2-3率 | 平均配当 |", "|---|---:|---:|---:|"]
    for k, c in top_km.head(7).items():
        s = d[km == k]
        L.append(f"| {k} | {c/N*100:.1f}% | {(s['ci'] == lab2['1-2-3']).mean()*100:.1f}% | {s['pay'].mean():,.0f}円 |")
    # 相互情報量 I(目; 決まり手)
    p_c = cnt / N
    H = -np.sum(p_c[p_c > 0] * np.log(p_c[p_c > 0]))
    Hc = 0.0
    for k, c in top_km.items():
        s = d.loc[km == k, "ci"].values
        pk = np.bincount(s, minlength=120) / len(s)
        Hc += (c / N) * (-np.sum(pk[pk > 0] * np.log(pk[pk > 0])))
    L += ["", f"- 120通りの不確かさ（エントロピー） **{H:.4f} nats**（無情報なら log120 = {np.log(120):.4f}）。",
          f"- 決まり手を知った後 **{Hc:.4f} nats**。→ **相互情報量 I(目; 決まり手) = {H - Hc:.4f} nats**。",
          f"- 市場が持っている情報は 0.6353 nats（1着基準）／3連単では {np.log(120) - 3.7058:.4f} nats。",
          f"  **決まり手は3連単の不確かさの {(H - Hc) / (np.log(120) - 3.7058) * 100:.0f}% を説明する。**",
          "",
          "つまり「決まり手が分かれば勝てる」は正しい。問題は**レース前に決まり手が当てられるか**。\n"]
    # レース前の情報から決まり手を当てられるか
    from sklearn.metrics import log_loss
    import lightgbm as lgb
    FEATS = ["l1_klass_n", "l1_nwr", "l1_lwr", "l1_motor", "l1_avg_st", "l1_ext", "l1_st_exh", "l1_ext_rank",
             "l1_stx_rank", "l1_nwr_rank", "l1_tilt", "oth_nwr_max", "oth_motor_max", "nwr_gap", "nwr_std",
             "n_a1", "n_b", "ext_rng", "n_maezuke", "n_parts", "ws", "wave", "temp_c", "water_temp_c",
             "race_no", "day_no", "night", "stadium_code"]
    keep = [c for c in FEATS if c in d.columns]
    km5 = km.where(km.isin(top_km.head(5).index), "その他")
    cats = sorted(km5.unique())
    ymap = {c: i for i, c in enumerate(cats)}
    y = km5.map(ymap).values
    tr = (d["year"] <= 2023).values
    te = (d["year"] >= 2024).values
    X = d[keep].astype(float).values
    mdl = lgb.train(dict(objective="multiclass", num_class=len(cats), learning_rate=0.08, num_leaves=63,
                         min_data_in_leaf=200, feature_fraction=0.8, verbose=-1, num_threads=2, seed=7),
                    lgb.Dataset(X[tr], label=y[tr]), num_boost_round=250)
    pr = mdl.predict(X[te])
    base = np.bincount(y[tr], minlength=len(cats)) / tr.sum()
    ll_base = log_loss(y[te], np.tile(base, (te.sum(), 1)), labels=list(range(len(cats))))
    ll_mdl = log_loss(y[te], pr, labels=list(range(len(cats))))
    L += ["### レース前の情報から決まり手は当てられるか（2018〜2023で学習 → 2024〜2026で評価）\n",
          f"- 決まり手の種類: {', '.join(cats)}",
          f"- 基準（出現率だけ）の対数損失 **{ll_base:.4f}** → モデル **{ll_mdl:.4f}**（**{ll_base - ll_mdl:+.4f} nats**）。",
          f"- 決まり手が完全に分かれば {H - Hc:.4f} nats 取れるところ、レース前に取れるのは "
          f"**{(ll_base - ll_mdl) / (H - Hc) * 100:.1f}%** にあたる {ll_base - ll_mdl:.4f} nats。",
          "",
          f"→ **決まり手経由で取れる上限は {ll_base - ll_mdl:.4f} nats 程度。必要量 0.2877 の "
          f"{(ll_base - ll_mdl) / 0.2877 * 100:.1f}%。**",
          "  ただしこれは「市場を条件づけていない」値で、市場も同じことを知っている可能性が高い（後段で確認）。\n"]

    # ================= 飛躍10: 市場の厚さ
    L += ["## 飛躍10: 市場が薄いレースほど歪みは大きいか\n",
          "Σ(1/オッズ) は丸めのぶん 1.333 より大きくなる。**薄いプールほど刻みが粗く丸めが効く**ので、",
          "この値は「市場の厚さ」の代理になる。厚さで層別して歪み（実測÷市場）を見る。\n"]
    z = np.load(CACHE, allow_pickle=True)
    O, W, P = z["O"], z["W"], z["P"]
    inv_sum = np.where(np.isfinite(O), 1.0 / np.nan_to_num(O, nan=1e9), 0.0).sum(1)
    Q = np.where(np.isfinite(O), 1.0 / np.nan_to_num(O, nan=1e9), 0.0) / inv_sum[:, None]
    n26 = len(O)
    i123 = PERM_LABELS.index("1-2-3")
    qs = np.quantile(inv_sum, [0.2, 0.4, 0.6, 0.8])
    edges = [-1e9] + list(qs) + [1e9]
    L += ["| Σ(1/オッズ)（大きいほど薄い） | レース数 | 実効の払戻率 | 人気1番の比 | 1-2-3 の回収率 | 最不人気1点の比 |",
          "|---|---:|---:|---:|---:|---:|"]
    top1 = np.argmax(Q, axis=1)
    bot1 = np.argmin(np.where(Q > 0, Q, 9), axis=1)
    for k in range(5):
        m = (inv_sum > edges[k]) & (inv_sum <= edges[k + 1])
        if m.sum() < 500:
            continue
        rate = 1.0 / inv_sum[m].mean()
        hit_t = (W[m] == top1[m])
        roi_t = np.where(hit_t, P[m], 0.0).sum() / (100.0 * m.sum())
        hit_b = (W[m] == bot1[m])
        roi_b = np.where(hit_b, P[m], 0.0).sum() / (100.0 * m.sum())
        roi_1 = np.where(W[m] == i123, P[m], 0.0).sum() / (100.0 * m.sum())
        L.append(f"| {edges[k] if k else 1.30:.4f}〜{edges[k+1] if k < 4 else inv_sum.max():.4f} | {int(m.sum()):,} | "
                 f"{rate*100:.2f}% | {roi_t/rate:.3f} | {roi_1*100:.1f}% | {roi_b/rate:.3f} |")
    L.append("\n（比は 回収率 ÷ そのレース群の実効払戻率。1.00 なら市場は正しい）\n")
    # 交絡の確認: Σ(1/オッズ) は本当に「厚さ」か
    fav_q = Q.max(1)
    L += ["### 交絡の確認: Σ(1/オッズ) は『厚さ』の代理になっていたか\n",
          f"- 全レースで120通りのオッズが揃っている（欠けは0件）。Σ の範囲は {inv_sum.min():.4f}〜{inv_sum.max():.4f}。",
          f"- **Σ と 本命の市場確率の相関は +{np.corrcoef(inv_sum, fav_q)[0,1]:.3f}**（本命オッズとは "
          f"{np.corrcoef(inv_sum, O.min(1))[0,1]:.3f}）。",
          f"- Σ下位20%の本命オッズ中央値 {np.median(O.min(1)[inv_sum <= np.quantile(inv_sum,0.2)]):.2f}倍 に対し、",
          f"  Σ上位20%は {np.median(O.min(1)[inv_sum >= np.quantile(inv_sum,0.8)]):.2f}倍。",
          "",
          "**つまり Σ(1/オッズ) は市場の厚さではなく『本命がどれだけ堅いか』の代理だった。**",
          "丸め損は低オッズの買い目で大きく、Σ は 1/オッズ で重みづけされるので、本命が短いほど Σ が膨らむ。",
          "上の勾配は既知の堅い／荒れの軸の言い換えで、**新しい歪みではない。飛躍10は空振り。**\n"]

    # ================= 飛躍9 の追い込み: 決まり手は市場の上に足せるか
    L += ["## 飛躍9の追い込み: 予測した決まり手は、市場の値段の上に足せるか\n",
          "上の 0.0619 nats は市場を条件づけていない。**市場も同じことを知っているなら上積みはゼロ。**",
          "2018〜2023 から `r_k(目) = P(目|決まり手k) / P(目)` を作り、2026年の各レースで",
          "`lift = Σ_k P(k|レース前の情報) × r_k` を計算して `p ∝ q × lift^γ` を当てる。",
          "**γ は2026年前半で推定し、後半で1回だけ評価する。**\n"]
    p_c = np.clip(cnt / N, 1e-9, None)
    R_k = np.zeros((len(cats), 120))
    ALPHA = 2000.0
    for ci_, c0 in enumerate(cats):
        m = (km5 == c0).values & tr
        nk = np.bincount(d.loc[m, "ci"].values, minlength=120).astype(float)
        R_k[ci_] = ((nk + ALPHA * p_c) / (nk.sum() + ALPHA)) / p_c
    rid26 = z["rid"].astype(np.int64)
    d26 = d[d["year"] == 2026].copy()
    pos26 = {int(r): i for i, r in enumerate(rid26)}
    d26["k"] = d26["race_id"].astype(np.int64).map(pos26)
    d26 = d26[d26["k"].notna()].copy(); d26["k"] = d26["k"].astype(int)
    Pk = mdl.predict(d26[keep].astype(float).values)
    lift = Pk @ R_k
    idx26 = d26["k"].values
    q26 = Q[idx26]; w26 = W[idx26]
    dt26 = pd.to_datetime(d26["race_date"])
    h1 = (dt26 <= "2026-05-31").values
    def ll_for(gam, m):
        lp = np.log(np.clip(q26[m], 1e-12, None)) + gam * np.log(np.clip(lift[m], 1e-6, None))
        lp -= lp.max(1, keepdims=True)
        pp = np.exp(lp); pp /= pp.sum(1, keepdims=True)
        return float(-np.mean(np.log(np.clip(pp[np.arange(m.sum()), w26[m]], 1e-12, None))))
    gams = np.round(np.arange(0.0, 0.81, 0.05), 2)
    ex_ll = [ll_for(g, h1) for g in gams]
    g_best = float(gams[int(np.argmin(ex_ll))])
    L += ["| γ | 2026前半（探索） | 2026後半（確認） |", "|---|---:|---:|"]
    for g, e in zip(gams, ex_ll):
        if g in (0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8) or g == g_best:
            L.append(f"| {'**' if g == g_best else ''}{g:g}{'**' if g == g_best else ''} | {e:.4f} | {ll_for(g, ~h1):.4f} |")
    L += ["", f"探索で最良の γ = **{g_best:g}**。確認期間の3連単対数損失は "
          f"**{ll_for(g_best, ~h1):.4f}**（市場のみ {ll_for(0.0, ~h1):.4f}、差 {ll_for(g_best, ~h1) - ll_for(0.0, ~h1):+.4f} nats）。",
          f"- 必要量 0.2877 nats に対して **{max(0.0, ll_for(0.0, ~h1) - ll_for(g_best, ~h1)) / 0.2877 * 100:.2f}%**。", ""]

    L += ["## まとめ: 飛躍8〜10",
          "",
          "### 飛躍8（整い方）— **1-2-3 は特別ではなかった。軸は『出やすさ』**",
          "転倒数（艇番順からの崩れ）で並べても勾配は出ない: 転倒0 **65.4%** / 転倒1 **66.6%** / 転倒2 57.7% / 転倒3 53.4%。",
          "1号艇が1着のものに限れば 転倒0 71.8% と 転倒1 72.5% で**ほぼ同じ**。整列そのものは効いていない。",
          "",
          "本当の軸は**出やすさ**だった。120通りを出現率で5分割すると回収率は",
          "**46.8% → 57.4% → 63.3% → 66.2% → 72.5%** と単調。出現率(log)との相関 **+0.875**。",
          "**人気薄バイアスが、レース単位ではなく『目』の単位で現れている。**",
          "1-2-3 が最強なのは、それが46万レースで最も出やすい目（7.15%）だから。",
          "",
          "実用上の含意: 1-2-3 に固執する理由はない。**出現率の高い目ほど安い**ので、",
          "「その日その条件で最も出やすい目」を当てる方が一般的な形になる。ただしそれは結局モデルの仕事で、",
          "固定の 1-2-3 が市場の1番人気（76.6%）を6pt上回るという事実は変わらない。",
          "",
          "### 飛躍9（決まり手）— **上限は巨大、市場を超える分はゼロ**",
          "- 決まり手を知ると120通りの不確かさが **0.7168 nats** 減る。**市場が3連単に持っている情報の66%**にあたる。",
          "  「決まり手が分かれば勝てる」は数字の上でも正しい。",
          "- レース前の情報から決まり手を当てると、出現率だけの基準より **+0.0619 nats**（必要量の21.5%）。",
          "  **これだけ見ると今までで最大。**",
          "- **しかし市場の値段の上に乗せると +0.0003 nats（必要量の0.10%）。** γ=0.15 で確認期間 3.6887 vs 市場 3.6890。",
          "  **市場は、我々が予測できる範囲の決まり手をすでに全部知っている。**",
          "",
          "ここは分けて理解する必要がある:",
          "- **0.0385 nats**（`leap_self.md`）＝ 我々のモデルの*構造*が捨てている分。**市場に近づくために取り戻せる。**",
          "- **0.0003 nats**（今回）＝ 決まり手予測が*市場を超える*分。**取り戻せない。壁はここ。**",
          "",
          "### 飛躍10（市場の厚さ）— **空振り。しかも代理変数が間違っていた**",
          "Σ(1/オッズ) を「薄さ」の代理にしたが、**本命の市場確率との相関が +0.803**（本命オッズとは −0.689）で、",
          "実体は『本命がどれだけ堅いか』だった。丸め損は低オッズの目で大きく、Σ は 1/オッズ で重みづけされるため。",
          "Σ上位20%で1-2-3が90.9%になるのは、**堅いレースで1-2-3が安いという既知の話の言い換え**。",
          "市場の厚さを測るには売上高そのものが要るが、公式は公表していない。**この線は測りようがない。**",
          ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
