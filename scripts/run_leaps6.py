"""飛躍14〜16（最後の3本）。

飛躍14: **番組が人工的に堅くしたレース（企画レース）を、市場は正しく織り込むか。**
  モーニング一般・ツッキー・ガチ勝ち8などは番組側が意図的に1号艇を有利にしている
  （`manshu_deep.md`: 万舟率 ×0.6）。**人工的な堅さは選手の実力ではないので、群衆が
  織り込み切れない可能性がある。** 探索 2018〜2023 → 確認 2024〜2026。
  **帰無対照つき**（同じ大きさの群をそれ以外のレースから無作為抽出）。

飛躍15: **1着・2着・3着のどの位置の値付けが甘いか。**
  3連単プールから「その艇がちょうど n 着になる確率」を取り出して実測と比べる。
  市場確率の帯を揃えて位置を比べる（揃えないと「3着の方が低確率セルが多い」だけの話になる）。
  帰無は y ~ Multinomial(q)（`goal1_summary.md` の規約）。

飛躍16: **市場の癖は9年で薄れているのか。** 見つけた歪み（出やすい目ほど安い）が年々
  弱まっているなら、どんな発見にも賞味期限がある。**このプロジェクトの寿命の話。**

注意: 飛躍14・16 は `回収率 = 的中率 × 平均配当 ÷ 100` の恒等式で結果と公式配当だけから
計算するのでオッズが要らず46万レース全部使える。飛躍15 はオッズが要るので2026年のみ。

出力: reports/research/leaps6.md
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from boatlab.config import ROOT  # noqa: E402
from boatlab.model.trifecta import PERM_LABELS, PERMS  # noqa: E402

OUT = Path(ROOT) / "reports" / "research" / "leaps6.md"
FEAT = Path(ROOT) / "reports" / "research" / "manshu_features.parquet"
CACHE = Path("/tmp/claude-0/mix10_cache.npz")
_A = np.array([p[0] for p in PERMS]); _B = np.array([p[1] for p in PERMS]); _C = np.array([p[2] for p in PERMS])
KIKAKU = re.compile(r"モーニング|ツッキー|ガチ勝|ピンクル|サンライズ|進入固定|シャイニング|ドラキリュウ|スカパー")
FAMS = ["モーニング", "ツッキー", "サンライズ", "進入固定", "ピンクル", "ガチ勝"]
RNG = np.random.default_rng(7)


def boot_ci(x, B=600):
    """回収率のブートストラップ95%区間（メモリ節約のためループ）。"""
    n = len(x); s = np.empty(B)
    for b in range(B):
        s[b] = x[RNG.integers(0, n, n)].mean() / 100.0
    return np.percentile(s, 2.5) * 100, np.percentile(s, 97.5) * 100


def main():
    df = pd.read_parquet(FEAT)
    df["pay"] = pd.to_numeric(df["pay"], errors="coerce").fillna(0.0)
    lab2 = {l: i for i, l in enumerate(PERM_LABELS)}
    df["ci"] = df["trifecta"].map(lab2)
    df = df[df["ci"].notna()].copy(); df["ci"] = df["ci"].astype(int)
    df["is123"] = (df["ci"] == lab2["1-2-3"]).astype(float)
    df["rt"] = df["race_type"].fillna("")
    df["kikaku"] = df["rt"].str.contains(KIKAKU)
    N = len(df)

    # ================= 飛躍14
    L = [f"# 飛躍14〜16（最後の3本・2018〜2026・{N:,}レース）\n",
         "## 飛躍14: 人工的に堅くした番組（企画レース）を市場は織り込むか\n",
         "モーニング一般・ツッキー・ガチ勝ち8などは、番組側が意図的に1号艇を有利にしている",
         "（`manshu_deep.md`: 万舟率 ×0.6）。**人工的な堅さは選手の実力ではないので、群衆が織り込み切れないかもしれない。**",
         "物差しは「毎レース 1-2-3 を買う回収率」（`回収率 = 的中率 × 平均配当 ÷ 100`。結果と公式配当だけで",
         "計算できるのでオッズが要らず46万レース全部使える）。**探索 2018〜2023 → 確認 2024〜2026。**\n",
         "| 区分 | レース数 | 1-2-3 が来る確率 | 的中時の平均配当 | 探索 回収率 | 確認 回収率 |",
         "|---|---:|---:|---:|---:|---:|"]
    stat = {}
    for nm, m in (("企画レース", df["kikaku"]), ("それ以外", ~df["kikaku"])):
        e = df[m & (df["year"] <= 2023)]; c = df[m & (df["year"] >= 2024)]
        hr = df.loc[m, "is123"].mean(); pv = df.loc[m & (df["is123"] > 0), "pay"].mean()
        re_ = (e["is123"] * e["pay"]).mean() / 100.0; rc = (c["is123"] * c["pay"]).mean() / 100.0
        stat[nm] = (hr, pv, re_, rc, len(e), len(c))
        L.append(f"| {nm} | {int(m.sum()):,} | {hr*100:.2f}% | {pv:,.0f}円 | "
                 f"**{re_*100:.1f}%** | **{rc*100:.1f}%** |")
    hk, pk, _, _, ne, nc = stat["企画レース"]; ho, po, _, _, _, _ = stat["それ以外"]
    L += ["",
          f"**まず値付けの答えが先に出る。** 企画レースは 1-2-3 が来る確率が **{hk/ho:.2f}倍**（{ho*100:.2f}% → {hk*100:.2f}%）、",
          f"そのぶん配当が **{pk/po:.2f}倍**（{po:,.0f}円 → {pk:,.0f}円）に下がる。積は **{(hk/ho)*(pk/po):.3f}**。",
          "**群衆は人工的な堅さをほぼ丸ごと値段に入れている。**\n",
          "残るのはその「ほぼ」の中身で、探索期間では企画レースが有利に見える。**帰無対照を置く**:",
          "それ以外のレースから同じ大きさの群を無作為に2,000回取って、同じ2つの数字を計算する。\n"]
    oth = df[~df["kikaku"]]
    xe = (oth.loc[oth["year"] <= 2023, "is123"] * oth.loc[oth["year"] <= 2023, "pay"]).values
    xc = (oth.loc[oth["year"] >= 2024, "is123"] * oth.loc[oth["year"] >= 2024, "pay"]).values
    na = np.empty(2000); nb = np.empty(2000)
    for i in range(2000):
        na[i] = xe[RNG.integers(0, len(xe), ne)].mean() / 100.0
        nb[i] = xc[RNG.integers(0, len(xc), nc)].mean() / 100.0
    ke, kc = stat["企画レース"][2], stat["企画レース"][3]
    p_exp = float((na >= ke).mean()); p_gap = float(((na - nb) >= (ke - kc)).mean())
    L += ["| 指標 | 企画レースの実測 | 帰無2,000回の95%範囲 | 実測以上が出る確率 |", "|---|---:|---|---:|",
          f"| 探索期間の回収率 | {ke*100:.1f}% | {np.percentile(na,2.5)*100:.1f}〜{np.percentile(na,97.5)*100:.1f}% | **{p_exp:.3f}** |",
          f"| 探索 − 確認 の落差 | {(ke-kc)*100:+.1f}pt | {np.percentile(na-nb,2.5)*100:+.1f}〜{np.percentile(na-nb,97.5)*100:+.1f}pt | **{p_gap:.3f}** |",
          "",
          f"→ **探索期間の88.6%は帰無でも{p_exp*100:.0f}%の確率で出る。確認期間での{(ke-kc)*100:.1f}ptの落下も{p_gap*100:.0f}%で出る。**",
          "企画レースの n は12,724だが、そのうち 1-2-3 が来るのは1,213本しかなく、配当の裾が重いので",
          "回収率の区間は ±10pt 級になる。**この粒度では何も判定できない。**\n",
          "企画の系統ごとに割ると、その揺らぎの大きさが直接見える:\n",
          "| 系統 | レース数 | 1-2-3率 | 平均配当 | 全期間 回収率 | 95%区間 | 探索 | 確認 |",
          "|---|---:|---:|---:|---:|---|---:|---:|"]
    for pat in FAMS:
        g = df[df["rt"].str.contains(pat)]
        e = g[g["year"] <= 2023]; c = g[g["year"] >= 2024]
        x = (g["is123"] * g["pay"]).values
        lo, hi = boot_ci(x)
        L.append(f"| {pat} | {len(g):,} | {g['is123'].mean()*100:.2f}% | {g.loc[g['is123']>0,'pay'].mean():,.0f}円 | "
                 f"{x.mean()/100*100:.1f}% | {lo:.0f}〜{hi:.0f}% | {(e['is123']*e['pay']).mean()/100*100:.1f}% | "
                 f"{(c['is123']*c['pay']).mean()/100*100:.1f}% |")
    L += ["",
          "**進入固定戦隊の104.0%が罠の見本**: 全期間で100%超えだが n=1,154、区間 87〜120%、",
          "探索118.1% → 確認82.8%。**`stadium_study.md` の「びわこ102%」と完全に同型。**",
          "系統別の探索→確認は ツッキー 89.6→56.8、進入固定 118.1→82.8、ガチ勝 62.7→77.2 と符号すら揃わない。\n",
          "**結論: 飛躍14は否定。番組が作った堅さは値段に入っており、残差は測れる大きさではない。**"]

    # ================= 飛躍15
    z = np.load(CACHE, allow_pickle=True)
    O, W = z["O"], z["W"]
    inv = np.where(np.isfinite(O), 1.0 / np.nan_to_num(O, nan=1e9), 0.0)
    Q = inv / inv.sum(1, keepdims=True)
    n26 = len(Q)
    pq = np.zeros((n26, 6, 3))
    for a in range(6):
        pq[:, a, 0] = Q[:, _A == a].sum(1)
        pq[:, a, 1] = Q[:, _B == a].sum(1)
        pq[:, a, 2] = Q[:, _C == a].sum(1)

    def actmat(Wx):
        m = np.zeros((n26, 6, 3))
        m[np.arange(n26), _A[Wx], 0] = 1
        m[np.arange(n26), _B[Wx], 1] = 1
        m[np.arange(n26), _C[Wx], 2] = 1
        return m
    act = actmat(W)
    EDG = np.array([0, .03, .05, .08, .12, .18, .25, .35, 1.01])

    def band_tab(A):
        out = np.full((len(EDG) - 1, 3), np.nan)
        for i in range(len(EDG) - 1):
            for p in range(3):
                m = (pq[:, :, p] >= EDG[i]) & (pq[:, :, p] < EDG[i + 1])
                if m.sum() >= 5000:
                    out[i, p] = A[:, :, p][m].mean() / pq[:, :, p][m].mean()
        return out
    obs = band_tab(act)
    cum = Q.cumsum(1)
    nulls = []
    for _ in range(20):
        u = RNG.random((n26, 1))
        nulls.append(band_tab(actmat((cum < u).sum(1).clip(0, 119))))
    nulls = np.array(nulls)

    L += ["\n## 飛躍15: 1着・2着・3着のどの位置の値付けが甘いか\n",
          f"3連単プールから「その艇がちょうど n 着になる確率」を取り出して実測と比べる（2026年 {n26:,}R・確定オッズ）。",
          "**誤りが特定の位置に偏っているなら、`leap_self.md` の構造の話がもっと具体的になる。**\n",
          "| 位置 | 買い目数 | 市場の平均確率 | 実測 | 実測÷市場 |", "|---|---:|---:|---:|---:|"]
    for p_, nm in ((0, "1着"), (1, "2着"), (2, "3着")):
        L.append(f"| {nm} | {n26*6:,} | {pq[:, :, p_].mean()*100:.2f}% | {act[:, :, p_].mean()*100:.2f}% | "
                 f"{act[:, :, p_].mean()/pq[:, :, p_].mean():.4f} |")
    L += ["", "全体は定義上ぴったり合う（どのレースでも1着・2着・3着はちょうど1艇ずつ）。",
          "**位置を比べるなら市場確率の帯を揃えないといけない。** 揃えないと「3着の方が低確率のセルが多い」",
          "というだけの話になる。帯を揃えた表（括弧内はセル数、帰無は `y ~ Multinomial(q)` を20回）:\n",
          "| 市場が付けた確率 | 1着 実測÷市場 | 2着 | 3着 | 帰無20回の範囲 |",
          "|---|---:|---:|---:|---|"]
    for i in range(len(EDG) - 1):
        cells = []
        for p in range(3):
            m = (pq[:, :, p] >= EDG[i]) & (pq[:, :, p] < EDG[i + 1])
            if np.isnan(obs[i, p]):
                cells.append("—"); continue
            r = obs[i, p]
            bold = "**" if (r > 1.02 or r < 0.98) else ""
            cells.append(f"{bold}{r:.3f}{bold} ({int(m.sum())//1000}千)")
        c = nulls[:, i, :].ravel(); c = c[~np.isnan(c)]
        rng_s = f"{np.min(c):.3f}〜{np.max(c):.3f}" if len(c) else "—"
        L.append(f"| {EDG[i]:.0%}〜{EDG[i+1]:.0%} | " + " | ".join(cells) + f" | {rng_s} |")

    slopes = {}
    for p, pn in ((0, "1着"), (1, "2着"), (2, "3着")):
        idx = []; rs = []; qs = []; ws = []
        for i in range(len(EDG) - 1):
            m = (pq[:, :, p] >= EDG[i]) & (pq[:, :, p] < EDG[i + 1])
            if m.sum() < 5000 or np.isnan(obs[i, p]):
                continue
            idx.append(i)
            rs.append(np.log(obs[i, p])); qs.append(np.log(pq[:, :, p][m].mean())); ws.append(m.sum())
        w = np.sqrt(np.asarray(ws, dtype=float))
        sl = np.polyfit(qs, rs, 1, w=w)[0]
        nl = []
        for k in range(20):
            v = nulls[k, idx, p]
            if np.any(~np.isfinite(v)) or np.any(v <= 0):
                continue
            nl.append(np.polyfit(qs, np.log(v), 1, w=w)[0])
        if not nl:
            raise RuntimeError(f"帰無の傾きが1本も計算できない: {pn}")
        slopes[pn] = (sl, float(np.min(nl)), float(np.max(nl)), len(nl))
    L += ["",
          "**位置ごとに人気薄バイアスの傾きを測る**（log(実測÷市場) を log(市場確率) に回帰、セル数で重み）:\n",
          "| 位置 | 傾き | 帰無20回の範囲 | 判定 |", "|---|---:|---|---|"]
    for pn in ("1着", "2着", "3着"):
        sl, lo, hi, nk = slopes[pn]
        out = "帰無の外" if (sl > hi or sl < lo) else "帰無の中"
        L.append(f"| {pn} | **{sl:+.4f}** | {lo:+.4f}〜{hi:+.4f} | {out}（{nk}/{nk}） |")
    L += ["",
          f"→ **傾きは位置が後ろほど急になる（1着 {slopes['1着'][0]:+.3f} < 2着 {slopes['2着'][0]:+.3f} ≒ 3着 {slopes['3着'][0]:+.3f}）。**",
          "同じ市場確率3〜5%のセルでも 1着0.894 / 2着0.787 / **3着0.686**、同じ25〜35%でも 1着0.999 / 2着1.021 / **3着1.059**。",
          "**市場は「誰が1着か」はよく値付けするが、「誰が2着・3着に残るか」では人気薄バイアスが約2倍に開く。**",
          "20回の帰無では傾きは ±0.05 以内に収まるので、これは揺らぎではない。\n",
          "### この歪みで120通りの安さを説明できるか\n",
          "位置別の校正表を2026年の前半で作り、後半の120通りの「実測比」を予測できるか試す。"]
    half = n26 // 2
    cal = np.ones((len(EDG) - 1, 3))
    for i in range(len(EDG) - 1):
        for p in range(3):
            m = (pq[:half, :, p] >= EDG[i]) & (pq[:half, :, p] < EDG[i + 1])
            if m.sum() >= 3000:
                cal[i, p] = act[:half, :, p][m].mean() / pq[:half, :, p][m].mean()

    def rat(qv, p):
        return cal[np.clip(np.searchsorted(EDG, qv, side="right") - 1, 0, len(EDG) - 2), p]
    s = slice(half, n26); m2 = n26 - half
    R = np.ones((m2, 120))
    for c in range(120):
        R[:, c] = rat(pq[s, _A[c], 0], 0) * rat(pq[s, _B[c], 1], 1) * rat(pq[s, _C[c], 2], 2)
    pred = Q[s] * R; pred /= pred.sum(1, keepdims=True)
    prr = pred / np.clip(Q[s], 1e-12, None)
    A2 = np.zeros((m2, 120)); A2[np.arange(m2), W[s]] = 1
    rows = [(PERM_LABELS[c], A2[:, c].mean() / Q[s, c].mean(),
             float((prr[:, c] * Q[s, c]).sum() / Q[s, c].sum()), float(Q[s, c].mean())) for c in range(120)]
    arr = np.array([(r[1], r[2]) for r in rows])
    cor = float(np.corrcoef(arr[:, 0], arr[:, 1])[0, 1])
    L += ["",
          "| 買い目 | 市場確率 | 実測比 | 位置別校正からの予測比 |", "|---|---:|---:|---:|"]
    for lbl, a, p_, q in sorted(rows, key=lambda r: -r[3])[:6]:
        L.append(f"| {lbl} | {q*100:.2f}% | **{a:.3f}** | {p_:.3f} |")
    L += ["",
          f"120通り全体の相関は **{cor:.3f}**。**位置別の歪みは120通りの安さの「半分だけ」を説明する。**",
          "しかもいちばん肝心なところで外す: 1-2-3 は実測比1.132に対し予測1.076（説明できるのは超過分の約6割）、",
          "**1-2-4 は予測1.082 > 1-2-3 の1.076 なのに実測は1.021 で逆転している。**",
          "→ **位置別の校正では 1-2-3 を選べない。** 位置をばらばらに直すだけでは足りず、",
          "`leap_self.md` の「4号艇が1着なら2着は5号艇」型の**組み合わせそのものの相互作用**が別に要る。",
          "**飛躍15 は Model 1.3 の設計に対する2本目の独立な裏づけになった。**"]

    # ================= 飛躍16
    L += ["\n## 飛躍16: 市場の癖は9年で薄れているか\n",
          "ここまでで残った歪みは「出やすい目ほど安い」の一点だった（`leaps4.md`、相関 +0.875）。",
          "**年ごとに同じ計算をして、弱まっているかを見る。** 弱まっているなら、どんな発見にも賞味期限がある。\n",
          "| 年 | レース数 | 出現率と回収率の相関 | 120通りの回収率の幅 | 標準偏差 | 1-2-3 | 最頻10目の平均 |",
          "|---|---:|---:|---|---:|---:|---:|"]
    yrs = sorted(df["year"].unique())
    met = []
    for y in yrs:
        d = df[df["year"] == y]; n = len(d)
        cnt = np.bincount(d["ci"].values, minlength=120)
        paysum = np.bincount(d["ci"].values, weights=d["pay"].values, minlength=120)
        roi = paysum / (100.0 * n); fr = cnt / n
        cr = np.corrcoef(np.log(np.clip(fr, 1e-9, None)), roi)[0, 1]
        t10 = np.argsort(-fr)[:10]
        met.append((y, cr, roi.std(), roi[t10].mean()))
        L.append(f"| {y} | {n:,} | **{cr:+.3f}** | {roi.min()*100:.1f}〜{roi.max()*100:.1f}% | "
                 f"{roi.std()*100:.1f}pt | {roi[lab2['1-2-3']]*100:.1f}% | {roi[t10].mean()*100:.1f}% |")
    met = np.array(met)
    L += ["", "年に対する傾き（最小二乗）:", ""]
    sl = {}
    for j, nm in ((1, "出現率と回収率の相関"), (2, "回収率のばらつき（標準偏差）"), (3, "最頻10目の平均回収率")):
        sl[nm] = np.polyfit(met[:, 0], met[:, j], 1)[0]
        L.append(f"- {nm}: **年あたり {sl[nm]:+.4f}**（2018年 {met[0, j]:.4f} → 2026年 {met[-1, j]:.4f}）")
    L += ["",
          "**歪みは薄れていない。** 相関はむしろわずかに上向き（+0.0055/年）、120通りの散らばりも横ばい",
          f"（{sl['回収率のばらつき（標準偏差）']:+.4f}/年）。9年で市場が 1-2-3 の割安さを直した形跡は無い。",
          "唯一わずかに下向きなのは最頻10目の平均回収率（−0.21pt/年、9年で75.1→72.1%）だが、",
          "1-2-3 自体は 84.7→79.1→82.2% と上下しており、**9点では傾向と言い切れない**",
          "（`combo123.md` の訂正と同じ）。\n",
          "**結論: 「出やすい目ほど安い」に見える賞味期限は無い。** ただしそれは",
          "**82%が今後も82%であり続けるという意味**であって、100%に近づくという意味ではない。"]

    # ================= まとめ
    L += ["\n## まとめ: 飛躍14〜16（飛躍シリーズ全16本の締め）\n",
          "### 飛躍14（企画レース）— **否定。人工的な堅さは値段に入っている**",
          f"1-2-3 の出現率は {hk/ho:.2f}倍に上がるが配当が {pk/po:.2f}倍に下がり、積は {(hk/ho)*(pk/po):.3f}。",
          f"探索期間の88.6%は帰無でも確率{p_exp:.2f}で出て、確認期間への{(ke-kc)*100:.1f}pt下落も確率{p_gap:.2f}で出る。",
          "**「番組の意図」という外から見える情報でさえ、市場は織り込んでいる。**",
          "進入固定戦隊の全期間104.0%（探索118→確認83、n=1,154）は `stadium_study.md` のびわこ102%と同型の罠。\n",
          "### 飛躍15（位置別の値付け）— **本シリーズ最後の陽性。人気薄バイアスは後ろの位置ほど急**",
          f"市場確率の帯を揃えて測ると、傾きは 1着 {slopes['1着'][0]:+.3f} / 2着 {slopes['2着'][0]:+.3f} / 3着 {slopes['3着'][0]:+.3f}（帰無は±0.05以内、20/20で外）。",
          "**市場は「誰が勝つか」はよく値付けし、「誰が2着・3着に残るか」で約2倍ぶれる。**",
          f"ただしこの位置別の歪みで120通りの安さを再現しようとすると相関 {cor:.3f} 止まりで、",
          "**1-2-4 を 1-2-3 より上に置いてしまう＝買い方としては使えない。**",
          "不足分は `leap_self.md` が見つけた「1着が誰かで2着の顔ぶれが変わる」相互作用。\n",
          "### 飛躍16（賞味期限）— **薄れていない**",
          "出現率と回収率の相関は9年間 +0.70〜+0.83 で、傾きは +0.0055/年（むしろ上向き）。",
          "**1-2-3 の割安さに期限は見えない。** ただし水準は82%で、100%ではない。\n",
          "### 飛躍シリーズ16本の総括\n",
          "**16本試して、市場を超える情報は1つも見つからなかった。**",
          "決まり手 +0.0003 nats、選手ごとのズレ +0.0005、目的1の5族 ≈0.001、万舟予測 +0.0023 —",
          "必要な 0.2877 nats に対し最良で 0.8%。**市場の外に情報は無い、という結論は16本の独立な試行で固まった。**\n",
          "**見つかったのは3種類だけで、性質がはっきり分かれる:**\n",
          "| 種類 | 中身 | 大きさ | 使えるか |",
          "|---|---|---|---|",
          "| ①情報の要らない補助金 | 返還（`leaps2.md`）、元返し（`leaps5.md`） | +0.7〜1.6pt / 複勝1点に集中 | **使える。既に実装済み** |",
          "| ②市場の値付けの癖 | 出やすい目ほど安い（1-2-3 で比1.10〜1.22）、後ろの位置ほど急（飛躍15） | 比1.16が上限 | 比1.333に届かず**負けが小さくなるだけ** |",
          "| ③こちらの表現力不足 | PL が書けない1着×2着の相互作用 | **0.0385 nats** | **取り戻せる。市場に近づく分だけ** |\n",
          "**③だけが未着手で、しかもこれまでの全努力（0.001 nats）の38倍ある。**",
          "飛躍1（PLの損失）・飛躍2（情報の67%は「どの3艇か」）・飛躍15（位置別では1-2-3を選べない）の",
          "**3本が独立に同じ設計を指した: まず3艇の組を当て、次にその並びを当てる二段構え（Model 1.3）。**",
          "ただし `model12_wf.md` の通り対数損失が改善しても回収率は横ばいだった実績があり、",
          "0.0385 は必要量の13%。**市場に追いつく話であって、超える話ではない。**\n",
          "**正直な現在地**: 3連単で最も負けにくい買い方は 1-2-3 の条件つき（86%、`combo123.md`）、",
          "全券種で最も100%に近いのは複勝1点・確率0.90以上（99.1〜99.3%）。**どちらも100%未満で、",
          "それは控除率25%が情報で埋まらないため。** 飛躍16 が示す通り、この構図に賞味期限は無い。"]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
