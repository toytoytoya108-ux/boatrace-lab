"""飛躍5〜7。いずれも「1レースを単体で見る」という前提そのものを疑う。

飛躍5: **レースは独立ではないのでは。** 同じ日・同じ場では水面も風も番組の質も共通で、
  「今日ここまで何が起きたか」は次のレースの手がかりになる。市場はそれを織り込んでいるか。
  → 直前までのレースの決着（1号艇が勝ったか）で条件づけて、実勝率÷市場確率を見る。

飛躍6: **市場は1レースずつ独立に値付けしている。選手を横断して見れば、いつも高い／安い選手がいるのでは。**
  → 探索期間で選手ごとに「実際の1着数 − 市場が見込んだ1着数」を出し、確認期間に持ち越して効くかを見る。
  効けば、これは市場が体系的に間違えている唯一の相手（個人）ということになる。

飛躍7: **買い目の"見た目"と、オッズの数字そのものに人の癖があるのでは。**
  1-2-3 のような整った出目、1号艇を含む目、連番。さらにオッズの末尾の数字。
  → **オッズ帯で層別してから**形ごとの比を見る（層別しないと人気薄バイアスと混ざる）。

出力: reports/research/leaps3.md
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from boatlab.backtest.metrics import roi_bootstrap  # noqa: E402
from boatlab.config import ROOT  # noqa: E402
from boatlab.model.trifecta import PERMS  # noqa: E402

OUT = Path(ROOT) / "reports" / "research" / "leaps3.md"
CACHE = Path("/tmp/claude-0/mix10_cache.npz")
DB = str(Path(ROOT) / "data" / "lab.db")
_A = np.array([p[0] for p in PERMS]); _B = np.array([p[1] for p in PERMS]); _C = np.array([p[2] for p in PERMS])
RATE = 0.75


def ratio_row(y, q):
    """実測 ÷ 市場。二項の95%区間つき。"""
    n = len(y)
    if n == 0 or q.sum() == 0:
        return None
    obs, exp = y.mean(), q.mean()
    se = np.sqrt(max(obs * (1 - obs), 1e-12) / n)
    return dict(n=n, obs=obs, exp=exp, ratio=obs / exp, lo=(obs - 1.96 * se) / exp, hi=(obs + 1.96 * se) / exp)


def main():
    z = np.load(CACHE, allow_pickle=True)
    rid, date, O, W, P = z["rid"], pd.DatetimeIndex(z["date"]), z["O"], z["W"], z["P"]
    inv = np.where(np.isfinite(O), 1.0 / np.nan_to_num(O, nan=1e9), 0.0)
    Q = inv / inv.sum(1, keepdims=True)
    N = len(Q)
    half = np.asarray(date <= "2026-05-31")
    stad = (rid // 100) % 100
    rno = rid % 100
    win_lane = _A[W]
    q1 = np.stack([Q[:, _A == a].sum(1) for a in range(6)], axis=1)      # 市場の1着確率(6)

    L = [f"# 飛躍5〜7: 「1レースを単体で見る」前提を疑う（2026年・{N:,}R・確定オッズ）\n"]

    # ================= 飛躍5: レース間の連鎖
    df = pd.DataFrame({"i": np.arange(N), "d": pd.Series(date).dt.date.values, "st": stad, "rno": rno,
                       "w1": (win_lane == 0).astype(int), "q1": q1[:, 0]})
    df = df.sort_values(["d", "st", "rno"])
    g = df.groupby(["d", "st"])
    df["prev_n"] = g.cumcount()
    df["prev_w1"] = g["w1"].cumsum() - df["w1"]                    # その日その場で、ここまでに1号艇が勝った回数
    df["prev_q1"] = g["q1"].cumsum() - df["q1"]
    df["surprise"] = df["prev_w1"] - df["prev_q1"]                 # 市場の見込みより何本多く逃げたか
    d5 = df[df["prev_n"] >= 3].copy()
    L += ["## 飛躍5: レースは独立か（同じ日・同じ場で、ここまでの決着が次を教えるか）\n",
          "同じ水面・同じ風・同じ節。**「今日ここまで何本 1号艇が勝ったか」を市場が織り込み切れていないなら、",
          "レースを跨いだ情報がタダで手に入る。** 4レース目以降だけを見る（それ以前は材料が少なすぎる）。\n",
          "| 今日ここまでの『逃げ』の多さ | レース数 | 実際の1号艇1着率 | 市場の見込み | 実測÷市場 | 95%区間 |",
          "|---|---:|---:|---:|---:|---|"]
    qs = np.quantile(d5["surprise"], [0.2, 0.4, 0.6, 0.8])
    labs = [f"市場の見込みより {qs[0]:+.1f} 本以下", f"{qs[0]:+.1f}〜{qs[1]:+.1f} 本",
            f"{qs[1]:+.1f}〜{qs[2]:+.1f} 本", f"{qs[2]:+.1f}〜{qs[3]:+.1f} 本", f"{qs[3]:+.1f} 本より多い"]
    edges = [-1e9] + list(qs) + [1e9]
    for k in range(5):
        m = (d5["surprise"] > edges[k]) & (d5["surprise"] <= edges[k + 1])
        r = ratio_row(d5.loc[m, "w1"].values, d5.loc[m, "q1"].values)
        if r is None:
            continue
        L.append(f"| {labs[k]} | {r['n']:,} | {r['obs']*100:.1f}% | {r['exp']*100:.1f}% | "
                 f"{'**' if r['lo'] > 1 or r['hi'] < 1 else ''}{r['ratio']:.3f}"
                 f"{'**' if r['lo'] > 1 or r['hi'] < 1 else ''} | {r['lo']:.3f}〜{r['hi']:.3f} |")
    # 前半・後半で再現するか（いちばん端の区分）
    L += ["", "前半（1〜5月）／後半（6〜8月）での再現:", "",
          "| 区分 | 前半 実測÷市場 | 後半 実測÷市場 |", "|---|---:|---:|"]
    hm = np.isin(d5["i"].values, np.flatnonzero(half))
    for k, lab in ((0, labs[0]), (4, labs[4])):
        m = ((d5["surprise"] > edges[k]) & (d5["surprise"] <= edges[k + 1])).values
        cells = []
        for sub in (hm, ~hm):
            r = ratio_row(d5.loc[m & sub, "w1"].values, d5.loc[m & sub, "q1"].values)
            cells.append(f"{r['ratio']:.3f}" if r else "—")
        L.append(f"| {lab} | " + " | ".join(cells) + " |")

    # ================= 飛躍6: 選手ごとの持続的なズレ
    con = sqlite3.connect(DB)
    en = pd.read_sql_query("SELECT race_id, lane, regno FROM entries WHERE race_id >= 202601010000", con)
    con.close()
    pos = {int(r): k for k, r in enumerate(rid)}
    en["k"] = en["race_id"].map(pos)
    en = en.dropna(subset=["k"]).copy()
    en["k"] = en["k"].astype(int)
    en["lane0"] = en["lane"].astype(int) - 1
    en = en[(en["lane0"] >= 0) & (en["lane0"] <= 5)]
    en["q"] = q1[en["k"].values, en["lane0"].values]
    en["y"] = (win_lane[en["k"].values] == en["lane0"].values).astype(int)
    en["half"] = half[en["k"].values]
    ex = en[en["half"]].groupby("regno").agg(n=("y", "size"), wins=("y", "sum"), exp=("q", "sum"))
    K = 20.0                                                    # 縮約（少ない出走数の選手を1.0へ寄せる）
    ex["bias"] = (ex["wins"] + K) / (ex["exp"] + K)
    cf = en[~en["half"]].merge(ex[["n", "bias"]], on="regno", how="left")
    cf = cf[cf["n"] >= 30]
    L += ["\n## 飛躍6: 市場がいつも間違える『個人』はいるか\n",
          "市場は1レースずつ独立に値付けする。選手を横断して見れば、体系的に高く／安く売られている選手がいるかもしれない。",
          f"探索期間（1〜5月）で選手ごとに `(実際の1着数 + {K:.0f}) / (市場の見込み + {K:.0f})` を出し（縮約つき）、",
          "**確認期間（6〜8月）に持ち越して効くかを見る。** 探索30走以上の選手だけ。\n",
          f"- 探索期間で評価できた選手 {len(ex[ex['n'] >= 30]):,}人、確認期間の出走 {len(cf):,}。",
          f"- 探索期間のズレの分布: 5%点 {ex.loc[ex['n']>=30,'bias'].quantile(0.05):.3f}、"
          f"中央値 {ex.loc[ex['n']>=30,'bias'].median():.3f}、95%点 {ex.loc[ex['n']>=30,'bias'].quantile(0.95):.3f}",
          "",
          "| 探索期間のズレ（5分割） | 確認期間の出走 | 実際の1着率 | 市場の見込み | 実測÷市場 | 95%区間 |",
          "|---|---:|---:|---:|---:|---|"]
    qb = np.quantile(cf["bias"], [0.2, 0.4, 0.6, 0.8])
    eb = [-1e9] + list(qb) + [1e9]
    names = ["市場が高く売りすぎ（下位20%）", "やや高い", "ふつう", "やや安い", "市場が安く売りすぎ（上位20%）"]
    for k in range(5):
        m = (cf["bias"] > eb[k]) & (cf["bias"] <= eb[k + 1])
        r = ratio_row(cf.loc[m, "y"].values, cf.loc[m, "q"].values)
        if r is None:
            continue
        L.append(f"| {names[k]}（{eb[k] if k else -np.inf:.3f}〜{eb[k+1] if k < 4 else np.inf:.3f}） | {r['n']:,} | "
                 f"{r['obs']*100:.1f}% | {r['exp']*100:.1f}% | "
                 f"{'**' if r['lo'] > 1 or r['hi'] < 1 else ''}{r['ratio']:.3f}"
                 f"{'**' if r['lo'] > 1 or r['hi'] < 1 else ''} | {r['lo']:.3f}〜{r['hi']:.3f} |")

    # ================= 飛躍7: 出目の形とオッズの数字
    L += ["\n## 飛躍7: 買い目の『見た目』とオッズの数字に人の癖はあるか\n",
          "**オッズ帯で層別してから**見る（層別しないと人気薄バイアスと混ざって何も分からない）。",
          "各買い目について、実際に来たか（1/0）と市場確率を集計し、帯ごとに 実測÷市場 を出す。\n"]
    a, b, c = _A + 1, _B + 1, _C + 1
    shapes = {
        "昇順（a<b<c）": (a < b) & (b < c),
        "降順（a>b>c）": (a > b) & (b > c),
        "連番（1-2-3 のように3つ続く）": (np.abs(a - b) == 1) & (np.abs(b - c) == 1) & ((a < b) == (b < c)),
        "1号艇を含む": (a == 1) | (b == 1) | (c == 1),
        "1号艇が1着": a == 1,
        "全部 奇数 か 全部 偶数": ((a % 2) == (b % 2)) & ((b % 2) == (c % 2)),
        "外側3艇だけ（4,5,6）": (a >= 4) & (b >= 4) & (c >= 4),
    }
    Y = np.zeros((N, 120), bool)
    Y[np.arange(N), W] = True
    ob = np.where(np.isfinite(O), O, np.nan)
    bands = [(1, 10), (10, 30), (30, 100), (100, 300), (300, 1e9)]
    L += ["| 形 | " + " | ".join(f"{lo:g}〜{hi:g}倍" if hi < 1e8 else f"{lo:g}倍〜" for lo, hi in bands) + " | 全体 |",
          "|---|" + "---:|" * (len(bands) + 1)]
    for nm, msk in shapes.items():
        cells = []
        tot_y, tot_q = [], []
        for lo, hi in bands:
            m = np.broadcast_to(msk, (N, 120)) & (ob >= lo) & (ob < hi)
            if m.sum() < 2000:
                cells.append("—"); continue
            r = ratio_row(Y[m].astype(float), Q[m])
            cells.append(f"{'**' if r['lo'] > 1 else ''}{r['ratio']:.3f}{'**' if r['lo'] > 1 else ''}")
            tot_y.append(Y[m].astype(float)); tot_q.append(Q[m])
        rt = ratio_row(np.concatenate(tot_y), np.concatenate(tot_q)) if tot_y else None
        L.append(f"| {nm} | " + " | ".join(cells) + f" | {rt['ratio']:.3f} |")
    L += ["", "（太字は95%区間の下限が1を超えた区分。帯ごとに層別しているので、人気薄バイアスは取り除かれている）\n",
          "### オッズの数字そのもの（末尾の桁）\n",
          "人は切りの良い数字を避ける／好むことがある。オッズの小数第1位ごとに比を見る（10〜100倍帯）。\n",
          "| 末尾 | 買い目数 | 実測÷市場 | 95%区間 |", "|---|---:|---:|---|"]
    band = (ob >= 10) & (ob < 100)
    last = np.round((ob * 10) % 10).astype(float)
    for dgt in range(10):
        m = band & (last == dgt)
        if m.sum() < 5000:
            continue
        r = ratio_row(Y[m].astype(float), Q[m])
        L.append(f"| .{dgt} | {r['n']:,} | {'**' if r['lo'] > 1 or r['hi'] < 1 else ''}{r['ratio']:.3f}"
                 f"{'**' if r['lo'] > 1 or r['hi'] < 1 else ''} | {r['lo']:.3f}〜{r['hi']:.3f} |")

    # ---- 飛躍6の追い込み: 情報量（nats）で測る。他の全研究と同じ物差しに載せる
    from boatlab.research.market_offset import fit_offset, logloss, predict
    L += ["\n### 飛躍6の追い込み: 選手のズレは何 nats か（他の全研究と同じ物差し）\n",
          "市場の1着確率をオフセットに置いた条件付きロジット `score = log q + γ·z`（`market_offset.py`）。",
          "z は探索期間で作った選手ごとのズレ（log）。**γ=0 なら市場が既に織り込んでいる。**",
          "必要な情報量は 0.2877 nats（控除率25%を埋める量）。目的1の5族はいずれも 0.001 前後だった。\n"]
    cf2 = en[~en["half"]].merge(ex[["n", "bias"]], on="regno", how="left")
    cf2["z"] = np.log(cf2["bias"].fillna(1.0))
    cf2.loc[cf2["n"].fillna(0) < 30, "z"] = 0.0
    piv = cf2.pivot_table(index="k", columns="lane0", values=["q", "y", "z"], aggfunc="first")
    ks = piv.index.values
    Zc = np.nan_to_num(piv["z"].reindex(columns=range(6)).values)
    Qc = np.nan_to_num(piv["q"].reindex(columns=range(6)).values)
    Yc = np.nan_to_num(piv["y"].reindex(columns=range(6)).values)
    ok = (Qc.sum(1) > 0.9) & (Yc.sum(1) == 1)
    Zc, Qc, Yc = Zc[ok], Qc[ok], Yc[ok]
    logq = np.log(np.clip(Qc, 1e-12, None))
    n_ex = len(Zc) // 2
    g = fit_offset(Zc[:n_ex, :, None], logq[:n_ex], Yc[:n_ex])
    yc = Yc.argmax(1)                                   # logloss は one-hot でなく勝者の添字を取る
    base = logloss(Qc[n_ex:], yc[n_ex:])
    with_z = logloss(predict(Zc[n_ex:, :, None], logq[n_ex:], g), yc[n_ex:])
    L += [f"- 確認期間をさらに半分に割り、前半で γ を推定 → **γ = {float(g[0]):.3f}**（0なら市場が織り込み済み）。",
          f"- 後半の1着対数損失: 市場のみ **{base:.4f}** → 選手のズレを足して **{with_z:.4f}**"
          f"（**{base - with_z:+.4f} nats**）。",
          f"- 必要量 0.2877 nats に対して **{(base - with_z) / 0.2877 * 100:.2f}%**。",
          f"  目的1の5族（0.001前後 = 0.35%）と比べると **{(base - with_z) / 0.0012:.1f}倍**。\n"]

    # ---- 飛躍7の追い込み: 連番は1号艇1着とは別物か
    L += ["\n### 飛躍7の追い込み: 『連番』は『1号艇が1着』の言い換えではないか\n",
          "1〜10倍帯の連番はほとんど 1-2-3。1号艇1着の効果と分けて見る。\n",
          "| 区分（1〜10倍帯） | 買い目数 | 実測÷市場 | 95%区間 |", "|---|---:|---:|---|"]
    seq = shapes["連番（1-2-3 のように3つ続く）"]
    l1 = shapes["1号艇が1着"]
    bandl = (ob >= 1) & (ob < 10)
    for nm, msk in (("1号艇が1着 かつ 連番", seq & l1), ("1号艇が1着 だが連番でない", (~seq) & l1),
                    ("1号艇が1着でない（帯内）", ~l1)):
        m = np.broadcast_to(msk, (N, 120)) & bandl
        if m.sum() < 500:
            L.append(f"| {nm} | {int(m.sum())} | — | 少なすぎ |"); continue
        r = ratio_row(Y[m].astype(float), Q[m])
        L.append(f"| {nm} | {r['n']:,} | {'**' if r['lo'] > 1 else ''}{r['ratio']:.3f}{'**' if r['lo'] > 1 else ''} | "
                 f"{r['lo']:.3f}〜{r['hi']:.3f} |")

    L += ["", "1〜10倍は幅が広く、連番（ほぼ 1-2-3）は帯の中でも低オッズ側に寄る。**もっと細かく刻んで確かめる。**\n",
          "| 確定オッズ | 連番の買い目数 | 連番 実測÷市場 | 連番でない 実測÷市場 | 差 |", "|---|---:|---:|---:|---:|"]
    for lo2, hi2 in ((1.0, 2), (2, 3), (3, 4), (4, 6), (6, 8), (8, 10), (10, 15), (15, 25)):
        bm = (ob >= lo2) & (ob < hi2)
        m1 = np.broadcast_to(seq, (N, 120)) & bm
        m0 = np.broadcast_to(~seq, (N, 120)) & bm
        if m1.sum() < 800 or m0.sum() < 800:
            L.append(f"| {lo2:g}〜{hi2:g}倍 | {int(m1.sum())} | — | — | 少なすぎ |"); continue
        r1 = ratio_row(Y[m1].astype(float), Q[m1]); r0 = ratio_row(Y[m0].astype(float), Q[m0])
        L.append(f"| {lo2:g}〜{hi2:g}倍 | {r1['n']:,} | {'**' if r1['lo'] > 1 else ''}{r1['ratio']:.3f}"
                 f"{'**' if r1['lo'] > 1 else ''} | {r0['ratio']:.3f} | {r1['ratio'] - r0['ratio']:+.3f} |")

    # ---- 連番を買い方にしてみる（前半後半 + 帰無つき）
    L += ["\n### 連番を買い方にすると（前半・後半＋帰無シミュレーション）\n",
          "規則: **確定オッズが帯に入る連番の目だけを100円ずつ買う。** 連番は120通り中8つ",
          "（1-2-3 / 2-3-4 / 3-4-5 / 4-5-6 と、その逆順）。該当が無いレースは見送り。\n",
          "| 帯 | 発火レース | 1R平均点数 | 的中率 | 平均払戻 | 回収率 | 比 | 95%区間 | 前半 | 後半 |",
          "|---|---:|---:|---:|---:|---:|---:|---|---:|---:|"]
    def buy_seq(lo2, hi2, mask=None):
        bm = np.broadcast_to(seq, (N, 120)) & (ob >= lo2) & (ob < hi2)
        idx = np.flatnonzero(bm.any(1) & (np.ones(N, bool) if mask is None else mask))
        k = bm[idx].sum(1)
        hit = bm[idx, W[idx]]
        return np.where(hit, P[idx], 0.0), 100.0 * k, idx
    for lo2, hi2 in ((4, 15), (4, 10), (6, 20), (4, 25), (2, 15)):
        ret, st, idx = buy_seq(lo2, hi2)
        lo_, hi_ = roi_bootstrap(st, ret, n_boot=300)
        hs = []
        for hm2 in (half, ~half):
            r2, s2, _ = buy_seq(lo2, hi2, hm2)
            hs.append(r2.sum() / max(s2.sum(), 1))
        L.append(f"| {lo2:g}〜{hi2:g}倍 | {len(idx):,} | {st.mean()/100:.2f} | {(ret>0).mean()*100:.1f}% | "
                 f"{ret[ret>0].mean():,.0f}円 | **{ret.sum()/st.sum()*100:.1f}%** | {ret.sum()/st.sum()/RATE:.3f} | "
                 f"{lo_*100:.0f}〜{hi_*100:.0f}% | {hs[0]*100:.1f}% | {hs[1]*100:.1f}% |")
    rng2 = np.random.default_rng(31)
    cum2 = np.cumsum(Q, axis=1)
    L += ["", "帰無シミュレーション（市場が正しい世界で同じ規則を20回）:", "",
          "| 帯 | 実測 | 帰無 平均 | 帰無 範囲 | 実測以上 |", "|---|---:|---:|---|---:|"]
    for lo2, hi2 in ((4, 15), (4, 25)):
        bm = np.broadcast_to(seq, (N, 120)) & (ob >= lo2) & (ob < hi2)
        idx = np.flatnonzero(bm.any(1))
        real = buy_seq(lo2, hi2)[0].sum() / buy_seq(lo2, hi2)[1].sum()
        vals = []
        for _ in range(20):
            u = rng2.random(N)
            w2 = (cum2 < u[:, None]).sum(1).clip(0, 119)
            hit = bm[idx, w2[idx]]
            pay = np.where(hit, RATE * 100 / np.clip(Q[idx, w2[idx]], 1e-12, None), 0.0)
            vals.append(pay.sum() / (100.0 * bm[idx].sum(1).sum()))
        v = np.array(vals)
        L.append(f"| {lo2:g}〜{hi2:g}倍 | {real*100:.1f}% | {v.mean()*100:.1f}% | "
                 f"{v.min()*100:.1f}〜{v.max()*100:.1f}% | {(v >= real).sum()}/20 |")

    # ---- 中身を割る: 連番は実質 1-2-3 か
    from boatlab.model.trifecta import PERM_LABELS as _PLB
    L += ["\n### 中身を割る: 「連番」は実質 1-2-3 か\n",
          "| 目 | 4〜15倍で発火した回数 | 的中 |", "|---|---:|---:|"]
    bm = np.broadcast_to(seq, (N, 120)) & (ob >= 4) & (ob < 15)
    for i in np.flatnonzero(seq):
        L.append(f"| {_PLB[i]} | {int(bm[:, i].sum()):,} | {int((bm[:, i] & (W == i)).sum()):,} |")
    i123 = _PLB.index("1-2-3")
    j = np.flatnonzero(bm[:, i123])
    roi123 = np.where(W[j] == i123, P[j], 0.0).sum() / (100.0 * len(j))
    bm2 = bm.copy(); bm2[:, i123] = False
    j2 = np.flatnonzero(bm2.any(1))
    roi_rest = np.where(bm2[np.arange(N), W][j2], P[j2], 0.0).sum() / (100.0 * bm2.sum(1)[j2].sum())
    L += ["", f"**発火のほぼ全部が 1-2-3**（{int(bm[:, i123].sum()):,} / {int(bm.sum()):,}）。",
          f"1-2-3 だけなら回収率 **{roi123*100:.1f}%**、1-2-3 を除いた残りは {roi_rest*100:.1f}%（{len(j2):,}R）。",
          "→ **「連番が効く」ではなく「1-2-3 が効く」。**\n",
          "## 独立の検証: 2018〜2025 の公式払戻だけで確かめる\n",
          "2026年のオッズを一切使わず、**結果と公式配当だけ**で「毎レース 1-2-3 を買う」回収率を出せる",
          "（`回収率 = 1-2-3 が来る確率 × 1-2-3 のときの平均配当 ÷ 100`）。**発見に使ったデータと完全に別。**\n",
          "| 年 | レース数 | 1-2-3 が来る確率 | 平均配当 | 毎レース買う回収率 |", "|---|---:|---:|---:|---:|",
          "| 2018 | 53,099 | 6.75% | 1,254円 | **84.7%** |",
          "| 2019 | 53,155 | 7.19% | 1,175円 | **84.5%** |",
          "| 2020 | 53,300 | 7.37% | 1,140円 | **84.0%** |",
          "| 2021 | 53,316 | 7.27% | 1,148円 | **83.4%** |",
          "| 2022 | 53,670 | 7.24% | 1,141円 | **82.6%** |",
          "| 2023 | 53,558 | 7.22% | 1,155円 | **83.3%** |",
          "| 2024 | 53,579 | 6.95% | 1,176円 | **81.8%** |",
          "| 2025 | 53,599 | 6.99% | 1,132円 | **79.1%** |",
          "| 2026（1〜8月） | 36,251 | 7.44% | 1,105円 | **82.2%** |",
          "| **全期間** | **463,527** | **7.15%** | **1,159円** | **82.9%** |",
          "",
          "**9年間ずっと79〜85%。帰無は75%。** 比にして 1.05〜1.13 が9年連続で出ている。",
          "他の目と比べると 1-2-3 だけが突出: 1-3-2 80.7%、1-2-4 73.8%、2-1-3 76.8%、1-4-2 73.5%、1-3-4 74.1%。",
          "",
          "**「市場の1番人気の目」を毎回買うと 76.6%（2026年）。固定で 1-2-3 を買う方が 6pt 高い。**",
          "市場の人気は毎レース動くが、1-2-3 は動かない。**市場が本命を過小評価する癖の、いちばん濃い場所が 1-2-3。**",
          "",
          "注意: **2018年84.7% → 2025年79.1% とゆるやかに下がっている。** 市場が少しずつ直しているのか、",
          "単なる揺らぎかは、この9点では決められない。\n"]

    L += ["\n## まとめ: 飛躍5〜7",
          "",
          "### 飛躍5（レース間の連鎖）— **完全な空振り。だが空振り方が情報**",
          "「今日ここまで市場の見込みより何本多く逃げたか」で5分割しても、実測÷市場は **1.025〜1.042 で真っ平ら**。",
          "勾配がまったく無い。全区分が1.03前後なのは `crowd_bias.md` の1号艇過小評価そのもので、",
          "**セッション内の連鎖は市場が完全に織り込んでいる。** 前半・後半でも同じ。",
          "水面の状態や風の傾向は、我々が思いつく前に値段に入っている。",
          "",
          "### 飛躍6（選手ごとの持続的なズレ）— **向きは本物。ただし量は微小**",
          "探索期間（1〜5月）の選手ごとのズレが、**3か月空けた確認期間（6〜8月）でも単調に効く**:",
          "",
          "| 探索期間のズレ | 確認期間の 実測÷市場 | 95%区間 |",
          "|---|---:|---|",
          "| 市場が高く売りすぎ（下位20%） | **0.955** | 0.920〜0.989 |",
          "| やや高い | 0.977 | 0.943〜1.011 |",
          "| ふつう | 0.995 | 0.959〜1.031 |",
          "| やや安い | **1.035** | 1.003〜1.068 |",
          "| 市場が安く売りすぎ（上位20%） | 1.031 | 0.999〜1.062 |",
          "",
          "**単調で、両端が有意。** 各区分16,600走。市場は選手を1レースずつしか見ていないが、",
          "横断して見ると**同じ選手を同じ向きに誤り続けている**。これは「市場が体系的に間違えている相手」を",
          "初めて名指しできた例で、**向きとしては本物**（γ=0.702、3か月空けて持続）。",
          "",
          "**ただし量が微小だった。** 同じ物差し（市場オフセット型ロジット）に載せると **+0.0005 nats**、",
          "必要量 0.2877 の **0.18%**。目的1の5族（0.001前後）**より小さい**。",
          "理由は単純で、選手ごとのズレの幅が 0.86〜1.15 しかなく、log にすると ±0.14 程度の微調整にしかならないから。",
          "**「市場はこの選手を間違え続ける」は正しいが、間違いの大きさが3%では足りない。**",
          "",
          "### 飛躍7（見た目の癖）— **本命。1-2-3 が突出して安い**",
          "オッズ帯で層別しても「連番」が残り、細かく刻んでも残った（4〜6倍 +0.131、6〜8倍 +0.078、",
          "8〜10倍 +0.124、10〜15倍 +0.118 が同じオッズの他の目との差）。中身を割ると**実質すべて 1-2-3**。",
          "",
          "| 買い方 | レース数 | 的中率 | 回収率 | 比 | 前半/後半 | 帰無 |",
          "|---|---:|---:|---:|---:|---|---:|",
          "| 1-2-3 を オッズ4〜15倍のときだけ買う | 19,124 | 10.8% | **86.9%** | 1.159 | 85.9/86.9 | 0/20 |",
          "| 1-2-3 を毎レース買う（2026） | 36,251 | 7.4% | **82.2%** | 1.096 | — | — |",
          "| 1-2-3 を毎レース買う（2018〜2026・46万R） | 463,527 | 7.15% | **82.9%** | 1.105 | 9年連続79〜85% | — |",
          "| 市場の1番人気の目を毎レース買う（2026・対照） | 36,924 | 10.8% | 76.6% | 1.021 | 77.7/74.7 | 6/20 |",
          "",
          "**2018〜2025 の検証は、発見に使った2026年のオッズを一切使っていない**（結果と公式配当だけで計算できる）。",
          "完全に独立なデータで9年連続。これは本プロジェクトで見つかった中で**最も頑健な単純規則**。",
          "",
          "それでも **86.9% < 100%**。必要な比 1.333 に対して 1.16。**勝てる規則ではない。**",
          "オッズの末尾は .0 が 1.048 で有意に見えるが、10桁試して1つなので偶然の範囲。",
          ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
