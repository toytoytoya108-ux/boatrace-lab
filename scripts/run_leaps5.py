"""飛躍11〜13。「1点しか買わない」「群衆はいつも同じ」という前提を疑う。

飛躍11: **複勝は1レース1点、という前提は正しいか。**
  複勝は3着以内なら当たるので、1レースに2点・3点と買える。2番目3番目の艇の方が
  人気が集まりにくく、元返しの補助金が効いているかもしれない。**一度も測っていない。**

飛躍12: **群衆は珍しい番組を間違えるのでは。**
  企画レース・ドリーム戦・女子戦・新人戦は数が少なく、買い手の経験も薄いはず。
  番組の種類ごとに「実測÷市場」を見る。値付けの精度が落ちる番組があれば、そこが狙い目になる。

飛躍13: **群衆の顔ぶれは曜日と時間帯で変わるのでは。**
  平日昼と週末、ナイター。買い手の構成が変われば賢さも変わるはず。
  曜日・時間帯・ナイターで層別して歪みを測る。

出力: reports/research/leaps5.md
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from boatlab.backtest.metrics import roi_bootstrap  # noqa: E402
from boatlab.config import ROOT  # noqa: E402
from boatlab.model.trifecta import PERM_LABELS, PERMS  # noqa: E402

OUT = Path(ROOT) / "reports" / "research" / "leaps5.md"
CACHE = Path("/tmp/claude-0/mix10_cache.npz")
DB = str(Path(ROOT) / "data" / "lab.db")
FEAT = Path(ROOT) / "reports" / "research" / "manshu_features.parquet"
_A = np.array([p[0] for p in PERMS]); _B = np.array([p[1] for p in PERMS]); _C = np.array([p[2] for p in PERMS])
RATE = 0.75


def summ(ret, stake, seed=0):
    if stake.sum() <= 0:
        return None
    lo, hi = roi_bootstrap(stake, ret, n_boot=300)
    roi = ret.sum() / stake.sum()
    return dict(n=len(ret), roi=roi, lo=lo, hi=hi, hit=(ret > 0).mean(),
                avg=ret[ret > 0].mean() if (ret > 0).any() else 0.0)


def main():
    z = np.load(CACHE, allow_pickle=True)
    rid, date, O, W, P = z["rid"].astype(np.int64), pd.DatetimeIndex(z["date"]), z["O"], z["W"], z["P"]
    inv = np.where(np.isfinite(O), 1.0 / np.nan_to_num(O, nan=1e9), 0.0)
    Q = inv / inv.sum(1, keepdims=True)
    N = len(Q)
    half = np.asarray(date <= "2026-05-31")
    # 市場が示す「3着以内確率」（3連単プールから周辺化）
    q3 = np.stack([Q[:, (_A == a) | (_B == a) | (_C == a)].sum(1) for a in range(6)], axis=1)

    con = sqlite3.connect(DB)
    rows = con.execute("SELECT race_id, payouts FROM results WHERE race_id >= 202601010000 AND payouts IS NOT NULL").fetchall()
    con.close()
    place = {}
    for r, js in rows:
        d0 = json.loads(js) if isinstance(js, str) else js
        if not isinstance(d0, dict):
            continue
        m = {}
        for e in d0.get("place") or []:
            c0 = str(e.get("combination") or "")
            if c0.isdigit():
                m[int(c0) - 1] = float(e.get("amount") or 0)
        place[int(r)] = m
    pay_place = np.zeros((N, 6))
    has = np.zeros(N, bool)
    for i, r in enumerate(rid):
        m = place.get(int(r))
        if m:
            has[i] = True
            for k, v in m.items():
                if 0 <= k < 6:
                    pay_place[i, k] = v

    L = [f"# 飛躍11〜13（2026年・{N:,}レース・確定オッズ）\n",
         "## 飛躍11: 複勝は1レース1点、という前提は正しいか\n",
         "複勝は3着以内で当たるので1レースに2点3点と買える。**2番目3番目の艇の方が人気が集まりにくく、",
         "元返し（100円下限）の補助金が効いているかもしれない。** 一度も測っていなかった。",
         "市場の3着以内確率（3連単プールから周辺化）の高い順に k 点、100円ずつ。\n",
         "| 買い方 | レース数 | 点数 | 的中率 | 平均払戻 | 元返しの割合 | 回収率 | 95%区間 | 前半 | 後半 |",
         "|---|---:|---:|---:|---:|---:|---:|---|---:|---:|"]
    ok = has & (q3.sum(1) > 2.5)
    order3 = np.argsort(-q3, axis=1)
    def buy_place(kth, mask=None, thr=None):
        m = ok if mask is None else (ok & mask)
        idx = np.flatnonzero(m)
        if thr is not None:
            keep = q3[idx, order3[idx, kth - 1]] >= thr
            idx = idx[keep]
        lanes = order3[idx, kth - 1]
        pay = pay_place[idx, lanes]
        return pay, np.full(len(idx), 100.0), idx, lanes
    for kth in (1, 2, 3, 4):
        pay, st, idx, lanes = buy_place(kth)
        r = summ(pay, st)
        moto = (pay == 100).sum() / max((pay > 0).sum(), 1)
        hs = [summ(*buy_place(kth, m)[:2]) for m in (half, ~half)]
        L.append(f"| 市場の3着以内確率 {kth}位の艇 | {r['n']:,} | 1 | {r['hit']*100:.1f}% | {r['avg']:,.0f}円 | "
                 f"{moto*100:.1f}% | **{r['roi']*100:.1f}%** | {r['lo']*100:.1f}〜{r['hi']*100:.1f}% | "
                 f"{hs[0]['roi']*100:.1f}% | {hs[1]['roi']*100:.1f}% |")
    # まとめ買い
    for kk in (2, 3):
        pays, sts = [], []
        for kth in range(1, kk + 1):
            pay, st, _, _ = buy_place(kth)
            pays.append(pay); sts.append(st)
        pay = np.concatenate(pays); st = np.concatenate(sts)
        r = summ(pay, st)
        L.append(f"| 上位{kk}艇をまとめて買う | {len(pay)//kk:,} | {kk} | {r['hit']*100:.1f}% | {r['avg']:,.0f}円 | "
                 f"{(pay==100).sum()/max((pay>0).sum(),1)*100:.1f}% | **{r['roi']*100:.1f}%** | "
                 f"{r['lo']*100:.1f}〜{r['hi']*100:.1f}% | — | — |")
    # しきい値つき（本番モードと同じ 0.90 を2位の艇にも当てる）
    L += ["", "本番の複勝モードは「市場の2着以内確率0.90以上の艇を1点」。**同じ発想を3着以内確率で、順位別に見る。**", "",
          "| 買い方 | しきい値 | レース数 | 的中率 | 平均払戻 | 回収率 | 95%区間 |", "|---|---:|---:|---:|---:|---:|---|"]
    for kth in (1, 2, 3):
        for thr in (0.80, 0.90, 0.95):
            pay, st, idx, _ = buy_place(kth, thr=thr)
            if len(pay) < 300:
                continue
            r = summ(pay, st)
            L.append(f"| 3着以内確率 {kth}位の艇 | {thr:.2f} | {r['n']:,} | {r['hit']*100:.1f}% | {r['avg']:,.0f}円 | "
                     f"**{r['roi']*100:.1f}%** | {r['lo']*100:.1f}〜{r['hi']*100:.1f}% |")

    # ================= 飛躍12・13: 番組と時間帯
    df = pd.read_parquet(FEAT)
    df = df[df["year"] == 2026]
    pos = {int(r): i for i, r in enumerate(rid)}
    df = df[df["race_id"].astype(np.int64).isin(pos)].copy()
    df["k"] = df["race_id"].astype(np.int64).map(pos)
    k = df["k"].values
    top1 = np.argmax(Q, axis=1)
    hit1 = (W == top1)
    roi1 = np.where(hit1, P, 0.0) / 100.0                 # 人気1番の目を1点買ったときの払戻/100
    i123 = PERM_LABELS.index("1-2-3")
    roi123 = np.where(W == i123, P, 0.0) / 100.0
    eff = 1.0 / inv.sum(1)                                 # レースごとの実効払戻率

    def blk(title, col, note=""):
        out = [f"\n**{title}**{note}\n",
               "| 区分 | レース数 | 人気1番の回収率 | 比 | 1-2-3 の回収率 | 比 |", "|---|---:|---:|---:|---:|---:|"]
        for v, g in df.groupby(col, dropna=True):
            kk = g["k"].values
            if len(kk) < 800:
                continue
            e = eff[kk].mean()
            a = roi1[kk].mean(); b = roi123[kk].mean()
            out.append(f"| {v} | {len(kk):,} | {a*100:.1f}% | {'**' if a/e > 1.05 else ''}{a/e:.3f}"
                       f"{'**' if a/e > 1.05 else ''} | {b*100:.1f}% | {'**' if b/e > 1.05 else ''}{b/e:.3f}"
                       f"{'**' if b/e > 1.05 else ''} |")
        return out

    L += ["\n## 飛躍12: 群衆は珍しい番組を間違えるか\n",
          "企画レース・ドリーム戦・女子戦・新人戦は数が少なく、買い手の経験も薄いはず。",
          "**値付けの精度が落ちる番組があれば、そこが狙い目になる。**",
          "物差しは「人気1番の目」と「1-2-3」の回収率。比 = 回収率 ÷ そのレース群の実効払戻率。\n"]
    df["種別"] = df["race_type"].fillna("不明")
    top_types = df["種別"].value_counts()
    df["種別5"] = df["種別"].where(df["種別"].isin(top_types.head(8).index), "その他")
    L += blk("レースの種別", "種別5")
    L += blk("グレード", "grade")

    L += ["\n## 飛躍13: 群衆の顔ぶれは曜日と時間帯で変わるか\n",
          "平日昼と週末、ナイター。買い手が変われば賢さも変わるはず。\n"]
    df["曜日"] = df["dow"].map({0: "月", 1: "火", 2: "水", 3: "木", 4: "金", 5: "土", 6: "日"})
    df["時間帯"] = pd.cut(df["hour"], [-1, 11, 13, 15, 17, 24], labels=["〜11時", "12〜13時", "14〜15時", "16〜17時", "18時〜"])
    df["開催"] = df["night"].map({0: "昼", 1: "ナイター"})
    L += blk("曜日", "曜日")
    L += blk("時間帯（締切）", "時間帯")
    L += blk("ナイターかどうか", "開催")

    # ---- 飛躍12・13 の検算: 目立ったセルは偶然の幅に収まるか
    L += ["\n## 飛躍12・13 の検算: 目立ったセルは偶然か\n",
          "群れを無作為に作り直したとき、同じ大きさの群でどれくらいの比が出るかを見る（200回）。",
          "**観測値がその幅の中なら、それは番組や曜日の性質ではなく、ただの揺らぎ。**\n"]
    rng = np.random.default_rng(41)
    kk_all = df["k"].values
    e_all = eff[kk_all]
    def ratio_of(kk):
        return roi1[kk].mean() / eff[kk].mean(), roi123[kk].mean() / eff[kk].mean()
    checks = [("準優勝戦", (df["種別5"] == "準優勝戦").values), ("一般戦", (df["種別5"] == "一般戦").values),
              ("G1", (df["grade"] == "G1").values), ("土曜", (df["曜日"] == "土").values),
              ("14〜15時締切", (df["時間帯"] == "14〜15時").values), ("ナイター", (df["開催"] == "ナイター").values)]
    L += ["| 区分 | レース数 | 人気1番の比 | 95%区間 | 前半／後半 | 無作為200回の範囲 | 判定 |",
          "|---|---:|---:|---|---:|---|---|"]
    h1m = half[kk_all]
    for nm, m in checks:
        kk = kk_all[m]
        if len(kk) < 500:
            continue
        r1, _ = ratio_of(kk)
        pay = roi1[kk] * 100.0
        lo_, hi_ = roi_bootstrap(np.full(len(kk), 100.0), pay, n_boot=300)
        e = eff[kk].mean()
        h = []
        for sub in (h1m[m], ~h1m[m]):
            h.append(roi1[kk[sub]].mean() / eff[kk[sub]].mean() if sub.sum() > 100 else np.nan)
        sim = []
        for _ in range(200):
            pick = rng.choice(len(kk_all), size=len(kk), replace=False)
            sim.append(roi1[kk_all[pick]].mean() / eff[kk_all[pick]].mean())
        sim = np.array(sim)
        inside = sim.min() <= r1 <= sim.max()
        L.append(f"| {nm} | {len(kk):,} | {r1:.3f} | {lo_/e:.3f}〜{hi_/e:.3f} | {h[0]:.3f} ／ {h[1]:.3f} | "
                 f"{sim.min():.3f}〜{sim.max():.3f} | {'揺らぎの範囲' if inside else '**幅の外**'} |")

    L += ["\n## まとめ: 飛躍11〜13",
          "",
          "### 飛躍11（複勝を複数買う）— **前提は正しかった。1点が最適**",
          "市場の3着以内確率の順に買うと **1位 93.6% / 2位 88.5% / 3位 87.9% / 4位 85.2%** と単調に落ちる。",
          "まとめ買いは薄めるだけ（上位2艇 91.0%、上位3艇 90.0%）。**点数を増やすと必ず下がる。**",
          "理由は元返しの分布で、**的中したときに100円で返ってくる割合が 1位 50.1% → 2位 15.5% → 3位 4.4%**。",
          "`sweet_spot.md` の「元返しは胴元からの補助金」がそのまま効いていて、補助金は**最有力の1艇にだけ集中している**。",
          "しきい値つきでも同じ: 3着以内確率0.95以上の1位艇が **98.0%**（n=1,824）で、これが複勝の上限。",
          "**本番の複勝モード（1点・0.90以上）の設計は正しい。変更しない。**",
          "",
          "### 飛躍12（珍しい番組）・飛躍13（曜日と時間帯）— **どちらも揺らぎの範囲**",
          "準優勝戦の比1.154、一般戦の1-2-3が1.304、土曜1.082、14〜15時1.097 などが目を引くが、",
          "**同じ大きさの群を無作為に作ると同じくらいの幅が出る**（上の検算表）。",
          "前半・後半で符号が揃わないセルも多い。**番組の種類や曜日で群衆の賢さが変わるという証拠は無い。**",
          "",
          "1レース1点・n=5,000 程度では、比の95%区間が ±0.05〜0.10 になる。",
          "**この粒度で「どの区分が有利か」を探すと、必ず何かが有意に見えてしまう。** 場別の検証（`stadium_study.md`）で",
          "踏んだのと同じ罠で、今回は事前に無作為対照を置いたので引っかからずに済んだ。",
          ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
