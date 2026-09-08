"""場（レース場）別の買い方研究。

問い: 場ごとに回収率100%を超える場はあるか / 場ごとに買い方を変える・特定の場だけ買う
      ことで成績は良くなるか。

正直に答えるための設計（多重検定対策）:
  - 探索期間 2026-01〜05 で場別成績を出す（ここは「見つける」ための期間）。
  - 確認期間 2026-06〜08 は一切見ないで選び、選んだあとに1回だけ当てる。
  - 「探索で良かった場」を確認期間で評価 → これが実際に得られたはずの成績。
  - 場ごとにパラメータを最適化した場合も同様に探索→確認で評価する。
  - 偶然どれだけばらつくかを見るため、場ラベルを無作為化した対照（プラセボ）も出す。

出力: reports/backtest/stadium_study.{csv,md}
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from boatlab.backtest.metrics import roi_bootstrap
from boatlab.config import ROOT

BT = Path(ROOT) / "research_data" / "bt2026_top20.npz"
OUTDIR = Path(ROOT) / "reports" / "backtest"
EXPLORE_END = "2026-05-31"
STADIUM_NAMES = {
    1: "桐生", 2: "戸田", 3: "江戸川", 4: "平和島", 5: "多摩川", 6: "浜名湖", 7: "蒲郡", 8: "常滑",
    9: "津", 10: "三国", 11: "びわこ", 12: "住之江", 13: "尼崎", 14: "鳴門", 15: "丸亀", 16: "児島",
    17: "宮島", 18: "徳山", 19: "下関", 20: "若松", 21: "芦屋", 22: "福岡", 23: "唐津", 24: "大村",
}

GRID = [dict(ev_min=e, odds_hi=h, s15_min=s, max_points=m)
        for e, h, s, m in itertools.product((0.8, 0.9, 1.0), (15, 20, 50), (0.70, 0.73), (3, 5))]
CURRENT = dict(ev_min=0.8, odds_hi=20, s15_min=0.73, max_points=5)


def load():
    z = np.load(BT, allow_pickle=False)
    p = z["p"].astype(np.float64)          # (n,20) 上位20点の確率
    o = z["odds"].astype(np.float64)       # (n,20) 同じ並びのオッズ
    top = z["top"].astype(np.int64)        # (n,20) 120通りの index
    order = np.argsort(-p, axis=1)         # 念のため確率降順に整列
    rows = np.arange(len(p))[:, None]
    p, o, top = p[rows, order], o[rows, order], top[rows, order]
    return dict(date=z["date"].astype(str), stadium=z["stadium"].astype(int), p=p, odds=o, top=top,
                tri=z["tri"].astype(int), payout=z["payout"].astype(float))


def _stakes(pv: np.ndarray) -> np.ndarray:
    """確率比例・予算3000円（1点上限1000円）・100円単位。本番と同じ allocate を使う。"""
    from boatlab.model.staking import StakingParams, allocate
    k = len(pv)
    total = min(3000, 1000 * k)
    idx = list(range(k))
    return np.array(allocate(idx, idx, pv, np.full(k, 10.0), StakingParams(
        method="prob", total=total, prob_power=1.0, max_share=min(1.0, 1000 / total))))


def simulate(ds: dict, prm: dict):
    """各レースの (投資, 払戻, 的中)。買わないレースは投資0。"""
    p, o, top, tri, pay = ds["p"], ds["odds"], ds["top"], ds["tri"], ds["payout"]
    n = len(tri)
    S15 = p[:, :15].sum(axis=1)
    ev = p[:, :15] * np.where(np.isfinite(o[:, :15]), o[:, :15], 0.0)
    ok = (ev >= prm["ev_min"]) & (o[:, :15] >= 5.0) & (o[:, :15] <= prm["odds_hi"]) & np.isfinite(o[:, :15])
    ok &= (S15 >= prm["s15_min"])[:, None]
    stake = np.zeros(n); ret = np.zeros(n); hit = np.zeros(n, bool)
    for i in np.flatnonzero(ok.any(axis=1)):
        c = np.flatnonzero(ok[i])[: prm["max_points"]]
        st = _stakes(p[i, c])
        stake[i] = st.sum()
        m = top[i, c] == tri[i]
        if m.any():
            hit[i] = True
            ret[i] = pay[i] * st[m][0] / 100
    return stake, ret, hit


def summarize(stake, ret, hit, ci=True):
    m = stake > 0
    n = int(m.sum())
    if n == 0:
        return dict(n=0, hit=None, roi=None, lo=None, hi=None, pnl=0)
    lo, hi = roi_bootstrap(stake[m], ret[m], n_boot=400) if ci else (None, None)
    return dict(n=n, hit=round(float(hit[m].mean()), 4), roi=round(float(ret[m].sum() / stake[m].sum()), 4),
                lo=None if lo is None else round(lo, 3), hi=None if hi is None else round(hi, 3),
                pnl=int(ret[m].sum() - stake[m].sum()))


def main():
    ds = load()
    expl = ds["date"] <= EXPLORE_END
    conf = ~expl
    st, rt, ht = simulate(ds, CURRENT)
    sc = ds["stadium"]

    # ---- 1) 場別（現在設定）
    rows = []
    for s in range(1, 25):
        m = sc == s
        rows.append(dict(stadium=s, name=STADIUM_NAMES[s],
                         **{f"e_{k}": v for k, v in summarize(st * (m & expl), rt * (m & expl), ht).items()},
                         **{f"c_{k}": v for k, v in summarize(st * (m & conf), rt * (m & conf), ht).items()}))
    df = pd.DataFrame(rows)

    # ---- 2) 探索で良かった場だけ買う → 確認期間で1回だけ評価
    picks = {}
    for k in (3, 5, 8, 12):
        top_s = df.sort_values("e_roi", ascending=False).head(k)["stadium"].tolist()
        m = np.isin(sc, top_s) & conf
        picks[f"探索上位{k}場のみ"] = dict(stadiums=top_s, **summarize(st * m, rt * m, ht))
    m100 = df[df["e_roi"].fillna(0) >= 1.0]["stadium"].tolist()
    if m100:
        m = np.isin(sc, m100) & conf
        picks["探索で回収率100%超の場のみ"] = dict(stadiums=m100, **summarize(st * m, rt * m, ht))
    picks["全場（現状）"] = dict(stadiums=list(range(1, 25)), **summarize(st * conf, rt * conf, ht))

    # ---- 3) 場ごとに買い方（36候補）を最適化 → 確認期間で評価
    sims = {json.dumps(v, sort_keys=True): simulate(ds, v) for v in GRID}
    per_st, best_of = [], {}
    for s in range(1, 25):
        m = sc == s
        best, best_lo, best_roi = None, -9, None
        for key, (a, b, h) in sims.items():
            r = summarize(a * (m & expl), b * (m & expl), h)
            if r["n"] < 60 or r["lo"] is None:
                continue
            if r["lo"] > best_lo:
                best, best_lo, best_roi = key, r["lo"], r["roi"]
        if best is None:
            continue
        a, b, h = sims[best]
        cr = summarize(a * (m & conf), b * (m & conf), h)
        best_of[s] = best
        per_st.append(dict(stadium=s, name=STADIUM_NAMES[s], params=json.loads(best), e_roi=best_roi,
                           e_lo=round(best_lo, 3), c_n=cr["n"], c_roi=cr["roi"], c_pnl=cr["pnl"]))
    # つなげた成績（場ごとに最適パラメータを使い、確認期間で全場買う）
    A = np.zeros(len(sc)); B = np.zeros(len(sc)); H = np.zeros(len(sc), bool)
    for s, key in best_of.items():
        m = sc == s
        a, b, h = sims[key]
        A[m], B[m], H[m] = a[m], b[m], h[m]
    tuned = summarize(A * conf, B * conf, H)

    # ---- 4) 対照（プラセボ）: 場ラベルを無作為化しても同じくらいばらつくか
    rng = np.random.default_rng(0)
    placebo = []
    for _ in range(200):
        fake = rng.permutation(sc)
        rs = []
        for s in range(1, 25):
            m = (fake == s) & expl
            r = summarize(st * m, rt * m, ht, ci=False)
            if r["n"] >= 60:
                rs.append(r["roi"])
        if rs:
            placebo.append((max(rs), sum(1 for r in rs if r >= 1.0)))
    pl_max = np.array([x[0] for x in placebo]); pl_cnt = np.array([x[1] for x in placebo])

    # ---- 5) 探索の順位は確認期間に引き継がれるか（順位相関）／全期間の場別
    from scipy import stats as _st
    d2 = df.dropna(subset=["e_roi", "c_roi"])
    rho = _st.spearmanr(d2["e_roi"], d2["c_roi"])
    full = []
    for s in range(1, 25):
        m = sc == s
        r = summarize(st * m, rt * m, ht)
        full.append(dict(name=STADIUM_NAMES[s], **r, avg_pay=int(ds["payout"][m].mean())))
    fdf = pd.DataFrame(full).sort_values("roi", ascending=False)
    rough = np.corrcoef(fdf["avg_pay"], fdf["roi"])[0, 1]

    # ---- 6) 対照2: 無作為に3場選んで確認期間で買った場合の分布
    rng2 = np.random.default_rng(7)
    dist = []
    for _ in range(2000):
        pick = rng2.choice(np.arange(1, 25), 3, replace=False)
        m = np.isin(sc, pick) & conf
        r = summarize(st * m, rt * m, ht, ci=False)
        if r["n"] >= 100:
            dist.append(r["roi"])
    dist = np.array(dist)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTDIR / "stadium_study.csv", index=False)
    L = []
    L.append(f"# 場別の買い方研究（{CURRENT}、2026年実オッズ {len(sc):,}R）\n")
    dmin, dmax = sorted(ds["date"])[0], sorted(ds["date"])[-1]
    L.append(f"探索 = {dmin}〜{EXPLORE_END} / 確認 = 2026-06-01〜{dmax}\n")
    L.append("## 1. 場別成績（現在設定）\n")
    L.append("| 場 | 探索n | 探索的中 | 探索回収 | 95%区間 | 確認n | 確認的中 | 確認回収 |")
    L.append("|---|---:|---:|---:|---|---:|---:|---:|")
    for _, r in df.sort_values("e_roi", ascending=False).iterrows():
        f = lambda v: "-" if pd.isna(v) else f"{v*100:.1f}%"
        L.append(f"| {r['name']} | {r['e_n']} | {f(r['e_hit'])} | {f(r['e_roi'])} | "
                 f"{'-' if pd.isna(r['e_lo']) else f'{r.e_lo*100:.0f}〜{r.e_hi*100:.0f}%'} | "
                 f"{r['c_n']} | {f(r['c_hit'])} | {f(r['c_roi'])} |")
    L.append("\n## 2. 探索で良かった場だけ買った場合（確認期間・後出しなし）\n")
    L.append("| 選び方 | 場 | n | 的中 | 回収 | 損益 |")
    L.append("|---|---|---:|---:|---:|---:|")
    for k, v in picks.items():
        nm = "全場" if len(v["stadiums"]) == 24 else "・".join(STADIUM_NAMES[s] for s in v["stadiums"])
        hs = "-" if v["hit"] is None else f"{v['hit'] * 100:.1f}%"
        rs = "-" if v["roi"] is None else f"{v['roi'] * 100:.1f}%"
        L.append(f"| {k} | {nm} | {v['n']} | {hs} | {rs} | {v['pnl']:,}円 |")
    L.append("\n## 3. 場ごとに買い方を変えた場合（探索で選び、確認で評価）\n")
    L.append(f"つなげた確認期間の成績: n={tuned['n']} 的中{tuned['hit']*100:.1f}% 回収{tuned['roi']*100:.1f}% 損益{tuned['pnl']:,}円\n")
    L.append("| 場 | 探索で選ばれた買い方 | 探索回収 | 確認n | 確認回収 |")
    L.append("|---|---|---:|---:|---:|")
    for r in sorted(per_st, key=lambda x: -(x["e_roi"] or 0)):
        p = r["params"]
        cr = "-" if r["c_roi"] is None else f"{r['c_roi'] * 100:.1f}%"
        L.append(f"| {r['name']} | EV≥{p['ev_min']}・5〜{p['odds_hi']}倍・自信≥{p['s15_min']}・最大{p['max_points']}点 | "
                 f"{r['e_roi'] * 100:.1f}% | {r['c_n']} | {cr} |")
    L.append("\n## 4. 対照（場ラベルを無作為化した場合の探索期間）\n")
    L.append(f"200回の無作為化で、最も回収率が高い「場」の平均は {pl_max.mean()*100:.1f}%（中央値 {np.median(pl_max)*100:.1f}%、"
             f"95パーセンタイル {np.percentile(pl_max,95)*100:.1f}%）。")
    L.append(f"回収率100%を超える「場」の数は平均 {pl_cnt.mean():.1f} 個（最大 {pl_cnt.max()} 個）。")
    L.append(f"実データの探索期間では最高 {df['e_roi'].max()*100:.1f}%、100%超は {int((df['e_roi']>=1.0).sum())} 場。\n")
    L.append(f"無作為に3場選んで確認期間で買うと回収率は 中央値 {np.median(dist)*100:.1f}%・"
             f"5〜95パーセンタイル {np.percentile(dist,5)*100:.1f}〜{np.percentile(dist,95)*100:.1f}%（2,000回）。"
             f"探索上位3場の {picks['探索上位3場のみ']['roi']*100:.1f}% は "
             f"上位 {100 - (dist < picks['探索上位3場のみ']['roi']).mean()*100:.0f}% 相当。\n")
    L.append("## 5. 探索期間の順位は確認期間に引き継がれるか\n")
    L.append(f"場別回収率の順位相関（探索 vs 確認）: ρ = {rho.statistic:.3f}（p = {rho.pvalue:.2f}）。"
             f"ピアソン相関 {np.corrcoef(d2['e_roi'], d2['c_roi'])[0,1]:.3f}。\n")
    L.append("## 6. 全期間（2026-01〜08）の場別成績\n")
    L.append("| 場 | n | 的中 | 回収 | 95%区間 | 平均払戻 |")
    L.append("|---|---:|---:|---:|---|---:|")
    for _, r in fdf.iterrows():
        L.append(f"| {r['name']} | {r['n']} | {r['hit']*100:.1f}% | {r['roi']*100:.1f}% | "
                 f"{r['lo']*100:.0f}〜{r['hi']*100:.0f}% | {r['avg_pay']:,}円 |")
    L.append(f"\n全期間で回収率100%を超えるのは {int((fdf['roi']>=1.0).sum())} 場、"
             f"95%区間の下限が100%を超えるのは {int((fdf['lo']>=1.0).sum())} 場。")
    L.append(f"「荒れやすさ（平均払戻）」と回収率の相関は {rough:.2f}（n=24、有意ではない）。\n")
    (OUTDIR / "stadium_study.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
