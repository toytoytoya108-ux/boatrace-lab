"""市場は「何を」過大評価しているか（アイデアB）。

これまでは「モデル vs 市場」の総合点で比べて負けていた。総合点の勝負では、LightGBM の
高分散な反論がノイズごと優位を飲み込む（実測: モデルが期待値1.0超えと見た買い目の回収率は69.8%）。

ここでは代わりに、**同じ特徴量で2つの線形モデル**を作り、係数を比べる。

  モデルA: 特徴量 → **実際の1着**（何が本当に効くか）
  モデルB: 特徴量 → **市場が付けた1着確率**（群衆が何を見ているか）

β_B > β_A の特徴量は群衆の過大評価、β_B < β_A は過小評価。パラメータは十数個の線形なので
過学習の余地が小さく、ズレが出れば「たまたま」ではなく群衆の癖と解釈できる。

探索 2026-01〜05 で係数を推定し、確認 2026-06〜08 で1回だけ評価する。

出力: reports/backtest/crowd_bias.md
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from boatlab.backtest.metrics import roi_bootstrap
from boatlab.config import ROOT
from boatlab.model.trifecta import PERM_LABELS, PERMS

OUT = Path(ROOT) / "reports" / "backtest" / "crowd_bias.md"
DB = str(Path(ROOT) / "data" / "lab.db")
EXPLORE_END = "2026-05-31"
FIRST = np.array([p[0] for p in PERMS])
KLASS = {"A1": 3.0, "A2": 2.0, "B1": 1.0, "B2": 0.0}

FEATS = [("nat_win_rate", "全国勝率"), ("loc_win_rate", "当地勝率"), ("nat_rate2", "全国2連率"),
         ("loc_rate2", "当地2連率"), ("motor_rate2", "モーター2連率"), ("boat_rate2", "ボート2連率"),
         ("avg_st", "平均ST"), ("klass_n", "級別"), ("f_count", "F数"), ("age", "年齢"),
         ("weight", "体重"), ("exhibition_time", "展示タイム"), ("st_exh", "展示ST"), ("tilt", "チルト")]


def load():
    con = sqlite3.connect(DB)
    q = """
      SELECT e.race_id, e.lane, e.age, e.weight, e.klass, e.f_count, e.avg_st,
             e.nat_win_rate, e.nat_rate2, e.loc_win_rate, e.loc_rate2,
             e.motor_rate2, e.boat_rate2, r.race_date, res.trifecta,
             (SELECT p.exhibition_time FROM preview_snapshots p WHERE p.race_id=e.race_id AND p.lane=e.lane
                ORDER BY p.fetched_at DESC LIMIT 1) AS exhibition_time,
             (SELECT p.st_exh FROM preview_snapshots p WHERE p.race_id=e.race_id AND p.lane=e.lane
                ORDER BY p.fetched_at DESC LIMIT 1) AS st_exh,
             (SELECT p.tilt FROM preview_snapshots p WHERE p.race_id=e.race_id AND p.lane=e.lane
                ORDER BY p.fetched_at DESC LIMIT 1) AS tilt
      FROM entries e JOIN races r ON r.id=e.race_id JOIN results res ON res.race_id=e.race_id
      WHERE r.race_date >= '2026-01-01' AND res.trifecta IS NOT NULL AND res.is_irregular=0
        AND e.is_absent=0
      ORDER BY e.race_id, e.lane"""
    df = pd.read_sql_query(q, con)
    lab2 = {l: i for i, l in enumerate(PERM_LABELS)}
    mk = {}
    for rid, js in con.execute("SELECT race_id, odds FROM odds_snapshots WHERE bet_type='3t' AND source='turnmark_final'"):
        d = json.loads(js) if isinstance(js, str) else js
        inv = np.zeros(120)
        for k, v in d.items():
            j = lab2.get(k)
            if j is not None and v and float(v) > 0:
                inv[j] = 1.0 / float(v)
        if (inv > 0).sum() >= 110:
            qq = inv / inv.sum()
            mk[int(rid)] = np.array([qq[FIRST == a].sum() for a in range(6)])
    con.close()
    df["klass_n"] = df["klass"].map(KLASS)
    cnt = df.groupby("race_id")["lane"].count()
    keep = set(cnt[cnt == 6].index) & set(mk)
    df = df[df["race_id"].isin(keep)].copy()
    cols = [f for f, _ in FEATS]
    df[cols] = df[cols].astype(float)
    med = df[cols].median()
    df[cols] = df[cols].fillna(med)
    rid = df["race_id"].values[::6]
    X = df[cols].values.reshape(-1, 6, len(cols))
    mu, sd = X.reshape(-1, len(cols)).mean(0), X.reshape(-1, len(cols)).std(0) + 1e-9
    X = (X - mu) / sd
    lane = np.zeros((len(rid), 6, 5))                      # 1号艇を基準にしたコースのダミー
    for a in range(1, 6):
        lane[:, a, a - 1] = 1.0
    X = np.concatenate([X, lane], 2)
    y = np.array([int(str(t).split("-")[0]) - 1 for t in df["trifecta"].values[::6]])
    Q = np.stack([mk[int(r)] for r in rid])
    date = df["race_date"].values[::6].astype("U10")
    names = [n for _, n in FEATS] + [f"{a+1}号艇" for a in range(1, 6)]
    return X, Q, y, date, names, rid


def fit(X, target):
    """条件付きロジット。target は 1着のone-hot（モデルA）か市場確率（モデルB）。"""
    n, k, d = X.shape

    def f(w):
        s = X @ w
        s -= s.max(1, keepdims=True)
        ex = np.exp(s)
        p = ex / ex.sum(1, keepdims=True)
        loss = -np.mean(np.sum(target * np.log(np.clip(p, 1e-12, None)), 1))
        g = np.einsum("nkd,nk->d", X, p - target) / n
        return loss, g

    r = minimize(f, np.zeros(d), jac=True, method="L-BFGS-B", options={"maxiter": 500})
    return r.x


def probs(X, w):
    s = X @ w
    s -= s.max(1, keepdims=True)
    ex = np.exp(s)
    return ex / ex.sum(1, keepdims=True)


def main():
    X, Q, y, date, names, rid = load()
    e, c = date <= EXPLORE_END, date > EXPLORE_END
    Y = np.zeros_like(Q)
    Y[np.arange(len(y)), y] = 1.0
    wA = fit(X[e], Y[e])
    wB = fit(X[e], Q[e])
    L = [f"# 市場は何を過大評価しているか（{len(y):,}R・2026年）\n",
         "同じ特徴量で2つの条件付きロジットを推定する。A＝実際の1着を当てる。B＝市場の1着確率を再現する。",
         f"探索 〜{EXPLORE_END} で係数を推定し、確認期間で1回評価。係数は標準化済みで比較可能。\n",
         "| 特徴量 | A（実際に効く） | B（市場が見ている） | 差 B−A | 解釈 |",
         "|---|---:|---:|---:|---|"]
    order = np.argsort(-np.abs(wB - wA))
    for i in order:
        dlt = wB[i] - wA[i]
        tag = "市場が**過大評価**" if dlt > 0.02 else ("市場が**過小評価**" if dlt < -0.02 else "ほぼ一致")
        L.append(f"| {names[i]} | {wA[i]:+.3f} | {wB[i]:+.3f} | {dlt:+.3f} | {tag} |")
    llA = -np.mean(np.log(np.clip(probs(X[c], wA)[np.arange(int(c.sum())), y[c]], 1e-12, None)))
    llQ = -np.mean(np.log(np.clip(Q[c][np.arange(int(c.sum())), y[c]], 1e-12, None)))
    L.append(f"\n確認期間の1着対数損失: 線形モデルA **{llA:.4f}** / 市場 **{llQ:.4f}**"
             f"（{'Aの勝ち' if llA < llQ else '市場の勝ち'}）\n")

    # ズレの方向にだけ賭ける: スコア = (β_A − β_B)·x が大きい艇＝市場が過小評価している艇
    L.append("## ズレの方向に賭けるとどうなるか\n")
    L.append("スコア = (β_A − β_B)·x。大きいほど「市場が過小評価している」艇。単勝で買った場合。\n")
    L.append("| スコア上位 | n | 的中 | 市場確率 | 実勝率÷市場確率 | 回収率 | 95%区間 |")
    L.append("|---|---:|---:|---:|---:|---:|---|")
    sc = X @ (wA - wB)
    o = 0.75 / np.clip(Q, 1e-12, None)
    win = np.zeros_like(Q, bool)
    win[np.arange(len(y)), y] = True
    for frac in (0.10, 0.05, 0.02, 0.01):
        th = np.quantile(sc[e], 1 - frac)
        s = c[:, None] & (sc >= th)
        n = int(s.sum())
        if n < 100:
            continue
        ret = np.where(win[s], o[s] * 100, 0.0)
        lo, hi = roi_bootstrap(np.full(n, 100.0), ret, n_boot=400)
        L.append(f"| 上位{frac*100:g}% | {n:,} | {win[s].mean()*100:.1f}% | {Q[s].mean()*100:.1f}% | "
                 f"{win[s].mean()/max(Q[s].mean(),1e-9):.3f} | **{ret.sum()/(100*n)*100:.1f}%** | "
                 f"{lo*100:.0f}〜{hi*100:.0f}% |")
    L.append("\n回収率100%には「実勝率÷市場確率 > 1.333」が必要。\n")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
