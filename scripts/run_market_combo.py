"""「市場と組み合わせる」筋道の天井テスト（Benter 1994 の二段階ロジット）。

狙い: 回収率100%を超える道が残っているかを、いちばん有利な条件で先に確かめる。
  - 券種は控除率の低い 単勝（払戻率80%）。ここが最も100%に近い。
  - 市場確率は **確定オッズ**（turnmark_final）から作る。実際には締切前オッズしか使えないので
    これは「反則つきの上限」。ここで超えないなら、この筋道は現実にはもっと超えない。
  - 二段階ロジット: p_i ∝ f_i^a * x_i^b（f=モデル確率, x=市場確率）を条件付きロジットで a,b を推定。
    これまで試した線形ブレンド（β縮約・w混合）とは別物で、Benter が実際に使った形。

探索 2026-01〜05 で a,b と購入閾値を決め、確認 2026-06〜08 で1回だけ評価する。
出力: reports/backtest/market_combo.md
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from boatlab.backtest.metrics import roi_bootstrap
from boatlab.config import ROOT
from boatlab.model.trifecta import PERMS

DB = str(Path(ROOT) / "data" / "lab.db")
OUT = Path(ROOT) / "reports" / "backtest" / "market_combo.md"
CACHE = Path("/tmp/claude-0")
EXPLORE_END = "2026-05-31"


def load():
    z = np.load(CACHE / "test_lgb_probs.npz")
    ids, P, tri = z["ids"], z["P"].astype(np.float64), z["tri"]
    A = np.array([p[0] for p in PERMS])
    F = np.stack([P[:, A == a].sum(1) for a in range(6)], 1)      # モデルの単勝確率
    pos = {int(r): i for i, r in enumerate(ids)}
    con = sqlite3.connect(DB)
    X = np.full((len(ids), 6), np.nan)                            # 市場（確定オッズ）
    for rid, js in con.execute("SELECT race_id, odds FROM odds_snapshots WHERE bet_type='win' AND source='turnmark_final'"):
        i = pos.get(int(rid))
        if i is None:
            continue
        d = json.loads(js)
        for k in range(6):
            v = d.get(str(k + 1))
            if v:
                X[i, k] = float(v)
    win_pay = np.zeros((len(ids), 6))
    date = np.empty(len(ids), dtype="U10")
    for rid, js, rd in con.execute(
            "SELECT res.race_id, res.payouts, r.race_date FROM results res JOIN races r ON r.id=res.race_id "
            "WHERE res.payouts IS NOT NULL AND res.is_irregular=0"):
        i = pos.get(int(rid))
        if i is None:
            continue
        date[i] = str(rd)[:10]
        for e in json.loads(js).get("win") or []:
            try:
                win_pay[i, int(e["combination"]) - 1] = float(e["amount"] or 0)
            except Exception:
                pass
    con.close()
    winner = np.array([PERMS[int(t)][0] for t in tri])
    ok = np.isfinite(X).all(1) & (X > 0).all(1) & (win_pay.sum(1) > 0) & (date != "")
    return F[ok], X[ok], winner[ok], win_pay[ok], date[ok]


def implied(odds):
    inv = 1.0 / odds
    return inv / inv.sum(1, keepdims=True)


def fit(F, Q, y, init=(1.0, 1.0)):
    """条件付きロジット: p_i ∝ exp(a·log f_i + b·log x_i)。a,b を最尤で推定。"""
    lf, lx = np.log(np.clip(F, 1e-9, None)), np.log(np.clip(Q, 1e-9, None))
    idx = np.arange(len(y))

    def nll(w):
        s = w[0] * lf + w[1] * lx
        s -= s.max(1, keepdims=True)
        e = np.exp(s)
        return float(-np.mean(s[idx, y] - np.log(e.sum(1))))

    r = minimize(nll, np.array(init), method="Nelder-Mead")
    return r.x, r.fun


def probs(F, Q, w):
    s = w[0] * np.log(np.clip(F, 1e-9, None)) + w[1] * np.log(np.clip(Q, 1e-9, None))
    s -= s.max(1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(1, keepdims=True)


def bet(P, odds, pay, y, th, cap=1e9):
    """各レースで期待値が最大の1点を、期待値がth以上のときだけ100円買う。cap=買う上限オッズ。"""
    ev = np.where(odds <= cap, P * odds, -1)
    j = ev.argmax(1)
    i = np.arange(len(y))
    m = ev[i, j] >= th
    if m.sum() == 0:
        return None
    ret = pay[i, j][m]
    stake = np.full(int(m.sum()), 100.0)
    lo, hi = roi_bootstrap(stake, ret, n_boot=500)
    top5 = np.sort(ret)[::-1][:5].sum()
    return dict(n=int(m.sum()), share=round(float(m.mean()), 3), hit=round(float((ret > 0).mean()), 4),
                roi=round(float(ret.sum() / stake.sum()), 4), lo=round(lo, 3), hi=round(hi, 3),
                roi_x5=round(float((ret.sum() - top5) / stake.sum()), 4), pnl=int(ret.sum() - stake.sum()))


def main():
    F, X, y, pay, date = load()
    Q = implied(X)
    e = date <= EXPLORE_END
    c = ~e
    L = [f"# 市場と組み合わせる（Benter流の二段階ロジット）— 単勝・確定オッズ使用の天井テスト\n",
         f"対象 {len(y):,}R（探索 {int(e.sum()):,} / 確認 {int(c.sum()):,}）。市場確率は**確定オッズ**から作っており、",
         "実際の購入時には手に入らない。ここで100%を超えなければ、現実の運用ではさらに届かない。\n"]

    w, nll_c = fit(F[e], Q[e], y[e])
    _, nll_f = fit(F[e], Q[e], y[e], init=(1.0, 0.0))
    nll_model = fit(F[e], Q[e] * 0 + 1 / 6, y[e], init=(1.0, 0.0))
    # 単独の対数損失（確認期間）
    def ll(P_):
        return float(-np.mean(np.log(np.clip(P_[np.arange(len(y[c])), y[c]], 1e-12, None))))
    L.append("## 1. 予測精度（確認期間の対数損失。小さいほど良い）\n")
    L.append(f"- モデル単独: {ll(F[c] / F[c].sum(1, keepdims=True)):.4f}")
    L.append(f"- 市場単独（確定オッズ）: {ll(Q[c]):.4f}")
    L.append(f"- 二段階ロジット a={w[0]:.3f}（モデル） b={w[1]:.3f}（市場）: {ll(probs(F[c], Q[c], w)):.4f}\n")

    L.append("## 2. 単勝の回収率（期待値が最大の1点を、閾値以上のときだけ100円）\n")
    L.append("`上位5本除外` は、最も高配当だった5本を除いたときの回収率。ここが大きく下がるものは")
    L.append("「たまたま数本の万舟が当たっただけ」で、再現しない。\n")
    L.append("| 確率の作り方 | 上限オッズ | 閾値 | 買う割合 | 確認n | 的中 | 回収 | 95%区間 | 上位5本除外 |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---|---:|")
    for cap in (1e9, 50, 20):
        for name, wv in (("モデルのみ", (1.0, 0.0)), (f"二段階ロジット a={w[0]:.2f} b={w[1]:.2f}", tuple(w))):
            Pc = probs(F, Q, wv)
            best = None
            for th in (0.0, 0.9, 1.0, 1.05, 1.1, 1.2, 1.3):
                re_ = bet(Pc[e], X[e], pay[e], y[e], th, cap)
                if re_ and re_["n"] >= 200 and (best is None or re_["roi"] > best[1]["roi"]):
                    best = (th, re_)
            if best is None:
                continue
            rc = bet(Pc[c], X[c], pay[c], y[c], best[0], cap)
            cs = "なし" if cap > 1e8 else f"{int(cap)}倍"
            L.append(f"| {name} | {cs} | {best[0]} | {rc['share']*100:.0f}% | {rc['n']} | {rc['hit']*100:.1f}% | "
                     f"{rc['roi']*100:.1f}% | {rc['lo']*100:.0f}〜{rc['hi']*100:.0f}% | {rc['roi_x5']*100:.1f}% |")
    L.append("\n閾値は探索期間で最も回収率が高いものを選び、確認期間には1回だけ当てている。\n")
    L.append("## 2-1. オッズ帯ごとの内訳（確認期間・全艇を均等に買った場合）\n")
    L.append("| オッズ帯 | n | 実勝率 | 市場の想定 | モデルの想定 | 回収率 |")
    L.append("|---|---:|---:|---:|---:|---:|")
    Fn = F / F.sum(1, keepdims=True)
    win = (np.arange(6)[None, :] == y[:, None])
    for lo_, hi_ in [(1, 1.5), (1.5, 3), (3, 5), (5, 10), (10, 20), (20, 50), (50, 1e9)]:
        s_ = c[:, None] & (X >= lo_) & (X < hi_)
        if s_.sum() < 200:
            continue
        lab = f"{lo_}〜{hi_}倍" if hi_ < 1e8 else f"{lo_}倍〜"
        L.append(f"| {lab} | {int(s_.sum())} | {win[s_].mean()*100:.2f}% | {(0.75/X[s_]).mean()*100:.2f}% | "
                 f"{Fn[s_].mean()*100:.2f}% | {pay[s_ & win].sum()/(100*s_.sum())*100:.1f}% |")
    L.append("")
    # 3. モデルは市場に何を足しているか: 市場のみとの差
    Pc = probs(F, Q, w)
    L.append("## 3. モデルは市場に何を足しているか\n")
    L.append(f"二段階ロジットの係数は モデル {w[0]:.3f} / 市場 {w[1]:.3f}。")
    L.append("市場の係数が1に近くモデルの係数が小さいほど、「市場でほぼ説明でき、モデルの上積みは小さい」ことを意味する。\n")
    OUT.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
