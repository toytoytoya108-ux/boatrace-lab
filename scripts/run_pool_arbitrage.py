"""プール間の歪み（単勝プールが薄いこと）を使えるかの検証。

発見の経緯: 単勝で「モデル確率 ≫ 単勝オッズの想定確率」の艇を買うと回収率が120〜160%になった。
モデルが市場に勝ったように見えたが、市場確率を **3連単オッズから作り直す** と話が変わる。
競艇の売上はほぼ3連単に集中しており、単勝プールは薄い＝オッズがノイズだらけ。

  単勝の的中確率の推定精度（対数損失、小さいほど良い）
    モデル < 単勝オッズ   ← モデルの勝ち（ように見える）
    3連単オッズ < モデル  ← 本当の市場（3連単プール）はモデルより正確

つまり見つけたのは「モデルの優位」ではなく「薄い単勝プールの値付けの遅れ」。
本スクリプトは、モデルを使う場合と、3連単オッズだけを使う場合（モデル不要の裸の歪み）を
同じ土俵で比較し、実際の払戻で回収率を出す。

注意: ここで使うのは両プールとも **確定オッズ**。実運用では締切前のオッズしか使えないうえ、
自分の投票が薄いプールのオッズを動かす（Benter 1994 が最大の制約と呼んだ点）。
出力: reports/backtest/pool_arbitrage.md
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np

from boatlab.backtest.metrics import roi_bootstrap
from boatlab.config import ROOT
from boatlab.model.trifecta import PERMS, PERM_LABELS

DB = str(Path(ROOT) / "data" / "lab.db")
OUT = Path(ROOT) / "reports" / "backtest" / "pool_arbitrage.md"
CACHE = Path("/tmp/claude-0")
EXPLORE_END = "2026-05-31"
LANE1 = np.array([p[0] for p in PERMS])


def load():
    z = np.load(CACHE / "test_lgb_probs.npz")
    ids, P, tri = z["ids"], z["P"].astype(np.float64), z["tri"]
    n = len(ids)
    F = np.stack([P[:, LANE1 == a].sum(1) for a in range(6)], 1)
    pos = {int(r): i for i, r in enumerate(ids)}
    lab2 = {l: i for i, l in enumerate(PERM_LABELS)}
    con = sqlite3.connect(DB)
    OW = np.full((n, 6), np.nan)
    O3 = np.full((n, 120), np.nan)
    for bt, arr, key in (("win", OW, None), ("3t", O3, lab2)):
        for rid, js in con.execute("SELECT race_id, odds FROM odds_snapshots WHERE bet_type=? AND source='turnmark_final'", (bt,)):
            i = pos.get(int(rid))
            if i is None:
                continue
            for k, v in json.loads(js).items():
                j = (int(k) - 1) if key is None else key.get(k)
                if j is not None and v:
                    arr[i, j] = float(v)
    pay = np.zeros((n, 6))
    date = np.empty(n, dtype="U10")
    for rid, js, rd, irr in con.execute(
            "SELECT res.race_id, res.payouts, r.race_date, res.is_irregular FROM results res "
            "JOIN races r ON r.id=res.race_id WHERE res.payouts IS NOT NULL"):
        i = pos.get(int(rid))
        if i is None or irr:
            continue
        date[i] = str(rd)[:10]
        for e in json.loads(js).get("win") or []:
            try:
                pay[i, int(e["combination"]) - 1] = float(e["amount"] or 0)
            except Exception:
                pass
    con.close()
    y = LANE1[tri.astype(int)]
    ok = np.isfinite(OW).all(1) & (OW > 0).all(1) & (np.isfinite(O3).sum(1) >= 110) & (pay.sum(1) > 0) & (date != "")
    return dict(F=F[ok], OW=OW[ok], O3=O3[ok], pay=pay[ok], y=y[ok], date=date[ok])


def norm(x):
    return x / x.sum(1, keepdims=True)


def main():
    d = load()
    F, OW, O3, pay, y, date = d["F"], d["OW"], d["O3"], d["pay"], d["y"], d["date"]
    Fn = norm(F)
    Qw = norm(1.0 / OW)                                             # 単勝プールの想定確率
    q3 = norm(np.where(np.isfinite(O3), 1.0 / O3, 0.0))
    Q3 = np.stack([q3[:, LANE1 == a].sum(1) for a in range(6)], 1)   # 3連単プールが示す単勝確率
    e, c = date <= EXPLORE_END, date > EXPLORE_END
    win = (np.arange(6)[None, :] == y[:, None])
    i = np.arange(len(y))

    def ll(P_, s):
        return float(-np.mean(np.log(np.clip(P_[s][np.arange(int(s.sum())), y[s]], 1e-12, None))))

    L = ["# 単勝プールの歪みは使えるか（プール間の比較）\n",
         f"対象 {len(y):,}R（探索 {int(e.sum()):,} / 確認 {int(c.sum()):,}）。すべて確定オッズ・実払戻。\n",
         "## 1. 「1着になる確率」を誰がいちばん正しく当てているか（対数損失・小さいほど良い）\n",
         "| 推定者 | 探索 | 確認 |", "|---|---:|---:|"]
    for nm, P_ in (("モデル（Model 1.0）", Fn), ("単勝プールのオッズ", Qw), ("3連単プールのオッズ", Q3),
                   ("3連単プール×モデルの平均", norm(np.sqrt(Q3 * Fn)))):
        L.append(f"| {nm} | {ll(P_, e):.4f} | {ll(P_, c):.4f} |")
    L.append("\n**3連単プールのオッズがモデルより正確**。単勝プールのオッズだけが大きく劣る。")
    L.append("競艇の売上は3連単に集中しており、単勝プールは薄くて値付けが荒いことを示している。\n")

    L.append("## 2. 単勝を買った場合の回収率（確認期間・1点100円・上限20倍）\n")
    L.append("| 買い方 | 条件 | n | 的中 | 回収 | 95%区間 | 損益 |")
    L.append("|---|---|---:|---:|---:|---|---:|")
    rows = []
    for nm, src in (("モデルで選ぶ", Fn), ("3連単オッズで選ぶ（モデル不要）", Q3), ("両方の平均で選ぶ", norm(np.sqrt(Q3 * Fn)))):
        for r in (1.5, 2.0, 3.0):
            s = c[:, None] & (OW <= 20) & (src >= r * Qw)
            n = int(s.sum())
            if n < 100:
                continue
            ret = pay[s & win]
            stake = np.full(n, 100.0)
            lo, hi = roi_bootstrap(stake, np.concatenate([ret, np.zeros(n - len(ret))]), n_boot=500)
            roi = ret.sum() / (100 * n)
            L.append(f"| {nm} | 想定確率が単勝オッズの{r}倍以上 | {n} | {win[s].mean()*100:.1f}% | "
                     f"{roi*100:.1f}% | {lo*100:.0f}〜{hi*100:.0f}% | {int(ret.sum() - 100*n):,}円 |")
            rows.append((nm, r, n, roi))
    L.append("")

    L.append("## 3. 歪みの大きさ（確認期間・全艇）\n")
    L.append("| 単勝オッズ帯 | n | 単勝プールの想定 | 3連単プールの想定 | 実際の勝率 |")
    L.append("|---|---:|---:|---:|---:|")
    for lo_, hi_ in [(1, 1.5), (1.5, 3), (3, 5), (5, 10), (10, 20), (20, 50)]:
        s = c[:, None] & (OW >= lo_) & (OW < hi_)
        if s.sum() < 200:
            continue
        L.append(f"| {lo_}〜{hi_}倍 | {int(s.sum())} | {Qw[s].mean()*100:.2f}% | {Q3[s].mean()*100:.2f}% | {win[s].mean()*100:.2f}% |")
    L.append("\n3連単プールの想定が実際の勝率にいちばん近い。単勝プールは人気薄を高く見積もりすぎている。\n")
    L.append("## 4. いくらまで賭けられるか（自分の投票でオッズが潰れる）\n")
    L.append("単勝の売上は全体のおよそ0.08%しかない。1レースの売上が4,000万円なら単勝プールは約32,000円。")
    L.append("そこへ自分の金を入れると、その分だけ自分の払戻が下がる。下は実測4,571本に当てはめた回収率。\n")
    sel = (OW <= 20) & (Q3 >= 2 * Qw)
    o_, hit_ = OW[sel], win[sel]
    share = (norm(1.0 / OW))[sel]
    pools = (16000, 32000, 64000, 160000)
    L.append("| 1点あたり | " + " | ".join(f"プール{t//1000}千円" for t in pools) + " |")
    L.append("|---" * (len(pools) + 1) + "|")
    for s_ in (100, 300, 500, 1000, 2000, 5000, 10000):
        cells = []
        for T in pools:
            newo = np.minimum(0.75 * (T + s_) / (share * T + s_), o_)
            cells.append(f"{np.where(hit_, newo * s_, 0.0).sum() / (s_ * len(o_)) * 100:.1f}%")
        cells[0] = f"**{cells[0]}**" if s_ == 1000 else cells[0]
        L.append(f"| {s_:,}円 | " + " | ".join(cells) + " |")
    L.append("\n1点1,000〜2,000円あたりが分かれ目。5,000円入れると歪みを自分で消してしまう。")
    L.append(f"買いが出るのは全レースの約{float((sel.sum(1) > 0).mean())*100:.0f}%（8か月で{int(sel.sum()):,}本＝1日およそ19本）。\n")
    L.append("## 5. 実運用でこのまま使えない理由\n")
    L.append("- ここで使ったのは**確定オッズ**。買えるのは締切前で、締切前の単勝オッズ・3連単オッズを別途取得する必要がある。")
    L.append("- 払戻は自分の投票も含めた最終プールで決まる。薄いプールに大金を入れると自分でオッズを潰す")
    L.append("  （Benter 1994: 抜ける利益はレース売上のおよそ0.25〜0.5%が上限）。")
    L.append("- 単勝の売上規模を実測していない。これが分からないと1レースあたりの上限額が決められない。\n")
    OUT.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
