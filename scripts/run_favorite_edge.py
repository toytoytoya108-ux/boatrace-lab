"""人気側の端（＝favorite-longshot bias）はどこまで100%に近づくか。

これまでに分かったこと:
  - モデルは市場より不正確（3連単 対数損失 3.786 vs 3.708）。だから「誰が堅いか」の判定は
    モデルより市場に任せた方がよい。
  - 人気薄は大幅に買われすぎ（3連単の全通り均等買いは60%、払戻率75%に対して大きく負ける）。
    その裏返しで人気側は買われなさすぎ＝ここだけ歪みが集中している。
  - 複勝1点は94.8%、自信上位1%に絞ると98.7%（bettype_study）。あと1.3%。

ここでは選定を**市場（3連単オッズ）**に任せ、市場がいちばん確信しているレースだけ、
その艇の複勝・単勝を買う。払戻は公式の確定配当。探索1〜5月で絞り方を決め、確認6〜8月で1回評価。

元返し（払戻100円）の割合も出す。確信が高いほど払戻は100円に張り付くので、
この構造が100%の天井になっていないかを確かめる。

出力: reports/backtest/favorite_edge.md
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np

from boatlab.backtest.metrics import roi_bootstrap
from boatlab.config import ROOT
from boatlab.model.trifecta import PERM_LABELS, PERMS

OUT = Path(ROOT) / "reports" / "backtest" / "favorite_edge.md"
CACHE = Path("/tmp/claude-0")
DB = str(Path(ROOT) / "data" / "lab.db")
EXPLORE_END = "2026-05-31"
FIRST = np.array([p[0] for p in PERMS])
SECOND = np.array([p[1] for p in PERMS])


def load():
    z = np.load(CACHE / "test_lgb_probs.npz")
    ids, P, tri = z["ids"], z["P"].astype(np.float64), z["tri"].astype(int)
    n = len(ids)
    pos = {int(r): i for i, r in enumerate(ids)}
    lab2 = {l: i for i, l in enumerate(PERM_LABELS)}
    O3 = np.full((n, 120), np.nan)
    payW = np.zeros((n, 6))
    payP = np.zeros((n, 6))
    date = np.empty(n, dtype="U10")
    con = sqlite3.connect(DB)
    for rid, js in con.execute("SELECT race_id, odds FROM odds_snapshots WHERE bet_type='3t' AND source='turnmark_final'"):
        i = pos.get(int(rid))
        if i is None:
            continue
        for k, v in (json.loads(js) if isinstance(js, str) else js).items():
            j = lab2.get(k)
            if j is not None and v:
                O3[i, j] = float(v)
    for rid, js, rd, irr in con.execute(
            "SELECT res.race_id, res.payouts, r.race_date, res.is_irregular FROM results res "
            "JOIN races r ON r.id=res.race_id WHERE res.payouts IS NOT NULL"):
        i = pos.get(int(rid))
        if i is None or irr:
            continue
        date[i] = str(rd)[:10]
        d = json.loads(js) if isinstance(js, str) else js
        for key, arr in (("win", payW), ("place", payP)):
            for e in d.get(key) or []:
                try:
                    arr[i, int(str(e["combination"]).strip()) - 1] = float(e["amount"] or 0)
                except Exception:
                    pass
    con.close()
    ok = (np.isfinite(O3).sum(1) >= 110) & (date != "") & (payW.sum(1) > 0) & (payP.sum(1) > 0)
    O3, P, tri, date, payW, payP = O3[ok], P[ok], tri[ok], date[ok], payW[ok], payP[ok]
    inv = np.where(np.isfinite(O3), 1.0 / O3, 0.0)
    Q3 = inv / inv.sum(1, keepdims=True)
    mk = dict(win=np.stack([Q3[:, FIRST == a].sum(1) for a in range(6)], 1),
              place=np.stack([Q3[:, (FIRST == a) | (SECOND == a)].sum(1) for a in range(6)], 1))
    Pn = P / P.sum(1, keepdims=True)
    md = dict(win=np.stack([Pn[:, FIRST == a].sum(1) for a in range(6)], 1),
              place=np.stack([Pn[:, (FIRST == a) | (SECOND == a)].sum(1) for a in range(6)], 1))
    return dict(mk=mk, md=md, pay=dict(win=payW, place=payP), date=date, n=len(tri))


def evaluate(conf, pay, sel, frac_mask):
    """conf: 各レースの「いちばん堅い艇」の確率、sel: その艇番。frac_mask で対象レースを絞る。"""
    i = np.arange(len(sel))[frac_mask]
    ret = pay[i, sel[frac_mask]]
    stake = np.full(len(i), 100.0)
    if not len(i):
        return None
    lo, hi = roi_bootstrap(stake, ret, n_boot=500)
    return dict(n=len(i), hit=float((ret > 0).mean()), roi=float(ret.sum() / stake.sum()),
                lo=lo, hi=hi, motogaeshi=float((ret == 100).mean()),
                pay_hit=float(ret[ret > 0].mean()) if (ret > 0).any() else 0.0)


def main():
    d = load()
    date = d["date"]
    e, c = date <= EXPLORE_END, date > EXPLORE_END
    L = [f"# 人気側の端はどこまで100%に近づくか（{d['n']:,}R、2026年・確定配当）\n",
         "「誰が堅いか」の判定を**市場（3連単オッズ）**に任せ、市場がいちばん確信しているレースだけ買う。",
         f"探索 〜{EXPLORE_END} で絞り方を決め、確認期間で1回評価。\n"]
    fracs = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005]
    for bt in ("place", "win"):
        name = "複勝" if bt == "place" else "単勝"
        for who, src in (("市場", d["mk"]), ("モデル", d["md"])):
            conf = src[bt].max(1)
            sel = src[bt].argmax(1)
            L.append(f"## {name} ・ {who}がいちばん確信している艇を1点\n")
            L.append("| 対象 | 確認n | 的中 | 平均払戻 | 元返しの割合 | 回収率 | 95%区間 |")
            L.append("|---|---:|---:|---:|---:|---:|---|")
            for f in fracs:
                th = np.quantile(conf[e], 1 - f) if f < 1 else -1
                r = evaluate(conf, d["pay"][bt], sel, c & (conf >= th))
                if not r or r["n"] < 100:
                    continue
                lab = "全レース" if f == 1 else f"上位{f*100:g}%"
                L.append(f"| {lab} | {r['n']:,} | {r['hit']*100:.1f}% | {r['pay_hit']:.0f}円 | "
                         f"{r['motogaeshi']*100:.1f}% | **{r['roi']*100:.1f}%** | "
                         f"{r['lo']*100:.0f}〜{r['hi']*100:.0f}% |")
            L.append("")
    L.append("## 読み方\n")
    L.append("- 「元返しの割合」は的中しても払戻が100円＝増えなかったレースの割合。確信が高いほど増える。")
    L.append("- これが100%の天井になっているなら、人気側をどこまで詰めても100%は超えない。")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
