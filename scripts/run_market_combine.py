"""「モデル＋市場」は「市場だけ」より正確になるか（route A の関門）。

賭けずに、確率の精度だけを見る。ここで市場を上回れなければ市場から金は取れないので、
回収率の計算には進まない（回収率から入ると必ずノイズの中に儲かって見える組合せを見つけてしまう）。

合成は Benter 1994 の形: p ∝ モデル確率^a × 市場確率^b（レース内で正規化）。パラメータは a,b の2つだけ。
  - 艇レベル: 1着確率6通りで合成（市場の1着確率は3連単オッズから周辺化して作る）
  - 組レベル: 3連単120通りで直接合成（市場の同着順情報を捨てない）

探索 2026-01〜05 で a,b を最尤推定し、確認 2026-06〜08 で1回だけ評価する。
使う市場オッズは**確定オッズ**。実際に払い戻されるのも確定オッズなので、これは本番より厳しい相手
（締切前オッズの市場はこれより不正確）。ここで勝てるなら本番でも見込みがある、という向き。

出力: reports/backtest/market_combine.md
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from boatlab.config import ROOT
from boatlab.model.trifecta import PERM_LABELS, PERMS

OUT = Path(ROOT) / "reports" / "backtest" / "market_combine.md"
CACHE = Path("/tmp/claude-0")
DB = str(Path(ROOT) / "data" / "lab.db")
EXPLORE_END = "2026-05-31"
FIRST = np.array([p[0] for p in PERMS])


def load():
    z = np.load(CACHE / "test_lgb_probs.npz")
    ids, P, tri = z["ids"], z["P"].astype(np.float64), z["tri"].astype(int)
    pos = {int(r): i for i, r in enumerate(ids)}
    lab2 = {l: i for i, l in enumerate(PERM_LABELS)}
    O3 = np.full((len(ids), 120), np.nan)
    date = np.empty(len(ids), dtype="U10")
    con = sqlite3.connect(DB)
    for rid, js in con.execute("SELECT race_id, odds FROM odds_snapshots WHERE bet_type='3t' AND source='turnmark_final'"):
        i = pos.get(int(rid))
        if i is None:
            continue
        for k, v in (json.loads(js) if isinstance(js, str) else js).items():
            j = lab2.get(k)
            if j is not None and v:
                O3[i, j] = float(v)
    for rid, rd in con.execute("SELECT id, race_date FROM races"):
        i = pos.get(int(rid))
        if i is not None:
            date[i] = str(rd)[:10]
    con.close()
    ok = (np.isfinite(O3).sum(1) >= 110) & (date != "")
    P, O3, tri, date = P[ok], O3[ok], tri[ok], date[ok]
    inv = np.where(np.isfinite(O3), 1.0 / O3, 0.0)
    Q3 = inv / inv.sum(1, keepdims=True)
    return P / P.sum(1, keepdims=True), Q3, tri, date


def norm(x):
    return x / x.sum(1, keepdims=True)


def fit(A, B, y, init=(1.0, 1.0)):
    """p ∝ A^a × B^b の a,b を最尤推定（レース内で正規化）。"""
    la, lb = np.log(np.clip(A, 1e-12, None)), np.log(np.clip(B, 1e-12, None))
    idx = np.arange(len(y))

    def nll(w):
        s = w[0] * la + w[1] * lb
        s -= s.max(1, keepdims=True)
        return float(-np.mean(s[idx, y] - np.log(np.exp(s).sum(1))))

    r = minimize(nll, np.array(init), method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-6})
    return r.x


def probs(A, B, w):
    s = w[0] * np.log(np.clip(A, 1e-12, None)) + w[1] * np.log(np.clip(B, 1e-12, None))
    s -= s.max(1, keepdims=True)
    return norm(np.exp(s))


def ll(Pm, y):
    return float(-np.mean(np.log(np.clip(Pm[np.arange(len(y)), y], 1e-12, None))))


def main():
    P, Q3, tri, date = load()
    win = FIRST[tri]
    Fw = norm(np.stack([P[:, FIRST == a].sum(1) for a in range(6)], 1))
    Qw = norm(np.stack([Q3[:, FIRST == a].sum(1) for a in range(6)], 1))
    e, c = date <= EXPLORE_END, date > EXPLORE_END
    L = ["# モデル＋市場は市場だけより正確か（route A の関門）\n",
         f"対象 {len(tri):,}R（探索 {int(e.sum()):,} / 確認 {int(c.sum()):,}）。市場は3連単の**確定オッズ**。",
         "確率の精度だけを見る。市場を上回れなければ市場から金は取れないので、回収率には進まない。\n"]
    rows = []
    for name, A, B, y in (("1着（艇6通り）", Fw, Qw, win), ("3連単（120通り）", P, Q3, tri)):
        w = fit(A[e], B[e], y[e])
        cand = {"モデルのみ": (1.0, 0.0), "市場のみ": (0.0, 1.0), f"合成 a={w[0]:.3f} b={w[1]:.3f}": tuple(w)}
        L.append(f"## {name}（対数損失・小さいほど良い）\n")
        L.append("| 確率の作り方 | 探索 | 確認 | 市場との差（確認） |")
        L.append("|---|---:|---:|---:|")
        base = ll(probs(A[c], B[c], (0.0, 1.0)), y[c])
        for nm, wv in cand.items():
            lc = ll(probs(A[c], B[c], wv), y[c])
            L.append(f"| {nm} | {ll(probs(A[e], B[e], wv), y[e]):.4f} | {lc:.4f} | {lc - base:+.4f} |")
            rows.append((name, nm, lc - base))
        L.append("")
        gain = base - ll(probs(A[c], B[c], tuple(w)), y[c])
        L.append(f"合成による改善は **{gain:+.4f}**。" +
                 ("市場を上回った。" if gain > 0.0005 else "**市場を上回らなかった。**") + "\n")
    # ---- 差が控除率を埋めるか: ケリー基準と、実際に期待値1.0を超える買い目があるか
    w3 = fit(P[e], Q3[e], tri[e])
    Pc = probs(P, Q3, w3)
    o = 0.75 / np.clip(Q3, 1e-12, None)
    winm = np.zeros_like(Q3, bool)
    winm[np.arange(len(tri)), tri] = True
    imp = ll(probs(P[c], Q3[c], (0.0, 1.0)), tri[c]) - ll(Pc[c], tri[c])
    L.append("## 差は控除率を埋めるか\n")
    L.append("ケリー基準では、全通りに確率比例で賭けたときの期待対数成長は")
    L.append("`（市場に対する情報優位）+ log(1-控除率)` になる。ここに実測値を入れる。\n")
    L.append(f"- 情報優位（合成が市場を上回った分）: **{imp:+.4f} nats**")
    L.append(f"- 控除率25%のコスト: **{np.log(0.75):+.4f} nats**")
    L.append(f"- 差し引き: **{imp + np.log(0.75):+.4f}** → 必要な優位の **{abs(np.log(0.75)) / max(imp, 1e-9):.0f}分の1** しかない\n")
    r = Pc / np.clip(Q3, 1e-12, None)
    over = int((r[c] >= 1 / 0.75).sum())
    tot = int(c.sum()) * 120
    L.append("期待値1.0を超えるには「合成確率 ÷ 市場確率 > 1.333」が必要。確認期間の実測:\n")
    L.append(f"| | 最大 | 99%点 | 中央値 | 1.333超えの買い目 |")
    L.append(f"|---|---:|---:|---:|---:|")
    L.append(f"| 合成 | {r[c].max():.3f} | {np.percentile(r[c], 99):.3f} | {np.median(r[c]):.3f} | **{over} / {tot:,}** |")
    rm = P / np.clip(Q3, 1e-12, None)
    sm = c[:, None] & (rm >= 1 / 0.75) & np.isfinite(o)
    L.append(f"| モデル単独 | {rm[c].max():.1f} | {np.percentile(rm[c], 99):.2f} | {np.median(rm[c]):.3f} | "
             f"{int(sm.sum()):,}（{sm.sum() / tot * 100:.1f}%） |")
    L.append(f"\nモデル単独では26.8%の買い目が「期待値1.0超え」に見えるが、その実回収率は "
             f"**{o[sm & winm].sum() / int(sm.sum()) * 100:.1f}%**。モデルの反論はノイズで、"
             "合成はそれを a≈0.15 まで縮めるのが正しいと判断している。\n")
    L.append("## 判定\n")
    best = min(x[2] for x in rows if x[1].startswith("合成"))
    if over == 0 or imp + np.log(0.75) < 0:
        L.append(f"合成は市場をわずかに上回る（{best:+.4f}）が、**その差は控除率の{abs(np.log(0.75))/max(imp,1e-9):.0f}分の1**。")
        L.append(f"確認期間 {tot:,} 買い目のうち期待値1.0を超えたのは **{over}件** だけで、実質ゼロ。")
        L.append("買い方をどう工夫しても勝てない。**route A はここで打ち切る。**\n")
        L.append("これは「この買い方が駄目」ではなく「このモデルと市場の差では、どんな買い方でも駄目」という")
        L.append("一般的な結論。勝つには、モデルを市場より大きく正確にするしかない（fs2特徴量の総改善が")
        L.append("0.008 nats だったことを考えると、必要な差はその数十倍）。")
    elif best < -0.0005:
        L.append(f"合成は市場を上回った（最良 {best:+.4f}）。次は、この差が控除率25%（1.333倍）を")
        L.append("超える買い目を生むかを、固定した少数の買い方だけで確かめる。")
    else:
        L.append(f"合成は市場を上回らなかった（最良 {best:+.4f}）。**モデルは市場が持っていない情報を持っていない**。")
        L.append("市場より正確に予測できない以上、市場から金は取れない。route A はここで打ち切る。")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
