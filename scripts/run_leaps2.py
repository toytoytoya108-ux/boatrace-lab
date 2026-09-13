"""飛躍の2本目・3本目・4本目。いずれも「市場を情報で超える」以外の経路を探す。

飛躍2: **市場の誤りは『どの3艇か』と『どの順か』のどちらに宿るか。**
  3連単120通りは「3艇の組（20通り）」×「その並び（6通り）」に分解できる。
  対数損失は完全に分解される: LL(3連単) = LL(組) + LL(並び|組)。
  群衆は「この艇が勝つ」では考えても「この正確な順序」では考えにくいはず。
  もし誤りが**並びの側**に偏っているなら、組を市場に任せたまま並びだけを買い直せる。
  → 市場の情報を超えずに、市場の中の**不均一な精度**を突く。

飛躍3: **返還は「情報優位を要しない唯一の補助金」。**
  レース中止や出走取消の返還は**払戻100%**。したがって
      期待回収率 = 0.75 + 0.25 × P(返還)
  が厳密に成り立つ。返還確率を高められれば、市場に勝たなくても回収率が上がる。
  さらに重要: **これまでの検証はすべて中止レースを母数から外していた**ので、
  実運用の回収率は測定値よりわずかに高いはず。その分をここで測る。

飛躍4: **オッズの丸めは帯によって取り分が違う。**
  表示オッズは切り捨てなので、胴元は端数を得る。1.5倍の0.1は6.7%、300倍の1は0.3%。
  **同じ控除率25%の裏で、実効の取り分は帯ごとに違う**はず。その形を測る。

出力: reports/research/leaps2.md
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
from boatlab.model.trifecta import PERMS  # noqa: E402

OUT = Path(ROOT) / "reports" / "research" / "leaps2.md"
CACHE = Path("/tmp/claude-0/mix10_cache.npz")
DB = str(Path(ROOT) / "data" / "lab.db")
_A = np.array([p[0] for p in PERMS]); _B = np.array([p[1] for p in PERMS]); _C = np.array([p[2] for p in PERMS])
RATE = 0.75
SETS = sorted({frozenset(p) for p in PERMS}, key=lambda s: sorted(s))
SET_ID = np.array([SETS.index(frozenset(p)) for p in PERMS])       # 120 → 0..19


def summ(ret, stake):
    if stake.sum() <= 0:
        return None
    lo, hi = roi_bootstrap(stake, ret, n_boot=300)
    roi = ret.sum() / stake.sum()
    return dict(n=len(ret), roi=roi, lo=lo, hi=hi, hit=(ret > 0).mean(), ratio=roi / RATE,
                avg=ret[ret > 0].mean() if (ret > 0).any() else 0.0)


def main():
    z = np.load(CACHE, allow_pickle=True)
    rid, date, O, W, P = z["rid"], pd.DatetimeIndex(z["date"]), z["O"], z["W"], z["P"]
    inv = np.where(np.isfinite(O), 1.0 / np.nan_to_num(O, nan=1e9), 0.0)
    Q = inv / inv.sum(1, keepdims=True)
    N = len(Q)
    half = np.asarray(date <= "2026-05-31")

    # ---------------- 飛躍2: 組 × 並び
    QS = np.zeros((N, 20))
    for k in range(20):
        QS[:, k] = Q[:, SET_ID == k].sum(1)
    win_set = SET_ID[W]
    q_set_w = QS[np.arange(N), win_set]
    q_ord_w = Q[np.arange(N), W] / np.clip(q_set_w, 1e-12, None)
    ll_set = -np.log(np.clip(q_set_w, 1e-12, None))
    ll_ord = -np.log(np.clip(q_ord_w, 1e-12, None))
    L = [f"# 飛躍2〜4: 市場を情報で超えない経路を探す（2026年・{N:,}R・確定オッズ）\n",
         "## 飛躍2: 市場の誤りは『どの3艇か』と『どの順か』のどちらにあるか\n",
         "3連単は「3艇の組（20通り）」×「並び（6通り）」に分解でき、対数損失も厳密に分解される。",
         "群衆は『この艇が勝つ』では考えても『この正確な順序』では考えにくい。**誤りが並び側に偏っていれば、",
         "組は市場に任せたまま、並びだけを買い直せる。**\n",
         "| | 対数損失（確認 6〜8月） | 無情報なら | 市場が持つ情報 |", "|---|---:|---:|---:|",
         f"| 3連単ぜんぶ | {(ll_set + ll_ord)[~half].mean():.4f} | {np.log(120):.4f} | {np.log(120) - (ll_set + ll_ord)[~half].mean():.4f} |",
         f"| どの3艇か（組） | {ll_set[~half].mean():.4f} | {np.log(20):.4f} | {np.log(20) - ll_set[~half].mean():.4f} |",
         f"| その並び（組を当てた上で） | {ll_ord[~half].mean():.4f} | {np.log(6):.4f} | {np.log(6) - ll_ord[~half].mean():.4f} |",
         "",
         f"市場の情報の **{(np.log(20) - ll_set[~half].mean()) / (np.log(120) - (ll_set + ll_ord)[~half].mean())*100:.0f}%** は「どの3艇か」に、",
         f"**{(np.log(6) - ll_ord[~half].mean()) / (np.log(120) - (ll_set + ll_ord)[~half].mean())*100:.0f}%** は「その並び」にある。\n",
         "### 並びの値付けは校正できているか（勝った組の中で）\n",
         "| 市場が付けた並びの確率 | 買い目数 | 実際に来た割合 | 実測÷市場 |", "|---|---:|---:|---:|"]
    # 勝った組に含まれる6通りについて、市場の条件付き確率と実際の的中を比べる
    rows_q, rows_y = [], []
    for i in range(N):
        m = np.flatnonzero(SET_ID == win_set[i])
        qq = Q[i, m] / max(QS[i, win_set[i]], 1e-12)
        rows_q.append(qq)
        rows_y.append((m == W[i]).astype(float))
    rq = np.concatenate(rows_q); ry = np.concatenate(rows_y)
    for lo, hi in ((0, .05), (.05, .1), (.1, .15), (.15, .25), (.25, .4), (.4, .6), (.6, 1.01)):
        m = (rq >= lo) & (rq < hi)
        if m.sum() < 500:
            continue
        L.append(f"| {lo:.0%}〜{hi:.0%} | {m.sum():,} | {ry[m].mean()*100:.1f}% | "
                 f"{'**' if ry[m].mean()/rq[m].mean()>1 else ''}{ry[m].mean()/rq[m].mean():.3f}"
                 f"{'**' if ry[m].mean()/rq[m].mean()>1 else ''} |")
    # 並びだけを買い直す: 市場の組の上位k組について、その中で条件付き確率が最小の並びを買う
    L += ["\n### 並びだけを買い直す（組は市場の言うとおり、並びは市場が嫌っている方を買う）\n",
          "| 買い方 | 点数 | 的中率 | 平均払戻 | 回収率 | 比 | 95%区間 | 前半 | 後半 |", "|---|---:|---:|---:|---:|---:|---|---:|---:|"]
    ordr = np.argsort(-QS, axis=1)
    def buy_orders(nsets, which, mask=None):
        idx = np.arange(N) if mask is None else np.flatnonzero(mask)
        ret, stake = [], []
        for i in idx:
            pts = []
            for k in ordr[i, :nsets]:
                m = np.flatnonzero(SET_ID == k)
                c = Q[i, m]
                pts.append(int(m[np.argmin(c)] if which == "min" else m[np.argmax(c)]))
            ret.append(P[i] if W[i] in pts else 0.0); stake.append(100.0 * len(pts))
        return np.array(ret), np.array(stake)
    for nsets in (1, 3, 5, 10):
        for which, nm in (("min", "市場がいちばん嫌う並び"), ("max", "市場がいちばん好む並び（対照）")):
            ret, st = buy_orders(nsets, which)
            r = summ(ret, st)
            hs = [summ(*buy_orders(nsets, which, m)) for m in (half, ~half)]
            L.append(f"| 上位{nsets}組 × {nm} | {nsets} | {r['hit']*100:.2f}% | {r['avg']:,.0f}円 | "
                     f"**{r['roi']*100:.1f}%** | {r['ratio']:.3f} | {r['lo']*100:.0f}〜{r['hi']*100:.0f}% | "
                     f"{hs[0]['roi']*100:.1f}% | {hs[1]['roi']*100:.1f}% |")

    # ---------------- 飛躍3: 返還という補助金
    con = sqlite3.connect(DB)
    rc = pd.read_sql_query("""
        SELECT r.id race_id, r.race_date, r.status, r.stadium_code,
               -- **必ず preview（レース前）相を使う。** 最新相を取ると、中止レースだけ preview、
               -- 完走レースは result（レース後）を見ることになり、比較が壊れる（最初これで「強風ほど中止しない」と出た）
               (SELECT c.wind_speed_m FROM race_conditions c WHERE c.race_id=r.id AND c.phase='preview' LIMIT 1) ws,
               (SELECT c.wave_cm FROM race_conditions c WHERE c.race_id=r.id AND c.phase='preview' LIMIT 1) wave,
               (SELECT c.weather FROM race_conditions c WHERE c.race_id=r.id AND c.phase='preview' LIMIT 1) weather,
               (SELECT res.refunds FROM results res WHERE res.race_id=r.id) refunds,
               (SELECT res.trifecta FROM results res WHERE res.race_id=r.id) tri
        FROM races r WHERE r.race_date >= '2026-01-01' AND r.status != 'scheduled'""", con)
    con.close()
    def nref(x):
        try:
            v = json.loads(x) if isinstance(x, str) else x
            return len(v) if v else 0
        except Exception:
            return 0
    rc["nref"] = rc["refunds"].map(nref)
    rc["cancelled"] = (rc["status"] == "cancelled") | rc["tri"].isna()
    # 買い目1点が返還される確率: 中止なら1、艇が抜けたなら その艇を含む買い目の割合
    frac = {0: 0.0, 1: 60 / 120, 2: 1 - (4 * 3 * 2) / 120, 3: 1 - (3 * 2 * 1) / 120, 4: 1.0, 5: 1.0, 6: 1.0}
    rc["p_refund"] = np.where(rc["cancelled"], 1.0, rc["nref"].map(frac).fillna(0.0))
    pr = rc["p_refund"].mean()
    L += ["\n## 飛躍3: 返還は『情報優位を要しない唯一の補助金』\n",
          "レース中止も出走取消も**払戻100%**。したがって厳密に `期待回収率 = 0.75 + 0.25 × P(返還)`。",
          "市場に勝つ必要が一切ない、唯一の経路。\n",
          f"- 2026年の {len(rc):,}レースのうち **中止（3連単の払戻なし） {int(rc['cancelled'].sum()):,}件（{rc['cancelled'].mean()*100:.2f}%）**。",
          f"- 出走取消などで一部の艇が返還されたレース **{int((rc['nref']>0).sum()):,}件（{(rc['nref']>0).mean()*100:.2f}%）**。",
          f"- 買い目1点あたりの返還確率の平均 **{pr*100:.2f}%** → 補助金は **+{0.25*pr*100:.2f}pt**。",
          "",
          "**注意: これまでの検証はすべて中止レースを母数から外していた。** 実運用ではその分だけ回収率が上がる"
          f"（全レースを買う前提で +{0.25*pr*100:.2f}pt 程度）。\n",
          "### 返還は予測できるか（風・波・天候で絞る）\n",
          "| 条件 | レース数 | 中止率 | 一部返還率 | 1点あたり返還確率 | 補助金 |", "|---|---:|---:|---:|---:|---:|"]
    rc["ws"] = pd.to_numeric(rc["ws"], errors="coerce"); rc["wave"] = pd.to_numeric(rc["wave"], errors="coerce")
    conds = [("全レース", np.ones(len(rc), bool)),
             ("風 6m以上", rc["ws"] >= 6), ("風 8m以上", rc["ws"] >= 8), ("風 10m以上", rc["ws"] >= 10),
             ("波 8cm以上", rc["wave"] >= 8), ("波 12cm以上", rc["wave"] >= 12),
             ("風8m以上 かつ 波8cm以上", (rc["ws"] >= 8) & (rc["wave"] >= 8)),
             ("雨または雪", rc["weather"].isin(["雨", "雪"])),
             ("霧", rc["weather"] == "霧")]
    for nm, m in conds:
        m = np.asarray(m.fillna(False) if hasattr(m, "fillna") else m)
        if m.sum() < 30:
            L.append(f"| {nm} | {int(m.sum())} | — | — | — | 少なすぎ |"); continue
        p = rc.loc[m, "p_refund"].mean()
        L.append(f"| {nm} | {int(m.sum()):,} | {rc.loc[m,'cancelled'].mean()*100:.2f}% | "
                 f"{(rc.loc[m,'nref']>0).mean()*100:.2f}% | {p*100:.2f}% | **+{0.25*p*100:.2f}pt** |")

    # ---------------- 飛躍4: オッズの丸め
    L += ["\n## 飛躍4: オッズの丸めは帯によって取り分が違う\n",
          "表示オッズは切り捨て。1.5倍の刻み0.1は6.7%、300倍の刻み1は0.3%。**同じ控除率25%の裏で、",
          "実効の取り分は帯ごとに違う**はず。刻み幅を推定して丸め損を測る。\n",
          "| 確定オッズ | 買い目数 | 刻み（推定） | 丸めの期待損 | 実効の払戻率 |", "|---|---:|---:|---:|---:|"]
    ov = O[np.isfinite(O)]
    for lo, hi in ((1.0, 10), (10, 100), (100, 1000), (1000, 1e9)):
        m = (ov >= lo) & (ov < hi)
        if m.sum() < 100:
            continue
        v = ov[m]
        step = 0.1 if hi <= 100 else (0.1 if hi <= 1000 else 1.0)
        # 表示は切り捨てなので真のオッズは [v, v+step)。期待損は (step/2)/v
        loss = float(np.mean((step / 2) / v))
        lab = f"{lo:g}〜{hi:g}倍" if hi < 1e8 else f"{lo:g}倍以上"
        L.append(f"| {lab} | {int(m.sum()):,} | {step} | {loss*100:.2f}% | {RATE*(1+loss)*100:.2f}% |")
    L.append("\n（「実効の払戻率」は、丸めが無ければ得られたはずの払戻。低オッズ帯ほど丸めで削られている）\n")

    # ---------------- 飛躍5: 均す量を測る（飛躍1の反省: 均しすぎた）
    L += ["## 飛躍5: どこまで均すのが最適か（`leap_self.md` の反省）\n",
          "飛躍1は周辺確率まで均して失敗した（均しすぎ）。今度は**組と並びの間だけ**で均す量を測る。",
          "`q'(o) ∝ q_set(S)^α × q(o|S)^β`。α=β=1 が市場そのまま。**β<1 が最適なら「並びの値付けは雑」**、",
          "**α>1 が最適なら「組の値付けの方が確か」**ということになる。探索期間で (α,β) を選び、確認期間で評価する。\n",
          "| α（組） | β（並び） | 探索 対数損失 | 確認 対数損失 |", "|---|---|---:|---:|"]
    QSET = QS[:, SET_ID]                                   # 各買い目が属する組の確率
    QORD = Q / np.clip(QSET, 1e-12, None)
    lQS = np.log(np.clip(QSET, 1e-12, None)); lQO = np.log(np.clip(QORD, 1e-12, None))
    best = (None, 1e9); tbl = []
    for a in (0.9, 1.0, 1.1, 1.2):
        for b in (0.7, 0.85, 1.0, 1.15):
            lp = a * lQS + b * lQO
            lp = lp - lp.max(1, keepdims=True)
            M = np.exp(lp); M = M / M.sum(1, keepdims=True)
            ll = -np.log(np.clip(M[np.arange(N), W], 1e-12, None))
            tbl.append((a, b, ll[half].mean(), ll[~half].mean()))
            if ll[half].mean() < best[1]:
                best = ((a, b), ll[half].mean())
    for a, b, e, c in tbl:
        mark = "**" if (a, b) == best[0] else ""
        L.append(f"| {mark}{a}{mark} | {mark}{b}{mark} | {e:.4f} | {c:.4f} |")
    (ba, bb), _ = best
    base_c = [c for a, b, e, c in tbl if (a, b) == (1.0, 1.0)][0]
    bst_c = [c for a, b, e, c in tbl if (a, b) == (ba, bb)][0]
    L += ["", f"探索期間で最良は **α={ba}, β={bb}**。確認期間の対数損失は {bst_c:.4f}（市場そのまま {base_c:.4f}、"
          f"差 {bst_c - base_c:+.4f} nats）。\n",
          "### 「組で選んでから並びを選ぶ」と「そのまま最有力を選ぶ」の一騎打ち\n",
          "2つが違う目を指すレースだけを取り出して比べる（同じレース・同じ1点・同じ物差し）。\n",
          "| 買い方 | レース数 | 的中率 | 平均払戻 | 回収率 | 比 | 95%区間 | 前半 | 後半 |", "|---|---:|---:|---:|---:|---:|---|---:|---:|"]
    top_combo = np.argmax(Q, axis=1)
    top_set = np.argmax(QS, axis=1)
    sel_so = np.array([int(np.flatnonzero(SET_ID == top_set[i])[np.argmax(Q[i, SET_ID == top_set[i]])]) for i in range(N)])
    diff = sel_so != top_combo
    for nm, sel in (("最有力の目をそのまま（従来）", top_combo), ("最有力の『組』→その中の最有力の並び", sel_so)):
        for sub, sn in ((np.ones(N, bool), "全レース"), (diff, "2つが食い違うレースだけ")):
            idx = np.flatnonzero(sub)
            ret = np.where(sel[idx] == W[idx], P[idx], 0.0); st = np.full(len(idx), 100.0)
            r = summ(ret, st)
            hs = []
            for hm in (half, ~half):
                j = np.flatnonzero(sub & hm)
                hs.append(summ(np.where(sel[j] == W[j], P[j], 0.0), np.full(len(j), 100.0)))
            L.append(f"| {nm}（{sn}） | {r['n']:,} | {r['hit']*100:.2f}% | {r['avg']:,.0f}円 | **{r['roi']*100:.1f}%** | "
                     f"{r['ratio']:.3f} | {r['lo']*100:.0f}〜{r['hi']*100:.0f}% | {hs[0]['roi']*100:.1f}% | {hs[1]['roi']*100:.1f}% |")
    L.append(f"\n食い違うのは {diff.mean()*100:.1f}% のレース（{int(diff.sum()):,}R）。\n")

    # ---------------- 飛躍6: 並べ替えを k 点に広げ、モデルとも比べる
    L += ["## 飛躍6: 『組を重く見る』並べ替えを k 点に広げる\n",
          f"探索期間で選ばれた α={ba} を使い、`q_set^α × q(並び|組)` で120通りを並べ替えて上位k点を買う。",
          "比較対象は「市場そのままの人気順」と「モデル確率順（本体 Model 1.0）」。\n",
          "| 並べ替え | 点数 | 的中率 | 平均払戻 | 回収率 | 比 | 95%区間 | 前半 | 後半 |", "|---|---:|---:|---:|---:|---:|---|---:|---:|"]
    score_so = np.exp(ba * lQS + lQO)
    ord_so = np.argsort(-score_so, axis=1)
    ord_q = np.argsort(-Q, axis=1)
    MT = z["MT"]
    def buy_rank(od, k, mask=None):
        idx = np.arange(N) if mask is None else np.flatnonzero(mask)
        sel = od[idx, :k]
        hit = (sel == W[idx][:, None]).any(1)
        return np.where(hit, P[idx], 0.0), np.full(len(idx), 100.0 * k)
    RANKS = [("市場そのまま人気順", ord_q), (f"市場・組を重く見る（α={ba}）", ord_so), ("モデル確率順（Model 1.0）", MT)]
    for k in (1, 3, 5, 10, 20):
        for nm, od in RANKS:
            if od.shape[1] < k:
                continue
            r = summ(*buy_rank(od, k))
            hs = [summ(*buy_rank(od, k, m)) for m in (half, ~half)]
            L.append(f"| {nm} | {k} | {r['hit']*100:.2f}% | {r['avg']:,.0f}円 | **{r['roi']*100:.1f}%** | "
                     f"{r['ratio']:.3f} | {r['lo']*100:.0f}〜{r['hi']*100:.0f}% | {hs[0]['roi']*100:.1f}% | {hs[1]['roi']*100:.1f}% |")
    # 帰無シミュレーション（市場が正しい世界で同じ並べ替えをやる）
    rng = np.random.default_rng(23)
    cum = np.cumsum(Q, axis=1)
    L += ["\n### 帰無シミュレーション（市場が正しい世界で同じ並べ替えを20回）\n",
          "| 並べ替え | 点数 | 実測 | 帰無 平均 | 帰無 範囲 | 実測以上 |", "|---|---:|---:|---:|---|---:|"]
    sims = np.zeros((20, N), int)
    for si in range(20):
        u = rng.random(N)
        sims[si] = (cum < u[:, None]).sum(1).clip(0, 119)
    for k in (1, 5, 10):
        for nm, od in RANKS[:2]:
            sel = od[:, :k]
            real = summ(*buy_rank(od, k))["roi"]
            vals = []
            for si in range(20):
                w2 = sims[si]
                hit = (sel == w2[:, None]).any(1)
                pay = np.where(hit, RATE * 100 / np.clip(Q[np.arange(N), w2], 1e-12, None), 0.0)
                vals.append(pay.sum() / (100.0 * k * N))
            v = np.array(vals)
            L.append(f"| {nm} | {k} | {real*100:.1f}% | {v.mean()*100:.1f}% | {v.min()*100:.1f}〜{v.max()*100:.1f}% | {(v >= real).sum()}/20 |")

    # ---------------- 飛躍6b: 「硬い」二段選択を k 点に広げる（α による滑らかな重みづけとは別物）
    L += ["\n### 硬い二段選択（組を先に決め切ってから並びを選ぶ）を k 点に広げる\n",
          "α による滑らかな重みづけでは k=1 の差が再現しなかった。**二段選択は α→∞ に相当する別物**なので、",
          "硬いまま点数を増やして確かめる。\n",
          "| 買い方 | 点数 | 的中率 | 平均払戻 | 回収率 | 比 | 95%区間 | 前半 | 後半 |", "|---|---:|---:|---:|---:|---:|---|---:|---:|"]
    set_order = np.argsort(-QS, axis=1)
    def two_stage(nsets, nord, mask=None):
        idx = np.arange(N) if mask is None else np.flatnonzero(mask)
        ret, st = [], []
        for i in idx:
            pts = []
            for k in set_order[i, :nsets]:
                m = np.flatnonzero(SET_ID == k)
                pts += [int(x) for x in m[np.argsort(-Q[i, m])[:nord]]]
            ret.append(P[i] if W[i] in pts else 0.0); st.append(100.0 * len(pts))
        return np.array(ret), np.array(st)
    for nsets, nord, nm in ((1, 1, "上位1組×1並び"), (2, 1, "上位2組×1並び"), (3, 1, "上位3組×1並び"),
                            (5, 1, "上位5組×1並び"), (1, 3, "上位1組×3並び"), (2, 2, "上位2組×2並び"),
                            (3, 2, "上位3組×2並び"), (5, 2, "上位5組×2並び")):
        r = summ(*two_stage(nsets, nord))
        hs = [summ(*two_stage(nsets, nord, m)) for m in (half, ~half)]
        L.append(f"| {nm} | {nsets*nord} | {r['hit']*100:.2f}% | {r['avg']:,.0f}円 | **{r['roi']*100:.1f}%** | "
                 f"{r['ratio']:.3f} | {r['lo']*100:.0f}〜{r['hi']*100:.0f}% | {hs[0]['roi']*100:.1f}% | {hs[1]['roi']*100:.1f}% |")
    L += ["", "同じ点数の「市場そのまま人気順」は k=1で76.6%、3で76.6%、5で77.0%、10で76.6%（上の表）。\n"]

    L += ["## まとめ: 4本の飛躍から拾えたもの",
          "",
          "### 飛躍2（組 × 並び）— **拾えた。ただし1点だけ**",
          "市場の情報は **67% が「どの3艇か」、33% が「その並び」**。並びの側にも同じ向きの人気薄バイアスがある",
          "（条件付き確率5%未満で比0.895 → 60%超で1.052）。**誤りは片側に偏っていない。**",
          "",
          "ただし選び方としては違いが出た。**「最有力の組」を決め切ってから「その中の最有力の並び」を買う**と、",
          "「最有力の目をそのまま買う」より良い:",
          "",
          "| | 全レース | 2つが食い違う11.8%のレースだけ |",
          "|---|---:|---:|",
          "| 最有力の目をそのまま | 76.8%（比1.024） | **66.2%**（比0.883） |",
          "| 最有力の組 → その中の最有力の並び | **78.8%**（比1.051） | **83.0%**（比1.107） |",
          "",
          "食い違うレースでの差 **+16.8pt**（前半 +18.7 / 後半 +13.4、両方で同じ向き）。",
          "理屈: 組の確率は並び6通りを合計するのでノイズが均される。**組の値付けの方が確か**で、",
          "組が2番手の目に人気が集まっているときは、その人気は信号でなく雑音のことが多い。",
          "",
          "**ただし1点でしか出ない。** 点数を増やすと消える（上位2組×1並び 76.5%、3組 76.3%、5組 74.5%）。",
          "α による滑らかな重みづけ（`q_set^α × q(並び|組)`）でも再現しない（α=1.1 が最良だが k=1 で 76.5%）。",
          "**買い方として採用はしない。** 拾うのは理屈の方（下の総括）。",
          "",
          "### 飛躍3（返還＝補助金）— **拾えた。小さいが確実**",
          "`期待回収率 = 0.75 + 0.25 × P(返還)` は厳密。市場に勝つ必要が一切ない唯一の経路。",
          "- 全体で1点あたり返還確率 2.90% → **+0.72pt**。**これまでの検証は中止レースを母数から外していたので、",
          "  実運用の回収率はすべて 0.7pt ほど高いはず。**",
          "- 予測もできる: **風8m以上で中止率3.57%（全体1.43%の2.5倍）→ 補助金 +1.35pt**、",
          "  風8m以上かつ波8cm以上で +1.61pt、風10m以上では中止18.9%（n=37）で +4.73pt。",
          "- **測り方の罠を1つ踏んだ**: 気象を「最新の観測」で取ると、完走レースはレース後の値、中止レースはレース前の値を",
          "  見ることになり「強風ほど中止しない」と出た。**preview 相に統一**して逆転した。",
          "",
          "### 飛躍4（丸め）— **小さい。ただし向きは本命に不利**",
          "表示オッズの切り捨てで、1〜10倍帯は払戻の 0.70%、10〜100倍帯は 0.15%、100倍以上は 0.02% を失う。",
          "**低オッズ帯ほど丸めで削られる**ので、本命端の優位は見た目より 0.5pt ほど小さい。",
          "",
          "### 総括: 2本の飛躍が同じ場所を指している",
          "飛躍1（`leap_self.md`）は「Plackett–Luce は『誰が1着かで2着の顔ぶれが変わる』を書けず 0.0385 nats 失う」",
          "と示した。飛躍2は「市場の情報の67%は**どの3艇か**にあり、そこの値付けの方が確か」と示した。",
          "",
          "**2つは同じ設計を指している: まず『どの3艇が3着以内に入るか』を当て、次に『その並び』を当てる二段構え。**",
          "いまの Model 1.0 は 1着確率・2着以内確率・3着以内確率から一気に120通りを作る（PL）ので、",
          "この二段構えになっていない。**Model 1.3 の形として、これが今いちばん筋の通った候補。**",
          "",
          "正直な但し書き: 二段構えにしても回収率が動く保証はない（`model12_wf.md` で精度改善が回収率に出なかった）。",
          "今回拾えた確実な増分は、返還の +0.7〜1.6pt だけ。**それでも、モデルの形を変える理由が2本の独立な測定から出たのは初めて。**",
          ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
