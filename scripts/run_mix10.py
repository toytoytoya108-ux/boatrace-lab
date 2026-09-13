"""本命6点＋穴4点＝10点固定、「当たってもマイナス」が起きない配分での検証。

ユーザー指定（2026-09-13）:
  - 買い目は10点固定。内訳は本命6・穴4。
  - 金額は「当たってもマイナス」というパターンが無くなるようにする。
  - 穴の選び方は要相談（本スクリプトで候補を総当たり）。
  - まずは過去の確定オッズで検証する。

■ 配分の原理（`no_hit_loss.md` と同じ）
  どの点が当たっても 払戻 ≥ 投資 にするには、予算Bに対し 賭け金_i = ceil(B ÷ オッズ_i ÷100)×100 とし、
  Σ賭け金 ≤ B が成り立てばよい。成立の条件は **Σ(1/オッズ) ≤ 1**（＝選んだ10点の市場確率の合計 ≤ 0.75）。
  成立しないレースは10点固定を崩さないため**見送り**にする。

■ 先に確認しておく理屈（結果を読み違えないため）
  市場が正しい（p=q、オッズ=0.75/q）なら、どの点でも p×オッズ＝0.75。
  したがって**どの10点をどう配分しても期待回収率は75%**になる。
  配分は「当たっても赤字」の有無と分散だけを変え、回収率は変えない。
  回収率を動かせるのは**選んだ点の 実勝率÷市場確率（比）だけ**。
  さらに、穴側の賭け金は100円に張り付くので、穴1点あたりの期待払戻は 75円×比 で、オッズの大小に依存しない。
  → **穴は「配当が大きい点」ではなく「比が高い点」を選ぶべき**、というのが事前の見立て。

出力: reports/research/mix10.md
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from boatlab.backtest.metrics import roi_bootstrap
from boatlab.config import ROOT
from boatlab.model.trifecta import PERM_LABELS, PERMS

OUT = Path(ROOT) / "reports" / "research" / "mix10.md"
DB = str(Path(ROOT) / "data" / "lab.db")
NPZ = Path(ROOT) / "research_data" / "bt2026_top20.npz"
LAB2 = {l: i for i, l in enumerate(PERM_LABELS)}
A = np.array([p[0] for p in PERMS])
UNIT = 100


# ------------------------------------------------------------------ データ
def load():
    con = sqlite3.connect(DB)
    rid, ODD = [], []
    for r, js in con.execute("SELECT race_id, odds FROM odds_snapshots WHERE bet_type='3t' AND source='turnmark_final'"):
        d0 = json.loads(js) if isinstance(js, str) else js
        o = np.full(120, np.nan)
        for k, v in d0.items():
            j = LAB2.get(k)
            if j is not None and v and float(v) > 0:
                o[j] = float(v)
        if np.isfinite(o).sum() >= 110:
            rid.append(int(r)); ODD.append(o)
    pay, win = {}, {}
    for r, js in con.execute("SELECT race_id, payouts FROM results WHERE race_id >= 202601010000 AND payouts IS NOT NULL"):
        d0 = json.loads(js) if isinstance(js, str) else js
        if not isinstance(d0, dict):
            continue
        t = (d0.get("trifecta") or [{}])[0]
        j = LAB2.get(str(t.get("combination") or ""))
        if j is not None:
            pay[int(r)] = float(t.get("amount") or 0); win[int(r)] = j
    con.close()
    keep = [i for i, r in enumerate(rid) if r in pay]
    rid = np.array([rid[i] for i in keep]); O = np.stack([ODD[i] for i in keep])
    W = np.array([win[r] for r in rid]); P = np.array([pay[r] for r in rid])
    date = pd.to_datetime(rid // 10000, format="%Y%m%d")
    z = np.load(NPZ, allow_pickle=True)
    mi = {int(r): k for k, r in enumerate(z["race_id"])}
    idx = np.array([mi.get(int(r), -1) for r in rid])
    MT = np.where(idx[:, None] >= 0, z["top"][np.maximum(idx, 0)], -1).astype(int)
    MP = np.where(idx[:, None] >= 0, z["p"][np.maximum(idx, 0)].astype(float), np.nan)
    return rid, date, O, W, P, MT, MP


# ------------------------------------------------------------------ 選び方
def take(order, honmei, k):
    """order の先頭から、本命と重複しない点を k 個取る。"""
    out = []
    for j in order:
        if j not in honmei:
            out.append(int(j))
            if len(out) == k:
                break
    return out


HONMEI = {
    "市場人気 上位6": lambda o, q, mt, mp, order: list(order[:6]),
    "モデル確率 上位6": lambda o, q, mt, mp, order: (list(mt[:6]) if mt[0] >= 0 else list(order[:6])),
}


def ana_rules():
    R = {}
    for lo, hi, nm in ((7, 10, "人気7〜10"), (10, 13, "人気10〜13"), (15, 18, "人気15〜18"),
                       (20, 23, "人気20〜23"), (25, 28, "人気25〜28"), (30, 33, "人気30〜33"),
                       (40, 43, "人気40〜43"), (60, 63, "人気60〜63")):
        R[nm] = (lambda lo, hi: lambda o, q, mt, mp, order, hm: take(order[lo - 1:], hm, 4))(lo, hi)
    R["人気20・27・34・40（帯に散らす）"] = lambda o, q, mt, mp, order, hm: take(order[[19, 26, 33, 39]], hm, 4)
    R["100倍超で最も堅い4点"] = lambda o, q, mt, mp, order, hm: take(order[np.isin(order, np.flatnonzero(o >= 100))], hm, 4)
    R["モデル確率 7〜10位"] = lambda o, q, mt, mp, order, hm: (take(mt[mt >= 0], hm, 4) if mt[0] >= 0 else [])
    def ev(o, q, mt, mp, order, hm):
        if mt[0] < 0:
            return []
        c = [j for j in mt if j >= 0 and j not in hm]
        c.sort(key=lambda j: -(mp[list(mt).index(j)] / max(q[j], 1e-9)))
        return c[:4]
    R["モデル÷市場（期待値）上位4"] = ev
    def notfav(o, q, mt, mp, order, hm):
        lane = A[hm[0]]
        return take(order[A[order] != lane], hm, 4)
    R["本命の1着艇を変えた目の人気上位4"] = notfav
    return R


# ------------------------------------------------------------------ 配分
def guaranteed_stakes(odds, budget):
    """全点で 払戻 ≥ budget ≥ Σ賭け金 になる賭け金。成立しなければ None。"""
    st = np.ceil(budget / odds / UNIT) * UNIT
    return st if st.sum() <= budget else None


def evaluate(sel_pts, O, W, P, budget):
    """sel_pts: 各レースの買い目インデックス（空なら見送り）。"""
    n = len(O)
    ret = np.zeros(n); stake = np.zeros(n); fired = np.zeros(n, bool)
    hit = np.zeros(n, bool); hitloss = np.zeros(n, bool); npts = np.zeros(n, int)
    for i, pts in enumerate(sel_pts):
        if not pts or len(pts) < 10:
            continue
        od = O[i, pts]
        if not np.isfinite(od).all():
            continue
        st = guaranteed_stakes(od, budget)
        if st is None:
            continue
        fired[i] = True; stake[i] = st.sum(); npts[i] = len(pts)
        if W[i] in pts:
            k = pts.index(W[i])
            ret[i] = P[i] * st[k] / 100.0
            hit[i] = True
            hitloss[i] = ret[i] < stake[i]
    return dict(fired=fired, ret=ret, stake=stake, hit=hit, hitloss=hitloss)


def summarize(res, mask=None, seed=0):
    f = res["fired"] if mask is None else (res["fired"] & mask)
    if f.sum() == 0:
        return None
    ret, stake, hit, hl = res["ret"][f], res["stake"][f], res["hit"][f], res["hitloss"][f]
    lo, hi = roi_bootstrap(stake, ret, n_boot=300)
    cur = best = 0
    for r in ret:
        cur = cur + 1 if r <= 0 else 0
        best = max(best, cur)
    return dict(n=int(f.sum()), roi=ret.sum() / stake.sum(), lo=lo, hi=hi, hit=hit.mean(),
                hitloss=(hl.sum() / max(hit.sum(), 1)), stake=stake.mean(), pnl=ret.sum() - stake.sum(),
                streak=best, avg=ret[ret > 0].mean() if (ret > 0).any() else 0.0)


def guaranteed_max(odds, spend):
    """支出の上限 spend のもとで「保証水準B」を最大にする賭け金。返り値 (B, stakes) or (None, None)。
    保証が成り立つ条件は **Σ賭け金 ≤ B**（Bが払戻の下限なので、投資がBを超えたら意味がない）。
    Σ賭け金 ≈ B×Σ(1/オッズ) なので、Σ(1/オッズ) ≤ 1 のとき B=spend が上限。
    Σ(1/オッズ) > 1 のレースはどんなBでも成立しないので見送りになる。
    ※ 保証を守る限り、支出は必ず B を下回る（＝予算を使い切れない）。ここがこの配分の代償。"""
    B = int(spend // UNIT) * UNIT
    while B >= UNIT:
        st = np.ceil(B / odds / UNIT) * UNIT
        if st.sum() <= min(B, spend):
            return B, st
        B -= UNIT
    return None, None


# ------------------------------------------------------------------ 本体
def build(O, q, MT, MP, hon_fn, ana_fn):
    order_all = np.argsort(-q, 1)
    sel = []
    for i in range(len(O)):
        o, qq, order = O[i], q[i], order_all[i]
        hm = hon_fn(o, qq, MT[i], MP[i], order)
        if len(hm) < 6:
            sel.append([]); continue
        hm = [int(x) for x in hm[:6]]
        an = ana_fn(o, qq, MT[i], MP[i], order, hm)
        sel.append(hm + [int(x) for x in an[:4]] if len(an) >= 4 else [])
    return sel


def evaluate2(sel_pts, O, W, P, spend, mode="max"):
    """mode='max': 支出 spend 以内で保証水準を最大化 / mode=数値: その額を保証水準にする。"""
    n = len(O)
    out = dict(fired=np.zeros(n, bool), ret=np.zeros(n), stake=np.zeros(n), hit=np.zeros(n, bool),
               hitloss=np.zeros(n, bool), floor=np.zeros(n), ana_hit=np.zeros(n, bool))
    for i, pts in enumerate(sel_pts):
        if len(pts) != 10:
            continue
        od = O[i, pts]
        if not np.isfinite(od).all():
            continue
        if mode == "max":
            B, st = guaranteed_max(od, spend)
        else:
            B = mode; st = guaranteed_stakes(od, B)
        if st is None:
            continue
        out["fired"][i] = True; out["stake"][i] = st.sum(); out["floor"][i] = B
        if W[i] in pts:
            k = pts.index(W[i])
            out["ret"][i] = P[i] * st[k] / 100.0
            out["hit"][i] = True
            out["ana_hit"][i] = k >= 6
            out["hitloss"][i] = out["ret"][i] < out["stake"][i]
    return out


def summ2(res, mask=None, seed=0):
    f = res["fired"] if mask is None else (res["fired"] & mask)
    if f.sum() == 0:
        return None
    ret, stake, hit, hl, ah = res["ret"][f], res["stake"][f], res["hit"][f], res["hitloss"][f], res["ana_hit"][f]
    lo, hi = roi_bootstrap(stake, ret, n_boot=300)
    cur = best = 0
    for r in ret:
        cur = cur + 1 if r <= 0 else 0
        best = max(best, cur)
    return dict(n=int(f.sum()), roi=ret.sum() / stake.sum(), lo=lo, hi=hi, hit=hit.mean(),
                hitloss=hl.sum() / max(hit.sum(), 1), stake=stake.mean(), floor=res["floor"][f].mean(),
                pnl=ret.sum() - stake.sum(), streak=best, ana=ah.sum() / max(hit.sum(), 1),
                avg=ret[ret > 0].mean() if (ret > 0).any() else 0.0,
                avg_ana=ret[ah].mean() if ah.any() else 0.0)


def main():
    cache = Path("/tmp/claude-0/mix10_cache.npz")
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        rid, date, O, W, P, MT, MP = z["rid"], pd.DatetimeIndex(z["date"]), z["O"], z["W"], z["P"], z["MT"], z["MP"]
    else:
        rid, date, O, W, P, MT, MP = load()
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, rid=rid, date=np.asarray(date).astype(str), O=O, W=W, P=P, MT=MT, MP=MP)
        date = pd.DatetimeIndex(date)
    inv = np.where(np.isfinite(O), 1.0 / np.nan_to_num(O, nan=1e9), 0.0)
    q = inv / inv.sum(1, keepdims=True)
    half = np.asarray(date <= "2026-05-31")
    N = len(rid); days = pd.Series(date).dt.date.nunique()
    q_man = np.where(q <= 0.0075, q, 0.0).sum(1)
    q1 = q.max(1)
    SPEND = 3000
    RULES = ana_rules()
    HON = HONMEI["市場人気 上位6"]

    def row(r):
        return (f"{r['n']:,} | {r['stake']:,.0f}円 | {r['floor']:,.0f}円 | {r['hit']*100:.1f}% | {r['hitloss']*100:.1f}% | "
                f"{r['avg']:,.0f}円 | {r['streak']}R | **{r['roi']*100:.1f}%** | {r['lo']*100:.0f}〜{r['hi']*100:.0f}%")

    L = [f"# 本命6点＋穴4点＝10点固定・「当たっても赤字」ゼロの配分（2026年・{N:,}R・確定オッズ）\n",
         "指定: 10点固定（本命6・穴4）、当たってマイナスになるパターンを無くす、穴の選び方は要検討、まず確定オッズで検証。\n",
         "## 0. 先に置いておく理屈（結果を読み違えないため）\n",
         "配分は **保証つき**: 保証水準Bに対し 賭け金_i = ceil(B ÷ オッズ_i ÷100)×100。Σ賭け金 ≤ B なら",
         "**どの点が当たっても 払戻 ≥ B ≥ 投資**。ここでは支出の上限を1レース3,000円とし、その範囲で**Bを最大化**する",
         "（`no_hit_loss.md` の予算固定版より保証水準が高くなる。同じ支出で当たったときの下限が上がる）。",
         "",
         "**市場が正しいなら、どの10点をどう配分しても期待回収率は75%になる。**",
         "オッズ=0.75/市場確率なので、どの点でも 実勝率×オッズ = 0.75×比。配分は分散と「当たって赤字」の有無だけを変える。",
         "**回収率を動かせるのは、選んだ点の 実勝率÷市場確率（比）だけ。**",
         "さらに穴側の賭け金は100円に張り付くので、**穴1点の期待払戻は 75円×比 で、配当の大小に依存しない。**",
         "→ 事前の見立ては「穴は配当が大きい点ではなく**比が高い点**を選ぶべき」。比は人気薄ほど下がる（`longshot.md`）。\n",
         "## 1. 穴4点の選び方（本命＝市場人気 上位6点で固定・全レース）\n",
         "| 穴の選び方 | 発火 | 1R投資 | 保証水準 | 的中率 | 当たって赤字 | 平均払戻 | 最長連敗 | 回収率 | 95%区間 | 前半 | 後半 |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|"]
    best = (None, -1)
    for nm, fn in RULES.items():
        sel = build(O, q, MT, MP, HON, fn)
        res = evaluate2(sel, O, W, P, SPEND)
        r = summ2(res)
        if r is None:
            continue
        h1, h2 = summ2(res, half), summ2(res, ~half)
        L.append(f"| {nm} | {row(r)} | {h1['roi']*100:.1f}% | {h2['roi']*100:.1f}% |")
        if r["roi"] > best[1] and min(h1["roi"], h2["roi"]) > 0.75:
            best = (nm, r["roi"])

    # 穴の寄与
    L += ["\n### 穴4点は何を買っているのか\n",
          "| 穴の選び方 | 穴が当たった割合（的中のうち） | 穴的中時の平均払戻 | 本命的中時の平均払戻 |", "|---|---:|---:|---:|"]
    for nm in ("人気7〜10", "人気20〜23", "人気30〜33", "人気60〜63", "100倍超で最も堅い4点"):
        sel = build(O, q, MT, MP, HON, RULES[nm])
        res = evaluate2(sel, O, W, P, SPEND)
        f = res["fired"]; hit = res["hit"][f]; ah = res["ana_hit"][f]; ret = res["ret"][f]
        L.append(f"| {nm} | {ah.sum()/hit.sum()*100:.1f}% | {ret[ah].mean():,.0f}円 | {ret[hit & ~ah].mean():,.0f}円 |")

    # 2. 本命の決め方
    L += ["\n## 2. 本命6点の決め方\n",
          "| 本命 | 穴 | 発火 | 1R投資 | 保証水準 | 的中率 | 当たって赤字 | 平均払戻 | 最長連敗 | 回収率 | 95%区間 | 前半 | 後半 |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|"]
    for hn, hf in HONMEI.items():
        for an in ("人気20〜23", "人気7〜10", "モデル確率 7〜10位"):
            sel = build(O, q, MT, MP, hf, RULES[an])
            res = evaluate2(sel, O, W, P, SPEND)
            r = summ2(res)
            if r is None:
                continue
            h1, h2 = summ2(res, half), summ2(res, ~half)
            L.append(f"| {hn} | {an} | {row(r)} | {h1['roi']*100:.1f}% | {h2['roi']*100:.1f}% |")

    # 3. レースの選び方
    L += ["\n## 3. レースの選び方（本命＝モデル確率上位6、穴＝人気20〜23）\n",
          "**既存モードの回収率（80〜83%）は買い方でなくレース選択から来ている。** 同じ10点でも選ぶレースで動く。\n",
          "| レース選択 | 発火 | 1日あたり | 1R投資 | 保証水準 | 的中率 | 当たって赤字 | 平均払戻 | 最長連敗 | 回収率 | 95%区間 | 前半 | 後半 | 月の期待損失 |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|"]
    sel_best = build(O, q, MT, MP, HONMEI["モデル確率 上位6"], RULES["人気20〜23"])
    res_best = evaluate2(sel_best, O, W, P, SPEND)
    SELR = {"全レース": np.ones(N, bool),
            "市場万舟確率 上位10%（荒れ・穴モードの条件）": q_man >= np.quantile(q_man, 0.90),
            "市場万舟確率 上位30%": q_man >= np.quantile(q_man, 0.70),
            "市場万舟確率 下位30%（堅い）": q_man <= np.quantile(q_man, 0.30),
            "市場万舟確率 下位10%（最も堅い）": q_man <= np.quantile(q_man, 0.10),
            "1番人気の市場確率 上位30%": q1 >= np.quantile(q1, 0.70),
            "1番人気の市場確率 上位10%": q1 >= np.quantile(q1, 0.90)}
    for rn, rm in SELR.items():
        r = summ2(res_best, rm)
        if r is None:
            continue
        h1, h2 = summ2(res_best, rm & half), summ2(res_best, rm & ~half)
        per_day = r["n"] / days
        loss = (1 - r["roi"]) * r["stake"] * per_day * 30
        L.append(f"| {rn} | {row(r)} | {h1['roi']*100:.1f}% | {h2['roi']*100:.1f}% | {loss:,.0f}円 |".replace(
            f"{r['n']:,} | ", f"{r['n']:,} | {per_day:.1f}R | ", 1))

    # 3b. 同じレース群での直接比較（穴を入れる代金はいくらか）
    conf = np.nansum(MP[:, :10], axis=1)
    CONF = {"全レース": np.ones(N, bool),
            "モデル信頼度（上位10点の確率合計）上位30%": conf >= np.quantile(conf, 0.70),
            "モデル信頼度 上位10%": conf >= np.quantile(conf, 0.90)}
    L += ["\n### 3b. 穴を入れる代金はいくらか（本命はモデル確率上位6で固定。**3通りとも成立したレースだけ**で比較）\n",
          "点数を10に固定すると、7〜10点目に何を入れるかで成立するレースが変わる（Σ(1/オッズ) ≤ 1 の条件）。",
          "母集団の違いで比較が濁らないよう、**3通りすべてで保証が成立したレース**に揃えて集計する。\n",
          "| レース群 | 7〜10点目 | 発火 | 1R投資 | 的中率 | 下位4点の的中寄与 | 平均払戻 | 最長連敗 | 回収率 | 95%区間 |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    VAR = {}
    for an in ("モデル確率 7〜10位", "人気20〜23", "人気7〜10"):
        sel = build(O, q, MT, MP, HONMEI["モデル確率 上位6"], RULES[an])
        VAR[an] = evaluate2(sel, O, W, P, SPEND)
    common = VAR["モデル確率 7〜10位"]["fired"] & VAR["人気20〜23"]["fired"] & VAR["人気7〜10"]["fired"]
    for cn, cm in CONF.items():
        for an, res in VAR.items():
            r = summ2(res, cm & common)
            if r is None:
                continue
            L.append(f"| {cn} | {an} | {r['n']:,} | {r['stake']:,.0f}円 | {r['hit']*100:.1f}% | {r['ana']*100:.1f}% | "
                     f"{r['avg']:,.0f}円 | {r['streak']}R | **{r['roi']*100:.1f}%** | {r['lo']*100:.0f}〜{r['hi']*100:.0f}% |")

    # 4. 日次の形
    L += ["\n## 4. 1日の収支の形（全レース買った場合・本命モデル上位6＋人気20〜23）\n"]
    f = res_best["fired"]
    dd = pd.DataFrame({"d": pd.Series(date)[f].dt.date, "ret": res_best["ret"][f], "st": res_best["stake"][f]})
    g = dd.groupby("d").sum()
    g["pnl"] = g["ret"] - g["st"]
    L += [f"- 買えた日数 {len(g)}日、**プラスの日 {(g.pnl>0).sum()}日（{(g.pnl>0).mean()*100:.1f}%）**。",
          f"- 1日の収支: 中央値 {g.pnl.median():,.0f}円、最良 {g.pnl.max():,.0f}円、最悪 {g.pnl.min():,.0f}円。",
          f"- 1日の投資は平均 {g.st.mean():,.0f}円（{f.sum()/len(g):.0f}レース）。**全レース買う前提の額なので、実運用では絞る前提。**\n"]

    # 5. 既存モードとの比較
    r = summ2(res_best)
    L += ["## 5. 既存モードとの比較（すべて2026年・確定オッズ）\n",
          "| 買い方 | 点数 | 1R投資 | 的中率 | 当たって赤字 | 最長連敗 | 回収率 |", "|---|---:|---:|---:|---:|---:|---:|",
          "| 穴狙い（人気20〜40の21点・市場万舟確率上位10%） | 21 | 2,100円 | 22.1% | — | 32R | **80.5%** |",
          "| 堅い予想（保証つき・予算3,000円・信頼度0.70以上） | 8.8 | 2,841円 | 66.2% | 1.0% | 6R | **81.4%** |",
          "| 堅い予想（上位厚め・10点3,000円固定・信頼度0.70以上） | 10 | 3,000円 | 70.0% | 46.9% | 6R | **83.6%** |",
          f"| **本命6＋穴4（保証つき・全レース）** | 10 | {r['stake']:,.0f}円 | {r['hit']*100:.1f}% | {r['hitloss']*100:.1f}% | {r['streak']}R | **{r['roi']*100:.1f}%** |"]
    rr = summ2(res_best, q_man <= np.quantile(q_man, 0.30))
    L.append(f"| **本命6＋穴4（保証つき・堅いレース下位30%）** | 10 | {rr['stake']:,.0f}円 | {rr['hit']*100:.1f}% | "
             f"{rr['hitloss']*100:.1f}% | {rr['streak']}R | **{rr['roi']*100:.1f}%** |")
    L.append("\n※ 既存3モードは発火条件つきで母集団が違うので厳密な横並びではない。")

    # 6. データの注意
    ow = O[np.arange(N), W] * 100
    bad = (P < ow * 0.9)
    L += ["\n## 6. データ上の注意\n",
          f"- 公式の確定配当が `確定オッズ×100` を1割以上下回るレースが **{bad.sum():,}件（{bad.mean()*100:.2f}%）** ある",
          "  （同着で配当が分割された、返還でオッズが計算し直された、など）。**保証は「見えているオッズ」に対するもので、",
          f"  公式配当がそれを下回ると保証は破れる。** 上の「当たって赤字」{r['hitloss']*100:.1f}% はすべてこの原因。",
          "- この1.7%は既存の全バックテストにも同じだけ効いている（どのモードの回収率も同じ分だけ低めに出る）。\n"]

    L += ["",
          "## 7. 結論とおすすめ",
          "",
          "### 配分は成立する。ただし予算は使い切れない",
          "保証つき配分で「当たって赤字」は **0.6%** まで落ちる。しかもその0.6%は配分のせいではなく、",
          "**公式配当が確定オッズ×100を下回ったレース**（同着・返還）で、原理的にこちらから防げない。",
          "代償は使い切れないこと: 保証水準3,000円に対し実際の投資は平均2,100〜2,800円。",
          "`Σ賭け金 ≤ 保証水準` が保証の定義そのものなので、**予算を使い切ったら保証は必ず壊れる**。",
          "支出を増やしたいなら保証水準を上げる（＝5,000円予算にする）しかない。",
          "",
          "### 穴4点は『配当の大きさ』で選んではいけない",
          "13通り試して回収率は **75.0〜77.6%** に収まり、順序は理屈どおり**本命寄りほど良い**:",
          "人気7〜10 77.0% ＞ 人気20〜23 76.1% ＞ 人気30〜33 75.9% ＞ 人気60〜63 75.0%。前半・後半でも同じ順序。",
          "穴の賭け金は100円に張り付くので、1点の期待払戻は `75円 × 比` で配当に依存せず、比は人気薄ほど下がるため。",
          "**「穴らしさ」と回収率は正面から対立する**（穴的中時の平均払戻は 人気7〜10で4,330円、人気60〜63で26,219円）。",
          "",
          "### おすすめ: 本命＝モデル確率 上位6、穴＝**人気20〜23**（4点）",
          "- 穴の帯として実測の裏づけがあるのは `two_modes.md` の **人気20〜40** だけ（18の通説構造すべてに勝っている）。",
          "  その帯の中で最も本命寄りの4点を取る、というのが比の理屈とも一致する。",
          "- 前半77.1% / 後半77.5% と安定。穴的中は的中の約10%で、そのときの平均払戻は **6,097円**（本命的中は3,527円）。",
          "- 回収率だけなら 人気7〜10 の方が0.9pt高いが、それは実質「本命10点」で穴ではない（穴的中時4,330円）。",
          "",
          "### 正直な代金: 穴を入れると3ポイント下がる",
          "同じレース・同じ本命6点で7〜10点目だけ差し替えた比較（§3b、3通りとも成立したレースに揃えた）:",
          "",
          "| レース群 | モデル7〜10位（＝本命10点） | 人気20〜23（穴） | 差 |",
          "|---|---:|---:|---:|",
          "| 全レース | 78.4% | 77.2% | −1.2pt |",
          "| モデル信頼度 上位30% | 80.8% | 77.7% | −3.1pt |",
          "| モデル信頼度 上位10% | 84.3% | 81.1% | −3.2pt |",
          "",
          "**実際に買う『自信のあるレース』ほど、穴を混ぜる代金は高くつく。** 自信のあるレースは本命が来るレースで、",
          "そこに穴を4点足すのは、当たる見込みの薄い4点に投資を割くことになるから。",
          "得られるのは「たまに6,000円級が当たる」という分散で、最長連敗も 6R → 8R に伸びる。",
          "",
          "### レース選択の方が効く（買い方より大きい）",
          "同じ10点でも 全レース77.2% → モデル信頼度上位10% 81.1%、逆に市場万舟確率上位10%（荒れ）では **74.0%**。",
          "**この10点構成は堅いレース向きで、荒れるレースでは負ける。**",
          "荒れると市場が見ているレースでは本命6点が来ず、穴4点も薄すぎて当たらないため。",
          "既存3モードの80〜83%も買い方ではなくレース選択から来ている（`condition_rules.md`）。",
          "",
          "### 判断材料のまとめ",
          "- 回収率を最優先するなら、この10点構成は既存の「堅い予想（保証つき）81.4%」に勝てない。",
          "- それでも穴を入れたいなら **人気20〜23**。代金は約3ポイント、見返りは的中の1割が6,000円級になること。",
          "- 発火は堅いレースに限る（モデル信頼度の上位30%で 77.7%、上位10%で 81.1%）。全レース買うと必ず負ける（プラスの日 0/241日）。",
          "- **以上はすべて確定オッズの数字。** 締切前オッズでは本命側のオッズが中央値で1割ほど高く見えており、",
          "  保証つき配分は削る点数が前後する。実運用に載せるなら締切前オッズでの再測定が要る。",
          ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
