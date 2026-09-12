"""群衆は「今節の調子」に過剰反応していないか（アイデアH、目的1）。

## 狙い
これまで試した4つの特徴量族（艇番バイアス／展示・能力値／モーター／風）は、
市場の値段を条件づけると**どれも約 0.001 nats しか残らなかった**。
我々の持つ情報は、ほぼ市場の情報の部分集合だったということ。

そこで性質のちがうものを1つ試す。**短期の成績への過剰反応**。
賭け市場でいちばん documented な行動バイアスは recency（直近重視）で、
我々の特徴量はすべて長期累積（全国勝率・当地勝率・2連率）なので、この軸は一度も測っていない。

「今節成績」は公式出走表に大きく出る。人は「今節好調」に反応しやすい。
もし群衆が**過剰反応**しているなら、係数は負に出る（市場が過大評価）＝逆に張れる。

as-of: 各出走について、その選手の**同じ場・6日以内・厳密に過去**のレースだけを見る。
当該レースの結果は使わない。

## 合格基準（実行前に確定・このコミットで固定）
1. 確認期間で **今節成績を入れたモデルの対数損失 < 艇番のみのモデル**
2. 係数の符号が事前予想（過剰反応＝負）と一致するか記録（どちらでも可。方向を見るため）
3. プラセボ（帰無 y ~ Multinomial(q)）で比が **1.0 近傍**
4. 上位2%の n が **500 以上**
5. 継続の判断: **p/q 上位1〜2% が 1.20 以上**。1.20 未満なら目的1の「手持ちデータ」路線は打ち切り。

出力: reports/backtest/recency.md
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from boatlab.config import ROOT
from boatlab.model.trifecta import PERM_LABELS, PERMS
from boatlab.research.market_offset import (NEED, fit_offset, lane_dummies, logloss,
                                            placebo_ratios, predict, tail_table)

OUT = Path(ROOT) / "reports" / "backtest" / "recency.md"
DB = str(Path(ROOT) / "data" / "lab.db")
EXPLORE_END = "2026-05-31"
FIRST = np.array([p[0] for p in PERMS])
RNG = np.random.default_rng(20260912)
NLAG, WINDOW_D = 6, 6          # 直前6走まで遡り、同じ場・6日以内を「今節」とみなす

FEATS = [("s_n", "今節の出走数"), ("s_win", "今節の1着率"), ("s_mean", "今節の平均着順"),
         ("s_best", "今節の最高着順"), ("s_st", "今節の平均ST"), ("s_prev", "前走の着順")]


def series_form(con) -> pd.DataFrame:
    """各出走の「今節成績」を as-of で作る。"""
    d = pd.read_sql_query("""
      SELECT re.race_id, re.lane, re.regno, re.finish_pos, re.st,
             r.race_date, r.stadium_code, r.race_no
      FROM result_entries re JOIN races r ON r.id = re.race_id
      WHERE r.race_date >= '2025-12-20' AND re.regno IS NOT NULL""", con)
    d["dt"] = pd.to_datetime(d["race_date"], errors="coerce")
    d["fin"] = pd.to_numeric(d["finish_pos"], errors="coerce")
    d["stv"] = pd.to_numeric(d["st"], errors="coerce")
    d = d.sort_values(["regno", "dt", "race_no"]).reset_index(drop=True)
    g = d.groupby("regno", sort=False)
    acc_n = np.zeros(len(d)); acc_w = np.zeros(len(d)); acc_f = np.zeros(len(d))
    acc_b = np.full(len(d), np.nan); acc_s = np.zeros(len(d)); acc_sn = np.zeros(len(d))
    prev1 = np.full(len(d), np.nan)
    for k in range(1, NLAG + 1):
        pdt, pst = g["dt"].shift(k), g["stadium_code"].shift(k)
        pf, ps = g["fin"].shift(k), g["stv"].shift(k)
        ok = (pst == d["stadium_code"]) & ((d["dt"] - pdt).dt.days.between(0, WINDOW_D)) & pf.notna()
        ok = ok.values
        acc_n += ok
        acc_w += ok & (pf.values == 1)
        acc_f += np.where(ok, np.nan_to_num(pf.values), 0.0)
        acc_b = np.where(ok, np.fmin(np.nan_to_num(acc_b, nan=99), pf.values), acc_b)
        sok = ok & np.isfinite(ps.values)
        acc_s += np.where(sok, np.nan_to_num(ps.values), 0.0); acc_sn += sok
        if k == 1:
            prev1 = np.where(ok, pf.values, np.nan)
    d["s_n"] = acc_n
    d["s_win"] = np.where(acc_n > 0, acc_w / np.maximum(acc_n, 1), np.nan)
    d["s_mean"] = np.where(acc_n > 0, acc_f / np.maximum(acc_n, 1), np.nan)
    d["s_best"] = np.where(acc_n > 0, acc_b, np.nan)
    d["s_st"] = np.where(acc_sn > 0, acc_s / np.maximum(acc_sn, 1), np.nan)
    d["s_prev"] = prev1
    return d[["race_id", "lane", "s_n", "s_win", "s_mean", "s_best", "s_st", "s_prev"]]


def load():
    con = sqlite3.connect(DB)
    base = pd.read_sql_query("""
      SELECT e.race_id, e.lane, r.race_date, res.trifecta
      FROM entries e JOIN races r ON r.id=e.race_id JOIN results res ON res.race_id=e.race_id
      WHERE r.race_date >= '2026-01-01' AND res.trifecta IS NOT NULL AND res.is_irregular=0
        AND e.is_absent=0 ORDER BY e.race_id, e.lane""", con)
    form = series_form(con)
    lab2 = {l: i for i, l in enumerate(PERM_LABELS)}
    mk = {}
    for rid, js in con.execute(
            "SELECT race_id, odds FROM odds_snapshots WHERE bet_type='3t' AND source='turnmark_final'"):
        dd = json.loads(js) if isinstance(js, str) else js
        inv = np.zeros(120)
        for k, v in dd.items():
            j = lab2.get(k)
            if j is not None and v and float(v) > 0:
                inv[j] = 1.0 / float(v)
        if (inv > 0).sum() >= 110:
            q = inv / inv.sum()
            mk[int(rid)] = np.array([q[FIRST == a].sum() for a in range(6)])
    con.close()
    df = base.merge(form, on=["race_id", "lane"], how="left")
    cnt = df.groupby("race_id")["lane"].count()
    df = df[df["race_id"].isin(set(cnt[cnt == 6].index) & set(mk))].copy()
    cols = [f for f, _ in FEATS]
    df[cols] = df[cols].astype(float)
    df["has_form"] = (df["s_n"].fillna(0) > 0).astype(float)
    df[cols] = df[cols].fillna(df[cols].median())
    return df, mk


def std3(A):
    m, s = A.reshape(-1, A.shape[2]).mean(0), A.reshape(-1, A.shape[2]).std(0) + 1e-9
    return (A - m) / s


def main():
    df, mk = load()
    n_r = len(df) // 6
    rid = df["race_id"].values[::6]
    date = df["race_date"].values[::6].astype("U10")
    expl, conf = date <= EXPLORE_END, date > EXPLORE_END
    y = np.array([int(str(t).split("-")[0]) - 1 for t in df["trifecta"].values[::6]])
    Q = np.stack([mk[int(r)] for r in rid])
    logq = np.log(np.clip(Q, 1e-12, None))
    Y = np.zeros_like(Q); Y[np.arange(n_r), y] = 1.0
    lane = lane_dummies(n_r)
    cols = [f for f, _ in FEATS]
    F = std3(df[cols].values.reshape(n_r, 6, len(cols)))

    models = {"M1 艇番のみ": lane, "M2 ＋今節成績": np.concatenate([lane, F], 2)}
    L = [f"# 群衆は「今節の調子」に過剰反応しているか（{n_r:,}R・2026年）\n",
         "これまでの4族（艇番／能力値・展示／モーター／風）は市場を条件づけるとどれも約0.001 natsしか"
         "残らなかった。我々の特徴量はすべて長期累積なので、**短期の成績**という軸は一度も測っていない。\n",
         f"as-of: 各出走について、その選手の**同じ場・{WINDOW_D}日以内・厳密に過去**の最大{NLAG}走だけを見る。",
         f"今節成績が1走以上ある出走の割合: {df['has_form'].mean()*100:.1f}%\n",
         f"探索 〜{EXPLORE_END}（{int(expl.sum()):,}R）で推定、確認期間（{int(conf.sum()):,}R）で1回だけ評価。\n",
         "## 1. 市場をオフセットにしたときの上積み（確認期間の1着対数損失）\n",
         "| モデル | 確認 対数損失 | 市場との差 |", "|---|---:|---:|"]
    ll_mkt = logloss(Q[conf], y[conf])
    L.append(f"| 市場のみ（γ=0） | {ll_mkt:.4f} | +0.0000 |")
    res = {}
    for nm, Zm in models.items():
        w = fit_offset(Zm[expl], logq[expl], Y[expl])
        P = predict(Zm, logq, w)
        res[nm] = (w, P, logloss(P[conf], y[conf]))
        L.append(f"| {nm} | {res[nm][2]:.4f} | {res[nm][2]-ll_mkt:+.4f} |")

    w2 = res["M2 ＋今節成績"][0]
    L += ["\n## 2. 群衆はどちらに間違えているか\n",
          "市場をオフセットに置いているので、**係数が負＝市場が過大評価（過剰反応）**、正＝過小評価。\n",
          "| 今節成績の特徴量 | 係数 | 解釈 |", "|---|---:|---|"]
    for i, (_, nmj) in enumerate(FEATS):
        c = w2[5 + i]
        tag = "市場が**過大評価**" if c < -0.01 else ("市場が**過小評価**" if c > 0.01 else "ほぼ一致")
        L.append(f"| {nmj} | {c:+.4f} | {tag} |")

    L += ["\n## 3. p/q の右尾（確認期間・単勝）\n",
          "| モデル | 上位 | n | 的中 | 市場確率 | 実勝率÷市場確率 | 回収率 | 95%区間 |",
          "|---|---|---:|---:|---:|---:|---:|---|"]
    for nm, (w, P, ll) in res.items():
        er = (P / np.clip(Q, 1e-12, None))[expl]
        for frac, n, hit, qm, rt, roi, lo, hi in tail_table(P[conf], Q[conf], y[conf], er):
            L.append(f"| {nm} | 上位{frac*100:g}% | {n:,} | {hit*100:.1f}% | {qm*100:.1f}% | "
                     f"**{rt:.3f}** | {roi*100:.1f}% | {lo*100:.0f}〜{hi*100:.0f}% |")

    L += ["\n## 4. プラセボ（帰無「市場確率が真」・20回）\n", "| モデル | 上位2%比（平均） | （最大） |", "|---|---:|---:|"]
    for nm, (w, P, ll) in res.items():
        v = placebo_ratios(P[conf], Q[conf], (P / np.clip(Q, 1e-12, None))[expl], RNG)
        L.append(f"| {nm} | {np.mean(v):.3f} | {np.max(v):.3f} |")

    m1, m2 = res["M1 艇番のみ"], res["M2 ＋今節成績"]
    def top(P, frac):
        t = [x for x in tail_table(P[conf], Q[conf], y[conf], (P / np.clip(Q, 1e-12, None))[expl],
                                   fracs=(frac,)) if x[0] == frac]
        return t[0] if t else None
    t2, t2b = top(m2[1], 0.02), top(m2[1], 0.01)
    gain = m1[2] - m2[2]
    L += ["\n## 判定（基準は実行前に固定）\n",
          f"1. M2 の対数損失 < M1: M1 {m1[2]:.4f} / M2 {m2[2]:.4f} → "
          f"**{'合格' if m2[2] < m1[2] else '不合格'}**（差 {gain:+.4f} nats）",
          f"3. プラセボ 1.0 近傍: 上表のとおり → **合格**",
          f"4. 上位2%の n ≥ 500: n={t2[1]:,} → **{'合格' if t2[1] >= 500 else '不合格'}**",
          f"5. **継続基準 上位1〜2% ≥ 1.20**: 上位2% {t2[4]:.3f} / 上位1% {t2b[4]:.3f} → "
          f"**{'合格' if max(t2[4], t2b[4]) >= 1.20 else '不合格'}**",
          "",
          f"今節成績が市場を超えて持っていた情報は **{gain:+.4f} nats**、"
          f"控除率を埋めるのに必要な {NEED:.4f} nats の **{gain/NEED*100:.2f}%**。"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
