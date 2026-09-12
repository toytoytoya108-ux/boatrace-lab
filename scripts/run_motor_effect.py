"""モーターの真の実力を分離して、市場を超えられるか（アイデアE、目的1）。

## 狙い
公式が出す「モーター2連率」は **誰が乗ったかに汚染されている**。A1級が乗れば上がり、
B2級が乗れば下がる。我々の `ms_90d`（場×モーターの90日累積）も同じ汚染を受けている。
群衆はこの汚染された公表値を見ている。実際、`reports/backtest/crowd_bias.md` は
**市場がモーター2連率を過小評価している（B−A = −0.051）** と実測している。

そこで、選手の腕前を先に説明したあとの**残差**からモーター効果を縮小推定する。

  1. 探索期間で、選手の能力特徴量＋艇番だけの条件付きロジットを当てる
  2. 各(場, モーター)について残差（実際の1着 − 予測確率）を合計し、
     出走数で縮小する: e_m = Σresid / (n_m + k)   ← 経験ベイズ的な縮小
  3. この e_m を、市場確率をオフセットに置いたロジットに入れて γ を見る

as-of の安全性: e_m は**探索期間（1〜5月）のみ**で作り、確認期間（6〜8月）に適用する。
確認期間の結果は一切使わない。モーターは場ごとに年単位で使われるので同一性は保たれる。

対照として**公表モーター2連率**も同時に入れる。「我々の分離推定が、群衆が見ている公表値を
超えて情報を持つか」を見たいので、公表値を条件づけた上での上積みを測る。

## 合格基準（実行前に確定・このコミットで固定）
1. 確認期間で **分離推定 e_m を入れたモデルの対数損失 < 入れないモデル**
2. 確認期間で **e_m 入りの p/q 上位2% が、対照（公表値のみ）の上位2% より高い**
3. プラセボ（帰無 y ~ Multinomial(q)）で比が **1.0 近傍**（手順に偏りがない）
4. 上位2%の n が **500 以上**
5. 継続の判断: **p/q 上位1〜2% が 1.20 以上**。本当に必要なのは 1.333。
   1.20 未満なら、モーター方面（部品交換・節内推移）には進まず打ち切る。

出力: reports/backtest/motor_effect.md
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
                                            per_boat, placebo_ratios, predict, tail_table)

OUT = Path(ROOT) / "reports" / "backtest" / "motor_effect.md"
DB = str(Path(ROOT) / "data" / "lab.db")
EXPLORE_END = "2026-05-31"
FIRST = np.array([p[0] for p in PERMS])
KLASS = {"A1": 3.0, "A2": 2.0, "B1": 1.0, "B2": 0.0}
RNG = np.random.default_rng(20260912)

# 選手の腕前（モーター・ボートは入れない＝残差にモーター効果を残すため）
SKILL = ["nat_win_rate", "loc_win_rate", "nat_rate2", "loc_rate2", "avg_st",
         "klass_n", "f_count", "age", "weight", "exhibition_time", "st_exh", "tilt"]


def load():
    con = sqlite3.connect(DB)
    df = pd.read_sql_query("""
      SELECT e.race_id, e.lane, e.age, e.weight, e.klass, e.f_count, e.avg_st,
             e.nat_win_rate, e.nat_rate2, e.loc_win_rate, e.loc_rate2,
             e.motor_rate2, e.motor_no, r.race_date, r.stadium_code, res.trifecta,
             (SELECT p.exhibition_time FROM preview_snapshots p
                WHERE p.race_id=e.race_id AND p.lane=e.lane ORDER BY p.fetched_at DESC LIMIT 1) exhibition_time,
             (SELECT p.st_exh FROM preview_snapshots p
                WHERE p.race_id=e.race_id AND p.lane=e.lane ORDER BY p.fetched_at DESC LIMIT 1) st_exh,
             (SELECT p.tilt FROM preview_snapshots p
                WHERE p.race_id=e.race_id AND p.lane=e.lane ORDER BY p.fetched_at DESC LIMIT 1) tilt
      FROM entries e JOIN races r ON r.id=e.race_id JOIN results res ON res.race_id=e.race_id
      WHERE r.race_date >= '2026-01-01' AND res.trifecta IS NOT NULL AND res.is_irregular=0
        AND e.is_absent=0
      ORDER BY e.race_id, e.lane""", con)
    lab2 = {l: i for i, l in enumerate(PERM_LABELS)}
    mk = {}
    for rid, js in con.execute(
            "SELECT race_id, odds FROM odds_snapshots WHERE bet_type='3t' AND source='turnmark_final'"):
        d = json.loads(js) if isinstance(js, str) else js
        inv = np.zeros(120)
        for k, v in d.items():
            j = lab2.get(k)
            if j is not None and v and float(v) > 0:
                inv[j] = 1.0 / float(v)
        if (inv > 0).sum() >= 110:
            q = inv / inv.sum()
            mk[int(rid)] = np.array([q[FIRST == a].sum() for a in range(6)])
    con.close()
    df["klass_n"] = df["klass"].map(KLASS)
    cnt = df.groupby("race_id")["lane"].count()
    df = df[df["race_id"].isin(set(cnt[cnt == 6].index) & set(mk))].copy()
    cols = SKILL + ["motor_rate2"]
    df[cols] = df[cols].astype(float)
    df[cols] = df[cols].fillna(df[cols].median())
    df["motor_key"] = (df["stadium_code"].astype(str) + "-"
                       + df["motor_no"].fillna(-1).astype(int).astype(str)).to_numpy(dtype=object)
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
    Y = np.zeros_like(Q)
    Y[np.arange(n_r), y] = 1.0
    lane = lane_dummies(n_r)

    # --- 1〜2. モーター効果を「交差適合」で作る
    # 注意: 探索期間の残差から e_m を作り、その同じ探索期間で縮小定数を選ぶと、
    # 特徴量の中に答えが入る（最初の実装がこれで、k=10 が選ばれ確認期間で市場より悪化した）。
    # 正しくは、探索期間を時系列で K 分割し、fold j の e_m は fold≠j からだけ作る。
    S = std3(df[SKILL].values.reshape(n_r, 6, len(SKILL)))
    mk_arr = np.asarray(df["motor_key"].tolist(), dtype=object).reshape(n_r, 6)
    mr = std3(df["motor_rate2"].values.reshape(n_r, 6, 1))
    K_FOLD = 5
    ei = np.where(expl)[0]
    fold = np.array_split(ei, K_FOLD)          # 日付順に並んでいるので時系列ブロック分割

    def motor_eff(train_idx, k_shrink):
        """train_idx のレースだけから (場,モーター) 効果を縮小推定した辞書を返す。"""
        Zs = np.concatenate([S, lane], 2)
        zero = np.zeros((len(train_idx), 6))
        w = fit_offset(Zs[train_idx], zero, Y[train_idx])
        Pk = predict(Zs[train_idx], zero, w)
        r = (Y[train_idx] - Pk).reshape(-1)
        kk = mk_arr[train_idx].reshape(-1)
        gg = pd.DataFrame({"k": kk, "r": r}).groupby("k")["r"].agg(["sum", "count"])
        return (gg["sum"] / (gg["count"] + k_shrink)).to_dict(), gg

    def lookup(d, idx):
        return np.vectorize(lambda s: d.get(s, 0.0))(mk_arr[idx])

    # 縮小定数は「fold外で作った e_m」で選ぶ（これで答えの漏れが消える）
    best = None
    for k_shrink in (10.0, 20.0, 50.0, 100.0, 200.0, 400.0):
        oof = np.zeros((n_r, 6))
        for j in range(K_FOLD):
            tr = np.concatenate([fold[i] for i in range(K_FOLD) if i != j])
            d, _ = motor_eff(tr, k_shrink)
            oof[fold[j]] = lookup(d, fold[j])
        Eo = std3(oof[:, :, None])
        Z = np.concatenate([lane, mr, Eo], 2)
        w = fit_offset(Z[expl], logq[expl], Y[expl])
        ll = logloss(predict(Z[expl], logq[expl], w), y[expl])
        if best is None or ll < best[0]:
            best = (ll, k_shrink, oof.copy())
    _, k_sel, oof = best
    # 確認期間には探索期間ぜんぶから作った e_m を使う（確認期間の結果は一切使わない）
    d_full, g = motor_eff(ei, k_sel)
    Efull = lookup(d_full, np.arange(n_r))
    raw = oof.copy()
    raw[conf] = Efull[conf]
    E = std3(raw[:, :, None])

    # --- 3. 市場をオフセットにした比較
    models = {
        "M0 市場のみ（γ=0）": None,
        "M1 艇番のみ": lane,
        "M2 ＋公表モーター2連率": np.concatenate([lane, mr], 2),
        "M3 ＋分離推定したモーター効果": np.concatenate([lane, mr, E], 2),
    }
    L = [f"# モーターの真の実力を分離して市場を超えられるか（{n_r:,}R・2026年）\n",
         "公表モーター2連率は「誰が乗ったか」に汚染されている。選手の腕前を先に説明した"
         "**残差**からモーター効果を縮小推定し、公表値を条件づけた上での上積みを測る。\n",
         f"探索 〜{EXPLORE_END}（{int(expl.sum()):,}R）で推定、確認期間（{int(conf.sum()):,}R）で1回だけ評価。",
         f"縮小定数 k は探索期間を5分割した**fold外**の e_m で選択 → **k={k_sel:g}**（場×モーター {g.shape[0]:,}種、"
         f"1モーターあたり平均 {g['count'].mean():.1f} 出走）\n",
         "## 1. 市場をオフセットにしたときの上積み（確認期間の1着対数損失）\n",
         "| モデル | 確認 対数損失 | 市場との差 |", "|---|---:|---:|"]
    res = {}
    ll_mkt = logloss(Q[conf], y[conf])
    for nm, Zm in models.items():
        if Zm is None:
            res[nm] = (None, Q, ll_mkt)
            L.append(f"| {nm} | {ll_mkt:.4f} | +0.0000 |")
            continue
        w = fit_offset(Zm[expl], logq[expl], Y[expl])
        P = predict(Zm, logq, w)
        ll = logloss(P[conf], y[conf])
        res[nm] = (w, P, ll)
        L.append(f"| {nm} | {ll:.4f} | {ll - ll_mkt:+.4f} |")
    w2, w3 = res["M2 ＋公表モーター2連率"][0], res["M3 ＋分離推定したモーター効果"][0]
    L.append(f"\nM2 の係数: 公表2連率 **{w2[5]:+.4f}**")
    L.append(f"M3 の係数: 公表2連率 **{w3[5]:+.4f}**、分離推定 **{w3[6]:+.4f}**")
    L.append("（市場をオフセットに置いているので、係数が負＝市場が過大評価、正＝過小評価）\n")

    L += ["## 2. p/q の右尾はどこまで伸びるか（確認期間・単勝）\n",
          "| モデル | 上位 | n | 的中 | 市場確率 | 実勝率÷市場確率 | 回収率 | 95%区間 |",
          "|---|---|---:|---:|---:|---:|---:|---|"]
    for nm, (w, P, ll) in res.items():
        if w is None:
            continue
        er = (P / np.clip(Q, 1e-12, None))[expl]
        for frac, n, hit, qm, rt, roi, lo, hi in tail_table(P[conf], Q[conf], y[conf], er):
            L.append(f"| {nm} | 上位{frac*100:g}% | {n:,} | {hit*100:.1f}% | {qm*100:.1f}% | "
                     f"**{rt:.3f}** | {roi*100:.1f}% | {lo*100:.0f}〜{hi*100:.0f}% |")

    L += ["\n## 3. プラセボ（帰無仮説「市場確率が真」・20回生成）\n",
          "| モデル | 上位2% 比（平均） | （最大） |", "|---|---:|---:|"]
    for nm, (w, P, ll) in res.items():
        if w is None:
            continue
        v = placebo_ratios(P[conf], Q[conf], (P / np.clip(Q, 1e-12, None))[expl], RNG)
        L.append(f"| {nm} | {np.mean(v):.3f} | {np.max(v):.3f} |")

    m2, m3 = res["M2 ＋公表モーター2連率"], res["M3 ＋分離推定したモーター効果"]
    def top(P, frac):
        t = [x for x in tail_table(P[conf], Q[conf], y[conf], (P / np.clip(Q, 1e-12, None))[expl],
                                   fracs=(frac,)) if x[0] == frac]
        return t[0] if t else None
    t2, t3, t3b = top(m2[1], 0.02), top(m3[1], 0.02), top(m3[1], 0.01)
    gain = m2[2] - m3[2]
    L += ["\n## 判定（基準は実行前に固定）\n",
          f"1. M3 の対数損失 < M2: M2 {m2[2]:.4f} / M3 {m3[2]:.4f} → "
          f"**{'合格' if m3[2] < m2[2] else '不合格'}**（差 {gain:+.4f} nats）",
          f"2. M3 の上位2%比 > M2: M2 {t2[4]:.3f} / M3 {t3[4]:.3f} → "
          f"**{'合格' if t3[4] > t2[4] else '不合格'}**",
          f"4. 上位2%の n ≥ 500: n={t3[1]:,} → **{'合格' if t3[1] >= 500 else '不合格'}**",
          f"5. **継続基準 上位1〜2% ≥ 1.20**: 上位2% {t3[4]:.3f} / 上位1% {t3b[4]:.3f} → "
          f"**{'合格' if max(t3[4], t3b[4]) >= 1.20 else '不合格'}**",
          "",
          f"分離推定が市場を超えて持っていた情報は **{gain:+.4f} nats**、"
          f"控除率を埋めるのに必要な {NEED:.4f} nats の **{gain/NEED*100:.2f}%**。"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
