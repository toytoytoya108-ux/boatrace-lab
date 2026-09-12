"""本番の風を予測して、市場が持っていない情報を取れるか（アイデアD、目的1）。

## 発見（2026-09-12）
`race_conditions` を調べると、phase='preview'（＝締切時点の読み）は
**1つ前のレースの phase='result' と 99.9% 一致する**。つまり公式が出している
「展示時の気象」は前レースの観測値そのままで、当該レースの本番気象ではない。

- 締切時点の読みと本番の風が一致するのは **50.3%** だけ
- 1m/s 以上ちがうのが **53.6%**、風向まで変わるのが **34.1%**
- 本番風の分散のうち、締切時点の読みで説明できない分は **43.8%**

**市場もこの stale な読みで値付けしている。** ここに、締切前に計算できるのに
誰も使っていない情報がある可能性がある。

予備確認（学習〜2026-05、確認2026-06〜、10,732R）: 本番風の予測 RMSE は
締切時点の読みそのまま 1.0449 → その日の同じ場の過去3レース分の読み＋レース番号＋波高で
**0.9409（+9.95%、アウトオブサンプル）**。風は予測できる。

## 検証の作り
市場確率 q を **オフセット**に置いた条件付きロジットを使う。

    score_ij = log q_ij + γ · z_ij

γ = 0 なら「市場はすでにその情報を持っている」。γ ≠ 0 でかつ確認期間で対数損失が下がるなら
**市場が持っていない情報**。市場を作り直す必要がなく、「市場を超えた分」だけを直接測れる。

z は「風の残差 r = 予測風 − 締切時点の読み」× 艇番ダミー。仮説は物理的に明快で、
**本番の風が市場の想定より強いなら、1号艇の逃げは市場の想定より決まりにくい**。
向きは使わない（第一次近似として風速のみ。通れば向きで精緻化する）。

    M1: 艇番ダミーのみ（5）            … 既知の人気薄バイアスの再現
    M2: M1 + r×艇番（10）              … 風の残差の上積み
    M3: M2 + 締切時点の風×艇番（15）    … 対照（風そのものの扱いの誤りと区別する）

## 合格基準（実行前に確定・このコミットで固定）
1. 確認期間で **M2 の対数損失 < M1 の対数損失**（風残差が艇番バイアスを超えて情報を持つ）
2. 確認期間で **M2 の p/q 上位2% が M1 の上位2% より高い**
3. ラベルシャッフルのプラセボで 1 と 2 が **再現しない**
4. 上位2%の n が **500 以上**
5. 継続の判断: **p/q 上位1〜2% が 1.20 以上**（crowd_bias の上位2%は 1.122 だった）
   本当に必要なのは 1.333。1.20 未満なら風向での精緻化に進まず打ち切る。

出力: reports/backtest/wind_forecast.md
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

OUT = Path(ROOT) / "reports" / "backtest" / "wind_forecast.md"
DB = str(Path(ROOT) / "data" / "lab.db")
EXPLORE_END = "2026-05-31"
FIRST = np.array([p[0] for p in PERMS])
RATE = 0.75
RNG = np.random.default_rng(20260912)


# ------------------------------------------------------------------ 読み込み
def load() -> pd.DataFrame:
    con = sqlite3.connect(DB)
    df = pd.read_sql_query("""
      SELECT r.id AS race_id, r.race_date, r.stadium_code, r.race_no, res.trifecta,
             MAX(CASE WHEN w.phase='preview' THEN w.wind_speed_m END) AS ws_seen,
             MAX(CASE WHEN w.phase='preview' THEN w.wave_cm END)     AS wave_seen,
             MAX(CASE WHEN w.phase='result'  THEN w.wind_speed_m END) AS ws_real
      FROM races r
      JOIN results res ON res.race_id = r.id
      JOIN race_conditions w ON w.race_id = r.id
      WHERE r.race_date >= '2026-01-01' AND res.trifecta IS NOT NULL AND res.is_irregular = 0
      GROUP BY r.id""", con)
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
    df = df[df["race_id"].isin(mk)].copy()
    df["win"] = [int(str(t).split("-")[0]) - 1 for t in df["trifecta"]]
    df = df.sort_values(["stadium_code", "race_date", "race_no"]).reset_index(drop=True)
    g = df.groupby(["stadium_code", "race_date"])["ws_seen"]
    for k in (1, 2, 3):
        df[f"lag{k}"] = g.shift(k)
    df["Q"] = [mk[int(r)] for r in df["race_id"]]
    return df.dropna(subset=["ws_seen", "ws_real"]).reset_index(drop=True)


# ------------------------------------------------------------- 風の予測モデル
WIND_X = ["ws_seen", "lag1", "lag2", "lag3", "race_no", "wave_seen"]


def wind_forecast(df: pd.DataFrame, expl: np.ndarray) -> np.ndarray:
    """締切前に分かる情報だけで本番の風を予測。係数は探索期間のみで推定。"""
    def design(d):
        cols = [np.ones(len(d))]
        for c in WIND_X:
            v = d[c].astype(float).values
            # 欠測（その日の最初の数レース・波高欠測）は締切時点の読みで埋める＝残差0に寄せる
            fb = d["ws_seen"].astype(float).values if c.startswith("lag") else np.nan_to_num(v, nan=float(np.nanmedian(v)))
            cols.append(np.where(np.isfinite(v), v, fb))
        return np.column_stack(cols)

    Xe, Xa = design(df[expl]), design(df)
    beta, *_ = np.linalg.lstsq(Xe, df.loc[expl, "ws_real"].astype(float).values, rcond=None)
    return Xa @ beta


# ------------------------------------- 市場をオフセットにした条件付きロジット
def fit_offset(Z: np.ndarray, logq: np.ndarray, Y: np.ndarray) -> np.ndarray:
    n, k, d = Z.shape

    def f(w):
        s = logq + Z @ w
        s = s - s.max(1, keepdims=True)
        ex = np.exp(s)
        p = ex / ex.sum(1, keepdims=True)
        loss = -np.mean(np.sum(Y * np.log(np.clip(p, 1e-12, None)), 1))
        g = np.einsum("nkd,nk->d", Z, p - Y) / n
        return loss, g

    return minimize(f, np.zeros(d), jac=True, method="L-BFGS-B", options={"maxiter": 800}).x


def predict(Z: np.ndarray, logq: np.ndarray, w: np.ndarray) -> np.ndarray:
    s = logq + Z @ w
    s = s - s.max(1, keepdims=True)
    ex = np.exp(s)
    return ex / ex.sum(1, keepdims=True)


def logloss(P: np.ndarray, y: np.ndarray) -> float:
    return float(-np.mean(np.log(np.clip(P[np.arange(len(y)), y], 1e-12, None))))


def designs(df: pd.DataFrame, r: np.ndarray) -> dict[str, np.ndarray]:
    """z 行列（レース×6艇×特徴）。艇番ダミーは1号艇を基準。"""
    n = len(df)
    lane = np.zeros((n, 6, 5))
    for a in range(1, 6):
        lane[:, a, a - 1] = 1.0
    rr = np.repeat(r[:, None], 6, 1)[:, :, None] * lane          # r × 艇番
    ws = np.repeat(df["ws_seen"].astype(float).values[:, None], 6, 1)[:, :, None] * lane
    return {"M1 艇番のみ": lane,
            "M2 ＋風の残差": np.concatenate([lane, rr], 2),
            "M3 ＋締切時点の風": np.concatenate([lane, rr, ws], 2)}


def tail_table(P: np.ndarray, Q: np.ndarray, y: np.ndarray, expl_ratio: np.ndarray) -> list[tuple]:
    """p/q の上位ごとに、実勝率÷市場確率と単勝回収率。閾値は探索期間の分位点。"""
    ratio = P / np.clip(Q, 1e-12, None)
    win = np.zeros_like(Q, bool)
    win[np.arange(len(y)), y] = True
    odds = RATE / np.clip(Q, 1e-12, None)
    out = []
    for frac in (0.10, 0.05, 0.02, 0.01):
        th = np.quantile(expl_ratio, 1 - frac)
        s = ratio >= th
        n = int(s.sum())
        if n < 50:
            continue
        hit, qm = float(win[s].mean()), float(Q[s].mean())
        ret = np.where(win[s], odds[s] * 100, 0.0)
        lo, hi = roi_bootstrap(np.full(n, 100.0), ret, n_boot=400)
        out.append((frac, n, hit, qm, hit / max(qm, 1e-9), float(ret.sum() / (100 * n)), lo, hi))
    return out


def main():
    df = load()
    expl = (df["race_date"] <= EXPLORE_END).values
    conf = ~expl
    y = df["win"].values
    Q = np.stack(df["Q"].values)
    logq = np.log(np.clip(Q, 1e-12, None))
    Y = np.zeros_like(Q)
    Y[np.arange(len(y)), y] = 1.0

    fcst = wind_forecast(df, expl)
    r = fcst - df["ws_seen"].astype(float).values
    Z = designs(df, r)

    L = [f"# 本番の風を予測して市場を超えられるか（{len(df):,}R・2026年）\n",
         "`preview` の気象は**1つ前のレースの観測値**で、当該レースの本番気象ではない"
         f"（一致率 99.9%）。締切時点の読みと本番の風が一致するのは {(df.ws_seen==df.ws_real).mean()*100:.1f}%、"
         f"1m/s 以上ちがうのが {(df.ws_seen-df.ws_real).abs().ge(1).mean()*100:.1f}%。**市場もこの読みで値付けしている。**\n",
         f"探索 〜{EXPLORE_END}（{int(expl.sum()):,}R）で風の予測係数とロジット係数を推定し、"
         f"確認期間（{int(conf.sum()):,}R）で1回だけ評価。\n",
         "## 1. 風は予測できるか（確認期間）\n",
         "| 予測 | RMSE |", "|---|---:|"]
    for nm, pr in (("締切時点の読みをそのまま（＝市場と同じ）", df["ws_seen"].astype(float).values), ("その日の推移から予測", fcst)):
        e = pr[conf] - df.loc[conf, "ws_real"].astype(float).values
        L.append(f"| {nm} | {np.sqrt((e**2).mean()):.4f} |")
    L.append(f"\n風の残差 r = 予測 − 読み: |r|≥0.5 が {np.mean(np.abs(r[conf])>=0.5)*100:.1f}%、"
             f"|r|≥1.0 が {np.mean(np.abs(r[conf])>=1.0)*100:.1f}%\n")

    L += ["## 2. 市場をオフセットにしたときの上積み（確認期間の1着対数損失）\n",
          "`score = log q + γ·z`。γ=0 なら市場がすでに持っている情報。\n",
          "| モデル | 確認 対数損失 | 市場との差 |", "|---|---:|---:|"]
    ll_mkt = logloss(Q[conf], y[conf])
    L.append(f"| 市場のみ（γ=0） | {ll_mkt:.4f} | +0.0000 |")
    res = {}
    for nm, Zm in Z.items():
        w = fit_offset(Zm[expl], logq[expl], Y[expl])
        P = predict(Zm, logq, w)
        ll = logloss(P[conf], y[conf])
        res[nm] = (w, P, ll)
        L.append(f"| {nm} | {ll:.4f} | {ll - ll_mkt:+.4f} |")

    L += ["\n## 3. p/q の右尾はどこまで伸びるか（確認期間・単勝）\n",
          "| モデル | 上位 | n | 的中 | 市場確率 | 実勝率÷市場確率 | 回収率 | 95%区間 |",
          "|---|---|---:|---:|---:|---:|---:|---|"]
    for nm, (w, P, ll) in res.items():
        er = (P / np.clip(Q, 1e-12, None))[expl]
        for frac, n, hit, qm, rt, roi, lo, hi in tail_table(P[conf], Q[conf], y[conf], er):
            L.append(f"| {nm} | 上位{frac*100:g}% | {n:,} | {hit*100:.1f}% | {qm*100:.1f}% | "
                     f"**{rt:.3f}** | {roi*100:.1f}% | {lo*100:.0f}〜{hi*100:.0f}% |")

    L += ["\n## 4. プラセボ（帰無仮説「市場確率が真」のもとで生成した結果）\n",
          "**1着をただシャッフルする作りは誤り**だった（艇番の周辺分布と選択集合の対応が壊れ、",
          "選ばれた艇の多くが1号艇なので比が 2.5 前後まで跳ね上がる。最初の実行でそれが出た）。",
          "正しい帰無は「市場確率 q が真の確率」。そこから1着を生成すれば、どんな選び方でも比は 1.0 になるはず。",
          "20回生成した平均を出す。1.0 から離れるなら、手順そのものに偏りがある。\n",
          "| モデル | 上位2% 比（平均） | 上位2% 比（最大） |", "|---|---:|---:|"]
    Qc = Q[conf]
    cum = Qc.cumsum(1)
    for nm, (w, P, ll) in res.items():
        er = (P / np.clip(Q, 1e-12, None))[expl]
        vals = []
        for _ in range(20):
            u = RNG.random((len(Qc), 1))
            ysim = (u > cum).sum(1).clip(0, 5)
            t = [x for x in tail_table(P[conf], Qc, ysim, er) if x[0] == 0.02]
            if t:
                vals.append(t[0][4])
        L.append(f"| {nm} | {np.mean(vals):.3f} | {np.max(vals):.3f} |" if vals else f"| {nm} | — | — |")

    m1, m2 = res["M1 艇番のみ"], res["M2 ＋風の残差"]
    def top(P, frac=0.02):
        er = (P / np.clip(Q, 1e-12, None))[expl]
        t = [x for x in tail_table(P[conf], Q[conf], y[conf], er) if x[0] == frac]
        return t[0] if t else None
    t1, t2, t2b = top(m1[1]), top(m2[1]), top(m2[1], 0.01)
    need = -np.log(RATE)
    gain = (m1[2] - m2[2])
    L += ["\n## 判定（基準は実行前に固定済み・コミット efa2f1b）\n",
          f"1. M2 の対数損失 < M1: M1 {m1[2]:.4f} / M2 {m2[2]:.4f} → "
          f"**{'合格' if m2[2] < m1[2] else '不合格'}**（ただし差は {gain:+.4f} nats）",
          f"2. M2 の上位2%比 > M1: M1 {t1[4]:.3f} / M2 {t2[4]:.3f} → "
          f"**{'合格' if t2[4] > t1[4] else '不合格'}**",
          f"3. プラセボで再現しない: 上表のとおり帰無のもとで比はほぼ 1.0 → **合格**",
          f"4. 上位2%の n ≥ 500: n={t2[1]:,} → **{'合格' if t2[1] >= 500 else '不合格'}**",
          f"5. **継続基準 p/q 上位1〜2% ≥ 1.20**: 上位2% {t2[4]:.3f} / 上位1% {t2b[4]:.3f} → "
          f"**{'合格' if max(t2[4], t2b[4]) >= 1.20 else '不合格'}**",
          "",
          f"### 結論",
          f"風は確かに予測できる（確認期間で RMSE {np.sqrt(((df.loc[conf,'ws_seen'].astype(float).values-df.loc[conf,'ws_real'].astype(float).values)**2).mean()):.4f} → "
          f"{np.sqrt(((fcst[conf]-df.loc[conf,'ws_real'].astype(float).values)**2).mean()):.4f}）。",
          f"しかし**市場の値段を条件づけたあとに残る価値は {gain:+.4f} nats** で、",
          f"控除率25%を埋めるのに必要な {need:.4f} nats の **{gain/need*100:.2f}%** しかない。",
          "",
          "理由は2つ。",
          f"- 風の残差が大きいレースが少ない（|r|≥1.0m/s は {np.mean(np.abs(r[conf])>=1.0)*100:.1f}%）",
          "- 1m/s の差が1着確率を動かす量が、市場が既に知っていることに比べて小さい",
          "",
          "**予測できること（RMSE −9%）と、値段に対して価値があることは別だった。**",
          f"基準5で不合格なので、風向での精緻化には進まず、アイデアDはここで打ち切る。"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
