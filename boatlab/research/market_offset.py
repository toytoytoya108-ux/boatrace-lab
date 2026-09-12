"""市場確率をオフセットに置いた条件付きロジット（目的1の検証の共通土台）。

    score_ij = log q_ij + γ · z_ij

γ=0 なら「市場はすでにその情報を持っている」。γ≠0 でかつ確認期間で対数損失が下がるなら
**市場が持っていない情報**。市場を作り直す必要がなく、「市場を超えた分」だけを直接測れる。

回収率100%には `実勝率 ÷ 市場確率 > 1/(1-控除率) = 1.333` が必要。
プラセボは必ず `placebo_ratios`（帰無仮説 y ~ Multinomial(q)）を使う。
**1着をシャッフルする作りは誤り**で、艇番の周辺分布と選択集合の対応が壊れて比が跳ね上がる
（`reports/backtest/wind_forecast.md` §4 に実例）。
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

RATE = 0.75
NEED = float(-np.log(RATE))          # 控除率25%を埋めるのに必要な情報量 0.2877 nats


def fit_offset(Z: np.ndarray, logq: np.ndarray, Y: np.ndarray, l2: float = 0.0) -> np.ndarray:
    """Z:(n,6,d) logq:(n,6) Y:(n,6) one-hot。γ を最尤（任意でL2）推定。"""
    n, _, d = Z.shape

    def f(w):
        s = logq + Z @ w
        s = s - s.max(1, keepdims=True)
        ex = np.exp(s)
        p = ex / ex.sum(1, keepdims=True)
        loss = -np.mean(np.sum(Y * np.log(np.clip(p, 1e-12, None)), 1)) + l2 * float(w @ w)
        g = np.einsum("nkd,nk->d", Z, p - Y) / n + 2 * l2 * w
        return loss, g

    return minimize(f, np.zeros(d), jac=True, method="L-BFGS-B", options={"maxiter": 800}).x


def predict(Z: np.ndarray, logq: np.ndarray, w: np.ndarray) -> np.ndarray:
    s = logq + Z @ w
    s = s - s.max(1, keepdims=True)
    ex = np.exp(s)
    return ex / ex.sum(1, keepdims=True)


def logloss(P: np.ndarray, y: np.ndarray) -> float:
    return float(-np.mean(np.log(np.clip(P[np.arange(len(y)), y], 1e-12, None))))


def lane_dummies(n: int) -> np.ndarray:
    """1号艇を基準にしたコースのダミー (n,6,5)。"""
    lane = np.zeros((n, 6, 5))
    for a in range(1, 6):
        lane[:, a, a - 1] = 1.0
    return lane


def per_boat(v: np.ndarray) -> np.ndarray:
    """レース単位の値 (n,) を (n,6,1) に広げる。"""
    return np.repeat(np.asarray(v, float)[:, None], 6, 1)[:, :, None]


def tail_table(P, Q, y, expl_ratio, fracs=(0.10, 0.05, 0.02, 0.01), n_min=50, boot=400):
    """p/q の上位ごとに (割合, n, 的中, 市場確率, 比, 単勝回収率, 下限, 上限)。閾値は探索期間の分位点。"""
    from boatlab.backtest.metrics import roi_bootstrap
    ratio = P / np.clip(Q, 1e-12, None)
    win = np.zeros_like(Q, bool)
    win[np.arange(len(y)), y] = True
    odds = RATE / np.clip(Q, 1e-12, None)
    out = []
    for frac in fracs:
        s = ratio >= np.quantile(expl_ratio, 1 - frac)
        n = int(s.sum())
        if n < n_min:
            continue
        hit, qm = float(win[s].mean()), float(Q[s].mean())
        ret = np.where(win[s], odds[s] * 100, 0.0)
        lo, hi = roi_bootstrap(np.full(n, 100.0), ret, n_boot=boot)
        out.append((frac, n, hit, qm, hit / max(qm, 1e-9), float(ret.sum() / (100 * n)), lo, hi))
    return out


def placebo_ratios(P, Q, expl_ratio, rng, frac=0.02, reps=20) -> list[float]:
    """帰無仮説「市場確率 q が真」のもとでの比。どんな選び方でも 1.0 になるはず。"""
    cum = Q.cumsum(1)
    out = []
    for _ in range(reps):
        ysim = (rng.random((len(Q), 1)) > cum).sum(1).clip(0, 5)
        t = [x for x in tail_table(P, Q, ysim, expl_ratio, fracs=(frac,), boot=1) if x[0] == frac]
        if t:
            out.append(t[0][4])
    return out
