"""[B] 120通り確率：位置別割引付き Plackett–Luce（docs/04 §4, Model 1.0）。

P(a→b→c) = w_a/Σw · s2_b^λ2/Σ_{j≠a} s2_j^λ2 · s3_c^λ3/Σ_{j≠a,b} s3_j^λ3
  w  = 1着モデルの確率（レース内で正規化）
  s2 = 2着以内モデルの確率、s3 = 3着以内モデルの確率（強さとして使う）
λ2, λ3 は検証データの3連単対数尤度を最大化して選ぶ。
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

PERMS = list(itertools.permutations(range(6), 3))  # 120 (a,b,c) 0-based
PERM_LABELS = [f"{a+1}-{b+1}-{c+1}" for a, b, c in PERMS]
_A = np.array([p[0] for p in PERMS])
_B = np.array([p[1] for p in PERMS])
_C = np.array([p[2] for p in PERMS])


def race_matrix(pred: pd.DataFrame, x: pd.DataFrame, col: str) -> tuple[np.ndarray, list]:
    """艇別予測 → (R,6) 行列（lane 順、欠場は 0）。"""
    df = pd.DataFrame({"race_id": x["race_id"].values, "lane": x["lane"].values, "v": pred[col].values})
    piv = df.pivot_table(index="race_id", columns="lane", values="v", aggfunc="first").reindex(columns=[1, 2, 3, 4, 5, 6])
    return piv.fillna(0.0).values, list(piv.index)


def trifecta_probs(p_win: np.ndarray, p_top2: np.ndarray, p_top3: np.ndarray,
                   lam2: float = 0.7, lam3: float = 0.6, eps: float = 1e-9) -> np.ndarray:
    """(R,6)×3 → (R,120) 確率。行ごとに合計 1。"""
    w = np.clip(p_win, 0, None) + eps
    w = w / w.sum(axis=1, keepdims=True)
    s2 = np.power(np.clip(p_top2, 0, None) + eps, lam2)
    s3 = np.power(np.clip(p_top3, 0, None) + eps, lam3)
    S2 = s2.sum(axis=1, keepdims=True)
    S3 = s3.sum(axis=1, keepdims=True)
    pa = w[:, _A]
    pb = s2[:, _B] / (S2 - s2[:, _A])
    pc = s3[:, _C] / (S3 - s3[:, _A] - s3[:, _B])
    p = pa * pb * pc
    # 欠場艇（強さ0）を含む組み合わせは 0 に、残りを正規化
    absent = (p_win <= 0) & (p_top2 <= 0)
    mask = absent[:, _A] | absent[:, _B] | absent[:, _C]
    p = np.where(mask, 0.0, p)
    return p / np.clip(p.sum(axis=1, keepdims=True), eps, None)


def combo_index(trifecta: str) -> int | None:
    try:
        return PERM_LABELS.index(trifecta)
    except ValueError:
        return None


def loglik(p: np.ndarray, actual_idx: np.ndarray) -> float:
    ok = actual_idx >= 0
    return float(np.mean(np.log(np.clip(p[ok, actual_idx[ok]], 1e-12, None))))


def fit_lambdas(p_win, p_top2, p_top3, actual_idx, grid=np.arange(0.3, 1.01, 0.1)) -> tuple[float, float, float]:
    """検証データで λ2, λ3 をグリッド探索（粗い 0.1 刻み。docs/04 §14）。"""
    best = (None, None, -np.inf)
    for l2 in grid:
        for l3 in grid:
            ll = loglik(trifecta_probs(p_win, p_top2, p_top3, l2, l3), actual_idx)
            if ll > best[2]:
                best = (float(l2), float(l3), ll)
    return best
