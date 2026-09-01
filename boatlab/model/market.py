"""[D] 市場オッズ推定モデル（docs/04 §6）。過去の締切前オッズが無い期間の穴判定・期待値に使う。

q_i：公開情報だけのモデル（M0b + PL）で推定した「市場が付けるであろう確率」
odds_est_i = exp(a + b·log(0.75 / q_i))
(a, b) は単勝・2連単・3連単の実払戻（=最終オッズ×100）を観測値として回帰（3連単の的中目だけより選択バイアスが小さい）。
2026 年以降の実オッズ（turnmark_final）は精度検証にのみ使う。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from boatlab.model.trifecta import PERMS, PERM_LABELS, trifecta_probs

_A = np.array([p[0] for p in PERMS])
_B = np.array([p[1] for p in PERMS])

TAKEOUT = 0.25


class MarketOddsModel:
    def __init__(self, lam2: float = 0.8, lam3: float = 0.7):
        self.lam2, self.lam3 = lam2, lam3
        self.a, self.b = 0.0, 1.0
        self.fit_report: dict = {}

    def q_from_public(self, p_win, p_top2, p_top3) -> np.ndarray:
        return trifecta_probs(p_win, p_top2, p_top3, self.lam2, self.lam3)

    def fit(self, q: np.ndarray, payouts: pd.DataFrame) -> "MarketOddsModel":
        """payouts: race 順に並んだ DataFrame（win_combo, win_amount, ex_combo, ex_amount, tri_idx, tri_amount）。"""
        xs, ys = [], []
        # 単勝
        q1 = np.zeros((len(q), 6))
        for k in range(6):
            q1[:, k] = q[:, _A == k].sum(axis=1)
        ok = payouts["win_lane"].notna() & payouts["win_amount"].notna()
        lanes = payouts.loc[ok, "win_lane"].astype(int).values - 1
        xs.append(np.log((1 - TAKEOUT) / np.clip(q1[ok.values, lanes], 1e-6, None)))
        ys.append(np.log(payouts.loc[ok, "win_amount"].values / 100.0))
        # 2連単
        ok = payouts["ex_a"].notna() & payouts["ex_amount"].notna()
        a = payouts.loc[ok, "ex_a"].astype(int).values - 1
        b = payouts.loc[ok, "ex_b"].astype(int).values - 1
        q2 = np.zeros(ok.sum())
        rows = np.where(ok.values)[0]
        for i, (ra, aa, bb) in enumerate(zip(rows, a, b)):
            q2[i] = q[ra, (_A == aa) & (_B == bb)].sum()
        xs.append(np.log((1 - TAKEOUT) / np.clip(q2, 1e-6, None)))
        ys.append(np.log(payouts.loc[ok, "ex_amount"].values / 100.0))
        # 3連単
        ok = (payouts["tri_idx"] >= 0) & payouts["tri_amount"].notna()
        rows = np.where(ok.values)[0]
        q3 = q[rows, payouts.loc[ok, "tri_idx"].astype(int).values]
        xs.append(np.log((1 - TAKEOUT) / np.clip(q3, 1e-6, None)))
        ys.append(np.log(payouts.loc[ok, "tri_amount"].values / 100.0))
        X = np.concatenate(xs)
        Y = np.concatenate(ys)
        b, a = np.polyfit(X, Y, 1)
        self.a, self.b = float(a), float(b)
        resid = Y - (a + b * X)
        self.fit_report = {"n_obs": int(len(X)), "a": self.a, "b": self.b, "resid_sd": float(resid.std())}
        return self

    def odds(self, q: np.ndarray) -> np.ndarray:
        est = np.exp(self.a + self.b * np.log((1 - TAKEOUT) / np.clip(q, 1e-6, None)))
        return np.where(q <= 0, np.nan, np.clip(est, 1.0, 100000.0))


def evaluate_against_real(est: np.ndarray, real: np.ndarray) -> dict:
    """全120通りの推定オッズ vs 実最終オッズ（2026〜）。log スケールの相関・MAE、帯別の偏り。"""
    ok = np.isfinite(est) & np.isfinite(real) & (real > 0) & (est > 0)
    le, lr = np.log(est[ok]), np.log(real[ok])
    df = pd.DataFrame({"le": le, "lr": lr})
    df["band"] = pd.cut(np.exp(df["lr"]), [0, 10, 20, 50, 100, 300, 1e6], labels=["<10", "10-20", "20-50", "50-100", "100-300", "300+"])
    band = df.groupby("band", observed=True).apply(lambda g: pd.Series({"n": len(g), "bias_log": float((g["le"] - g["lr"]).mean())}))
    return {"n": int(ok.sum()), "corr_log": float(np.corrcoef(le, lr)[0, 1]), "mae_log": float(np.abs(le - lr).mean()),
            "bias_log": float((le - lr).mean()), "by_band": band.reset_index().to_dict("records")}


def real_odds_matrix(odds_json: dict) -> np.ndarray:
    """{'1-2-3': 5.6, ...} → (120,) 配列（欠場は nan）。"""
    return np.array([np.nan if odds_json.get(k) is None else float(odds_json[k]) for k in PERM_LABELS])
