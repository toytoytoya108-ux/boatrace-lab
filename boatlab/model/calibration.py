"""[C] 校正：isotonic regression（docs/04 §5）。

- combo 校正：120通りの確率 p_i を「その組み合わせが的中したか」で校正し、レース内で再正規化。
- set 校正：S = Σ_{15点} p_i を「セットが的中したか」で校正 → 信頼度 C（docs/04 §8）。
評価：信頼性曲線・Brier・log-loss・ECE。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


class ComboCalibrator:
    def __init__(self):
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1.0)

    def fit(self, p: np.ndarray, actual_idx: np.ndarray) -> "ComboCalibrator":
        ok = actual_idx >= 0
        y = np.zeros_like(p[ok])
        y[np.arange(ok.sum()), actual_idx[ok]] = 1.0
        # 120×R 点。isotonic は単調写像なので順位は保存される
        self.iso.fit(p[ok].ravel(), y.ravel())
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        q = self.iso.predict(p.ravel()).reshape(p.shape)
        q = np.where(p <= 0, 0.0, q)
        return q / np.clip(q.sum(axis=1, keepdims=True), 1e-12, None)


class SetCalibrator:
    def __init__(self):
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)

    def fit(self, s: np.ndarray, hit: np.ndarray) -> "SetCalibrator":
        self.iso.fit(s, hit.astype(float))
        return self

    def transform(self, s: np.ndarray) -> np.ndarray:
        return self.iso.predict(s)


def reliability_table(pred: np.ndarray, y: np.ndarray, bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"p": pred, "y": y})
    df["bin"] = pd.qcut(df["p"], bins, duplicates="drop")
    g = df.groupby("bin", observed=True).agg(n=("y", "size"), p_mean=("p", "mean"), y_rate=("y", "mean"))
    g["gap"] = g["y_rate"] - g["p_mean"]
    return g.reset_index()


def calibration_metrics(pred: np.ndarray, y: np.ndarray) -> dict:
    pred = np.clip(pred, 1e-9, 1 - 1e-9)
    brier = float(np.mean((pred - y) ** 2))
    ll = float(-np.mean(y * np.log(pred) + (1 - y) * np.log(1 - pred)))
    t = reliability_table(pred, y)
    ece = float((t["n"] / t["n"].sum() * t["gap"].abs()).sum())
    return {"brier": brier, "logloss": ll, "ece": ece, "n": int(len(y))}
