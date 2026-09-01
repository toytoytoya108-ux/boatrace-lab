"""[E][F][G][H] 期待値・信頼度・15点選定・購入判定（docs/04 §7〜§10）。

selection_version = "sel1"
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from boatlab.model.trifecta import PERMS, PERM_LABELS

SELECTION_VERSION = "sel1"
_A = np.array([p[0] for p in PERMS])
TAKEOUT = 0.25


@dataclass
class SelectionParams:
    points: int = 15
    main_points: int = 10
    stake: int = 200
    alpha: float = 0.0            # 本線の EV 混合（0 = 確率順）
    beta: float = 0.3             # p̂ = (1-β)p + βq  市場側へ縮約
    hole_min_odds: float = 20.0   # バックテストで決める
    hole_p_min: float = 0.003
    hole_max_same_first: int = 2
    hole_min_non_lane1: int = 2
    odds_drift: dict = field(default_factory=dict)  # d(odds) 帯別補正。初期 1.0
    confidence_min: float = 0.70
    ev_min: float = 1.00
    completeness_min: float = 0.6


def implied_q(odds: np.ndarray) -> np.ndarray:
    """オッズ → 市場の暗黙確率（控除率を除いて正規化）。"""
    inv = np.where(np.isfinite(odds) & (odds > 0), 1.0 / odds, 0.0)
    s = inv.sum()
    return inv / s if s > 0 else inv


def drift_factor(odds: np.ndarray, table: dict) -> np.ndarray:
    if not table:
        return np.ones_like(odds)
    out = np.ones_like(odds)
    for band, f in table.items():  # band: "lo-hi"
        lo, hi = (float(v) for v in band.split("-"))
        out[(odds >= lo) & (odds < hi)] = f
    return out


@dataclass
class RaceSelection:
    main: list[int]               # 120通りの index
    hole: list[int]
    p_hat: np.ndarray
    ev: np.ndarray
    expected_return: float
    S: float                      # Σ p（校正前セット確率）
    hole_relaxed: bool
    hole_relaxed_to: float | None


def select_points(p: np.ndarray, odds: np.ndarray, prm: SelectionParams, absent: np.ndarray | None = None) -> RaceSelection:
    """p: 校正後確率(120)、odds: 予想時オッズ(120, 推定可)。"""
    p = np.asarray(p, float).copy()
    odds = np.asarray(odds, float)
    if absent is not None:
        p[absent] = 0.0
    q = implied_q(odds)
    p_hat = (1 - prm.beta) * p + prm.beta * q
    ev = p_hat * np.where(np.isfinite(odds), odds, 0.0) * drift_factor(odds, prm.odds_drift)

    score = p if prm.alpha <= 0 else np.power(np.clip(p, 1e-12, None), 1 - prm.alpha) * np.power(np.clip(ev, 1e-12, None), prm.alpha)
    order = np.argsort(-score)
    main = [int(i) for i in order if p[i] > 0][: prm.main_points]

    n_hole = prm.points - prm.main_points
    hole: list[int] = []
    relaxed = False
    relaxed_to = None
    thresholds = [prm.hole_min_odds] + [t for t in (15.0, 10.0, 5.0, 1.0) if t < prm.hole_min_odds]
    for th in thresholds:
        cand = [i for i in np.argsort(-ev) if i not in main and i not in hole and np.isfinite(odds[i]) and odds[i] >= th
                and p_hat[i] >= prm.hole_p_min]
        for i in cand:
            if len(hole) >= n_hole:
                break
            first = _A[i]
            if sum(1 for h in hole if _A[h] == first) >= prm.hole_max_same_first:
                continue
            # 1号艇1着以外を最低 n 点確保する（残り枠で調整）
            remaining = n_hole - len(hole)
            non1 = sum(1 for h in hole if _A[h] != 0)
            if first == 0 and non1 < prm.hole_min_non_lane1 and remaining <= (prm.hole_min_non_lane1 - non1):
                continue
            hole.append(int(i))
        if len(hole) >= n_hole:
            break
        relaxed = True
        relaxed_to = th
    if len(hole) < n_hole:  # 制約を外して埋める（必ず15点にする）
        for i in np.argsort(-ev):
            if len(hole) >= n_hole:
                break
            if i not in main and i not in hole and p[i] > 0:
                hole.append(int(i))
    sel = main + hole
    er = float(np.mean(ev[sel])) if sel else 0.0   # 均等200円なので平均でよい
    S = float(p[sel].sum())
    return RaceSelection(main, hole, p_hat, ev, er, S, relaxed, relaxed_to)


def decide(confidence: float, expected_return: float, completeness: float, flags: dict, prm: SelectionParams) -> tuple[str, str | None]:
    if completeness < prm.completeness_min:
        return "skip", "incomplete"
    if flags.get("low_agreement"):
        return "skip", "low_agreement"
    if confidence < prm.confidence_min:
        return "skip", "confidence"
    if expected_return < prm.ev_min:
        return "skip", "expected_return"
    return "buy", None


def labels(idx: list[int]) -> list[str]:
    return [PERM_LABELS[i] for i in idx]
