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


# ---------------------------------------------------------------- 絞り込み型（focused）
@dataclass
class FocusedParams:
    """絞り込み型の買い方（docs/04 §16、2026-09-05 採用）。

    1) レース選定: S15（上位15点の確率和）≥ s15_min（検証期間の上位20%≈0.73）
    2) 買い目: 上位 pool 点のうち EV=p×odds ≥ ev_min かつ odds∈[odds_lo, odds_hi] を確率順に最大 max_points
    3) 配分: 予算 budget、1点上限 cap、確率比例（power）、100円単位
    """
    enabled: bool = True
    s15_min: float = 0.73
    pool: int = 15
    ev_min: float = 1.0
    beta: float = 0.0             # EV に使う確率の市場側縮約（0=モデル確率そのまま。検証はこの設定）
    odds_lo: float = 5.0
    odds_hi: float = 50.0
    max_points: int = 5
    budget: int = 3000
    cap: int = 1000
    power: float = 1.0
    completeness_min: float = 0.6
    require_real_odds: bool = False

    @staticmethod
    def from_dict(d: dict | None) -> "FocusedParams":
        d = d or {}
        return FocusedParams(**{k: v for k, v in d.items() if k in FocusedParams().__dict__})


@dataclass
class FocusedSelection:
    points: list[int]             # 120通りの index（確率順）
    stakes: list[int]             # 点ごとの賭け金
    ev: np.ndarray
    S15: float
    decision: str                 # buy / skip
    skip_reason: str | None
    expected_return: float        # 賭け金加重の EV


def select_focused(p: np.ndarray, odds: np.ndarray, prm: FocusedParams, completeness: float = 1.0,
                   odds_estimated: bool = False) -> FocusedSelection:
    from boatlab.model.staking import StakingParams, allocate
    p = np.asarray(p, float)
    odds = np.asarray(odds, float)
    q = implied_q(odds)
    p_hat = (1 - prm.beta) * p + prm.beta * q
    ev = p_hat * np.where(np.isfinite(odds), odds, 0.0)
    order = np.argsort(-p)
    S15 = float(p[order[:15]].sum())
    pool = [int(i) for i in order[: prm.pool]]
    cand = [i for i in pool if ev[i] >= prm.ev_min and np.isfinite(odds[i]) and prm.odds_lo <= odds[i] <= prm.odds_hi][: prm.max_points]
    reason = None
    if completeness < prm.completeness_min:
        reason = "incomplete"
    elif prm.require_real_odds and odds_estimated:
        reason = "odds_estimated"
    elif S15 < prm.s15_min:
        reason = "confidence"
    elif not cand:
        reason = "no_value_points"
    if not cand:
        return FocusedSelection([], [], ev, S15, "skip", reason or "no_value_points", 0.0)
    # 見送りでも候補点は保存して仮想採点する（ゲートの妥当性を後から検証できる）
    budget = int(min(prm.budget, prm.cap * len(cand)))
    stakes = allocate(cand, cand, p, odds, StakingParams(method="prob", total=budget, prob_power=prm.power,
                                                          max_share=min(1.0, prm.cap / budget)))
    er = float(np.average(ev[cand], weights=stakes))
    return FocusedSelection(cand, [int(x) for x in stakes], ev, S15, "skip" if reason else "buy", reason, er)


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
