"""資金配分（docs/04 §12 追補）。15点の合計は固定（既定 3,000円）、100円単位。

方式:
  uniform      : 均等（既定。200円×15点）
  payout_equal : 払戻均等。オッズ反比例（どの点が的中してもほぼ同額の払戻）
  prob         : 確率比例。モデル確率 p^power に比例（本線に厚く）
  group        : 本線／穴で単価を変える（main_stake×10 + hole_stake×5 = total）

共通の制約: 各点 min_per_point 以上、各点 max_share×total 以下、合計 = total、unit 単位。
配分は選定済みの15点に対して行うだけで、どの点を買うか（選定）には影響しない。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

METHODS = ("uniform", "payout_equal", "prob", "group")


@dataclass
class StakingParams:
    method: str = "uniform"
    total: int = 3000
    unit: int = 100
    min_per_point: int = 100
    max_share: float = 0.30      # 1点の上限（total 比）
    prob_power: float = 1.0      # prob 方式の指数
    main_stake: int = 100        # group 方式（unit の倍数。10×main + 5×hole = total となる組合せは 200/200, 100/400 のみ）
    hole_stake: int = 400        # group 方式

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict | None) -> "StakingParams":
        d = d or {}
        return StakingParams(**{k: v for k, v in d.items() if k in StakingParams().__dict__})


def _largest_remainder(weights: np.ndarray, n_units: int) -> np.ndarray:
    """重みに比例して n_units 個の単位を整数配分（最大剰余法）。"""
    w = np.clip(np.asarray(weights, dtype=float), 0, None)
    if w.sum() <= 0 or n_units <= 0:
        return np.zeros(len(w), dtype=int)
    raw = w / w.sum() * n_units
    base = np.floor(raw).astype(int)
    rem = n_units - base.sum()
    if rem > 0:
        order = np.argsort(-(raw - base), kind="stable")
        base[order[:rem]] += 1
    return base


def allocate(sel_idx: list[int], main_idx: list[int], p: np.ndarray, odds: np.ndarray, prm: StakingParams) -> list[int]:
    """sel_idx（買い目の組番号）ごとの賭け金（円）。順序は sel_idx と同じ。合計は prm.total。"""
    n = len(sel_idx)
    if n == 0:
        return []
    unit = prm.unit
    total_units = prm.total // unit
    min_units = prm.min_per_point // unit
    max_units = max(min_units, int(np.floor(prm.max_share * prm.total / unit)))
    if prm.method == "uniform" or n * min_units >= total_units:
        return [int(x) * unit for x in _largest_remainder(np.ones(n), total_units)]
    if prm.method == "group":
        main_set = set(main_idx)
        ms = max(unit, (int(prm.main_stake) // unit) * unit)
        hs = max(unit, (int(prm.hole_stake) // unit) * unit)
        st = [ms if i in main_set else hs for i in sel_idx]
        diff = prm.total - sum(st)
        # 合計が合わない設定は本線の上位から unit 刻みで吸収（購入単位を崩さない）
        order = [j for j, i in enumerate(sel_idx) if i in main_set] or list(range(n))
        j = 0
        while diff != 0 and j < 10 * n:
            k = order[j % len(order)]
            if diff > 0:
                st[k] += unit; diff -= unit
            elif st[k] > unit:
                st[k] -= unit; diff += unit
            j += 1
        return [int(x) for x in st]
    if prm.method == "payout_equal":
        o = np.array([odds[i] for i in sel_idx], dtype=float)
        med = np.nanmedian(o) if np.isfinite(o).any() else 30.0
        o = np.where(np.isfinite(o) & (o > 0), o, med)
        w = 1.0 / o
    elif prm.method == "prob":
        w = np.array([max(float(p[i]), 1e-9) for i in sel_idx]) ** prm.prob_power
    else:
        raise ValueError(f"unknown staking method: {prm.method}")
    # 下限を先に配り、残りを重みで配分。上限超過分は再配分
    alloc = np.full(n, min_units, dtype=int)
    free = total_units - alloc.sum()
    w_eff = w.copy()
    for _ in range(n + 1):
        add = _largest_remainder(w_eff, free)
        cand = alloc + add
        over = cand > max_units
        if not over.any():
            alloc = cand
            break
        # 上限に達した点は固定し、残りを再配分
        alloc = np.where(over, max_units, alloc)
        w_eff = np.where(over, 0.0, w_eff)
        free = total_units - alloc.sum()
        if free <= 0 or w_eff.sum() <= 0:
            break
    # 端数調整（合計を必ず total に）
    diff = total_units - alloc.sum()
    if diff != 0:
        order = np.argsort(-w, kind="stable")
        j = 0
        while diff != 0 and j < 10 * n:
            k = order[j % n]
            if diff > 0 and alloc[k] < max_units:
                alloc[k] += 1
                diff -= 1
            elif diff < 0 and alloc[k] > min_units:
                alloc[k] -= 1
                diff += 1
            j += 1
    return [int(x) * unit for x in alloc]
