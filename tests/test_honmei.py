"""本命10点・×1.5保証モードの単体テスト（`min15.md` / `online5.md`）。

守りたい性質:
  1. 成立したら **どの点が当たっても 払戻 ≥ 投資 × 倍率**。
  2. 成立条件は Σ(1/オッズ) ≤ 1/倍率。これを満たさないレースは必ず見送り。
  3. 支出が上限を超えるなら見送り（点数を削らない。10点固定を崩さない）。
  4. しきい値は「残り枠 ÷ この先に残る本数」の分位点。枠が無ければ必ず見送り。
  5. 枠の残りは「今日ここまでに買った本数」から数える（先読みしない）。
"""
from __future__ import annotations

import importlib
from datetime import date, datetime

import numpy as np
import pytest

from boatlab.model.modes import ModeParams, honmei_threshold, min_guarantee_F, select_honmei
from boatlab.model.trifecta import PERM_LABELS

PRM = ModeParams()


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setenv("BOATLAB_DATABASE_URL", f"sqlite:///{tmp_path}/t.db")
    from boatlab import config as cfg
    importlib.reload(cfg)
    from boatlab.store import db as dbm
    importlib.reload(dbm)
    dbm.init_db()
    return dbm


def _odds120(head):
    o = np.full(120, 500.0)
    for i, v in enumerate(head):
        o[i] = v
    return o


def test_guarantee_holds():
    odds = np.array([6.0, 12.0, 18.0, 30.0, 40.0, 55.0, 70.0, 90.0, 130.0, 200.0])
    F = min_guarantee_F(odds, 1.5, 10000)
    st = np.ceil(F / odds / 100) * 100
    assert st.sum() * 1.5 <= F                      # 投資×1.5 ≤ 保証払戻
    assert (st * odds >= F - 1e-9).all()            # どの点でも払戻 ≥ 保証払戻


def test_infeasible_when_favourite_too_short():
    """本命が堅い（10点で市場確率の50%超）レースは、どんな金額でも×1.5にできない。"""
    odds = np.array([1.8] + [6.0] * 9)
    assert float(np.sum(1.0 / odds)) > 1 / 1.5
    assert min_guarantee_F(odds, 1.5, 10_000_000) is None


def test_cap_is_respected():
    odds = np.array([4.0, 8.0, 12.0, 20.0, 25.0, 30.0, 45.0, 60.0, 80.0, 120.0])
    assert min_guarantee_F(odds, 1.5, 3000) is None         # 上限が小さいと成立しない
    assert min_guarantee_F(odds, 1.5, 50000) is not None     # 上げれば成立する


@pytest.mark.parametrize("mult", [1.0, 1.2, 1.5, 2.0])
def test_multiple_is_configurable(mult):
    odds = np.array([8.0, 14.0, 22.0, 33.0, 44.0, 60.0, 80.0, 110.0, 160.0, 260.0])
    F = min_guarantee_F(odds, mult, 100000)
    st = np.ceil(F / odds / 100) * 100
    assert st.sum() * mult <= F


def test_threshold_is_monotone_and_blocks_when_full():
    assert honmei_threshold(0, 50, PRM) == float("inf")            # 枠なし → 絶対に買わない
    assert honmei_threshold(1, 60, PRM) > honmei_threshold(5, 60, PRM)   # 枠が少ないほど厳しい
    assert honmei_threshold(5, 60, PRM) > honmei_threshold(5, 10, PRM)   # 残りが少ないほど緩い


def test_select_reasons():
    odds = _odds120([6.0, 12.0, 18.0, 30.0, 40.0, 55.0, 70.0, 90.0, 130.0, 200.0])
    pts = list(range(10))
    hi = {PERM_LABELS[j]: 0.06 for j in pts}       # 合計0.60
    lo = {PERM_LABELS[j]: 0.02 for j in pts}       # 合計0.20
    assert select_honmei(pts, odds, hi, 5, 60, PRM)["fired"] is True
    assert select_honmei(pts, odds, lo, 5, 60, PRM)["reason"] == "confidence_low"
    assert select_honmei(pts, odds, hi, 0, 60, PRM)["reason"] == "slots_full"
    assert select_honmei(pts[:4], odds, hi, 5, 60, PRM)["reason"] == "too_few_points"
    assert select_honmei(pts, np.full(120, np.nan), hi, 5, 60, PRM)["reason"] == "odds_missing"
    assert select_honmei(pts, _odds120([1.8] + [6.0] * 9), hi, 5, 60, PRM)["reason"] == "no_guarantee"
    off = ModeParams(honmei_enabled=False)
    assert select_honmei(pts, odds, hi, 5, 60, off)["reason"] == "disabled"


def test_slots_are_counted_from_today_only(db):
    """残り枠は「今日すでに買った本数」から数える。前日の記録や skip は数えない。"""
    from boatlab.ops import daily as dl
    from boatlab.store import models as mm
    from boatlab.util import now_jst
    d = date(2026, 9, 13)
    with db.session_scope() as s:
        s.add(mm.SettingsVersion(id=1, points=15, stake_per_point=100, extra={}))
        s.add(mm.ModelVersion(version="t", feature_set_version="fs", selection_version="sel", params={}, status="active"))
        for k in range(4):
            s.add(mm.Race(id=202609130101 + k, race_date=d, stadium_code=1, race_no=k + 1, source="t",
                          closed_at=datetime(2026, 9, 13, 11 + k, 0)))
        s.add(mm.Race(id=202609120101, race_date=date(2026, 9, 12), stadium_code=1, race_no=1, source="t",
                      closed_at=datetime(2026, 9, 12, 11, 0)))
    with db.session_scope() as s:
        for rid, dec in ((202609130101, "buy"), (202609130102, "skip"), (202609120101, "buy")):
            s.add(mm.Prediction(race_id=rid, model_version="t", settings_id=1, stage="final", role="honmei",
                                created_at=now_jst(), asof_ts=datetime(2026, 9, 13, 10, 0),
                                post_time_at_pred=datetime(2026, 9, 13, 11, 0), completeness=1.0, decision=dec,
                                features_used={}, boat_eval={}, probs={}, odds_used={}, ev={}, confidence=0.5,
                                rationale={}, input_hash="h", expected_return=0.0, flags={"mode": "honmei"}))
    with db.session_scope() as s:
        r = s.get(mm.Race, 202609130103)
        slots, races = dl.honmei_context(s, r, PRM)
    assert slots == PRM.honmei_slots - 1            # 今日の buy 1件だけ（skip と前日は数えない）
    assert races == max(1, round(2 * PRM.honmei_feasible_rate))   # この先2レース × 成立率
