"""期待値1以上モード（ev1、2026-09-21）。

実績規則: 1号艇のモーター2連率1位＋展示タイム1位＋上位8場 → 複勝1点（オッズ不要）。
較正期待値: p ∝ モデル^0.15 × 市場^0.926、期待値 = p × 締切前オッズ × 0.90。1.0 以上の目だけ。
"""
from __future__ import annotations

import importlib
from datetime import date, datetime, timedelta

import numpy as np
import pytest

from boatlab.model.trifecta import PERM_LABELS


def _be(motor1=60.0, ext_rank1=1, others_motor=40.0):
    be = {}
    for k in range(1, 7):
        be[str(k)] = dict(motor_rate2=(motor1 if k == 1 else others_motor), exhibition_rank=(ext_rank1 if k == 1 else 3),
                          exhibition_time=6.7, klass="A1", name=f"b{k}")
    return be


def _odds(seed=0):
    rng = np.random.default_rng(seed)
    q = rng.dirichlet(np.ones(120) * 0.3)
    return 0.75 / q


def test_rule_fires_only_when_all_three_conditions_hold():
    from boatlab.model.modes import ModeParams, select_ev1
    prm = ModeParams()
    p = np.ones(120) / 120
    ok = select_ev1(_be(), 3, None, p, prm)            # 江戸川
    assert ok["rule"]["fired"] and ok["fired"] and ok["rule"]["motor_rank"] == 1 and ok["rule"]["stadium_ok"]
    assert ok["cal"]["reason"] == "odds_missing"       # オッズ無しでも規則は成立する
    bad_st = select_ev1(_be(), 1, None, p, prm)        # 桐生は対象外
    assert not bad_st["fired"] and bad_st["reason"] == "rule_not_met" and not bad_st["rule"]["stadium_ok"]
    bad_ext = select_ev1(_be(ext_rank1=2), 3, None, p, prm)
    assert not bad_ext["fired"] and bad_ext["reason"] == "rule_not_met"
    bad_motor = select_ev1(_be(motor1=40.0, others_motor=50.0), 3, None, p, prm)
    assert not bad_motor["fired"] and bad_motor["rule"]["motor_rank"] == 6
    tie = select_ev1(_be(motor1=50.0, others_motor=50.0), 3, None, p, prm)   # 同率は1位扱い（method='min'）
    assert tie["rule"]["motor_rank"] == 1 and tie["fired"]


def test_rule_reports_missing_data():
    from boatlab.model.modes import ModeParams, select_ev1
    prm = ModeParams()
    p = np.ones(120) / 120
    assert select_ev1({}, 3, None, p, prm)["reason"] == "boat_eval_missing"
    be = _be(); be["1"]["exhibition_rank"] = None
    assert select_ev1(be, 3, None, p, prm)["reason"] == "preview_missing"
    be = _be()
    for k in be:
        be[k]["motor_rate2"] = None
    assert select_ev1(be, 3, None, p, prm)["reason"] == "entries_missing"
    assert select_ev1(_be(), 3, None, p, ModeParams(ev1_enabled=False))["reason"] == "disabled"


def test_calibrated_probs_is_market_dominated_and_normalised():
    from boatlab.model.modes import calibrated_probs, market_probs
    odds = _odds(1)
    q = market_probs(odds)
    p = np.ones(120) / 120                             # モデルが何も知らない → 市場の形に近い
    pc = calibrated_probs(p, q)
    assert abs(pc.sum() - 1) < 1e-9
    assert np.corrcoef(np.log(pc), np.log(q))[0, 1] > 0.99
    # a=0.15 × b=0.926: 市場と完全一致のモデルなら p_cal ∝ q^1.076（合計1に正規化）
    pc2 = calibrated_probs(q, q)
    ref = q ** 1.076; ref /= ref.sum()
    assert np.allclose(pc2, ref, rtol=1e-6)


def test_cal_fires_only_above_threshold_and_records_reference_point():
    from boatlab.model.modes import ModeParams, market_probs, select_ev1
    prm = ModeParams()
    odds = _odds(2)
    q = market_probs(odds)
    # モデル = 市場 のとき、期待値 = q^1.076/Σ × オッズ × 0.90 ≈ 0.75×0.90×… で 1.0 を超えない
    r = select_ev1(_be(ext_rank1=2), 3, odds, q, prm)
    assert not r["cal"]["fired"] and r["cal"]["reason"] == "ev_below_min"
    assert len(r["cal"]["points"]) == 1 and r["cal"]["max_ev"] < 1.0      # 参考1点
    # モデルがある目を強く確信 → その目の較正期待値が 1.0 を超えて発火
    p = q.copy(); j = int(np.argmax(odds * q)); p[j] = 0.9; p /= p.sum()
    r2 = select_ev1(_be(ext_rank1=2), 3, odds, p, prm)
    ev = r2["cal"]["ev"]
    if r2["cal"]["fired"]:
        assert all(e >= prm.ev1_ev_min for e in ev) and len(ev) <= prm.ev1_points_max and r2["fired"]
    # 下落率を 1.0 にすると期待値は 1/0.9 倍
    r3 = select_ev1(_be(ext_rank1=2), 3, odds, p, ModeParams(ev1_odds_ratio=1.0))
    assert r3["cal"]["max_ev"] == pytest.approx(r2["cal"]["max_ev"] / 0.90, rel=1e-6)


def test_params_roundtrip_keeps_ev1_fields():
    from boatlab.model.modes import ModeParams
    d = ModeParams(ev1_ev_min=1.2, ev1_stadiums=[3, 24]).to_dict()
    p = ModeParams.from_dict(d)
    assert p.ev1_ev_min == 1.2 and list(p.ev1_stadiums) == [3, 24] and p.ev1_odds_ratio == 0.90


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setenv("BOATLAB_DATABASE_URL", f"sqlite:///{tmp_path}/t.db")
    from boatlab import config as cfg
    importlib.reload(cfg)
    from boatlab.store import db as dbm
    importlib.reload(dbm)
    dbm.init_db()
    return dbm


def _fake_o(odds, be, estimated=False):
    q = 1.0 / odds
    q = q / q.sum()
    order = np.argsort(-q)
    return dict(race_id=202609210301, completeness=1.0, flags={"odds_estimated": estimated}, boat_eval=be,
                probs={PERM_LABELS[i]: float(q[i]) for i in range(120)},
                odds_used={PERM_LABELS[i]: float(odds[i]) for i in range(120)}, ev={}, confidence=0.8,
                rationale={"summary": "x"}, input_hash="h", odds_source=("estimated" if estimated else "real"),
                selections=[dict(combo=PERM_LABELS[int(order[k])], rank=k + 1, kind=("main" if k < 10 else "hole"),
                                 stake=100, prob=float(q[order[k]]), odds=float(odds[order[k]]), ev=None) for k in range(15)])


def test_record_and_score_ev1(db):
    from boatlab.model.modes import ModeParams
    from boatlab.ops import daily as dl
    from boatlab.store import models as mm
    from boatlab.util import now_jst
    now = now_jst()
    rid = 202609210301
    with db.session_scope() as s:
        s.add(mm.ModelVersion(version="t", feature_set_version="fs", selection_version="sel", params={}, status="active"))
        s.add(mm.SettingsVersion(id=1, points=15, stake_per_point=100, extra={}))
        s.add(mm.Race(id=rid, race_date=date(2026, 9, 21), stadium_code=3, race_no=1, source="t",
                      closed_at=now + timedelta(minutes=10)))
    odds = _odds(3)
    o = _fake_o(odds, _be())
    with db.session_scope() as s:
        r = s.get(mm.Race, rid)
        dl._record_modes(s, o, r, "t", 1, now, {}, ModeParams(), {k: set() for k in dl.MODE_ROLES})
    with db.session_scope() as s:
        p = s.query(mm.Prediction).filter_by(role="ev1").one()
        assert p.decision == "buy" and p.flags["mode"] == "ev1" and p.flags["rule"]["fired"]
        sels = s.query(mm.PredictionSelection).filter_by(prediction_id=p.id).all()
        kinds = {x.kind for x in sels}
        assert "fukusho" in kinds and next(x for x in sels if x.kind == "fukusho").combo == "複1"
        # 較正は発火せず、規則が発火した行には参考点を混ぜない（投資に数えられるため）
        assert p.flags["cal"]["reason"] == "ev_below_min" and sum(1 for x in sels if x.kind == "ev3t") == 0
        assert p.flags["stake_total"] == 100 and len(sels) == 1
        # 結果: 1号艇が2着以内（複勝120円）、3連単は外れ
        s.add(mm.Result(race_id=rid, trifecta="1-2-3", trifecta_payout=1000, fetched_at=now + timedelta(hours=1),
                        payouts={"trifecta": [{"combination": "1-2-3", "amount": 1000}],
                                 "place": [{"combination": "1", "amount": 120}, {"combination": "2", "amount": 150}],
                                 "win": [{"combination": "1", "amount": 200}]}, refunds=[]))
    dl.score_pending()      # 締切(now+10分) < 結果取得(now+1h) なので有効
    with db.session_scope() as s:
        p = s.query(mm.Prediction).filter_by(role="ev1").one()
        sc = s.get(mm.Scoring, p.id)
        assert sc.valid and sc.hit and sc.hit_kind == "fukusho"
        assert sc.stake_total == 100 and sc.payout_total == 120 and sc.pnl == 20


def test_skip_row_keeps_reference_point_for_virtual_scoring(db):
    from boatlab.model.modes import ModeParams
    from boatlab.ops import daily as dl
    from boatlab.store import models as mm
    from boatlab.util import now_jst
    now = now_jst()
    rid = 202609210301
    with db.session_scope() as s:
        s.add(mm.ModelVersion(version="t", feature_set_version="fs", selection_version="sel", params={}, status="active"))
        s.add(mm.SettingsVersion(id=1, points=15, stake_per_point=100, extra={}))
        s.add(mm.Race(id=rid, race_date=date(2026, 9, 21), stadium_code=1, race_no=1, source="t",
                      closed_at=now + timedelta(minutes=10)))
    o = _fake_o(_odds(5), _be())          # 桐生 → 規則不成立、モデル=市場 → 較正も1.0未満
    with db.session_scope() as s:
        r = s.get(mm.Race, rid)
        dl._record_modes(s, o, r, "t", 1, now, {}, ModeParams(), {k: set() for k in dl.MODE_ROLES})
    with db.session_scope() as s:
        p = s.query(mm.Prediction).filter_by(role="ev1").one()
        sels = s.query(mm.PredictionSelection).filter_by(prediction_id=p.id).all()
        assert p.decision == "skip" and p.skip_reason == "rule_not_met" and [x.kind for x in sels] == ["ev3t"]
        assert p.flags["cal"]["max_ev"] < 1.0


def test_ev1_rule_records_on_estimated_odds(db):
    from boatlab.model.modes import ModeParams
    from boatlab.ops import daily as dl
    from boatlab.store import models as mm
    from boatlab.util import now_jst
    now = now_jst()
    rid = 202609210301
    with db.session_scope() as s:
        s.add(mm.ModelVersion(version="t", feature_set_version="fs", selection_version="sel", params={}, status="active"))
        s.add(mm.SettingsVersion(id=1, points=15, stake_per_point=100, extra={}))
        s.add(mm.Race(id=rid, race_date=date(2026, 9, 21), stadium_code=24, race_no=1, source="t",
                      closed_at=now + timedelta(minutes=10)))
    o = _fake_o(_odds(4), _be(), estimated=True)
    with db.session_scope() as s:
        r = s.get(mm.Race, rid)
        dl._record_modes(s, o, r, "t", 1, now, {}, ModeParams(), {k: set() for k in dl.MODE_ROLES})
    with db.session_scope() as s:
        p = s.query(mm.Prediction).filter_by(role="ev1").one()
        assert p.decision == "buy" and p.flags["cal"]["reason"] == "odds_missing"
        sels = s.query(mm.PredictionSelection).filter_by(prediction_id=p.id).all()
        assert [x.kind for x in sels] == ["fukusho"]
        ana = s.query(mm.Prediction).filter_by(role="ana").one()
        assert ana.decision == "skip"                   # 市場ベースのモードは推定オッズでは見送りのまま
