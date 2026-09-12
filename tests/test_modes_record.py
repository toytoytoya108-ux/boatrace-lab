"""3モードの記録（追記専用）と採点。"""
import importlib
from datetime import date, datetime, timedelta

import numpy as np
import pytest

from boatlab.model.trifecta import PERM_LABELS, PERMS


def _odds_rough():
    p = np.concatenate([np.full(20, 0.6 / 20), np.full(100, 0.4 / 100)])
    p = p / p.sum()
    return 0.75 / p


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setenv("BOATLAB_DATABASE_URL", f"sqlite:///{tmp_path}/t.db")
    from boatlab import config as cfg
    importlib.reload(cfg)
    from boatlab.store import db as dbm
    importlib.reload(dbm)
    dbm.init_db()
    return dbm


def _fake_o(odds, estimated=False, confidence=0.8):
    q = 1.0 / odds
    q = q / q.sum()
    order = np.argsort(-q)
    return dict(race_id=202609120101, completeness=1.0,
                flags={"odds_estimated": estimated}, boat_eval={}, probs={PERM_LABELS[i]: float(q[i]) for i in range(120)},
                odds_used={PERM_LABELS[i]: float(odds[i]) for i in range(120)}, ev={}, confidence=confidence,
                rationale={"summary": "x"}, input_hash="h", odds_source=("estimated" if estimated else "real"),
                selections=[dict(combo=PERM_LABELS[int(order[k])], rank=k + 1, kind=("main" if k < 10 else "hole"),
                                 stake=100, prob=float(q[order[k]]), odds=float(odds[order[k]]), ev=None) for k in range(15)])


def test_record_and_score_modes(db):
    from boatlab.model.modes import ModeParams
    from boatlab.ops import daily as dl
    from boatlab.store import models as mm
    now = datetime(2026, 9, 12, 12, 0)
    with db.session_scope() as s:
        s.add(mm.ModelVersion(version="t", feature_set_version="fs", selection_version="sel", params={}, status="active"))
        s.add(mm.SettingsVersion(id=1, points=15, stake_per_point=100, extra={}))
        s.add(mm.Race(id=202609120101, race_date=date(2026, 9, 12), stadium_code=1, race_no=1, source="t",
                      closed_at=now + timedelta(minutes=10)))
    odds = _odds_rough()
    o = _fake_o(odds)
    with db.session_scope() as s:
        r = s.get(mm.Race, 202609120101)
        dl._record_modes(s, o, r, "t", 1, now, {}, ModeParams(), {"ana": set(), "katai": set(), "place": set()})
    with db.session_scope() as s:
        rows = {p.role: p for p in s.query(mm.Prediction).all()}
        assert set(rows) == {"ana", "katai", "place"}
        assert rows["ana"].decision == "buy" and rows["ana"].flags["n_points"] == 21
        assert rows["katai"].decision == "buy" and rows["katai"].flags["min_payout"] >= 3000
        assert rows["place"].decision == "skip"                       # 荒れる分布なので複勝・単勝は見送り
        assert s.query(mm.PredictionSelection).filter_by(prediction_id=rows["place"].id).count() == 2  # 参考記録
        sels = s.query(mm.PredictionSelection).filter_by(prediction_id=rows["katai"].id).all()
        assert sum(x.stake for x in sels) <= 3000 and all(x.kind == "katai" for x in sels)
    # 追記専用: 更新は拒否される
    with pytest.raises(Exception):
        with db.session_scope() as s:
            s.query(mm.Prediction).filter_by(role="ana").update({"decision": "skip"})
    # 推定オッズなら市場ベースの2モードは skip（odds_estimated）で記録される
    with db.session_scope() as s:
        s.add(mm.Race(id=202609120102, race_date=date(2026, 9, 12), stadium_code=1, race_no=2, source="t",
                      closed_at=now + timedelta(minutes=30)))
    o2 = {**_fake_o(odds, estimated=True), "race_id": 202609120102}
    with db.session_scope() as s:
        r = s.get(mm.Race, 202609120102)
        dl._record_modes(s, o2, r, "t", 1, now, {}, ModeParams(), {"ana": set(), "katai": set(), "place": set()})
    with db.session_scope() as s:
        p = s.query(mm.Prediction).filter_by(race_id=202609120102, role="ana").one()
        assert p.decision == "skip" and p.skip_reason == "odds_estimated"
        assert s.query(mm.PredictionSelection).filter_by(prediction_id=p.id).count() == 0   # 推定オッズでは何も出さない


def test_score_place():
    from boatlab.ops.daily import score_place

    class S:  # PredictionSelection の最小形
        def __init__(self, combo, kind, stake):
            self.combo, self.kind, self.stake = combo, kind, stake

    payouts = {"place": [{"combination": "1", "amount": 110}, {"combination": "3", "amount": 250}],
               "win": [{"combination": "1", "amount": 150}]}
    sc = score_place([S("複1", "fukusho", 100), S("単1", "tansho", 100)], payouts, [])
    assert sc["valid"] and sc["hit"] and sc["stake_total"] == 200 and sc["payout_total"] == 260 and sc["pnl"] == 60
    sc = score_place([S("複4", "fukusho", 100)], payouts, [])
    assert sc["valid"] and not sc["hit"] and sc["pnl"] == -100
    sc = score_place([S("複1", "fukusho", 100)], payouts, [1])           # 返還
    assert sc["stake_total"] == 0 and sc["refunded_points"] == 1
    assert not score_place([S("複1", "fukusho", 100)], None, [])["valid"]
