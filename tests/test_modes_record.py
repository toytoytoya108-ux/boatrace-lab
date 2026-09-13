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


def test_record_late_modes(db):
    """締切2〜4分前の取り直しで ana_late/place_late が追記され、8分前との重なりが記録される。本体が無ければ何もしない。"""
    from boatlab.model.modes import ModeParams
    from boatlab.model.trifecta import PERM_LABELS
    from boatlab.ops import daily as dl
    from boatlab.store import models as mm
    from boatlab.util import now_jst
    now = now_jst()                       # created_at はトリガで「今」を強制されるので実時刻を使う
    rid = int(now.strftime("%Y%m%d")) * 10000 + 401
    with db.session_scope() as s:
        s.add(mm.ModelVersion(version="t", feature_set_version="fs", selection_version="sel", params={}, status="active"))
        s.add(mm.SettingsVersion(id=1, points=15, stake_per_point=100, extra={}))
        s.add(mm.Race(id=rid, race_date=now.date(), stadium_code=4, race_no=1, source="t", closed_at=now + timedelta(minutes=10)))
    # 同順位が無い分布にする（順位の入れ替えを正確に数えるため）
    pp = np.concatenate([np.full(20, 0.6 / 20) - np.arange(20) * 1e-5, np.full(100, 0.4 / 100) - np.arange(100) * 1e-6])
    pp = pp / pp.sum(); odds = 0.75 / pp
    o = {**_fake_o(odds), "race_id": rid}
    # 本体が無いうちは何も書かない
    assert dl.record_late_modes(rid, {PERM_LABELS[i]: float(odds[i]) for i in range(120)}, now, 3.0) == 0
    with db.session_scope() as s:
        r = s.get(mm.Race, rid)
        s.add(mm.Prediction(race_id=rid, model_version="t", settings_id=1, stage="final", role="active", created_at=now, asof_ts=now,
                            post_time_at_pred=r.closed_at, features_used=None, completeness=1.0, boat_eval={}, probs=o["probs"],
                            odds_used=o["odds_used"], ev={}, confidence=0.8, expected_return=0.0, decision="buy", rationale={"summary": "x"},
                            rationale_text="x", input_hash="h"))
        dl._record_modes(s, o, r, "t", 1, now, {}, ModeParams(), {"ana": set(), "katai": set(), "place": set()})
    # 直前のオッズ: 人気順を少し入れ替える（上位20通りの一部を入れ替え）
    late = {PERM_LABELS[i]: float(odds[i]) for i in range(120)}
    order = np.argsort(1.0 / odds)[::-1]  # 確率の高い順（オッズ低い順）
    a, b = PERM_LABELS[int(order[19])], PERM_LABELS[int(order[45])]
    late[a], late[b] = late[b], late[a]
    assert dl.record_late_modes(rid, late, now + timedelta(minutes=6), 3.5) == 2
    with db.session_scope() as s:
        p = s.query(mm.Prediction).filter_by(race_id=rid, role="ana_late").one()
        assert p.flags["late"] and p.flags["mode"] == "ana" and p.flags["minutes_before"] == 3.5
        assert p.flags["overlap_with_early"] == 20 and p.flags["early_decision"] == "buy"
        assert s.query(mm.PredictionSelection).filter_by(prediction_id=p.id).count() == 21
        assert s.query(mm.Prediction).filter_by(race_id=rid, role="place_late").count() == 1
    # 2回目は書かない（同一レースに1回）
    assert dl.record_late_modes(rid, late, now + timedelta(minutes=7), 2.5) == 0
    # 採点は既存経路に乗る
    with db.session_scope() as s:
        s.add(mm.Result(race_id=rid, trifecta=PERM_LABELS[int(order[25])], trifecta_payout=12000, payouts={"place": [], "win": []},
                        refunds=[], is_irregular=False, source="t", fetched_at=now + timedelta(minutes=30)))
    n = dl.score_pending()["scored"]
    assert n >= 5
    with db.session_scope() as s:
        p = s.query(mm.Prediction).filter_by(race_id=rid, role="ana_late").one()
        sc = s.get(mm.Scoring, p.id)
        assert sc.valid and sc.hit and sc.hit_kind == "ana"
