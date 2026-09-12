"""/api/modes と /api/today?mode=ana|katai|place。"""
from datetime import date, datetime, timedelta

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("BOATLAB_DATABASE_URL", f"sqlite:///{tmp_path}/t.db")
    monkeypatch.setenv("BOATLAB_DATA_DIR", str(tmp_path))
    import importlib

    from boatlab import config as cfg
    importlib.reload(cfg)
    from boatlab.store import db as dbm
    importlib.reload(dbm)
    from boatlab.api import app as appmod
    importlib.reload(appmod)
    dbm.init_db()
    return TestClient(appmod.app), dbm, appmod


def _auth(appmod):
    return {"Authorization": f"Bearer {appmod.PASSWORD}"} if getattr(appmod, "PASSWORD", None) else {}


def test_modes_endpoint(client):
    tc, dbm, appmod = client
    from boatlab.store import models as mm
    from boatlab.util import now_jst
    day = now_jst().date()
    rid = int(day.strftime("%Y%m%d")) * 10000 + 101
    now = now_jst().replace(tzinfo=None)
    with dbm.session_scope() as s:
        s.add(mm.ModelVersion(version="t", feature_set_version="fs", selection_version="sel", params={}, status="active"))
        s.add(mm.SettingsVersion(id=1, points=15, stake_per_point=100, extra={}))
        # created_at はトリガで「今」を強制されるので、締切と結果取得はその後の時刻にする
        s.add(mm.Race(id=rid, race_date=day, stadium_code=1, race_no=1, source="t", closed_at=now + timedelta(minutes=30)))
        s.add(mm.Result(race_id=rid, trifecta="1-2-3", trifecta_payout=650,
                        payouts={"place": [{"combination": "1", "amount": 110}], "win": [{"combination": "1", "amount": 150}]},
                        refunds=[], is_irregular=False, source="t", fetched_at=now + timedelta(minutes=60)))
    common = dict(race_id=rid, model_version="t", settings_id=1, stage="final", created_at=now,
                  asof_ts=now, post_time_at_pred=now + timedelta(minutes=30), features_used=None,
                  completeness=1.0, boat_eval={}, probs={}, odds_used={}, ev={}, confidence=0.8, expected_return=0.0,
                  rationale={"summary": "x"}, rationale_text="x", input_hash="h")
    with dbm.session_scope() as s:
        p = mm.Prediction(**common, role="place", decision="buy", flags={"mode": "place"})
        s.add(p); s.flush()
        s.add(mm.PredictionSelection(prediction_id=p.id, combo="1", rank=1, kind="fukusho", stake=100, prob=0.95))
        s.add(mm.Prediction(**common, role="ana", decision="skip", skip_reason="q_man_low", flags={"mode": "ana"}))
        s.add(mm.Prediction(**common, role="katai", decision="skip", skip_reason="confidence_low", flags={"mode": "katai"}))
    from boatlab.ops import daily
    assert daily.score_pending()["scored"] == 3
    h = _auth(appmod)
    r = tc.get("/api/modes", headers=h)
    assert r.status_code == 200, r.text
    j = r.json()
    assert set(j["modes"]) == {"ana", "katai", "place"} and j["params"]["fukusho_q_min"] == 0.90
    pl = j["modes"]["place"]
    assert pl["today"]["n_fired"] == 1 and pl["today"]["hits"] == 1 and pl["today"]["pnl"] == 10
    assert pl["cumulative"]["n"] == 1 and pl["cumulative"]["roi"] == pytest.approx(1.1)
    assert "fukusho" in pl["measured"] and pl["expected_monthly_loss"]["fukusho"] >= 0
    assert j["modes"]["ana"]["today"]["n_fired"] == 0 and j["modes"]["ana"]["today"]["n_recorded"] == 1
    # /api/today はモードごとに role を切り替える
    t = tc.get("/api/today?mode=place", headers=h).json()
    assert t["n_buy"] == 1 and t["races"][0]["pred_role"] == "place"
    t = tc.get("/api/today?mode=ana", headers=h).json()
    assert t["n_skip"] == 1 and t["races"][0]["skip_reason"] == "q_man_low"
