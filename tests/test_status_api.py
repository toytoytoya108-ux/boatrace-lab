"""/api/status: 「上流が遅れている」のか「取り込みが壊れている」のかを画面で判別できること。

2026-09-11 に、レース一覧が「次 12:44締切」のまま止まって見える事象があった。実際は上流が
結果を出しておらず、こちらは設計どおりスキップしていただけだったが、画面からは判別できなかった。
"""
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
    appmod._upstream_cache.update(at=None, value=None)
    return TestClient(appmod.app), dbm, appmod


def _seed(dbm, n_races=12, n_results=5, closed_past=True):
    from boatlab.store import models as mm
    from boatlab.util import now_jst
    day = now_jst().date()
    base = int(day.strftime("%Y%m%d")) * 10000 + 100
    t0 = now_jst().replace(tzinfo=None) - timedelta(hours=3 if closed_past else -3)
    with dbm.session_scope() as s:
        for i in range(1, n_races + 1):
            s.add(mm.Race(id=base + i, race_date=day, stadium_code=1, race_no=i, source="t",
                          closed_at=t0 + timedelta(minutes=5 * i)))
    with dbm.session_scope() as s:
        for i in range(1, n_results + 1):
            s.add(mm.Result(race_id=base + i, trifecta="1-2-3", trifecta_payout=2500.0, source="t",
                            payouts={"win": [{"combination": "1", "amount": 130}]}))
    return day, base


def test_upstream_lag_is_reported_as_waiting(client, monkeypatch):
    c, dbm, appmod = client
    _seed(dbm)
    # 上流にも結果が無い＝こちらは正常。「上流待ち」と出る
    monkeypatch.setattr(appmod, "_upstream_results", lambda day: {"races": 12, "filled": 5})
    t = c.get("/api/status").json()["today"]
    assert t["verdict"] == "upstream" and t["waiting"] == 7
    assert "上流" in t["note"]


def test_ingest_problem_is_reported_as_error(client, monkeypatch):
    c, dbm, appmod = client
    _seed(dbm)
    # 上流には12件あるのにDBは5件＝こちらの取り込みがおかしい
    monkeypatch.setattr(appmod, "_upstream_results", lambda day: {"races": 12, "filled": 12})
    t = c.get("/api/status").json()["today"]
    assert t["verdict"] == "ng" and "取り込み" in t["note"]


def test_all_results_in_is_ok(client, monkeypatch):
    c, dbm, appmod = client
    _seed(dbm, n_results=12)
    called = []
    monkeypatch.setattr(appmod, "_upstream_results", lambda day: called.append(1) or {"races": 12, "filled": 12})
    t = c.get("/api/status").json()["today"]
    assert t["verdict"] == "ok" and not called      # 待ちが無ければ上流を叩かない


def test_upstream_unreachable_is_unknown_not_error(client, monkeypatch):
    c, dbm, appmod = client
    _seed(dbm)
    monkeypatch.setattr(appmod, "_upstream_results", lambda day: {"error": "ConnectError()"})
    t = c.get("/api/status").json()["today"]
    assert t["verdict"] == "unknown"


def test_missing_official_odds_is_flagged(client, monkeypatch):
    c, dbm, appmod = client
    day, base = _seed(dbm)
    monkeypatch.setattr(appmod, "_upstream_results", lambda day: {"races": 12, "filled": 5})
    o = c.get("/api/status").json()["odds"]
    assert o["verdict"] == "ng" and "3連単" in o["note"]     # 1件も取れていない
    from boatlab.store import models as mm
    from boatlab.util import now_jst
    with dbm.session_scope() as s:
        for i in range(1, 6):
            for bt, v in (("3t", {"1-2-3": 4.6}), ("win", {"1": 1.5})):
                s.add(mm.OddsSnapshot(race_id=base + i, bet_type=bt, captured_at=now_jst(),
                                      source="official_web", odds=v))
    o = c.get("/api/status").json()["odds"]
    assert o["verdict"] == "ok" and o["trifecta"] == 5 and o["win"] == 5


def test_stale_running_job_becomes_stuck(client):
    c, dbm, appmod = client
    _seed(dbm)
    from boatlab.store import models as mm
    from boatlab.util import now_jst
    with dbm.session_scope() as s:                      # 落ちたまま残った古いジョブ
        s.add(mm.JobRun(job="morning", started_at=now_jst() - timedelta(hours=7), finished_at=None, ok=None))
    f = c.get("/api/status").json()["freshness"]
    assert f["running"] is None and f["stuck"] and f["stuck"]["job"] == "morning"
    with dbm.session_scope() as s:                      # 本当に実行中のものは running に出る
        s.add(mm.JobRun(job="intraday", started_at=now_jst() - timedelta(minutes=1), finished_at=None, ok=None))
    f = c.get("/api/status").json()["freshness"]
    assert f["running"] and f["running"]["job"] == "intraday"
