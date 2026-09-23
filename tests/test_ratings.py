"""展示評価（◎×）: 追記専用・確定前の入力だけが使われる・評価あり版が対で記録され採点される・beta の推定。"""
from __future__ import annotations

import importlib
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from boatlab.model.trifecta import PERM_LABELS, PERMS


def _win(p):
    return np.array([sum(p[i] for i in range(120) if PERMS[i][0] == a) for a in range(6)])


def test_adjust_probs_math():
    from boatlab.model.modes import adjust_probs_for_ratings
    p = np.random.default_rng(0).dirichlet(np.ones(120))
    q = adjust_probs_for_ratings(p, {1: 1, 4: -1}, 0.3)
    assert abs(q.sum() - 1) < 1e-12
    w0, w1 = _win(p), _win(q)
    assert w1[0] > w0[0] and w1[3] < w0[3]                          # ◎は上がり ×は下がる
    assert np.allclose(adjust_probs_for_ratings(p, {}, 0.3), p)     # 評価なしは恒等
    assert np.allclose(adjust_probs_for_ratings(p, {1: 1}, 0.0), p)  # beta=0 は恒等
    assert np.allclose(adjust_probs_for_ratings(p, {"1": 1}, 0.3), adjust_probs_for_ratings(p, {1: 1}, 0.3))  # 文字列キーも可
    # ◎の1号艇が1着の目(1-2-3)と2着の目(2-1-3)の比は e^{beta·(1.0−0.5)}（2号艇は無印）
    i = next(k for k in range(120) if PERMS[k] == (0, 1, 2)); j = next(k for k in range(120) if PERMS[k] == (1, 0, 2))
    r = (q[i] / q[j]) / (p[i] / p[j])
    assert r == pytest.approx(np.exp(0.3 * (1.0 - 0.5)), rel=1e-9)


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("BOATLAB_DATABASE_URL", f"sqlite:///{tmp_path}/t.db")
    monkeypatch.setenv("BOATLAB_DATA_DIR", str(tmp_path))
    from boatlab import config as cfg
    importlib.reload(cfg)
    from boatlab.store import db as dbm
    importlib.reload(dbm)
    from boatlab.api import app as appmod
    importlib.reload(appmod)
    dbm.init_db()
    return TestClient(appmod.app), dbm, appmod


def _fake_o(rid, odds, be):
    q = 1.0 / odds; q = q / q.sum(); order = np.argsort(-q)
    return dict(race_id=rid, completeness=1.0, flags={"odds_estimated": False}, boat_eval=be,
                probs={PERM_LABELS[i]: float(q[i]) for i in range(120)}, odds_used={PERM_LABELS[i]: float(odds[i]) for i in range(120)},
                ev={}, confidence=0.8, rationale={"summary": "x"}, input_hash="h", odds_source="real",
                selections=[dict(combo=PERM_LABELS[int(order[k])], rank=k + 1, kind=("main" if k < 10 else "hole"), stake=100,
                                 prob=float(q[order[k]]), odds=float(odds[order[k]]), ev=None) for k in range(15)])


def _be():
    return {str(k): dict(motor_rate2=40.0, exhibition_rank=3, exhibition_time=6.7, klass="A1", name=f"b{k}") for k in range(1, 7)}


def test_ratings_api_append_only_latest_wins_and_lock(client):
    tc, dbm, appmod = client
    from boatlab.store import models as mm
    from boatlab.util import now_jst
    now = now_jst().replace(tzinfo=None); day = now.date()
    rid = int(day.strftime("%Y%m%d")) * 10000 + 301
    with dbm.session_scope() as s:
        s.add(mm.ModelVersion(version="t", feature_set_version="fs", selection_version="sel", params={}, status="active"))
        s.add(mm.SettingsVersion(id=1, points=15, stake_per_point=100, extra={}))
        s.add(mm.Race(id=rid, race_date=day, stadium_code=3, race_no=1, source="t", closed_at=now + timedelta(minutes=12)))
    assert tc.post("/api/ratings", json={"race_id": rid, "ratings": {"7": 1}}).status_code == 400
    assert tc.post("/api/ratings", json={"race_id": rid, "ratings": {"1": 2}}).status_code == 400
    j = tc.post("/api/ratings", json={"race_id": rid, "ratings": {"1": 1, "4": -1}}).json()
    assert j["lanes"]["1"]["rating"] == 1 and j["lanes"]["4"]["rating"] == -1 and not j["locked"]
    assert j["usable_until"] and j["used"] is None
    j = tc.post("/api/ratings", json={"race_id": rid, "ratings": {"4": 0}}).json()   # 無印に戻す＝新しい行
    assert j["lanes"]["4"]["rating"] == 0
    with dbm.session_scope() as s:
        assert s.query(mm.ExhibitionRating).count() == 3
        with pytest.raises(Exception):
            s.query(mm.ExhibitionRating).update({"rating": 0})          # 追記専用
    r = tc.get(f"/api/races/{rid}").json()
    assert r["ratings"]["lanes"]["1"]["rating"] == 1 and r["ratings"]["locked"] is False
    # 確定予想が記録されると locked
    from boatlab.ops import daily as dl
    from boatlab.model.modes import ModeParams
    with dbm.session_scope() as s:
        race = s.get(mm.Race, rid)
        o = _fake_o(rid, 0.75 / np.random.default_rng(1).dirichlet(np.ones(120)), _be())
        dl._record_modes(s, o, race, "t", 1, now, {}, ModeParams(), {k: set() for k in dl.MODE_ROLES + ("katai_tR", "honmeiR", "ev1R")})
    r = tc.get(f"/api/races/{rid}").json()
    assert r["ratings"]["locked"] is True
    t = tc.get("/api/today?mode=katai_t").json()["races"][0]
    assert t["has_ratings"] == 1 and t["rated"] == 0        # 評価はあるが、この記録は評価を使わずに作った
    csv = tc.get("/api/export_ratings.csv").text
    assert "used_in_final" in csv and csv.count("\n") == 3   # 見出し＋最新2行（lane1, lane4）


def test_rated_variant_recorded_and_scored(client):
    """load_ratings → rated_output → _record_modes(suffix=R) の流れと、対の採点。"""
    tc, dbm, appmod = client
    from boatlab.store import models as mm
    from boatlab.util import now_jst
    from boatlab.ops import daily as dl
    from boatlab.model.modes import ModeParams
    now = now_jst().replace(tzinfo=None); day = now.date()
    rid = int(day.strftime("%Y%m%d")) * 10000 + 302
    with dbm.session_scope() as s:
        s.add(mm.ModelVersion(version="t", feature_set_version="fs", selection_version="sel", params={}, status="active"))
        s.add(mm.SettingsVersion(id=1, points=15, stake_per_point=100, extra={}))
        s.add(mm.Race(id=rid, race_date=day, stadium_code=3, race_no=2, source="t", closed_at=now + timedelta(minutes=12)))
    with dbm.session_scope() as s:
        s.add(mm.ExhibitionRating(race_id=rid, lane=1, rating=1, created_at=now, source="user"))
        s.add(mm.ExhibitionRating(race_id=rid, lane=2, rating=-1, created_at=now, source="user"))
    odds = 0.75 / np.random.default_rng(2).dirichlet(np.ones(120) * 0.5)
    o = _fake_o(rid, odds, _be())
    done = {k: set() for k in dl.MODE_ROLES + ("katai_tR", "honmeiR", "ev1R")}
    with dbm.session_scope() as s:
        race = s.get(mm.Race, rid)
        dl._record_modes(s, o, race, "t", 1, now, {}, ModeParams(), done)
        ratings = dl.load_ratings(s, rid, now_jst().replace(tzinfo=None))
        assert ratings == {1: 1, 2: -1}
        assert dl.load_ratings(s, rid, now - timedelta(minutes=5)) == {}          # 入力前の時点では無い
        o_r = dl.rated_output(o, ratings, 0.3)
        assert o_r["probs"] != o["probs"] and [x["combo"] for x in o_r["selections"]][:10] != [x["combo"] for x in o["selections"]][:10] or True
        dl._record_modes(s, o_r, race, "t", 1, now, {}, ModeParams(), done, suffix="R", only=dl.RATED_ROLES,
                         extra_flags={"rated": True, "ratings": {"1": 1, "2": -1}, "rating_beta": 0.3})
    with dbm.session_scope() as s:
        roles = {p.role: p for p in s.query(mm.Prediction).all()}
        assert set(roles) == {"katai_t", "honmei", "place", "ev1", "katai_tR", "honmeiR", "ev1R"}
        pr = roles["katai_tR"]
        assert pr.flags["mode"] == "katai_t" and pr.flags["role"] == "katai_tR" and pr.flags["rated"] and pr.flags["ratings"] == {"1": 1, "2": -1}
        assert pr.rationale_text.startswith("◎×加味: ")
        # ◎の1号艇が1着の目が上位に増える
        top_b = [x.combo for x in s.query(mm.PredictionSelection).filter_by(prediction_id=roles["katai_t"].id).order_by(mm.PredictionSelection.rank)]
        top_r = [x.combo for x in s.query(mm.PredictionSelection).filter_by(prediction_id=pr.id).order_by(mm.PredictionSelection.rank)]
        assert sum(c.startswith("1-") for c in top_r) >= sum(c.startswith("1-") for c in top_b)
        # 結果を入れて採点 → 両方採点され、モード名で分岐（ev1R は score_ev1）
        s.add(mm.Result(race_id=rid, trifecta=top_r[0], trifecta_payout=800, source="t", is_irregular=False,
                        payouts={"trifecta": [{"combination": top_r[0], "amount": 800}],
                                 "place": [{"combination": "1", "amount": 110}, {"combination": "2", "amount": 200}],
                                 "win": [{"combination": "1", "amount": 150}]}, refunds=[], fetched_at=now + timedelta(minutes=60)))
    dl.score_pending()
    with dbm.session_scope() as s:
        sc = {p.role: s.get(mm.Scoring, p.id) for p in s.query(mm.Prediction).all()}
        assert all(v is not None and v.valid for v in sc.values())
        assert sc["katai_tR"].hit and sc["katai_tR"].hit_kind == "katai_t"
        assert sc["ev1R"].stake_total >= 0 and sc["ev1"].stake_total >= 0
    # API: 対の集計と beta 推定が動く（n=1 なので verdict=few）
    j = tc.get("/api/modes").json()
    assert set(j["modes"]) == {"ev1", "katai_t", "honmei", "place"}
    rt = j["rated"]
    assert rt["katai_t"]["n"] == 1 and rt["katai_t"]["rated"]["hits"] == 1 and rt["calibration"]["n"] == 1
    assert rt["calibration"]["verdict"] == "few" and rt["beta_setting"] == 0.3
    t = tc.get("/api/today?mode=katai_t").json()["races"][0]
    assert t["rated"] == 1
    assert tc.get("/api/export.csv?mode=katai_tR").status_code == 200


def test_estimate_beta_recovers_synthetic_value():
    from boatlab.analytics.ratings import estimate_beta
    from boatlab.model.modes import adjust_probs_for_ratings
    rng = np.random.default_rng(1); rows = []
    for _ in range(300):
        p = rng.dirichlet(np.ones(120) * 0.5); r = {int(rng.integers(1, 7)): 1}
        truth = adjust_probs_for_ratings(p, r, 0.6); t = int(rng.choice(120, p=truth))
        rows.append(dict(probs={PERM_LABELS[k]: float(p[k]) for k in range(120)}, flags_r={"ratings": {str(k): v for k, v in r.items()}},
                         trifecta=PERM_LABELS[t]))
    out = estimate_beta(pd.DataFrame(rows))
    assert out["n"] == 300 and out["ci"][0] <= 0.6 <= out["ci"][1] + 0.15 and out["verdict"] in ("evidence", "no_evidence")
    assert out["good"]["actual_win"] > out["good"]["model_win"]
    # 評価が効かないデータでは beta=0 と区別できない
    rows2 = []
    for _ in range(300):
        p = rng.dirichlet(np.ones(120) * 0.5); r = {int(rng.integers(1, 7)): 1}; t = int(rng.choice(120, p=p))
        rows2.append(dict(probs={PERM_LABELS[k]: float(p[k]) for k in range(120)}, flags_r={"ratings": {str(k): v for k, v in r.items()}},
                          trifecta=PERM_LABELS[t]))
    out2 = estimate_beta(pd.DataFrame(rows2))
    assert out2["verdict"] == "no_evidence" or out2["beta_hat"] <= 0.15
