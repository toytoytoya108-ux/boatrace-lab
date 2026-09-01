import json
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import select, text

from boatlab.ingest.parsers import parse_v1_day
from boatlab.store import db as dbmod
from boatlab.store.models import Entry, ModelVersion, OddsSnapshot, Prediction, Race, Result, SettingsVersion
from boatlab.store.writer import write_bundle
from boatlab.util import now_jst

FX = Path(__file__).parent / "fixtures"


@pytest.fixture()
def db_url(tmp_path):
    url = f"sqlite:///{tmp_path / 't.db'}"
    dbmod.init_db(url)
    return url


def test_write_bundle_idempotent(db_url):
    doc = json.load(open(FX / "turnmark_20260501_s1.json", encoding="utf-8"))
    b = parse_v1_day(date(2026, 5, 1), doc, source_prefix="turnmark")
    with dbmod.session_scope(db_url) as s:
        c1 = write_bundle(s, b)
    with dbmod.session_scope(db_url) as s:
        write_bundle(s, b)  # 2回目は何も増えない
        assert s.scalar(select(text("count(*)")).select_from(Race)) == 12
        assert s.scalar(select(text("count(*)")).select_from(Entry)) == 72
        assert s.scalar(select(text("count(*)")).select_from(OddsSnapshot)) == c1["odds"]
        r = s.get(Result, b.results[0].race_id)
        assert r.trifecta == "4-3-1"
        race = s.get(Race, b.results[0].race_id)
        assert race.status == "finished"


def test_predictions_are_append_only(db_url):
    with dbmod.session_scope(db_url) as s:
        s.add(Race(id=1, race_date=date(2026, 1, 1), stadium_code=1, race_no=1, closed_at=datetime(2026, 1, 1, 12)))
        s.add(ModelVersion(version="0.0", feature_set_version="0", selection_version="0", params={}))
        s.add(SettingsVersion())
    common = dict(race_id=1, model_version="0.0", settings_id=1, stage="final", role="active",
                  asof_ts=datetime(2026, 1, 1, 11, 50), post_time_at_pred=datetime(2026, 1, 1, 12),
                  features_used={}, completeness=1.0, boat_eval={}, probs={}, odds_used={}, ev={},
                  confidence=0.5, expected_return=0.9, decision="skip", rationale={}, input_hash="x")
    with dbmod.session_scope(db_url) as s:
        s.add(Prediction(created_at=now_jst(), **common))
    eng = dbmod.get_engine(db_url)
    with pytest.raises(Exception, match="append-only"):
        with eng.begin() as conn:
            conn.execute(text("UPDATE predictions SET decision='buy'"))
    with pytest.raises(Exception, match="append-only"):
        with eng.begin() as conn:
            conn.execute(text("DELETE FROM predictions"))
    # created_at を偽装（過去時刻）した挿入は拒否される
    with pytest.raises(Exception, match="created_at"):
        with dbmod.session_scope(db_url) as s:
            s.add(Prediction(created_at=now_jst() - timedelta(hours=3), **dict(common, stage="program")))
    with dbmod.session_scope(db_url) as s:
        p = s.execute(select(Prediction)).scalar_one()
        assert p.decision == "skip"
