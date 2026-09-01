"""エンジン・セッション・初期化（テーブル作成＋追記専用トリガ）。"""
from __future__ import annotations

from contextlib import contextmanager

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from boatlab.config import DATABASE_URL, DATA_DIR, STADIUMS
from boatlab.store.models import Base, Stadium

_engine: Engine | None = None
_Session: sessionmaker | None = None


def _make_engine(url: str) -> Engine:
    if url.startswith("sqlite"):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        eng = create_engine(url, future=True)

        @event.listens_for(eng, "connect")
        def _pragma(dbapi_conn, _):
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA synchronous=NORMAL")
            cur.execute("PRAGMA foreign_keys=ON")
            cur.close()
        return eng
    return create_engine(url, future=True, pool_pre_ping=True)


def get_engine(url: str | None = None) -> Engine:
    """既定は BOATLAB_DATABASE_URL。url を渡すとそのDBに切り替える（テスト用）。"""
    global _engine, _Session
    if url is not None:
        if _engine is None or str(_engine.url) != url:
            _engine = _make_engine(url)
            _Session = sessionmaker(bind=_engine, expire_on_commit=False)
    elif _engine is None:
        _engine = _make_engine(DATABASE_URL)
        _Session = sessionmaker(bind=_engine, expire_on_commit=False)
    return _engine


@contextmanager
def session_scope(url: str | None = None):
    get_engine(url)
    s: Session = _Session()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()


# SQLite: 追記専用（UPDATE/DELETE 禁止）。created_at は「現在時刻±60秒」以外を拒否する
# （SQLite の BEFORE INSERT では NEW を書き換えられないため、検査で担保する）。
SQLITE_TRIGGERS = [
    """CREATE TRIGGER IF NOT EXISTS predictions_no_update BEFORE UPDATE ON predictions
       BEGIN SELECT RAISE(ABORT, 'predictions are append-only'); END;""",
    """CREATE TRIGGER IF NOT EXISTS predictions_no_delete BEFORE DELETE ON predictions
       BEGIN SELECT RAISE(ABORT, 'predictions are append-only'); END;""",
    """CREATE TRIGGER IF NOT EXISTS predictions_created_at_guard BEFORE INSERT ON predictions
       WHEN abs(strftime('%s', NEW.created_at) - strftime('%s', 'now', '+9 hours')) > 60
       BEGIN SELECT RAISE(ABORT, 'predictions.created_at must be now()'); END;""",
    """CREATE TRIGGER IF NOT EXISTS selections_no_update BEFORE UPDATE ON prediction_selections
       BEGIN SELECT RAISE(ABORT, 'prediction_selections are append-only'); END;""",
    """CREATE TRIGGER IF NOT EXISTS selections_no_delete BEFORE DELETE ON prediction_selections
       BEGIN SELECT RAISE(ABORT, 'prediction_selections are append-only'); END;""",
]

POSTGRES_TRIGGERS = """
CREATE OR REPLACE FUNCTION forbid_change() RETURNS trigger AS $$
BEGIN RAISE EXCEPTION 'append-only table'; END $$ LANGUAGE plpgsql;
DROP TRIGGER IF EXISTS predictions_immutable ON predictions;
CREATE TRIGGER predictions_immutable BEFORE UPDATE OR DELETE ON predictions
  FOR EACH ROW EXECUTE FUNCTION forbid_change();
DROP TRIGGER IF EXISTS selections_immutable ON prediction_selections;
CREATE TRIGGER selections_immutable BEFORE UPDATE OR DELETE ON prediction_selections
  FOR EACH ROW EXECUTE FUNCTION forbid_change();
CREATE OR REPLACE FUNCTION force_created_at() RETURNS trigger AS $$
BEGIN NEW.created_at := (now() AT TIME ZONE 'Asia/Tokyo'); RETURN NEW; END $$ LANGUAGE plpgsql;
DROP TRIGGER IF EXISTS predictions_created_at ON predictions;
CREATE TRIGGER predictions_created_at BEFORE INSERT ON predictions
  FOR EACH ROW EXECUTE FUNCTION force_created_at();
"""


def init_db(url: str | None = None) -> Engine:
    eng = get_engine(url)
    Base.metadata.create_all(eng)
    with eng.begin() as conn:
        if eng.dialect.name == "sqlite":
            for t in SQLITE_TRIGGERS:
                conn.execute(text(t))
        else:
            for stmt in [s for s in POSTGRES_TRIGGERS.split(";\n") if s.strip()]:
                conn.execute(text(stmt))
    with session_scope(url) as s:
        existing = {c for (c,) in s.execute(text("SELECT code FROM stadiums"))}
        for code, name in STADIUMS.items():
            if code not in existing:
                s.add(Stadium(code=code, name=name))
    return eng
